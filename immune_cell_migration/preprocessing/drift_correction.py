import os
import gc
import joblib
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import natsort

from glob import glob
from scipy.ndimage import shift
from scipy.ndimage import gaussian_filter
from skimage.registration import phase_cross_correlation


DEFAULT_FILE_PATTERNS = [
        '*rep*_pos{:02d}_x00_y00_mode0_zMaxProj.tif',
        '*rep*_pos{:02d}_x00_y00_mode0_zMinProj.tif',
        '*rep*_pos{:02d}_x00_y00_mode0_zMaxIndices*.tif',
        '*rep*_pos{:02d}_x00_y00_mode0_zMinIndices*.tif'
    ]


def read_images(filenames, step=1):
    return np.array([plt.imread(f) for f in filenames[::step]])


def preprocess_images(imgs):
    return np.array([img.astype(float) / gaussian_filter(img, 400, truncate=0.5) for img in imgs])


def calculate_masks(imgs2):
    return np.array([(img > np.percentile(img, 20)) for img in imgs2])


def calculate_drift(imgs2, masks):
    drift = [phase_cross_correlation(imgs2[0], img2, reference_mask=masks[0], moving_mask=mask)[0]
             for img2, mask in zip(imgs2[1:], masks[1:])]
    return np.array(drift)


def apply_shift(imgs, drift):
    # Fill the region exposed by the shift with the image mean, which for these
    # projections equals the bright background (~95), so the shifted-in border
    # blends into the background and stays visually invisible.
    return np.array([shift(img, drift_val, order=0, cval=int(np.mean(img))) for img, drift_val in zip(imgs, drift)])


def save_shifted_images(filenames, shifted, outfolder):
    for filename, img in zip(filenames, shifted):
        tifffile.imwrite(os.path.join(outfolder, os.path.basename(filename)), img)


def _subsample_indices(n, subsample):
    """Frame indices used to *estimate* drift: every ``subsample``-th, plus the
    first and last frame so the interpolation spans the whole measurement."""
    idx = list(range(0, n, max(1, int(subsample))))
    if idx[0] != 0:
        idx.insert(0, 0)
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return idx


def _interpolate_drift(sub_idx, sub_drift, n):
    """Linearly interpolate the (y, x) drift measured at ``sub_idx`` to all n frames."""
    all_idx = np.arange(n)
    dy = np.interp(all_idx, sub_idx, sub_drift[:, 0])
    dx = np.interp(all_idx, sub_idx, sub_drift[:, 1])
    return np.stack([dy, dx], axis=1)


def _prep_one(img):
    """Flat-field normalize a single image (matches preprocess_images)."""
    img = img.astype(float)
    return img / gaussian_filter(img, 400, truncate=0.5)


def _prep_masked(img, downsample):
    """Downsample + flat-field normalize + foreground mask for registration."""
    small = img[::downsample, ::downsample]
    norm = _prep_one(small)
    return norm, norm > np.percentile(norm, 20)


def _estimate_drift_sequential(proj_files, handover=10, downsample=4):
    """Cumulative drift measured for EVERY frame by chaining registrations.

    Every frame is registered (never interpolated) against a nearby reference
    frame, and the shifts are accumulated ("handed over") into the drift relative
    to frame 0. The reference re-anchors every ``handover`` frames, which keeps
    each measured shift small yet still large enough to resolve - with a reference
    that is too close the inter-frame shift is sub-pixel and would be lost.

    Registration uses the masked phase correlation (needed for accuracy on these
    images - the unmasked variant locks onto the static background and reports no
    shift) on ``downsample``-reduced images, which is ~20x faster at the same
    accuracy. Only two frames are held in memory at once.
    """
    n = len(proj_files)
    drift = np.zeros((n, 2))
    ref, ref_mask = _prep_masked(plt.imread(proj_files[0]), downsample)
    ref_base = np.zeros(2)      # cumulative drift of the current reference frame
    ref_idx = 0
    for i in range(1, n):
        cur, cur_mask = _prep_masked(plt.imread(proj_files[i]), downsample)
        step = phase_cross_correlation(ref, cur, reference_mask=ref_mask,
                                       moving_mask=cur_mask)[0] * downsample
        drift[i] = ref_base + step
        if (i - ref_idx) >= max(1, int(handover)):     # hand the reference over
            ref, ref_mask, ref_base, ref_idx = cur, cur_mask, drift[i], i
        del cur, cur_mask
    return drift


def _estimate_drift_interpolated(proj_files, subsample):
    """Old behavior: register a subsample of frames to frame 0, interpolate to all."""
    n = len(proj_files)
    sub_idx = _subsample_indices(n, subsample)
    sub_imgs = np.array([plt.imread(proj_files[i]) for i in sub_idx])
    sub_imgs2 = preprocess_images(sub_imgs)
    masks = calculate_masks(sub_imgs2)
    sub_drift = [phase_cross_correlation(sub_imgs2[0], img2, reference_mask=masks[0], moving_mask=m)[0]
                 for img2, m in zip(sub_imgs2[1:], masks[1:])]
    sub_drift = np.vstack((np.array([[0., 0.]]), np.array(sub_drift)))
    drift = _interpolate_drift(np.array(sub_idx), sub_drift, n)
    del sub_imgs, sub_imgs2, masks
    gc.collect()
    return drift


def correct_drift_longterm(folder, pos, outfolder, subsample=10, file_patterns=DEFAULT_FILE_PATTERNS,
                           sequential=True, handover=10, reg_downsample=4):
    """
    Drift correction for long continuous acquisitions (hundreds/thousands of frames).

    Unlike :func:`correct_drift`, this streams read -> shift -> save one frame at a
    time (flat memory, no dropped/mislabeled frames). Drift is estimated either:
      * ``sequential=True`` (default): from EVERY frame by chaining consecutive
        registrations with a ``handover`` re-anchor (robust; catches real jumps), or
      * ``sequential=False``: on every ``subsample``-th frame relative to frame 0,
        then interpolated (faster, assumes smooth drift).

    The saved ``drift_pos{pos}.pkl`` holds the full per-frame (y, x) drift.
    """
    mode = f"sequential (handover={handover})" if sequential else f"interpolate (subsample={subsample})"
    print(f"Processing position {pos} (long-term, {mode})")
    os.makedirs(outfolder, exist_ok=True)

    # --- 1) estimate drift on the zMaxProj frames ---
    proj_pattern = next((p for p in file_patterns if "zMaxProj" in p), file_patterns[0])
    proj_files = natsort.natsorted(glob(os.path.join(folder, proj_pattern.format(pos))))
    if not proj_files:
        print(f"No files found for pattern: {proj_pattern.format(pos)}")
        return
    n = len(proj_files)
    if sequential:
        drift = _estimate_drift_sequential(proj_files, handover, reg_downsample)
    else:
        drift = _estimate_drift_interpolated(proj_files, subsample)
    joblib.dump(drift, os.path.join(outfolder, f"drift_pos{str(pos).zfill(2)}.pkl"))
    gc.collect()

    # --- 2) apply drift to every frame of every pattern, streaming ---
    for pattern in file_patterns:
        filenames = natsort.natsorted(glob(os.path.join(folder, pattern.format(pos))))
        if not filenames:
            print(f"No files found for pattern: {pattern.format(pos)}")
            continue
        if len(filenames) != n:
            print(f"WARNING: {pattern.format(pos)} has {len(filenames)} frames, expected {n}; "
                  f"drift applied by index up to the shorter length")
        for i, fname in enumerate(filenames[:n]):
            img = plt.imread(fname)
            # cval = image mean (~background), so the exposed border blends in
            shifted = shift(img, drift[i], order=0, cval=int(np.mean(img)))
            tifffile.imwrite(os.path.join(outfolder, os.path.basename(fname)), shifted)
            del img, shifted
        gc.collect()


def correct_drift(folder, pos, outfolder, long_measurements=False, file_patterns=DEFAULT_FILE_PATTERNS):
    print(f"Processing position {pos}")

    try:
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)
    except FileExistsError:
        pass

    for pattern in file_patterns:
        filenames = natsort.natsorted(glob(os.path.join(folder, pattern.format(pos))))
        if not filenames:
            print(f"No files found for pattern: {pattern.format(pos)}")
            continue

        step = 10 if long_measurements else 1
        imgs = read_images(filenames, step)

        if "zMaxProj" in pattern:
            imgs2 = preprocess_images(imgs)
            masks = calculate_masks(imgs2)
            drift = calculate_drift(imgs2, masks)
            drift = np.vstack((np.array([[0., 0.]]), drift))

            joblib.dump(drift, os.path.join(outfolder, f'drift_pos{str(pos).zfill(2)}.pkl'))

        else:
            drift = joblib.load(os.path.join(outfolder, f'drift_pos{str(pos).zfill(2)}.pkl'))

        shifted = apply_shift(imgs, drift)
        save_shifted_images(filenames, shifted, outfolder)

        del imgs
        if "zMaxProj" in pattern:
            del imgs2
        gc.collect()
