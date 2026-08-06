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


def correct_drift_longterm(folder, pos, outfolder, subsample=10, file_patterns=DEFAULT_FILE_PATTERNS):
    """
    Drift correction for long continuous acquisitions (hundreds/thousands of frames).

    Unlike :func:`correct_drift`, this:
      * estimates drift on a subsample of frames (every ``subsample``-th, relative
        to frame 0) and then *interpolates* it to every frame, and
      * streams read -> shift -> save one frame at a time, keeping each frame's own
        filename, so memory stays flat and no frames are dropped or mislabeled.

    The saved ``drift_pos{pos}.pkl`` holds the full per-frame (y, x) drift, so it is
    compatible with the standard downstream stages.
    """
    print(f"Processing position {pos} (long-term, subsample={subsample})")
    os.makedirs(outfolder, exist_ok=True)

    # --- 1) estimate drift on the zMaxProj frames (subsampled) ---
    proj_pattern = next((p for p in file_patterns if "zMaxProj" in p), file_patterns[0])
    proj_files = natsort.natsorted(glob(os.path.join(folder, proj_pattern.format(pos))))
    if not proj_files:
        print(f"No files found for pattern: {proj_pattern.format(pos)}")
        return
    n = len(proj_files)
    sub_idx = _subsample_indices(n, subsample)

    sub_imgs = np.array([plt.imread(proj_files[i]) for i in sub_idx])
    sub_imgs2 = preprocess_images(sub_imgs)
    masks = calculate_masks(sub_imgs2)
    # drift of each subsampled frame relative to the first (== frame 0)
    sub_drift = [phase_cross_correlation(sub_imgs2[0], img2, reference_mask=masks[0], moving_mask=m)[0]
                 for img2, m in zip(sub_imgs2[1:], masks[1:])]
    sub_drift = np.vstack((np.array([[0., 0.]]), np.array(sub_drift)))
    drift = _interpolate_drift(np.array(sub_idx), sub_drift, n)
    joblib.dump(drift, os.path.join(outfolder, f"drift_pos{str(pos).zfill(2)}.pkl"))
    del sub_imgs, sub_imgs2, masks
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
