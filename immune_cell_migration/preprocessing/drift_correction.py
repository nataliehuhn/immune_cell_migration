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


def correct_drift(folder, pos, outfolder, long_measurements=False, file_patterns=DEFAULT_FILE_PATTERNS):
    print(f"Processing position {pos}")

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

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
