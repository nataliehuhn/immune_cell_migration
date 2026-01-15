import os
import re
import numpy as np
from PIL import Image
from tifffile import imwrite
from scipy.ndimage import zoom
from collections import defaultdict


def downsample(img, factor):
    """Downsample image by given factor."""
    return zoom(img, zoom=1 / factor, order=1)


def normalize_to_uint8(img):
    """Normalize image to uint8 [0, 255]."""
    img = img.astype(np.float32)
    img -= img.min()
    img /= (img.max() + 1e-8)
    return (img * 255).astype(np.uint8)


def get_triplet_flexible(t, spacing, min_rep, max_rep):
    """Return [t_m, t, t_p] with adjusted spacing if at edges."""
    t_m = max(t - spacing, min_rep)
    t_p = min(t + spacing, max_rep)
    return [t_m, t, t_p]


def prepare_unet_input(pathlist, rep_spacing=3, downsampling_factor=3):
    """
    Prepares composite TIFFs for UNET training/prediction.

    Parameters
    ----------
    pathlist : list of [str, dict]
        List of measurement folders in the format returned by name_glob: [folder_path, extra_dict].
    rep_spacing : int, default=3
        Temporal spacing (in frames) for computing motion images.
    downsampling_factor : int, default=3
        Factor by which images are downsampled.
    """
    for item in pathlist:
        # extract the folder string
        if isinstance(item, (list, tuple)):
            base_dir = item[0]
        else:
            base_dir = item

        output_dir = os.path.join(base_dir, "composites")
        os.makedirs(output_dir, exist_ok=True)

        # --- File indexing ---
        file_index = defaultdict(lambda: defaultdict(str))
        required_patterns = {
            "Imin": "_zMinProj",
            "Imax": "_zMaxProj",
            "Zmin": "_zMinIndices",
            "Zmax": "_zMaxIndices",
        }

        for fname in sorted(os.listdir(base_dir)):
            if not fname.endswith(".tif") or "rep" not in fname:
                continue

            match = re.search(r"rep(\d+)", fname)
            if not match:
                continue
            rep = int(match.group(1))

            for key, pattern in required_patterns.items():
                if pattern in fname:
                    file_index[rep][key] = os.path.join(base_dir, fname)
                    break

        if not file_index:
            print(f"No valid TIFF files found in {base_dir}")
            continue

        available_reps = sorted(file_index.keys())
        min_rep = min(available_reps)
        max_rep = max(available_reps)

        # --- Loop through all reps ---
        for rep_center in available_reps:
            rep_tm, rep_t0, rep_tp = get_triplet_flexible(rep_center, rep_spacing, min_rep, max_rep)

            try:
                # Load center time point
                Imin_t0 = np.array(Image.open(file_index[rep_t0]["Imin"]))
                Imax_t0 = np.array(Image.open(file_index[rep_t0]["Imax"]))
                Zmin_t0 = np.array(Image.open(file_index[rep_t0]["Zmin"]))
                Zmax_t0 = np.array(Image.open(file_index[rep_t0]["Zmax"]))

                # Load motion frames
                Imin_tm = np.array(Image.open(file_index[rep_tm]["Imin"]))
                Imin_tp = np.array(Image.open(file_index[rep_tp]["Imin"]))
                Imax_tm = np.array(Image.open(file_index[rep_tm]["Imax"]))
                Imax_tp = np.array(Image.open(file_index[rep_tp]["Imax"]))

                # Compute differences
                Imin_diff = (Imin_tp.astype(np.float32) - Imin_tm.astype(np.float32))
                Imax_diff = (Imax_tp.astype(np.float32) - Imax_tm.astype(np.float32))
                Zmin = Zmin_t0.astype(np.float32)
                Zmax = Zmax_t0.astype(np.float32)

                # Downsample + normalize
                ch1 = normalize_to_uint8(downsample(Imin_t0, downsampling_factor))
                ch2 = normalize_to_uint8(downsample(Imax_t0, downsampling_factor))
                ch3 = normalize_to_uint8(downsample(Imin_diff, downsampling_factor))
                ch4 = normalize_to_uint8(downsample(Imax_diff, downsampling_factor))
                ch5 = normalize_to_uint8(downsample(Zmin, downsampling_factor))
                ch6 = normalize_to_uint8(downsample(Zmax, downsampling_factor))

                # Stack and save
                multi_channel = np.stack([ch1, ch2, ch3, ch4, ch5, ch6], axis=0)
                base_out = os.path.basename(file_index[rep_t0]["Imin"]).split("_rep")[0]
                out_name = f"{base_out}_rep{rep_t0:04d}.tiff"
                out_path = os.path.join(output_dir, out_name)
                imwrite(out_path, multi_channel, photometric="minisblack")
                print(f"Saved: {out_path}")

            except Exception as e:
                print(f"Failed at rep{rep_center:04d} in {base_dir}: {e}")
                continue