import os
import clickpoints
from glob import glob
from joblib import Parallel, delayed
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tifffile import imread
from tqdm import tqdm
import matplotlib.cm as cm
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects, binary_opening, binary_closing, disk
from scipy.ndimage import zoom


TRAINED_NETWORKS = {"NK_old": {"trained_file": "best_model_finding.pth", "training_pixelsize": 4.56},
                    "Treg": {"trained_file": "best_model_finding_treg.pth", "training_pixelsize": 3.45},
                    "Treg_trick": {"trained_file": "best_model_finding_nk92_ds4_new.pth", "training_pixelsize": 3.45},
                    "NK": {"trained_file": "best_model_finding_nk92_ds4_new.pth", "training_pixelsize": 3.45},
                    "Jurkat": {"trained_file": "best_model_finding_treg.pth", "training_pixelsize": 3.45}}

# ----------------------------
# Utilities
# ----------------------------
def upsample(img, H_orig, W_orig):
    """Upsample image to original size."""
    H, W = img.shape[:2]
    h2, w2 = H_orig, W_orig
    z_up = (h2 / H, w2 / W) if img.ndim == 2 else (h2 / H, w2 / W, 1.0)
    return zoom(img, z_up, order=1, grid_mode=True)


def pad_to_divisible(img, div=32):
    """Pad image to multiples of div in H/W dimensions."""
    c, h, w = img.shape
    pad_h = (div - h % div) % div
    pad_w = (div - w % div) % div
    return np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')


def jet_palette_256():
    cmap = cm.get_cmap('jet', 256)
    lut = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    lut[0] = [0, 0, 0]  # background = black
    return lut.reshape(-1).tolist()


JET_PAL = jet_palette_256()


# ----------------------------
# Model definition (same as training)
# ----------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        f = 16
        super().__init__()
        self.down1 = DoubleConv(6, f)  # 6 input channels from composite TIFF
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(f, 2 * f)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(2 * f, 4 * f)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(4 * f, 8 * f)
        self.pool4 = nn.MaxPool2d(2)
        self.down5 = DoubleConv(8 * f, 16 * f)
        self.pool5 = nn.MaxPool2d(2)

        self.middle = nn.Sequential(
            nn.Conv2d(16 * f, 32 * f, 3, padding=1),
            nn.GroupNorm(4, 32 * f),
            nn.ReLU(inplace=True),
            nn.Conv2d(32 * f, 32 * f, 3, padding=1),
            nn.GroupNorm(4, 32 * f),
            nn.ReLU(inplace=True),
        )

        self.up5 = nn.ConvTranspose2d(32 * f, 16 * f, 2, stride=2)
        self.conv5 = DoubleConv(32 * f, 16 * f)
        self.up4 = nn.ConvTranspose2d(16 * f, 8 * f, 2, stride=2)
        self.conv4 = DoubleConv(16 * f, 8 * f)
        self.up3 = nn.ConvTranspose2d(8 * f, 4 * f, 2, stride=2)
        self.conv3 = DoubleConv(8 * f, 4 * f)
        self.up2 = nn.ConvTranspose2d(4 * f, 2 * f, 2, stride=2)
        self.conv2 = DoubleConv(4 * f, 2 * f)
        self.up1 = nn.ConvTranspose2d(2 * f, f, 2, stride=2)
        self.conv1 = DoubleConv(2 * f, f)
        self.out = nn.Conv2d(f, 1, 1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        d4 = self.down4(p3)
        p4 = self.pool4(d4)
        d5 = self.down5(p4)
        p5 = self.pool5(d5)
        m = self.middle(p5)
        u5 = self.up5(m)
        x5 = self.conv5(torch.cat([u5, d5], dim=1))
        u4 = self.up4(x5)
        x4 = self.conv4(torch.cat([u4, d4], dim=1))
        u3 = self.up3(x4)
        x3 = self.conv3(torch.cat([u3, d3], dim=1))
        u2 = self.up2(x3)
        x2 = self.conv2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(x2)
        x1 = self.conv1(torch.cat([u1, d1], dim=1))
        return torch.sigmoid(self.out(x1))


# ----------------------------
# Main function
# ----------------------------

def _predict_one_composite(f, model, device, mask_dir):
    """Run the finder on a single composite TIFF and save its mask."""
    try:
        img_stack = imread(f).astype(np.float32) / 255.0  # shape (6, H, W)
        H = img_stack.shape[1]
        W = img_stack.shape[2]

        img_stack = pad_to_divisible(img_stack)
        img_tensor = torch.from_numpy(img_stack).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(img_tensor)
            pred_prob = pred[0, 0].cpu().numpy()

        pred_bin = pred_prob > 0.5
        pred_bin = ndi.binary_fill_holes(pred_bin)
        pred_bin = binary_opening(pred_bin, footprint=disk(1))
        pred_bin = binary_closing(pred_bin, footprint=disk(1))
        pred_bin = remove_small_objects(pred_bin, min_size=10)

        # Save mask (use Zmin channel = channel 4 as base)
        mask_img = (np.clip(img_stack[4] * pred_bin, 0, 1) * 255).astype(np.uint8)
        # Remove the extra rows/columns added by pad_to_divisible
        mask_img = mask_img[:H, :W]

        pal_img = Image.fromarray(mask_img, mode='P')
        pal_img.putpalette(JET_PAL)
        pal_img.save(os.path.join(mask_dir, os.path.basename(f)), compression='tiff_lzw')
    except Exception as e:
        print(f"Error processing {f}: {e}")


def _load_finder(finder_path, device):
    model = UNet().to(device)
    model.load_state_dict(torch.load(finder_path, map_location=device))
    model.eval()
    return model


def _predict_chunk(files, finder_path, mask_dir, threads):
    """Worker: load the model once and process a chunk of composites (CPU, limited threads)."""
    try:
        torch.set_num_threads(max(1, int(threads)))
    except Exception:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_finder(finder_path, device)
    for f in files:
        _predict_one_composite(f, model, device, mask_dir)
    return len(files)


def predict_cells_unet(pathlist, celltype, n_jobs=1):
    """
    Predict cells on composite TIFFs using the UNet finder model.
    Saves one mask per repetition/position in the /masks folder.

    ``n_jobs`` > 1 spreads the composites over that many worker processes (useful
    on CPU: detection is embarrassingly parallel per frame). The available torch
    threads are split across workers to avoid oversubscription. A CUDA GPU, when
    present, is used automatically and is fast enough that n_jobs=1 is fine.
    """
    import multiprocessing

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    finder_path = os.path.join(script_dir, TRAINED_NETWORKS[celltype]["trained_file"])

    parallel = (n_jobs and n_jobs > 1 and not torch.cuda.is_available())

    for item in pathlist:
        base_dir = item[0] if isinstance(item, (list, tuple)) else item
        composites_dir = os.path.join(base_dir, "composites")
        mask_dir = os.path.join(base_dir, "masks")
        os.makedirs(mask_dir, exist_ok=True)

        tiff_files = sorted(glob(os.path.join(composites_dir, "*.tif")))
        if not tiff_files:
            print(f"No composite TIFF files found in {composites_dir}")
            continue

        if parallel and len(tiff_files) > 1:
            nj = min(int(n_jobs), len(tiff_files))
            threads = max(1, multiprocessing.cpu_count() // nj)
            chunks = [tiff_files[i::nj] for i in range(nj)]  # round-robin split
            print(f"Detecting {len(tiff_files)} composites in {base_dir} "
                  f"with {nj} workers x {threads} threads")
            Parallel(n_jobs=nj)(
                delayed(_predict_chunk)(chunk, finder_path, mask_dir, threads)
                for chunk in chunks
            )
        else:
            model = _load_finder(finder_path, device)
            for f in tqdm(tiff_files, desc=f"Processing {base_dir}"):
                _predict_one_composite(f, model, device, mask_dir)


def write_masks_to_cdb(pathlist):
    # Imported lazily to avoid a circular import (utils -> tracking -> cell_finder
    # would otherwise pull in the preprocessing package before utils is ready).
    from ..preprocessing import borders as border_utils

    for item in pathlist:
        base_dir = item[0]
        cdb_files = sorted(glob(os.path.join(base_dir, '*.cdb')))

        for cdb_file in cdb_files:
            print("writing masks to cdb for file: ", cdb_file)

            # Border lines are drawn only in the 0h_corrected database of each
            # position. Look them up so detections outside the channel (walls,
            # debris) get clipped away. If none are found we keep the full frame.
            ref_path = border_utils.find_reference_cdb(cdb_file)
            if ref_path is not None and os.path.abspath(ref_path) != os.path.abspath(cdb_file):
                borders = border_utils.load_borders_from_path(ref_path)
            else:
                borders = None  # this cdb is (or has no) reference; read it below

            with clickpoints.DataFile(cdb_file) as cdb:
                if borders is None:
                    borders = border_utils.read_borders(cdb)
                if borders is None:
                    print(f"  no border lines found for {os.path.basename(cdb_file)} "
                          f"(ref: {ref_path}); keeping full frame")

                minproj_images = [x for x in cdb.getImages() if (x.layer.name == 'MinProj')]
                region_cache = {}

                for minproj_image in minproj_images:
                    mask_raw = plt.imread(os.path.join(base_dir, 'masks', minproj_image.filename))

                    H_orig, W_orig = minproj_image.data8.shape
                    mask = upsample((np.sum(mask_raw[:, :, :3], axis=2) > 0).astype(np.uint8), H_orig, W_orig)

                    if borders is not None:
                        key = (H_orig, W_orig)
                        if key not in region_cache:
                            region_cache[key] = border_utils.region_mask((H_orig, W_orig), borders)
                        mask = mask * region_cache[key]  # clip to between-borders region

                    cdb.setMask(image=minproj_image, data=mask.astype("uint8"))