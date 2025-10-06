import os
import clickpoints
from glob import glob
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


TRAINED_NETWORKS = {"NK": {"trained_file": "best_model_finding.pth", "training_pixelsize": 4.56}}

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

def predict_cells_unet(pathlist, celltype):
    """
    Predict cells on composite TIFFs using the UNet finder model.
    Saves one mask per repetition/position in the /Mask folder.
    """
    # Auto-detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve model path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    finder_path = os.path.join(script_dir, TRAINED_NETWORKS[celltype]["trained_file"])

    # Load finder model
    model_finder = UNet().to(device)
    model_finder.load_state_dict(torch.load(finder_path, map_location=device))
    model_finder.eval()

    for item in pathlist:
        base_dir = item[0] if isinstance(item, (list, tuple)) else item
        composites_dir = os.path.join(base_dir, "composites")
        mask_dir = os.path.join(base_dir, "masks")
        os.makedirs(mask_dir, exist_ok=True)

        tiff_files = sorted(glob(os.path.join(composites_dir, "*.tif")))
        if not tiff_files:
            print(f"No composite TIFF files found in {composites_dir}")
            continue

        # Process each composite TIFF (each rep/pos)
        for f in tqdm(tiff_files, desc=f"Processing {base_dir}"):
            try:
                img_stack = imread(f).astype(np.float32) / 255.0  # shape (6, H, W)
                img_stack = pad_to_divisible(img_stack)
                img_tensor = torch.from_numpy(img_stack).unsqueeze(0).to(device)

                with torch.no_grad():
                    pred = model_finder(img_tensor)
                    pred_prob = pred[0, 0].cpu().numpy()

                pred_bin = pred_prob > 0.5
                pred_bin = ndi.binary_fill_holes(pred_bin)
                pred_bin = binary_opening(pred_bin, footprint=disk(1))
                pred_bin = binary_closing(pred_bin, footprint=disk(1))
                pred_bin = remove_small_objects(pred_bin, min_size=10)

                # Save mask (use Zmin channel = channel 4 as base)
                mask_img = (np.clip(img_stack[4] * pred_bin, 0, 1) * 255).astype(np.uint8)
                mask_out = os.path.join(mask_dir, os.path.basename(f))

                pal_img = Image.fromarray(mask_img, mode='P')
                pal_img.putpalette(JET_PAL)
                pal_img.save(mask_out, compression='tiff_lzw')

            except Exception as e:
                print(f"Error processing {f}: {e}")


def write_masks_to_cdb(pathlist):
    for item in pathlist:
        base_dir = item[0]
        cdb_files = sorted(glob(os.path.join(base_dir, '*.cdb')))

        for cdb_file in cdb_files:
            with clickpoints.DataFile(cdb_file) as cdb:
                minproj_images = [x for x in cdb.getImages() if (x.layer.name == 'MinProj')]

                for minproj_image in minproj_images:
                    mask_raw = plt.imread(os.path.join(base_dir, 'masks', minproj_image.filename))

                    H_orig, W_orig = minproj_image.data8.shape
                    mask = upsample((np.sum(mask_raw[:, :, :3], axis=2) > 0).astype(np.uint8), H_orig, W_orig)

                    cdb.setMask(image=minproj_image, data=mask.astype("uint8"))