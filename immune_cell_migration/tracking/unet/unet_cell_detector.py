"""
This script takes folders with clickpoints databases within them. For every clickpoints database it applies a pre
trained network to create a mask segmenting the cell positions for every frame. It writes the created mask to the
clickpoints database.
"""
import numpy as np
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
# import tensorflow.compat.v1.keras.backend as K
# K.get_session(session)
from tensorflow.python.keras.backend import get_session
get_session(session)
from ... import utils
import tqdm
import cv2
import time
from . import unet_config as conf
from .unet_model_doubleseg import UNet


class CellDetector:
    def __init__(self, pre_trained_weights_name, zoom_factor=1, indices=12, factor=1, n_classes=2):
        # n_classes: number of output classes
        # indices: number of images per batch
        # factor: to multiply the number of filters with (must be the same factor the network was trained with) (usually 1)
        self.zoom_factor = zoom_factor
        self.indices = indices
        self.factor = factor
        self.n_classes = n_classes

        self.mask_types = ["NK", "K562", "Gaps", "Non motile NK", "Dead K562", "Rubish"]
        # colors for the different classes (blue, red, green, orange, dark goldenrod, purple)
        self.mask_colors = ["#0000ff", "#ff0000", "#b5f5c4", "#ff4500", "#b8860b", "#800080"]
        self.adapt_brightness_for_multiple_frames = False

        self.pre_trained_weights_name = pre_trained_weights_name
        self.model = None

        self.cached_frames = {}

    def initialize_model(self, image_shape):
        self.original_image_shape = image_shape

        # if true, apply the given zoom factor
        if self.zoom_factor != 1:
            image_shape = (int(image_shape[1] * self.zoom_factor) + 16 - int(image_shape[1] * self.zoom_factor) % 16,
                           int(image_shape[0] * self.zoom_factor) + 16 - int(image_shape[0] * self.zoom_factor) % 16)

        img_shape = (image_shape[0], image_shape[1], self.indices)  # y, x, indices
        self.y_pix_to_remove = int(((img_shape[0] % 16) + 1) / 2)  # corner pixel to remove
        self.x_pix_to_remove = int(((img_shape[1] % 16) + 1) / 2)  # corner pixel to remove

        print("image_shape", self.original_image_shape, image_shape, img_shape)

        print("Will remove %d x, %d y pixel from the edges" % (self.x_pix_to_remove, self.y_pix_to_remove))
        if self.indices == 12:  # check for the number of images per batch
            img_shape = (img_shape[0], img_shape[1], 4, 3)  # reshape the image to fit for the 3D network

        self.network_shape = img_shape

        self.model = UNet().create_model(img_shape, self.n_classes, self.factor)  # create the network
        self.model.load_weights(self.pre_trained_weights_name)  # load the given weights

    def get_imgs_of_frame(self, db, frame):
        if frame not in self.cached_frames:
            xdata = np.zeros([1, self.original_image_shape[0], self.original_image_shape[1], 4])  # batch, y, x, indices
            xdata[0, :, :, 0] = db.getImages(frame=frame, layer="MaxProj")[0].data
            xdata[0, :, :, 1] = db.getImages(frame=frame, layer="MinProj")[0].data
            xdata[0, :, :, 2] = db.getImages(frame=frame, layer="MaxIndices")[0].data
            xdata[0, :, :, 3] = db.getImages(frame=frame, layer="MinIndices")[0].data
            # take the whole images
            if self.zoom_factor != 1:  # if true, resize the images according to the defined zoom factor
                xdata = cv2.resize(xdata[0], (int(self.original_image_shape[0] * self.zoom_factor) + 16 - int(self.original_image_shape[0] * self.zoom_factor) % 16,
                                              int(self.original_image_shape[1] * self.zoom_factor) + 16 - int(self.original_image_shape[1] * self.zoom_factor) % 16))
                xdata = np.expand_dims(xdata, axis=0)  # add another dimension for the batch after resizing

            self.cached_frames[frame] = xdata.astype('float32')
            if frame-3 in self.cached_frames:
                del self.cached_frames[frame-3]
        return self.cached_frames[frame]

    def load_new_data(self, cdb_pol, frame):
        xdata = np.zeros([1, self.network_shape[0], self.network_shape[1], self.indices])  # batch, y, x, indices

        xdata[0, :, :, 0:4] = self.get_imgs_of_frame(cdb_pol, frame)

        if self.indices == 6:  # if true, load the 4 current and the previous and next maximum projection image
            xdata[0, :, :, 5] = self.get_imgs_of_frame(cdb_pol, frame-1)[..., 0]
            xdata[0, :, :, 6] = self.get_imgs_of_frame(cdb_pol, frame+1)[..., 0]

        elif self.indices == 12:  # if true, load the previous, current and next 4 images
            xdata[0, :, :, 4:8] = self.get_imgs_of_frame(cdb_pol, frame-1)

            xdata[0, :, :, 8:12] = self.get_imgs_of_frame(cdb_pol, frame+1)

        utils.norm(xdata)  # norm the images with the maximal possible value provided in the config script
        if self.indices == 6 or self.indices == 12:  # check for multiple frames
            # if true, equalize the median for the previous, current and next images
            if self.adapt_brightness_for_multiple_frames:
                utils.adapt_brightness_for_multi_frames(xdata)
            if self.indices == 12:  # if true, reshape the image data to fit for the 3D network
                xdata = xdata.reshape(xdata.shape[0], xdata.shape[1], xdata.shape[2], 4, 3, order="F")
        return xdata

    def rescale_mask(self, mask_edit):
        if self.y_pix_to_remove > 0:
            mask_edit[:self.y_pix_to_remove, :] = 0
            mask_edit[-self.y_pix_to_remove:, :] = 0
        if self.x_pix_to_remove > 0:
            mask_edit[:, :self.x_pix_to_remove] = 0
            mask_edit[:, -self.x_pix_to_remove:] = 0

        if self.zoom_factor != 1:  # if true, apply the given zoom factor
            # resize the mask, to fit in the images in the clickpoints database
            mask_edit = cv2.resize(mask_edit.astype("float64"),
                                   (self.original_image_shape[1], self.original_image_shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
        return mask_edit

    def initialize_mask_types(self, cdb):
        try:  # add all the needed mask types
            cdb.deleteMaskTypes(name="Ground Truth")
            for j, name in enumerate(self.mask_types):
                cdb.setMaskType(name=name, color=self.mask_colors[j], index=j+1)
        except:
            pass

    def set_masks(self, cdb):
        time_stamp1 = time.time()  # save the current time

        self.cached_frames = {}

        if self.model is None:
            # obtain the first image
            image = cdb.getImage(frame=0, layer=1).data.astype('float')
            # and initialize the model with the shape
            self.initialize_model(image.shape)

        # add the mask types to the clickpoints database
        self.initialize_mask_types(cdb)

        # iterate over all images (besides the first and the last)
        for i in tqdm.tqdm(range(1, cdb.getImageCount() - 1)):
            # load the current data
            Xdata = self.load_new_data(cdb, i)

            # predict the mask based on the current data (and convert it from a one-hot encoding to a number)
            mask = np.argmax(self.model.predict(Xdata).squeeze(), axis=2)

            # rescale the mask to fit the original image
            mask = self.rescale_mask(mask)

            # write the mask to the clickpoints database
            cdb.setMask(image=cdb.getImage(frame=i, layer="MinProj"), data=mask.astype("uint8"))

        time_stamp2 = time.time()  # save the current time
        print("Time in s:", time_stamp2 - time_stamp1)  # print the needed time to predict the current database
