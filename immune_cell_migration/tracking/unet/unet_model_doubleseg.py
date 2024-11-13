"""
Original provided by

 * @author [Zizhao Zhang]
 * @email [zizhao@cise.ufl.edu]
 * @create date 2017-05-25 02:21:13
 * @modify date 2017-05-25 02:21:13
 * @desc [description]
This script is used by the script playgroundUNET_doubleseg.py. It defines the model of the network. It uses the script
config.py for its parameters.
"""

# from tensorflow.keras import models
# from tensorflow.keras import layers
from keras import models
from keras import layers
from . import unet_config as conf


class UNet:
    def __init__(self):
        print('build UNet ...')

    def get_crop_shape(self, target, refer):
        # width, the 3rd dimension
        # cw = (target.get_shape()[2] - refer.get_shape()[2])#.value
        cw = (target.shape[2] - refer.shape[2])#.value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        # ch = (target.get_shape()[1] - refer.get_shape()[1])#.value
        ch = (target.shape[1] - refer.shape[1])#.value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)

    def create_model(self, img_shape, num_class, factor):
        factor1 = 1 * factor
        concat_axis = 3
        inputs = layers.Input(shape=img_shape)
        if conf.use_3d_network:
            time1 = layers.Conv3D(8, (1, 1, 1), activation="relu", padding="valid", name="time1")(inputs)
            reshape = layers.Reshape(img_shape[:-2] + (8 * 4,))(time1)
            conv1 = layers.Conv2D(int(32 * factor1), (3, 3), activation="relu", padding='same', name='conv1')(reshape)
        else:
            conv1 = layers.Conv2D(int(32 * factor1), (3, 3), activation="relu", padding='same', name='conv1')(inputs)

        conv1 = layers.Conv2D(int(32 * factor1), (3, 3), activation="relu", padding='same')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = layers.Conv2D(int(64*factor1), (3, 3), activation='relu', padding='same')(pool1)
        conv2 = layers.Dropout(0.2)(conv2)
        conv2 = layers.Conv2D(int(64*factor1), (3, 3), activation='relu', padding='same')(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = layers.Conv2D(int(128*factor1), (3, 3), activation='relu', padding='same')(pool2)
        conv3 = layers.Dropout(0.2)(conv3)
        conv3 = layers.Conv2D(int(128*factor1), (3, 3), activation='relu', padding='same')(conv3)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = layers.Conv2D(int(256*factor1), (3, 3), activation='relu', padding='same')(pool3)
        conv4 = layers.Dropout(0.2)(conv4)
        conv4 = layers.Conv2D(int(256*factor1), (3, 3), activation='relu', padding='same')(conv4)
        pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = layers.Conv2D(int(512*factor1), (3, 3), activation='relu', padding='same')(pool4)

        print(conv5.shape)
        drop1 = layers.Dropout(0.2)(conv5)
        conv5 = layers.Conv2D(int(512*factor1), (3, 3), activation='relu', padding='same')(drop1)

        up_conv5 = layers.UpSampling2D(size=(2, 2))(conv5)
        ch, cw = self.get_crop_shape(conv4, up_conv5)
        print(ch, cw)
        crop_conv4 = layers.Cropping2D(cropping=(ch, cw))(conv4)
        up6 = layers.concatenate([up_conv5, crop_conv4], axis=concat_axis)
        conv6 = layers.Conv2D(int(256*factor), (3, 3), activation='relu', padding='same')(up6)
        conv6 = layers.Dropout(0.2)(conv6)
        conv6 = layers.Conv2D(int(256*factor), (3, 3), activation='relu', padding='same')(conv6)

        up_conv6 = layers.UpSampling2D(size=(2, 2))(conv6)
        ch, cw = self.get_crop_shape(conv3, up_conv6)
        print(ch, cw)
        crop_conv3 = layers.Cropping2D(cropping=(ch, cw))(conv3)
        up7 = layers.concatenate([up_conv6, crop_conv3], axis=concat_axis)
        conv7 = layers.Conv2D(int(128*factor), (3, 3), activation='relu', padding='same')(up7)
        conv7 = layers.Dropout(0.2)(conv7)
        conv7 = layers.Conv2D(int(128*factor), (3, 3), activation='relu', padding='same')(conv7)

        up_conv7 = layers.UpSampling2D(size=(2, 2))(conv7)
        ch, cw = self.get_crop_shape(conv2, up_conv7)
        crop_conv2 = layers.Cropping2D(cropping=(ch, cw))(conv2)
        up8 = layers.concatenate([up_conv7, crop_conv2], axis=concat_axis)
        conv8 = layers.Conv2D(int(64*factor), (3, 3), activation='relu', padding='same')(up8)
        conv8 = layers.Dropout(0.2)(conv8)
        conv8 = layers.Conv2D(int(64*factor), (3, 3), activation='relu', padding='same')(conv8)

        up_conv8 = layers.UpSampling2D(size=(2, 2))(conv8)
        ch, cw = self.get_crop_shape(inputs, up_conv8)
        crop_conv1 = layers.Cropping2D(cropping=(ch, cw))(conv1)
        up9 = layers.concatenate([up_conv8, crop_conv1], axis=concat_axis)
        conv9 = layers.Conv2D(int(32*factor), (3, 3), activation='relu', padding='same')(up9)
        conv9 = layers.Dropout(0.2)(conv9)
        conv9 = layers.Conv2D(int(32*factor), (3, 3), activation='relu', padding='same')(conv9)

        ch, cw = self.get_crop_shape(inputs, conv9)
        up_conv9 = layers.ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv9)
        conv10 = layers.Conv2D(num_class, (1, 1), activation="sigmoid")(up_conv9)

        model = models.Model(inputs=inputs, outputs=conv10)
        return model


if __name__ == "__main__":
    Model = UNet().create_model((int(1024 / 2), int(1344 / 2), 4, 3), 7, 1)
    # Model = UNet().create_model((int(1024 / 2), int(1344 / 2), 5), 1, 1)
    # Model.load_weights("/home/lucas/Software/u-net-cell-detektion/sebweights.best.hdf5_lf21")
    Model.summary()
