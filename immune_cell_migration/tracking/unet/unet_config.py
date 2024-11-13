"""
This script contains all parameters which are necessary for training the neuronal network. If you want to train the
neuronal network you only have to change parameters within this script. The rest of the necessary scripts don't need to
be changed. (includes: playgroundUNET_doubleseg.py, DataLoader_doubleseg.py, ContrastMarker.py, Accuracies.py,
Callbacks.py, UNETmodel_doubleseg.py)
"""
# playgroundUNET
epochs = 200  # number of epochs to be trained
indeces = 12  # number of images per batch
train_batch = 1  # number of simultaneous train batches
val_batch = 1  # number of simultaneous validation batches
# if true, the last layer of the model will be removed and a new layer will be initialised
change_last_layer_of_model = True
# if you want to use pre trained weights, you can set the initial number of classes here (if change_last_layer_of_model
# = False, simply set both the initial_n_classes and n_classes to the same output class value you want)
initial_network_classes = 7
n_classes = 2  # number of classes
factor = 1  # factor to multiply the filters with
z_max = 75  # maximal z value
z_aug = False  # if true, use z augmentation (randomly increase/decrease the z-values of a picture as a whole)
augmentation = True  # if true apply the selected augmentations (only effects for flip, distort, zoom)
# (usually always true, unwanted augmentation methods can be turned off separately)
loss_function = 'categorical_crossentropy'  # function to calculate the loss with
flip_aug = True  # if true, use flipping the image as a method of augmentation
distort_aug = True  # if true, use distortion as a method of augmentation
zoom_aug = True  # if true, use zooming as a method of augmentation
brightness_aug = True  # if true, use brightness multiplication as a method of augmentation
x_grid_for_distort = 10  # pixel width for the distortion grid
y_grid_for_distort = int(10*0.76)  # pixel height for the distortion grid
magnitude_for_distort = 20  # magnitude of the distortion
images_per_batch_in_training_folder = 13  # number of images (+ mask) per frame, within the training folder
# subtract the median from the previous and next frame and adds the median from the current
adapt_brightness_for_multiple_frames = False  # was never used
load_pre_trained_weights = True  # if true, continue the training from pre trained weights
# name of the pre trained weights to continue the training from
pre_trained_weights = "tripleseg_with_gap_weights_1309.best.hdf5_lf1"
name_to_save_weights = "NK_cell_weights.hdf5_lf"  # name to save the best obtained weights
# if true, overwrites the weights after every epoch in a separate file compared to the best weights
save_weights_after_every_epoch = False
# name to save the weights after every epoch
name_to_save_weights_after_every_epoch = "tripleseg_with_gap_weights_1309_each.best.hdf5_lf"
# if true, use a 3d convolution in the first layer (currently only useful in combination with indices = 12)
use_3d_network = True
# if true, only use k562, nk and gaps as classes (non motile nk will be transferred to nk)
reduce_7_to_4_classes = False
# if true, only use k562 cells, dead k562 cells, nk and non motile nk will be transferred to k562 cells (the other
# classes will be removed) (make sure to only use one of the two variables reduce_7_to_4 ore reduce_7_to_2)
reduce_7_to_2_classes = False
# if true any double detections and overlapping ground truth cells will be weighed extra for the f1 score
extra_w_for_double_detections = True
# factor by which any double detections and overlapping ground truth cells will be weighed extra for the f1 score
factor_for_extra_w = 2
# lists the mask values which will be used to calculate the f1 score with (each number corresponds to an object type)
# (for example 1: K562 cells, 2: NK cells)
classes_to_calculate_f1_score_with = [1]

# Reading Images (the script DataLoader_doubleseg.py uses the following paths to read the images in)
train_data_path = r"E:\LucasScricpts\train_data"  # path to read the training images
val_data_path = r"E:\LucasScricpts\train_data"  # path to read the validation images
maxpro = "//frame%04d_MaxPro.tif"  # name for the current maximum projection
minpro = "//frame%04d_MinPro.tif"  # name for the current minimum projection
maxidx = "//frame%04d_MaxIdx.tif"  # name for the current maximum indices projection
minidx = "//frame%04d_MinIdx.tif"  # name for the current minimum indices projection
diffpro = "//frame%04d_DiffPro.tif"  # name for the drift projection (currently not used)
mask = "//frame%04d_Mask.tif"  # name for the mask
weight_map = "//frame%04d_Weight_map.tif"  # name for the weight map (currently not used)
prevmaxpro = "//frame%04d_prevMaxPro.tif"  # name for the previous maximum projection
prevminpro = "//frame%04d_prevMinPro.tif"  # name for the previous minimum projection
prevmaxidx = "//frame%04d_prevMaxIdx.tif"  # name for the previous maximum indices projection
prevminidx = "//frame%04d_prevMinIdx.tif"  # name for the previous minimum indices projection
nextmaxpro = "//frame%04d_nextMaxPro.tif"  # name for the next maximum projection
nextminpro = "//frame%04d_nextMinPro.tif"  # name for the next minimum projection
nextmaxidx = "//frame%04d_nextMaxIdx.tif"  # name for the next maximum indices projection
nextminidx = "//frame%04d_nextMinIdx.tif"  # name for the next minimum indices projection

# Data Loader
width = 672  # width of the random window to be cut from an image
height = 512  # height of the random window to be cut from an image
# if true, the validation images will be split in 4 images from the corners, all with the defined size
# if false, one random crop with the defined size will be used per image for the validation data
use_whole_images_in_4_parts_for_validation = True
# the following probabilities only take effect, if their augmentation is set to true
# can be done in the script playgroundUNET_doubleseg.py while choosing the parameter list
flip_left_right_prob = 0.5  # probability for left_right flip
flip_top_bottom_prob = 0.5  # probability for top_bottom flip
random_distortion_prob = 0.5  # probability for distortion
zoom_random_prob = 0.5  # probability for zoom
zoom_random_lower_limit = 0.7  # lower limit for zoom
# if true, all image will be loaded to the ram simultaneously instead of loading only the current images for every step
load_all_images_in_ram = True

# Contrast maker
brightness_max = 2**12  # maximum value for minimum/maximum projections (used for norming)
index_max = 255  # maximum value for minimum/maximum indices projections (used for norming)
min_measured = 350  # lowest value measured (shouldn't change)
