import re
import os
import numpy as np
from .tracking.unet import unet_config as conf


def get_subdirectories(start):
    directories = [start]
    for root, dirs, files in os.walk(start):
        directories.extend([os.path.join(root, dir) for dir in dirs])
    return directories


def get_files(start):
    all_files = []
    for root, dirs, files in os.walk(start):
        all_files.extend([os.path.join(root, file) for file in files])
    return all_files


def get_regex_groups(regex, text):
    import re
    match = re.match(regex, text)
    data = []
    if match:
        group = match.groups()
        for v in group:
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
            data.append(v)
        if len(data) == 1:
            return data[0]
        return data
    return None


def name_glob(pattern):
    index1 = pattern.find("{")
    index2 = pattern.find("*")
    if index1 == -1:
        index = index2
    elif index2 == -1:
        index = index1
    else:
        index = min([index1, index2])
    if index != -1:
        start, _ = os.path.split(pattern[:index])
    else:
        start = pattern
    regex = pattern[:]
    regex = re.sub("\*", "[^\/\\]*", regex)
    regex = regex.replace("[^\/\\]*[^\/\\]*", ".*")
    regex = re.sub("{([^}]*)}", "(?P<\\1>[^\/\\]*)", regex)

    regex = regex.replace("\\", "\\\\")
    regex += "$"
    directories = get_subdirectories(start)
    result = []
    for dir in directories:
        match = re.match(regex, dir)
        if match:
            properties = match.groupdict()
            result.append([dir, properties])
    return result


def name_glob_files(pattern):
    pattern = os.path.abspath(pattern)
    start, _ = os.path.split(pattern[:pattern.find("{")])
    regex = pattern[:]
    regex = re.sub("\*", "[^\/\\]*", regex)
    regex = regex.replace("[^\/\\]*[^\/\\]*", ".*")
    regex = re.sub("{([^}]*)}", "(?P<\\1>[^\/\\]*)", regex)

    regex = regex.replace("\\", "\\\\")
    regex += "$"
    files = get_files(start)
    result = []
    for file in files:
        match = re.match(regex, file)
        if match:
            properties = match.groupdict()
            result.append([file, properties])
    return result


def get_value(string, pattern):
    start, _ = os.path.split(pattern[:pattern.find("{")])
    regex = pattern[:]
    regex = re.sub("\*", "[^\/\\]*", regex)
    regex = regex.replace("[^\/\\]*[^\/\\]*", ".*")
    regex = re.sub("{([^}]*)}", "(?P<\\1>[^\/\\]*)", regex)

    regex = regex.replace("\\", "\\\\")
    regex += "$"
    match = re.match(regex, string)
    if match:
        properties = match.groupdict()
        return properties
    return {}



def norm(img, cap=conf.brightness_max, index_cap=conf.index_max):
    """
    Normalizes all the pictures with the cap
    :param img: (array) contains images (batch size, width, height, projection)
    :param cap: (int) maximal value for the maximum/minimum projection
    :param index_cap: (int) maximal value for the indices projections
    :return: Normalized images
    """
    for i in range(img.shape[0]):  # iterate over all batches
        img[i, :, :, 0] /= cap  # normalize the maximum projection
        img[i, :, :, 1] /= cap  # normalize the minimum projection
        # depending on the number of input images, normalize the maximum/minimum projections with the brightness cap and
        # the maximum/minimum indices projections with the index cap
        if img.shape[3] == 3:
            img[i, :, :, 2] /= cap
        elif img.shape[3] == 4:
            img[i, :, :, 2] /= index_cap
            img[i, :, :, 3] /= index_cap
        elif img.shape[3] == 5:
            img[i, :, :, 2] /= index_cap
            img[i, :, :, 3] /= index_cap
            img[i, :, :, 4] /= cap
        elif img.shape[3] == 6:
            img[i, :, :, 2] /= index_cap
            img[i, :, :, 3] /= index_cap
            img[i, :, :, 4] /= cap
            img[i, :, :, 5] /= cap
        elif img.shape[3] == 12:
            img[i, :, :, 2] /= index_cap
            img[i, :, :, 3] /= index_cap
            img[i, :, :, 4] /= cap
            img[i, :, :, 5] /= cap
            img[i, :, :, 6] /= index_cap
            img[i, :, :, 7] /= index_cap
            img[i, :, :, 8] /= cap
            img[i, :, :, 9] /= cap
            img[i, :, :, 10] /= index_cap
            img[i, :, :, 11] /= index_cap


def multiplic_brightness(img, normed=True, cap=conf.brightness_max, minimal_measured_value=conf.min_measured):
    """
    Augment the images by multiplying them with a random factor between to borders. These borders are defined in a
    way, so that the multiplied images can't go bigger than 1 or smaller than the defined minimal value
    :param img: (array) contains images (batch size, width, height, projection)
    :param normed: (bool) necessary to apply the correct cap
    :param cap: (int) maximal value for the maximum/minimum projection
    :param minimal_measured_value: (int) minimal ever measured value
    :return: Brightness augmented images
    """
    for i in range(img.shape[0]):  # iterate over all batches
        maxv = np.amax(img[i, :, :, 0])  # obtain the maximal value in the maximum projection
        minv = np.amin(img[i, :, :, 1])  # obtain the minimal value in the minimum projection
        if normed:  # if yes, the max cap will be 1 and the minimal cap will be divided by the brightness cap
            measured_minval = minimal_measured_value/cap  # minimal measured value
            max_multiplic = 1 / maxv  # maximal multiplication factor
        else:  # if no, the max cap will be the brightness cap
            measured_minval = minimal_measured_value
            max_multiplic = cap / maxv
        min_multiplic = measured_minval / minv  # minimal multiplication factor
        # choose a random multiplication factor between two borders
        multi = np.random.uniform(min_multiplic, max_multiplic)
        img[i, :, :, 0] *= multi  # multiply the maximum projection with the random factor
        img[i, :, :, 1] *= multi  # multiply the minimum projection with the random factor
        # depending on the number of input images, multiply the maximum/minimum projections with the random factor
        if img.shape[3] == 3 or img.shape[3] == 5:
            img[i, :, :, 2] *= multi
        if img.shape[3] == 5:
            img[i, :, :, 4] *= multi
        if img.shape[3] == 6:
            img[i, :, :, 4] *= multi
            img[i, :, :, 5] *= multi
        elif img.shape[3] == 12:
            img[i, :, :, 4] *= multi
            img[i, :, :, 5] *= multi
            img[i, :, :, 8] *= multi
            img[i, :, :, 9] *= multi


def adapt_brightness_for_multi_frames(img):
    """
    Adjust the median for different time steps, to be the same (only maximum/minimum projections)
    :param img: (array) contains images (batch size, width, height, projection)
    :return: input images with the same median
    """
    for i in range(img.shape[0]):  # iterate over all batches
        # depending on the number of input images, adjust the median of the maximum/minimum projections
        if img.shape[3] == 6:
            current_img_med_maxpro = np.median(img[i, :, :, 0])
            img[i, :, :, 4] += current_img_med_maxpro - np.median(img[i, :, :, 4])
            img[i, :, :, 5] += current_img_med_maxpro - np.median(img[i, :, :, 5])
            img[i, :, :, 4][img[i, :, :, 4] < 0] = 0
            img[i, :, :, 4][img[i, :, :, 4] > 1] = 1
            img[i, :, :, 5][img[i, :, :, 5] < 0] = 0
            img[i, :, :, 5][img[i, :, :, 5] > 1] = 1
        elif img.shape[3] == 12:
            current_img_med_maxpro = np.median(img[i, :, :, 0])
            current_img_med_minpro = np.median(img[i, :, :, 1])
            img[i, :, :, 4] += current_img_med_maxpro - np.median(img[i, :, :, 4])
            img[i, :, :, 5] += current_img_med_minpro - np.median(img[i, :, :, 5])
            img[i, :, :, 4][img[i, :, :, 4] < 0] = 0
            img[i, :, :, 4][img[i, :, :, 4] > 1] = 1
            img[i, :, :, 5][img[i, :, :, 5] < 0] = 0
            img[i, :, :, 5][img[i, :, :, 5] > 1] = 1
            img[i, :, :, 8] += current_img_med_maxpro - np.median(img[i, :, :, 8])
            img[i, :, :, 9] += current_img_med_minpro - np.median(img[i, :, :, 9])
            img[i, :, :, 8][img[i, :, :, 8] < 0] = 0
            img[i, :, :, 8][img[i, :, :, 8] > 1] = 1
            img[i, :, :, 9][img[i, :, :, 9] < 0] = 0
            img[i, :, :, 9][img[i, :, :, 9] > 1] = 1


def multiplic_brightness_ret(img, maximum):
    """
    Multiply all projection images, so that their maximum is equal to the defined maximum
    (works only for a specific number of images per batch)
    :param img: (array) contains images (batch size, width, height, projection)
    :param maximum: (int) maximal value within the maximum projection
    :return: Images with their maximum being the defined maximum
    """
    for i in range(img.shape[0]):  # iterate over all batches
        maxv = np.amax(img[i, :, :, 0])  # obtain the maximum value within the maximum projection
        multi = maximum/maxv  # calculate the factor to multiply the images with
        # multiply the projection images with the factor
        img[i, :, :, 0] *= multi
        img[i, :, :, 1] *= multi
        img[i, :, :, 2] *= multi
