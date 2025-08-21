from __future__ import division, print_function
import clickpoints
import glob
import peewee
import os
import numpy as np

from ..utils import name_glob, get_value, name_glob_files


def prep_clickpoints_databases(path_list):
    for p, extra in path_list:
        # check if config data exists
        measurements = glob.glob(os.path.join(p, "*_Config.txt"))
        # create config data if not available, "dummy config"
        if len(measurements) == 0:
            print('no config?')
            image_filenames = name_glob_files(os.path.join(p, "*.tif"))
            for img in image_filenames:
                head, tail = os.path.split(img[0])
                file = open((os.path.join(p, tail[:16] + 'Config' + r".txt")), "w")
                file.close()
                measurements = glob.glob(os.path.join(p, "*_Config.txt"))
                # continue
                break
        # if len(measurements) == 0:
        #     print('no config?')
        #     continue

        measurement = sorted(measurements)[-1]
        # split the date string from the name e.g. "20180205-103213"
        measurement_date_id = os.path.basename(measurement)[:15]

        # get all image filenames for that measurement
        # image_filenames = nameGlobFiles(os.path.join(path, measurement_date_id+"*_pos{pos}_*_z*.tif"))
        image_filenames = name_glob_files(os.path.join(p, measurement_date_id + "*_pos{pos}_*_mode{mode}_z*.tif"))

        # extract all unique position identifiers e.g. 000, 001, ...
        positions = np.unique([extra["pos"] for filename, extra in image_filenames])
        modes = np.unique([extra["mode"] for filename, extra in image_filenames])

        # modes = ['POL']
        for pos in positions:
            pos = "pos" + pos
            for mode in modes:
                mode = "mode" + mode
                final_name = os.path.join(p, measurement_date_id + "_" + pos + "_" + mode + ".cdb")
                if os.path.exists(final_name):
                    print('Existing pos ', pos)
                    continue
                pic_path = os.path.join(p, measurement_date_id + '*_' + pos + "*_" + mode + '*.tif')
                create_database(final_name, pic_path)
                print(final_name)

    print("-----Done-----")


def create_database(database_name, pic_path):
    try:
        db = clickpoints.DataFile(database_name, 'w')
    except peewee.OperationalError:
        print(database_name)
        raise

    # Workaround base layer issue
    base = db.getLayer('MinProj', create=True)
    db.getLayer('MinIndices', base_layer=base, create=True)

    db.getLayer('MaxProj', base_layer=base, create=True)
    db.getLayer('MaxIndices', base_layer=base, create=True)

    # get all images in the folder that match the path

    images = glob.glob(pic_path)
    # iterate over all images
    for image_path in images:
        image_filename = os.path.basename(image_path)
        print(image_filename)
        rep = get_value(image_filename, "*_rep{rep}_pos*")["rep"]
        if image_filename.count("MinProj"):
            # layer = 0
            layer = "MinProj"
        elif image_filename.count("MinIndices"):
            # layer = 1
            layer = "MinIndices"
        elif image_filename.count("MaxProj"):
            # layer = 2
            layer = "MaxProj"
        elif image_filename.count("MaxIndices"):
            # layer = 3
            layer = "MaxIndices"
        else:
            raise ValueError("No known layer!")
        image = db.setImage(filename=image_path, layer=layer)
        # if first image was deleted: image.sort_index = int(rep)-1
        image.sort_index = int(rep)
        image.save()
    db.db.close()


