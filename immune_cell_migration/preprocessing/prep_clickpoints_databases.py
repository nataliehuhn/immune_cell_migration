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
        print(image_filenames)

        # extract all unique position identifiers e.g. 000, 001, ...
        positions = np.unique([extra["pos"] for filename, extra in image_filenames])
        print(positions)
        modes = np.unique([extra["mode"] for filename, extra in image_filenames])
        print(modes)

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

    images = glob.glob(pic_path)
    print(images)

    rep_values = []

    # --------- REP EXTRACTION (supports both formats) ----------
    for image_path in images:
        fn = os.path.basename(image_path)

        if fn.startswith("rep"):
            # format: rep001_...
            rep = int(fn[3:6])   # assumes zero-padded 3 digits
        else:
            # format: *_rep001_pos*
            rep = int(get_value(fn, "*_rep{rep}_pos*")["rep"])

        rep_values.append(rep)

    unique_sorted_reps = sorted(set(rep_values))
    rep_to_index = {rep: i for i, rep in enumerate(unique_sorted_reps)}

    # --------- IMAGE INSERTION ----------
    for image_path in images:
        image_filename = os.path.basename(image_path)
        print("image_filename in prep_clickpoints_databasis:", image_filename)

        if image_filename.startswith("rep"):
            rep = int(image_filename[3:6])
        else:
            rep = int(get_value(image_filename, "*_rep{rep}_pos*")["rep"])

        if "MinProj" in image_filename:
            layer = "MinProj"
        elif "MinIndices" in image_filename:
            layer = "MinIndices"
        elif "MaxProj" in image_filename:
            layer = "MaxProj"
        elif "MaxIndices" in image_filename:
            layer = "MaxIndices"
        else:
            raise ValueError(f"No known layer in {image_filename}!")

        image = db.setImage(filename=image_path, layer=layer)

        image.sort_index = rep_to_index[rep]
        image.save()

    db.db.close()
    """
    for image_path in images:
        image_filename = os.path.basename(image_path)
        print("image_filename in prep_clickpoints_databasis: ", image_filename)
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
    """
