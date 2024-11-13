import clickpoints
import sys
from pathlib import Path
import os
from .tracking_functions import run_tracking
from .unet.unet_cell_detector import CellDetector

TRAINED_NETWORKS = {"NK": {"trained_file": "NK_cell_weights.h5", "training_pixelsize": 6.45},
                    "pigPBMCs": {"trained_file": "NK_cell_weights.h5", "training_pixelsize": 6.45},
                    "Jurkat": {"trained_file": "Jurkat_cell_weights.h5", "training_pixelsize": 4.0954}}

def track_cells(celltype, path_list, pixelsize_ccd=4.0954):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(os.path.join(script_dir, TRAINED_NETWORKS[celltype]["trained_file"]))
    trained_network = os.path.join(script_dir, TRAINED_NETWORKS[celltype]["trained_file"])
    # r"Z:\nhuhn\Python\LucasNKMasksAndTracking_RichiePandas\weightsLucas\NK_cell_weigths_copy.h5"

    zoomed = pixelsize_ccd / TRAINED_NETWORKS[celltype]["training_pixelsize"]  # Lumenera   #6.45/6.45  #Hamamatsu
    detector = CellDetector(trained_network, zoom_factor=zoomed)

    for path, _ in path_list:
        # iterate over all databases in all subfolders
        for database_name in Path(path).glob("*-*_pos*.cdb"):  # "**/*-*_pos*.cdb"
            print(database_name)
            print("Processing data:", database_name)
            # load database
            db = clickpoints.DataFile(str(database_name))
            # detect the cells (e.g. draw the masks)
            detector.set_masks(db)

            Frames = db.getImages(layer=1).count()
            # Frames = 272
            run_tracking(path, db, Frames, start_frame=1)

            db.db.close()
        print("--------all done------")
