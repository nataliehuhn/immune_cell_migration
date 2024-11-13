from immune_cell_migration.utils import name_glob, get_value, name_glob_files
import os
from glob import glob
import numpy as np
from joblib import Parallel, delayed
from ..preprocessing import correct_drift
from ..preprocessing import prep_clickpoints_databases
from ..tracking import cell_tracker


def complete_pipeline(folder, drift_corr=True, clickpoints_db=True, tracking=True, postprocessing=True, n_jobs=1):
    if drift_corr:
        pathlist = name_glob(os.path.join(folder, '*h'))
        for path, _ in pathlist:
            num_pos = len(glob(os.path.join(path, "*rep000_pos*zMaxProj.tif")))
            positions = np.arange(0, num_pos, 1)
            long_measurements = False
            outfolder = path + '_corrected'
            print(outfolder)
            Parallel(n_jobs=n_jobs)(delayed(correct_drift)(path, pos, outfolder, long_measurements) for pos in positions)

    if clickpoints_db:
        pathlist = name_glob(os.path.join(folder, '*h_corrected'))
        prep_clickpoints_databases(pathlist)

    if tracking:
        if len(name_glob(os.path.join(folder, '*h_corrected'))) != 0:
            pathlist = name_glob(os.path.join(folder, '*h_corrected'))
            print(pathlist)
        else:
            pathlist = name_glob(os.path.join(folder, '*h'))
            print(pathlist)
        cell_tracker.track_cells(celltype="NK", path_list=pathlist, pixelsize_ccd=4.0954)
