from immune_cell_migration.utils import name_glob
import os
from glob import glob
import numpy as np
from joblib import Parallel, delayed
from ..preprocessing import correct_drift
from ..preprocessing import prep_6layer_tiffs_for_detection_ben
from ..tracking import cell_tracker
from .. tracking import cell_finder_ben
from ..postprocessing import motility_filter_cdb
from ..postprocessing import write_to_excel
from .. plots import plot_kde_speed_pers
from .. plots import plot_kde_differences
from .. plots import plot_mf_speed_pers
from .. plots import plot_pf
from .. plots import plot_quadrants_stacked
from .. pooling import pool_kde_plots
from .. pooling import pool_mf_speed_pers


def complete_pipeline(folder, time_step, conditions, pos_num, celltype, acq_mode, savename, order, conds,
                      rep_spacing, downsampling_factor,
                      drift_corr=True, clickpoints_db=True, detection=True, tracking=True, postprocessing=True, plotting=True, n_jobs=1):
    if drift_corr:
        pathlist = name_glob(os.path.join(folder, '*h'))
        print(pathlist)
        for path, _ in pathlist:
            num_pos = len(conditions) * pos_num
            print(num_pos)
            positions = np.arange(0, num_pos, 1)
            long_measurements = False
            outfolder = path + '_corrected'
            print(outfolder)
            Parallel(n_jobs=n_jobs)(delayed(correct_drift)(path, pos, outfolder, long_measurements) for pos in positions)


    if clickpoints_db:
        pathlist = name_glob(os.path.join(folder, '*h_corrected'))
        print(pathlist)
        prep_6layer_tiffs_for_detection_ben.prepare_unet_input(pathlist, rep_spacing, downsampling_factor)

    if detection:
        if len(name_glob(os.path.join(folder, '*h_corrected'))) != 0:
            pathlist = name_glob(os.path.join(folder, '*h_corrected'))
            print(pathlist)
        else:
            pathlist = name_glob(os.path.join(folder, '*h'))
            print(pathlist)
        cell_finder_ben.predict_cells_unet(pathlist, celltype)

    if tracking:
        if len(name_glob(os.path.join(folder, '*h_corrected'))) != 0:
            pathlist = name_glob(os.path.join(folder, '*h_corrected'))
            print(pathlist)
        else:
            pathlist = name_glob(os.path.join(folder, '*h'))
            print(pathlist)
        cell_tracker.track_cells(celltype, path_list=pathlist, pixelsize_ccd=4.56) #4.56 Lumenera, 3.45 Basler
