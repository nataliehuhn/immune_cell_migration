from immune_cell_migration.utils import name_glob
import os
from glob import glob
import numpy as np
from joblib import Parallel, delayed
from ..preprocessing import correct_drift
from ..preprocessing import prep_clickpoints_databases
from ..tracking import cell_tracker
from ..postprocessing import motility_filter_cdb
from ..postprocessing import write_to_excel
from .. plots import plot_kde_speed_pers
from .. plots import plot_kde_differences
from .. plots import plot_mf_speed_pers
from .. plots import plot_pf
from .. plots import plot_quadrants_stacked
from .. pooling import pool_kde_plots
from .. pooling import pool_mf_speed_pers


def complete_pipeline(folder, time_step, conditions, pos_num, celltype, acq_mode, savename, order, conds, drift_corr=True, clickpoints_db=True, tracking=True, postprocessing=True, plotting=True, n_jobs=1):
    if drift_corr:
        pathlist = name_glob(os.path.join(folder, '*h'))
        print(pathlist)
        for path, _ in pathlist:
            num_pos = len(glob(os.path.join(path, "*rep*_pos*zMaxProj.tif")))
            print(num_pos)
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
        cell_tracker.track_cells(celltype, path_list=pathlist, pixelsize_ccd=3.45) #4.56 Lumenera

    if postprocessing:
        if len(name_glob(os.path.join(folder, '*h_corrected'))) != 0:
            pathlist = name_glob(os.path.join(folder, '*h_corrected'))
            print(pathlist)
        else:
            pathlist = name_glob(os.path.join(folder, '*h'))
            print(pathlist)
        # analyze cdb: set motile fraction definition etc
        motility_filter_cdb.filter_cdb(time_step=time_step, celltype=celltype, path_list=pathlist, pixelsize_ccd=3.45, objective=10) #4.56 Lumenera
        print("cdb filtering done")
        # extract excel files
        write_to_excel.excel_writer(celltype=celltype, path_list=pathlist, savename=savename, conditions=conditions, acquisition_mode=acq_mode, pos_num=pos_num)
        print("excel files written")

    if plotting:
        if len(name_glob(os.path.join(folder, '*h_corrected'))) != 0:
            pathlist = name_glob(os.path.join(folder, '*h_corrected'))
            print(pathlist)
        else:
            pathlist = name_glob(os.path.join(folder, '*h'))
            print(pathlist)
        # plot kde
        plot_kde_speed_pers.generate_kde_plot(celltype, path_list=pathlist, conditions=conditions, acquisition_mode=acq_mode, pos_num=pos_num, custom_order=order)
        # plot_kde_differences.generate_kde_plot(celltype, path_list=pathlist, savename=savename, conditions=conditions, acquisition_mode=acq_mode, pos_num=pos_num, custom_order=order, conds_to_compare=conds)

        # plot speed, persistence, and motile fraction
        plot_mf_speed_pers.plot_motile_fractions(parent_folder=folder, custom_order=order)
        plot_mf_speed_pers.plot_speed(parent_folder=folder, custom_order=order)
        plot_mf_speed_pers.plot_persistence(parent_folder=folder, custom_order=order)

        # plot persistence fraction (specifically for elexa, teza experiments)
        plot_pf.plot_persistent_fraction(parent_folder=folder, custom_order=order)

        # plot quadrants stacked (Q1, Q2)
        plot_quadrants_stacked.plot_quadrant_percentages(parent_folder=folder, custom_order=order)


def complete_pooled_pipeline(folders, celltype, acq_mode, pos_num, order, conditions, output_base):
    pool_mf_speed_pers.plot_pooled_motile_fraction(folders=folders, custom_order=order, output_base=output_base)
    pool_mf_speed_pers.plot_pooled_speed(folders=folders, custom_order=order, output_base=output_base)
    pool_mf_speed_pers.plot_pooled_persistence(folders=folders, custom_order=order, output_base=output_base)
    pool_kde_plots.generate_kde_plot(celltype=celltype, folders=folders, conditions=conditions, acquisition_mode=acq_mode, pos_num=pos_num, custom_order=order, output_base=output_base)
