from immune_cell_migration.utils import name_glob
import os
from glob import glob
import numpy as np
from joblib import Parallel, delayed
from ..preprocessing import correct_drift
from ..preprocessing import prep_clickpoints_databases
from ..preprocessing import prep_6layer_tiffs_for_detection_ben
from ..preprocessing import borders as border_utils
from ..tracking import cell_tracker_ben
from .. tracking import cell_finder_ben
from ..postprocessing import motility_filter_cdb
from ..postprocessing import write_to_excel
from .. plots import plot_kde_speed_pers
from .. plots import plot_kde_differences
from .. plots import plot_mf_speed_pers
from .. plots import plot_pf
from .. plots import plot_quadrants_stacked
from .. plots import plot_chemokine_assay
from .. pooling import pool_kde_plots
from .. pooling import pool_mf_speed_pers


def _is_position_corrected(outfolder, pos):
    return os.path.exists(os.path.join(outfolder, f"drift_pos{pos:02d}.pkl"))


def complete_pipeline(folder, time_step, conditions, pos_num, celltype, acq_mode, savename, order, conds, chem_dir,
                      rep_spacing, downsampling_factor, drift_corr=True, clickpoints_db=True, detection=True, tracking=True, postprocessing=True, plotting=True, n_jobs=1,
                      require_borders=False):
    if drift_corr:
        pathlist = name_glob(os.path.join(folder, '*h'))
        print(pathlist)
        for path, _ in pathlist:
            num_pos = len(conditions) * pos_num
            positions = np.arange(0, num_pos, 1)
            long_measurements = False
            outfolder = path + '_corrected'

            remaining_positions = [
                pos for pos in positions
                if not _is_position_corrected(outfolder, pos)
            ]

            if not remaining_positions:
                print(f"All positions already corrected, skipping: {outfolder}")
                continue

            print(f"Correcting {len(remaining_positions)}/{len(positions)} remaining positions in: {outfolder}")
            Parallel(n_jobs=n_jobs)(
                delayed(correct_drift)(path, pos, outfolder, long_measurements)
                for pos in remaining_positions
            )

    if clickpoints_db:
        pathlist = name_glob(os.path.join(folder, '*h_corrected'))
        prep_clickpoints_databases(pathlist)
        prep_6layer_tiffs_for_detection_ben.prepare_unet_input(pathlist, rep_spacing, downsampling_factor)

    # Border gate: don't detect/track until the border lines are drawn into the
    # reference (0h_corrected) databases. This lets the whole thing be one call
    # you run twice with identical flags: the 1st run does drift + cdb prep then
    # stops here; you draw the two 'boarder' lines in each 0h database; the 2nd
    # run skips the finished stages (drift/cdb are idempotent) and continues.
    if require_borders and (detection or tracking or postprocessing or plotting):
        ref_folder = os.path.join(folder, border_utils.REFERENCE_FOLDER_NAME)
        missing = border_utils.cdbs_missing_borders(ref_folder)
        if missing:
            print("\n=== Border lines not found yet ===")
            print(f"Draw two 'boarder' (TYPE_Line) markers into the FIRST frame of each "
                  f"database below (in {ref_folder}), then re-run this exact call:")
            for m in missing:
                print("   -", os.path.basename(m))
            print("Stopping before detection until borders are present.\n")
            return

    if detection:
        if len(name_glob(os.path.join(folder, '*h_corrected'))) != 0:
            pathlist = name_glob(os.path.join(folder, '*h_corrected'))
            print(pathlist)
        else:
            pathlist = name_glob(os.path.join(folder, '*h'))
            print(pathlist)
        cell_finder_ben.predict_cells_unet(pathlist, celltype)
        cell_finder_ben.write_masks_to_cdb(pathlist)

    if tracking:
        if len(name_glob(os.path.join(folder, '*h_corrected'))) != 0:
            pathlist = name_glob(os.path.join(folder, '*h_corrected'))
            print(pathlist)
        else:
            pathlist = name_glob(os.path.join(folder, '*h'))
            print(pathlist)
        cell_tracker_ben.track_cells(pathlist)

    if postprocessing:
        if len(name_glob(os.path.join(folder, '*h_corrected'))) != 0:
            pathlist = name_glob(os.path.join(folder, '*h_corrected'))
            print(pathlist)
        else:
            pathlist = name_glob(os.path.join(folder, '*h'))
            print(pathlist)
        # analyze cdb: set motile fraction definition etc
        motility_filter_cdb.filter_cdb(time_step=time_step, celltype=celltype, path_list=pathlist,
                                       pixelsize_ccd=3.45, objective=10)  # 4.56 Lumenera, 3.45 Basler
        print("cdb filtering done")
        # extract excel files
        write_to_excel.excel_writer(celltype=celltype, path_list=pathlist, savename=savename, conditions=conditions,
                                    acquisition_mode=acq_mode, pos_num=pos_num)
        print("excel files written")

    if plotting:
        if len(name_glob(os.path.join(folder, '*h_corrected'))) != 0:
            pathlist = name_glob(os.path.join(folder, '*h_corrected'))
            print(pathlist)
        else:
            pathlist = name_glob(os.path.join(folder, '*h'))
            print(pathlist)
        # plot kde
        # plot_kde_speed_pers.generate_kde_plot(celltype, path_list=pathlist, conditions=conditions,
        #                                      acquisition_mode=acq_mode, pos_num=pos_num, custom_order=order)
        # plot_kde_differences.generate_kde_plot(celltype, path_list=pathlist, savename=savename, conditions=conditions, acquisition_mode=acq_mode, pos_num=pos_num, custom_order=order, conds_to_compare=conds)

        # plot speed, persistence, and motile fraction
        # plot_mf_speed_pers.plot_motile_fractions(parent_folder=folder, custom_order=order)
        # plot_mf_speed_pers.plot_speed(parent_folder=folder, custom_order=order)
        # plot_mf_speed_pers.plot_persistence(parent_folder=folder, custom_order=order)

        # plot persistence fraction (specifically for elexa, teza experiments)
        # plot_pf.plot_persistent_fraction(parent_folder=folder, custom_order=order)

        # plot quadrants stacked (Q1, Q2)
        # plot_quadrants_stacked.plot_quadrant_percentages(parent_folder=folder, custom_order=order)

        # plot directional fraction (perpendicular-to-border axis when borders exist)
        plot_chemokine_assay.plot_fraction_toward_chemokine(celltype=celltype, path_list=pathlist, conditions=conditions, custom_order=order,
            chemokine_direction=chem_dir, acquisition_mode=acq_mode, pos_num=pos_num
        )

        # spatial distribution of cells along the chemokine axis over the hours
        plot_chemokine_assay.plot_cell_distribution_along_axis(celltype=celltype, path_list=pathlist, conditions=conditions, custom_order=order,
            acquisition_mode=acq_mode, pos_num=pos_num
        )


def complete_pooled_pipeline(folders, celltype, acq_mode, pos_num, order, conditions, output_base):
    pool_mf_speed_pers.plot_pooled_motile_fraction(folders=folders, custom_order=order, output_base=output_base)
    pool_mf_speed_pers.plot_pooled_speed(folders=folders, custom_order=order, output_base=output_base)
    pool_mf_speed_pers.plot_pooled_persistence(folders=folders, custom_order=order, output_base=output_base)
    pool_kde_plots.generate_kde_plot(celltype=celltype, folders=folders, conditions=conditions, acquisition_mode=acq_mode, pos_num=pos_num, custom_order=order, output_base=output_base)
