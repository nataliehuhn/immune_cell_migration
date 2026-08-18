"""
Long-term chemokine assay.

Same optics/cells as ``chemokine_assay_ben`` (identical UNet detection and
tracking), but the acquisition is one long continuous timelapse per position
(e.g. 1440 frames at 60 s = 24 h) instead of several discrete timepoint folders.
Consequently:

* There is a single measurement folder (no ``0h``/``1h``/... folders). Drift
  correction writes to ``<folder>_corrected``; all downstream stages run there.
* Border lines are drawn once per position, into the first frame of that
  position's database (there is no separate ``0h_corrected`` reference; the mask
  clip in ``write_masks_to_cdb`` reads borders from each database itself).
* The "over the hours" analysis comes from binning frames *within* each database
  (``bin_minutes``), via ``plots.plot_longterm_chemokine``.

Assumed file naming is the same as the standard assay
(``<date>_..._rep{rep}_pos{pos}_..._mode{mode}_z{Min,Max}{Proj,Indices}.tif``),
just with many more reps. If your long-term data is laid out differently, the
drift/prep stages are the only parts that need adjusting.
"""
import os
import numpy as np
from joblib import Parallel, delayed

from ..utils import name_glob
from ..preprocessing import correct_drift_longterm
from ..preprocessing import prep_clickpoints_databases
from ..preprocessing import prep_6layer_tiffs_for_detection_ben
from ..preprocessing import borders as border_utils
from ..tracking import cell_tracker_ben
from ..tracking import cell_finder_ben
from ..postprocessing import motility_filter_cdb
from ..postprocessing import write_to_excel
from ..plots import plot_mf_speed_pers
from ..plots import plot_longterm_chemokine
from ..plots import chemotaxis_metrics


def _is_position_corrected(outfolder, pos):
    return os.path.exists(os.path.join(outfolder, f"drift_pos{pos:02d}.pkl"))


def complete_pipeline(folder, time_step, conditions, pos_num, celltype, acq_mode, savename,
                      order, chem_dir, rep_spacing, downsampling_factor,
                      bin_minutes=60, pixelsize_ccd=3.45, objective=10,
                      drift_subsample=10, drift_sequential=True, drift_handover=10, drift_reg_downsample=4,
                      motility_window_min=5.5, metrics_max_minutes=120,
                      directionality_bin_min=30, color_window_min=15, border_edge_um=15.0, n_space_bins=6, angular_coarse_frames=3,
                      extra_plots=(),
                      drift_threshold=None, drift_correct=True,
                      slow_percentile=25, rose_window_min=None,
                      drift_corr=True, clickpoints_db=True, detection=True, tracking=True,
                      postprocessing=True, plotting=True, n_jobs=1, require_borders=False):
    """
    Run the long-term chemokine assay end to end on a single measurement folder.

    Parameters mirror ``chemokine_assay_ben.complete_pipeline`` with these extras:
        time_step         : seconds between consecutive frames (e.g. 60).
        bin_minutes       : width of each time bin for the over-time plots (60 = hourly).
        pixelsize_ccd     : camera pixel size (3.45 Basler, 4.56 Lumenera).
        drift_subsample   : estimate drift on every Nth frame, then interpolate to
                            all frames (memory-safe streaming; see correct_drift_longterm).
        require_borders   : if True, stop before detection until borders are drawn
                            in every position database (run the same call twice).
    """
    corrected = folder + "_corrected"

    if drift_corr:
        num_pos = len(conditions) * pos_num
        positions = np.arange(0, num_pos, 1)
        remaining = [p for p in positions if not _is_position_corrected(corrected, p)]
        if remaining:
            print(f"Correcting {len(remaining)}/{len(positions)} positions in: {corrected}")
            Parallel(n_jobs=n_jobs)(
                delayed(correct_drift_longterm)(folder, pos, corrected, drift_subsample,
                                                sequential=drift_sequential, handover=drift_handover,
                                                reg_downsample=drift_reg_downsample)
                for pos in remaining
            )
        else:
            print(f"All positions already corrected, skipping: {corrected}")

    # pathlist is the single corrected folder (fall back to raw if not corrected)
    pathlist = [(corrected, None)] if os.path.isdir(corrected) else [(folder, None)]

    if clickpoints_db:
        prep_clickpoints_databases(pathlist)
        prep_6layer_tiffs_for_detection_ben.prepare_unet_input(pathlist, rep_spacing, downsampling_factor)

    # Border gate: run this call twice with identical flags - the first run does
    # drift + cdb prep and stops here, you draw the two 'boarder' lines in the
    # first frame of each position database, the second run continues.
    if require_borders and (detection or tracking or postprocessing or plotting):
        target = pathlist[0][0]
        missing = border_utils.cdbs_missing_borders(target)
        if missing:
            print("\n=== Border lines not found yet ===")
            print(f"Draw two 'boarder' (TYPE_Line) markers into the FIRST frame of each "
                  f"database below (in {target}), then re-run this exact call:")
            for m in missing:
                print("   -", os.path.basename(m))
            print("Stopping before detection until borders are present.\n")
            return

    if detection:
        cell_finder_ben.predict_cells_unet(pathlist, celltype, n_jobs=n_jobs)
        cell_finder_ben.write_masks_to_cdb(pathlist)

    if tracking:
        cell_tracker_ben.track_cells(pathlist, n_jobs=n_jobs)

    if postprocessing:
        motility_filter_cdb.filter_cdb(time_step=time_step, celltype=celltype, path_list=pathlist,
                                       pixelsize_ccd=pixelsize_ccd, objective=objective,
                                       motility_window_min=motility_window_min,
                                       colorize_direction=True, chemokine_direction=chem_dir,
                                       color_window_min=color_window_min)  # cdb colouring window (e.g. 15 min)
        print("cdb filtering done")
        write_to_excel.excel_writer(celltype=celltype, path_list=pathlist, savename=savename,
                                    conditions=conditions, acquisition_mode=acq_mode, pos_num=pos_num)
        print("excel files written")

    if plotting:
        # speed / persistence / motile fraction (folder-based, reused as-is)
        plot_mf_speed_pers.plot_motile_fractions(parent_folder=folder, custom_order=order)
        plot_mf_speed_pers.plot_speed(parent_folder=folder, custom_order=order)
        plot_mf_speed_pers.plot_persistence(parent_folder=folder, custom_order=order)

        # time-resolved chemokine analysis (binned within each database)
        plot_longterm_chemokine.plot_distribution_over_time(
            celltype=celltype, path_list=pathlist, conditions=conditions, custom_order=order,
            acquisition_mode=acq_mode, pos_num=pos_num, time_step=time_step, bin_minutes=bin_minutes)
        # directionality of the motile cells over time, in its own (e.g. 30-min) windows
        plot_longterm_chemokine.plot_directional_over_time(
            celltype=celltype, path_list=pathlist, conditions=conditions, custom_order=order,
            acquisition_mode=acq_mode, pos_num=pos_num, time_step=time_step,
            bin_minutes=directionality_bin_min,
            pixelsize_ccd=pixelsize_ccd, objective=objective, motility_window_min=motility_window_min)

        # ---- optional extra analyses, selected via `extra_plots` ----
        # (drift is corrected inside the metrics via slow-cell subtraction)
        mkw = dict(celltype=celltype, path_list=pathlist, conditions=conditions, custom_order=order,
                   acquisition_mode=acq_mode, pos_num=pos_num, time_step=time_step,
                   pixelsize_ccd=pixelsize_ccd, objective=objective,
                   max_minutes=metrics_max_minutes,
                   directionality_window_min=directionality_bin_min,
                   motility_window_min=motility_window_min,
                   drift_threshold=drift_threshold, drift_correct=drift_correct,
                   slow_percentile=slow_percentile)
        extras = set(extra_plots or ())
        if "fmi" in extras:            # Forward Migration Index (parallel/perpendicular)
            chemotaxis_metrics.plot_fmi(**mkw)
        if "speed" in extras:
            # speed + straightness per direction class, on the SAME window as the cdb
            # colouring, so the bars match what you see coloured in clickpoints
            skw = dict(mkw, directionality_window_min=color_window_min)
            chemotaxis_metrics.plot_speed_gradient(**skw)
        if "rose" in extras:           # angle roses across time windows
            chemotaxis_metrics.plot_rose_over_time(rose_window_min=rose_window_min, **mkw)
        if "map" in extras:            # WHERE chemotaxis happens, and how it shifts in time
            chemotaxis_metrics.plot_chemotaxis_map(n_space_bins=n_space_bins, **mkw)
        if "angular" in extras:        # torque vs angular-noise decomposition (Jakuszeit et al.)
            chemotaxis_metrics.plot_angular_dynamics(
                celltype=celltype, path_list=pathlist, conditions=conditions, custom_order=order,
                acquisition_mode=acq_mode, pos_num=pos_num, time_step=time_step,
                pixelsize_ccd=pixelsize_ccd, objective=objective, max_minutes=metrics_max_minutes,
                coarse_frames=angular_coarse_frames, drift_correct=drift_correct,
                slow_percentile=slow_percentile)
        if "exits" in extras:          # cells escaping through the right vs left border
            chemotaxis_metrics.plot_border_exits(
                celltype=celltype, path_list=pathlist, conditions=conditions, custom_order=order,
                acquisition_mode=acq_mode, pos_num=pos_num, time_step=time_step,
                pixelsize_ccd=pixelsize_ccd, objective=objective, max_minutes=metrics_max_minutes,
                edge_um=border_edge_um)
