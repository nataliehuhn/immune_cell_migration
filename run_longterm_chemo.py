"""
Caller for the LONG-TERM chemokine assay on
20260805_jurkat_chemo_conc_cxcl12 (one continuous 481-frame timelapse @ 60 s,
3 positions). Run this file directly in PyCharm.

Workflow for this dataset:
  1. Phase 1 (drift correction + clickpoints dbs + composites) is ALREADY DONE.
  2. Draw TWO border lines (marker type "border" or "boarder") into the FIRST
     frame of each position database in ...\0h_corrected:
        20260805-144238_pos00_x00_mode0.cdb
        20260805-144238_pos01_x00_mode0.cdb
        20260805-144238_pos02_x00_mode0.cdb
  3. Run this file -> detection (masks clipped to the borders) -> tracking ->
     motility filter + Excel -> time-binned distribution/directional plots.

If borders are still missing, the run stops before detection and lists which
databases need them (require_borders=True), so you can just run it again.
"""
import immune_cell_migration as icm

FOLDER = r"Z:\nhuhn\Microscopy\mic2_mic3\Asal\20260819_human_tcells_ccl19\start"

icm.pipelines.longterm_chemokine_assay.complete_pipeline(
    folder=FOLDER,
    time_step=60,
    conditions=["CCL19_10ng", "CCL19_100ng", "CCL19_100ng_DCPIB"],   # <- set real CXCL12 concentrations
    pos_num=1, celltype="Treg_trick", acq_mode="sequential", savename="results",
    order=["CCL19_10ng", "CCL19_100ng", "CCL19_100ng_DCPIB"],
    chem_dir="right", rep_spacing=1, downsampling_factor=2,
    drift_sequential=True, drift_handover=10, drift_reg_downsample=4, border_edge_um=30.0,
    metrics_max_minutes=240, color_window_min=15, directionality_bin_min=30,
    extra_plots=("fmi", "speed", "rose", "map", "exits", "angular"),
    rose_window_min=None, bin_minutes=15, pixelsize_ccd=3.45, #drift_subsample=10,
    drift_corr=False, clickpoints_db=False, detection=False, tracking=False, postprocessing=True, plotting=True,
    require_borders=True, n_jobs=8)
"""
FOLDER = r"Z:\nhuhn\Microscopy\mic2_mic3\Asal\20260819_human_tcells_ccl19\before"

icm.pipelines.longterm_chemokine_assay.complete_pipeline(
    folder=FOLDER,
    time_step=60,
    conditions=["no_treatment"],   # <- set real CXCL12 concentrations
    pos_num=1, celltype="Treg_trick", acq_mode="sequential", savename="results",
    order=["no_treatment"],
    chem_dir="right", rep_spacing=1, downsampling_factor=2,
    drift_sequential=True, drift_handover=10, drift_reg_downsample=4, border_edge_um=30.0,
    metrics_max_minutes=240, color_window_min=15, directionality_bin_min=30,
    extra_plots=("fmi", "speed", "rose", "map", "exits", "angular"),
    rose_window_min=None, bin_minutes=15, pixelsize_ccd=3.45, #drift_subsample=10,
    drift_corr=False, clickpoints_db=True, detection=True, tracking=True, postprocessing=True, plotting=True,
    require_borders=True, n_jobs=8)
"""
