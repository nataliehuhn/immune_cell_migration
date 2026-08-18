"""
Caller for the LONG-TERM chemokine assay on
20260805_jurkat_chemo_conc_cxcl12 (one continuous 481-frame timelapse @ 60 s,
3 positions). Run this file directly in PyCharm.

Border lines (marker type "border" or "boarder") must be drawn into the FIRST
frame of each position database in ...\0h_corrected. If they are missing the run
stops before detection and lists which databases still need them.

Time windows (independent):
  color_window_min       = 15  -> cdb TRACK COLOURING (direction must hold 15 min)
  directionality_bin_min = 30  -> rose plot grouping / FMI / directional-over-time
  rose_window_min        = None-> override just the rose columns (else 30)
"""
from immune_cell_migration.pipelines import longterm_chemokine_assay

# NOTE: pass the 0h folder; drift correction writes to <folder>_corrected (= 0h_corrected)
FOLDER = r"Z:\nhuhn\Microscopy\mic2_mic3\chemotaxis\20260805_jurkat_chemo_conc_cxcl12\0h"

if __name__ == "__main__":
    longterm_chemokine_assay.complete_pipeline(
        folder=FOLDER,
        time_step=60,
        conditions=["Jurkat_15ng", "Jurkat_30ng", "Jurkat_100ng"],   # CXCL12 concentrations
        pos_num=1, celltype="NK", acq_mode="sequential", savename="results",
        order=["Jurkat_15ng", "Jurkat_30ng", "Jurkat_100ng"],
        chem_dir="right", rep_spacing=1, downsampling_factor=4,
        drift_sequential=True, drift_handover=10, drift_reg_downsample=4,
        metrics_max_minutes=90,
        color_window_min=15,          # cdb track colouring window
        directionality_bin_min=30,    # rose / FMI / directional-over-time grouping
        rose_window_min=None,
        border_edge_um=15.0,          # exit zone = within 15 um of a border (~1 cell diameter)
        # optional extra analyses (leave as () to run only the standard plots):
        #   "fmi"   Forward Migration Index (parallel / perpendicular)
        #   "speed" speed + straightness split by migration direction
        #   "rose"  angle roses across time windows (+ Rayleigh test)
        #   "map"   where in the channel chemotaxis happens, over time
        #   "exits" cells vanishing at the right vs the left border
        #   "angular" torque vs angular-noise decomposition (Jakuszeit et al. 2025)
        extra_plots=("fmi", "speed", "rose", "map", "exits", "angular"),
        angular_coarse_frames=3,      # direction averaged over 3 frames before turning rates
        bin_minutes=15, pixelsize_ccd=3.45, #drift_subsample=10,
        drift_corr=False, clickpoints_db=False, detection=False, tracking=False, postprocessing=True, plotting=True,
        require_borders=True, n_jobs=8)
