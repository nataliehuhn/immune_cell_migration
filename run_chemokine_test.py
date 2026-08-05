"""
Caller for the chemokine assay on the 20270728 jurkat/nk92 CXCL12 test dataset.

Run this file directly in PyCharm.

Border lines: draw TWO lines (TYPE_Line, e.g. named "boarder") into the FIRST
frame of every position's database in `0h_corrected`, marking the left/right
channel walls. Detection then clips masks to the region between them, tracking
stays inside, and the chemokine axis is taken perpendicular to the borders.
Those 0h borders are automatically reused for all later timepoints of the same
position. If a position has no borders, it falls back to full-frame + `chem_dir`.
"""
from immune_cell_migration.pipelines import chemokine_assay_ben

# ---------------------------------------------------------------- experiment
FOLDER = r"Z:\nhuhn\Microscopy\mic2_mic3\chemotaxis\20270728_jurkat_nk92_test_cxcl12\data"

TIME_STEP   = 15                       # s between reps (real 5-min / ~22-frame movie)
CELLTYPE    = "NK"                     # UNet model + 6.5 um motility threshold
ACQ_MODE    = "skip"                   # interleaved: condition = position % n_conditions
CONDITIONS  = ["Jurkat_FKS", "Jurkat_CXCL12", "NK92_CXCL12"]
ORDER       = ["Jurkat_FKS", "Jurkat_CXCL12", "NK92_CXCL12"]   # plot order
POS_NUM     = 2                        # positions per condition
CHEM_DIR    = "right"                  # only used where borders are missing
SAVENAME    = "20260728_jurkat_nk92_test_cxcl12"

# detection composite settings (recording 15 s, model trained at 45 s -> spacing 3)
REP_SPACING        = 3
DOWNSAMPLING_FACTOR = 3

if __name__ == "__main__":
    chemokine_assay_ben.complete_pipeline(
        folder=FOLDER,
        time_step=TIME_STEP,
        conditions=CONDITIONS,
        pos_num=POS_NUM,
        celltype=CELLTYPE,
        acq_mode=ACQ_MODE,
        savename=SAVENAME,
        order=ORDER,
        conds=None,                    # only used by the disabled kde-differences plot
        chem_dir=CHEM_DIR,
        rep_spacing=REP_SPACING,
        downsampling_factor=DOWNSAMPLING_FACTOR,
        # ---- stage flags: this dataset already has drift + cdbs + composites ----
        drift_corr=False,      # drift .pkl files already exist
        clickpoints_db=False,  # cdbs + composites already built
        detection=True,        # re-run UNet + clip masks to the borders
        tracking=True,         # re-track inside the borders
        postprocessing=True,   # motility filter + Excel
        plotting=True,         # KDE / speed / persistence / directional + distribution
        n_jobs=1,
        require_borders=True,  # stop before detection if borders aren't drawn yet
    )

# ---------------------------------------------------------------------------
# FRESH DATASET (no cdbs yet) -- no flag flipping needed:
#   Set ALL stages True and require_borders=True, then run this file TWICE.
#     * 1st run: does drift correction + clickpoints dbs + composites, then stops
#       with a message listing the 0h_corrected databases that still need borders.
#     * You draw the two 'boarder' lines in the first frame of each 0h database.
#     * 2nd run (identical call): drift + cdb prep are skipped (already done),
#       the border gate now passes, and it continues detection -> tracking ->
#       postprocessing -> plotting.
#
#   chemokine_assay_ben.complete_pipeline(
#       folder=FOLDER, time_step=..., conditions=..., pos_num=..., celltype=...,
#       acq_mode=..., savename=..., order=..., conds=None, chem_dir=...,
#       rep_spacing=REP_SPACING, downsampling_factor=DOWNSAMPLING_FACTOR,
#       drift_corr=True, clickpoints_db=True, detection=True, tracking=True,
#       postprocessing=True, plotting=True, require_borders=True, n_jobs=1)
# ---------------------------------------------------------------------------
