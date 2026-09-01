"""
Phase 1 (drift correction + ClickPoints databases) for the 20260828 division
experiments:

    20260828_nk92_ctrl_b1inhib_b1b2inhib     (Mic3)
    20260828_jurkat_ctrl_b1inhib_b1b2inhib   (Mic2)

Both folders are 6 positions at 120 s/frame:
    pos00, pos01 -> control
    pos02, pos03 -> b1 inhibition
    pos04, pos05 -> b1b2 inhibition

The NK92 folder holds TWO acquisitions in the same directory - a short aborted
one (20260828-130005, 48 frames/pos) and the real 24 h run (20260828-143815,
720 frames/pos). `correct_drift_longterm` globs `*rep*_pos{pos}_...` without a
timestamp, so running it unmodified would concatenate both runs into one
768-frame sequence and interleave them. Each acquisition is therefore corrected
separately here, by prefixing the file patterns with its own timestamp and
writing to its own output folder. No raw file is moved or modified.

Only drift correction and database creation run - no UNet detection, no
tracking. The resulting .cdb files are the ones to annotate divisions in.

Run directly:  <repo>/.conda/python.exe run_longterm_division_prep.py
Safe to re-run: finished positions are skipped, so an interrupted run resumes.
"""
import os
import shutil
from glob import glob

from joblib import Parallel, delayed

from immune_cell_migration.preprocessing import correct_drift_longterm
from immune_cell_migration.preprocessing import prep_clickpoints_databases
from immune_cell_migration.preprocessing.drift_correction import DEFAULT_FILE_PATTERNS

BASE = r"Z:\nhuhn\Microscopy\mic2_mic3\cell_division\longterm_division"
NK92 = os.path.join(BASE, "20260828_nk92_ctrl_b1inhib_b1b2inhib")
JURKAT = os.path.join(BASE, "20260828_jurkat_ctrl_b1inhib_b1b2inhib")

# (raw folder, acquisition timestamp, output folder). NK92 first.
RUNS = [
    (NK92, "20260828-143815", NK92 + "_corrected_143815"),   # the 720-frame run
    (NK92, "20260828-130005", NK92 + "_corrected_130005"),   # the short aborted run
    (JURKAT, "20260828-124112", JURKAT + "_corrected"),
]

POSITIONS = range(6)
N_JOBS = 6              # one worker per position; the network share is the bottleneck
DRIFT_SEQUENTIAL = True  # register every frame, chaining with a handover re-anchor
DRIFT_HANDOVER = 10
DRIFT_REG_DOWNSAMPLE = 4


def _done_flag(outfolder, pos):
    """Marker written only after every TIFF of a position has been shifted.

    `drift_pos{pos}.pkl` is dumped *before* the images are written, so it cannot
    be used to tell a finished position from one interrupted halfway through.
    """
    return os.path.join(outfolder, f"done_pos{pos:02d}.flag")


def correct_one(folder, timestamp, outfolder, pos):
    if os.path.exists(_done_flag(outfolder, pos)):
        print(f"[{timestamp}] pos{pos:02d} already done, skipping")
        return
    patterns = [timestamp + p for p in DEFAULT_FILE_PATTERNS]
    correct_drift_longterm(folder, pos, outfolder,
                           file_patterns=patterns,
                           sequential=DRIFT_SEQUENTIAL,
                           handover=DRIFT_HANDOVER,
                           reg_downsample=DRIFT_REG_DOWNSAMPLE)
    open(_done_flag(outfolder, pos), "w").close()
    print(f"[{timestamp}] pos{pos:02d} finished")


def copy_config(folder, timestamp, outfolder):
    """prep_clickpoints_databases identifies the measurement from a *_Config.txt
    in the target folder, so give each corrected folder its own."""
    src = os.path.join(folder, timestamp + "_Config.txt")
    dst = os.path.join(outfolder, timestamp + "_Config.txt")
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)


def run(folder, timestamp, outfolder):
    n_raw = len(glob(os.path.join(folder, timestamp + "*_pos00_*_zMaxProj.tif")))
    print("=" * 70)
    print(f"{timestamp}: {n_raw} frames/position -> {outfolder}")
    os.makedirs(outfolder, exist_ok=True)

    Parallel(n_jobs=N_JOBS)(
        delayed(correct_one)(folder, timestamp, outfolder, pos) for pos in POSITIONS
    )

    copy_config(folder, timestamp, outfolder)
    prep_clickpoints_databases([(outfolder, None)])
    print(f"{timestamp}: databases written to {outfolder}")


if __name__ == "__main__":
    import sys
    # optional: pass one or more timestamps to process only those runs
    wanted = sys.argv[1:]
    for folder, timestamp, outfolder in RUNS:
        if wanted and timestamp not in wanted:
            continue
        run(folder, timestamp, outfolder)
    print("\nAll runs complete.")
