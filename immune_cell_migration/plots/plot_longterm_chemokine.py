"""
Time-resolved chemokine plots for the long-term assay.

Unlike the standard chemokine assay (which compares discrete timepoint folders),
the long-term assay is a single continuous acquisition per position (e.g. 1440
frames at 60 s = 24 h). The "over the hours" dimension therefore comes from
binning frames *within* each database.

Both plots project every detection onto the border-perpendicular axis
(0 = left border, 1 = right border) and bin by acquisition time:

* ``plot_distribution_over_time`` - heatmap + overlay of the cell distribution
  along the axis for each time bin (does the population drift toward a border?).
* ``plot_directional_over_time`` - signed chemotaxis index and fraction toward
  each border per time bin (do the cells *move* toward a border, and when?).

Borders are read from each position's own database (long-term data has no
separate 0h_corrected reference folder).
"""
import os
import re
import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..preprocessing import borders as border_utils
from . import _axis_distribution

MOTILITY_DEFINITION = {"NK": 6.5, "pigPBMCs": 6.0, "Jurkat": 4.0, "NK_day14": 13, "Treg": 13, "Treg_trick": 13}
ACQUISITION_MODE = {"skip": 0, "sequential": 1}


def _cdb_for_csv(csv_path):
    """Map '<cdb>_<thresh>umin5min.csv' back to '<cdb>.cdb' in the same folder."""
    base = re.sub(r"_[0-9.]+umin5min\.csv$", "", csv_path)
    cand = base + ".cdb"
    return cand if os.path.exists(cand) else None


def _iter_condition_csvs(path_list, thresh, acq_sequential, num_conditions, pos_num, conditions):
    """Yield (condition, pos, csv_path, borders) for every position CSV found."""
    for path, _ in path_list:
        for f in glob.glob(os.path.join(path, "*" + str(thresh) + "umin*.csv")):
            pos = int(f.split("_")[-4][3:])
            cond_idx = pos // pos_num if acq_sequential else pos % num_conditions
            if cond_idx >= num_conditions:
                continue
            cdb = _cdb_for_csv(f)
            borders = border_utils.load_borders_from_path(cdb)
            if borders is None:
                print(f"  WARNING: no border lines for pos{pos:02d} "
                      f"(looked in {path} and its 0h_corrected sibling) - skipping")
                continue
            yield conditions[cond_idx], pos, f, borders


def _time_bin(frames, time_step, bin_minutes):
    """Frame index -> integer time-bin index (frames are 0-based sort_index)."""
    return (np.asarray(frames) * time_step / 60.0 // bin_minutes).astype(int)


def _bin_labels(n_bins, bin_minutes):
    if bin_minutes % 60 == 0:
        step = bin_minutes // 60
        return [f"{i * step}-{(i + 1) * step}h" for i in range(n_bins)]
    return [f"{i * bin_minutes}-{(i + 1) * bin_minutes}min" for i in range(n_bins)]


def plot_distribution_over_time(celltype, path_list, conditions, custom_order,
                                acquisition_mode, pos_num, time_step,
                                bin_minutes=60, n_bins_axis=20, motile_only=False):
    """Cell distribution along the chemokine axis, per time bin, per condition."""
    thresh = MOTILITY_DEFINITION[celltype]
    acq_sequential = ACQUISITION_MODE[acquisition_mode]
    num_conditions = len(conditions)

    collected = defaultdict(list)  # (condition, time_bin) -> [u, ...]
    max_bin = 0
    for condition, pos, f, borders in _iter_condition_csvs(
            path_list, thresh, acq_sequential, num_conditions, pos_num, conditions):
        df = pd.read_csv(f, index_col=0)
        if not {"frame", "x", "y"}.issubset(df.columns):
            continue
        if motile_only and "motile" in df.columns:
            df = df[df["motile"] == True]
        if df.empty:
            continue
        u = border_utils.normalized_position(borders, df["x"].values, df["y"].values)
        tb = _time_bin(df["frame"].values, time_step, bin_minutes)
        ok = np.isfinite(u) & (u >= 0.0) & (u <= 1.0)
        for uu, bb in zip(u[ok], tb[ok]):
            collected[(condition, int(bb))].append(uu)
            max_bin = max(max_bin, int(bb))

    if not collected:
        print("plot_distribution_over_time: no data collected")
        return

    n_time = max_bin + 1
    time_labels = _bin_labels(n_time, bin_minutes)
    out_dir = os.path.dirname(os.path.normpath(path_list[0][0]))
    order = [c for c in custom_order if any((c, b) in collected for b in range(n_time))]

    _axis_distribution.render_distribution(
        collected, order, labels=time_labels, out_dir=out_dir,
        prefix="longterm_distribution_over_time", n_bins=n_bins_axis, time_title="time",
    )


def plot_directional_over_time(celltype, path_list, conditions, custom_order,
                               acquisition_mode, pos_num, time_step,
                               bin_minutes=60, cos_threshold=0.5,
                               pixelsize_ccd=3.45, objective=10, motility_window_min=5.5):
    """Signed chemotaxis index & fraction toward each border, resolved in time.

    Directionality is measured on SHORT fixed sub-segments of each track (one
    ``motility_window_min`` window, e.g. 5.5 min), NOT on the whole track: a long
    track that chemotaxes early then stops would otherwise smear its net
    displacement across its whole lifetime and flatten the early-vs-late signal.
    Each segment's net-displacement direction is attributed to its own time, then
    averaged per ``bin_minutes`` bin. Only segments in which the cell actually
    translocates >= the motility threshold are counted (so a cell that only starts
    moving later never contributes to an earlier bin).
    """
    thresh = MOTILITY_DEFINITION[celltype]
    res = pixelsize_ccd / objective
    step_frames = max(2, int(round(motility_window_min * 60.0 / time_step)))
    acq_sequential = ACQUISITION_MODE[acquisition_mode]
    num_conditions = len(conditions)

    # (condition, time_bin) -> list of signed cosines (one per moving segment)
    cos_by = defaultdict(list)
    max_bin = 0
    for condition, pos, f, borders in _iter_condition_csvs(
            path_list, thresh, acq_sequential, num_conditions, pos_num, conditions):
        df = pd.read_csv(f, index_col=0)
        if not {"frame", "id", "x", "y"}.issubset(df.columns):
            continue
        if df.empty:
            continue
        axis = border_utils.perpendicular_vector(borders, hint=None)
        axis = axis / (np.linalg.norm(axis) or 1.0)

        for cell_id, g in df.groupby("id"):
            g = g.sort_values("frame")
            xy = g[["x", "y"]].values
            frames_arr = g["frame"].values
            # walk non-overlapping short segments along the track
            for s in range(0, len(xy) - 1, step_frames):
                seg = xy[s:s + step_frames]
                if len(seg) < 2:
                    continue
                dx = seg[-1, 0] - seg[0, 0]
                dy = seg[-1, 1] - seg[0, 1]
                n = np.hypot(dx, dy)
                if n * res < thresh:        # no real translocation in this segment
                    continue
                cos = (dx * axis[0] + dy * axis[1]) / n
                tb = int(frames_arr[s] * time_step / 60.0 // bin_minutes)  # bin by segment start time
                cos_by[(condition, tb)].append(cos)
                if tb > max_bin:
                    max_bin = tb

    if not cos_by:
        print("plot_directional_over_time: no data collected")
        return

    n_time = max_bin + 1
    time_labels = _bin_labels(n_time, bin_minutes)
    x = np.arange(n_time)
    out_dir = os.path.dirname(os.path.normpath(path_list[0][0]))
    order = [c for c in custom_order if any((c, b) in cos_by for b in range(n_time))]

    rows = []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(1.4 * n_time, 8), 4.6))
    cmap = plt.get_cmap("tab10")
    for ci, condition in enumerate(order):
        idx = np.full(n_time, np.nan)
        fr = np.full(n_time, np.nan)
        fl = np.full(n_time, np.nan)
        for b in range(n_time):
            c = np.asarray(cos_by.get((condition, b), []))
            if c.size == 0:
                continue
            idx[b] = c.mean()
            fr[b] = np.mean(c >= cos_threshold)
            fl[b] = np.mean(c <= -cos_threshold)
            rows.append({"condition": condition, "time_bin": time_labels[b],
                         "index": idx[b], "frac_right": fr[b], "frac_left": fl[b],
                         "n_segments": int(c.size)})
        color = cmap(ci % 10)
        ax1.plot(x, idx, "-o", color=color, label=condition)
        ax2.plot(x, fr, "-o", color=color, label=f"{condition} → right")
        ax2.plot(x, fl, "--s", color=color, alpha=0.6, label=f"{condition} → left")

    ax1.axhline(0.0, color="black", lw=0.8)
    ax1.set_ylim(-1, 1)
    ax1.set_xticks(x); ax1.set_xticklabels(time_labels, rotation=45, ha="right", fontsize=7)
    ax1.set_ylabel("Chemotaxis index (+ = toward right border)")
    ax1.set_title("Signed index over time")
    ax1.legend(fontsize=7)

    ax2.axhline(0.33, color="black", ls="--", lw=1, alpha=0.6)
    ax2.set_ylim(0, 1)
    ax2.set_xticks(x); ax2.set_xticklabels(time_labels, rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("Fraction of motile cells (±60°)")
    ax2.set_title("Fraction toward each border over time")
    ax2.legend(fontsize=6, ncol=2)

    plt.tight_layout()
    outpath = os.path.join(out_dir, "longterm_directional_over_time.png")
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"Saved: {outpath}")
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "longterm_directional_over_time.csv"), index=False)
