import os
import re
import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ..preprocessing import borders as border_utils
from . import _axis_distribution

MOTILITY_DEFINITION = {"NK": 6.5, "pigPBMCs": 6.0, "Jurkat": 4.0, "NK_day14": 13, "Treg": 13}
ACQUISITION_MODE = {"skip": 0, "sequential": 1}
CHEMOTACTIC_GRADIENT = {"up": 0, "down": 1, "right": 2, "left": 3}
CHEMO_VECTORS = {
    "up":    np.array([0, 1]),
    "down":  np.array([0, -1]),
    "right": np.array([1, 0]),
    "left":  np.array([-1, 0]),
}


def extract_motile_directions(files):
    """Extract dx/dy per motile cell from CSV files."""
    all_entries = []

    for f in files:
        df = pd.read_csv(f, index_col=0)

        required = {'id', 'x', 'y', 'motile'}
        if not required.issubset(df.columns):
            print(f"Skipping {f}: missing columns {required - set(df.columns)}")
            continue

        motile_df = df[df['motile'] == True]
        if motile_df.empty:
            continue

        grouped = motile_df.groupby('id')
        disp = grouped.apply(
            lambda g: pd.Series({
                "x_start": g["x"].iloc[0],
                "x_end":   g["x"].iloc[-1],
                "y_start": g["y"].iloc[0],
                "y_end":   g["y"].iloc[-1],
            })
        ).reset_index()

        disp["dx"] = disp["x_end"] - disp["x_start"]
        disp["dy"] = disp["y_end"] - disp["y_start"]

        all_entries.append(disp[["id", "dx", "dy"]])

    if not all_entries:
        return pd.DataFrame(columns=["id", "dx", "dy"])

    return pd.concat(all_entries, ignore_index=True)


def compute_fraction_toward(df, chemokine_direction, gradient_vector=None):
    """
    Compute fraction of cells moving toward chemokine.
    cos_threshold = 0.5 corresponds to ±60° around gradient direction.

    If ``gradient_vector`` is given (a 2D vector, typically the perpendicular to
    the drawn border lines) it is used instead of the fixed cardinal
    ``CHEMO_VECTORS[chemokine_direction]`` lookup.
    """
    cos_threshold = 0.5
    if df.empty:
        return np.nan

    if gradient_vector is not None:
        g = np.asarray(gradient_vector, dtype=float)
        n = np.linalg.norm(g)
        if n == 0:
            return np.nan
        g = g / n  # ensure unit length
    else:
        g = CHEMO_VECTORS[chemokine_direction]

    # displacement vectors
    v = df[["dx", "dy"]].values

    # vector norms
    v_norm = np.linalg.norm(v, axis=1)

    # avoid division by zero
    valid = v_norm > 0
    if not np.any(valid):
        return np.nan

    v = v[valid]
    v_norm = v_norm[valid]

    # cosine of angle between displacement and gradient
    cos_theta = (v @ g) / v_norm  # g is unit length

    toward = cos_theta >= cos_threshold

    return toward.mean()


def compute_direction_metrics(df, axis, cos_threshold=0.5):
    """
    Project motile-cell displacements onto ``axis`` and return three numbers:

        (index, frac_pos, frac_neg)

    * ``index``    - mean signed cosine of the displacement vs ``axis``
                     (a chemotaxis index; >0 = net bias along +axis).
    * ``frac_pos`` - fraction of cells within +/-60 deg of +axis.
    * ``frac_neg`` - fraction of cells within +/-60 deg of -axis.

    ``axis`` is the border-perpendicular vector (oriented left border -> right
    border) when borders exist, otherwise the fixed ``CHEMO_VECTORS`` direction.
    """
    nan3 = (np.nan, np.nan, np.nan)
    if df.empty:
        return nan3

    g = np.asarray(axis, dtype=float)
    n = np.linalg.norm(g)
    if n == 0:
        return nan3
    g = g / n

    v = df[["dx", "dy"]].values
    v_norm = np.linalg.norm(v, axis=1)
    valid = v_norm > 0
    if not np.any(valid):
        return nan3
    v = v[valid]
    v_norm = v_norm[valid]

    cos_theta = (v @ g) / v_norm
    index = float(np.mean(cos_theta))
    frac_pos = float(np.mean(cos_theta >= cos_threshold))
    frac_neg = float(np.mean(cos_theta <= -cos_threshold))
    return index, frac_pos, frac_neg


"""
def compute_fraction_toward(df, chemokine_direction):
    # Compute fraction of cells moving toward chemokine

    if df.empty:
        return np.nan

    if chemokine_direction in ["up", "down"]:
        axis = "dy"
        if chemokine_direction == "up":
            toward = df[axis] > 0
        else:
            toward = df[axis] < 0

    elif chemokine_direction in ["right", "left"]:
        axis = "dx"
        if chemokine_direction == "right":
            toward = df[axis] > 0
        else:
            toward = df[axis] < 0
    else:
        raise ValueError("Unknown chemokine direction")

    return toward.mean()  # fraction
"""


def plot_fraction_toward_chemokine(
    celltype,
    path_list,
    conditions,
    custom_order,
    chemokine_direction,
    acquisition_mode,
    pos_num
):

    thresh_motile = MOTILITY_DEFINITION[celltype]
    acq_sequential = ACQUISITION_MODE[acquisition_mode]

    for path, _ in path_list:
        print(f"Processing path: {path}")

        # Cache of per-position (axis, has_borders). When border lines are drawn
        # in the position's 0h_corrected database, the axis is perpendicular to
        # them, oriented left border -> right border (direction-agnostic: +axis =
        # toward the right border). When no borders exist we fall back to the
        # explicit cardinal chemokine_direction so the plot still works.
        axis_by_pos = {}

        def axis_for_pos(pos):
            if pos not in axis_by_pos:
                ref_cdb = border_utils.reference_cdb_for_pos(path, pos)
                borders = border_utils.load_borders_from_path(ref_cdb)
                if borders is None:
                    axis_by_pos[pos] = (CHEMO_VECTORS[chemokine_direction], False)
                else:
                    axis_by_pos[pos] = (border_utils.perpendicular_vector(borders, hint=None), True)
            return axis_by_pos[pos]

        cond_sets = [[d] for d in conditions]
        num_conditions = len(conditions)

        count_cond = 0
        condition_files = [[] for _ in range(num_conditions)]

        # Match files to conditions (per position group)
        for d_set in cond_sets:
            filenames = glob.glob(os.path.join(path, "*" + str(thresh_motile) + "umin*.csv"))

            for filename in filenames:
                position = int(filename.split("_")[-4][3:])

                if acq_sequential:
                    if position // pos_num != count_cond:
                        continue
                else:
                    if position % num_conditions != count_cond:
                        continue

                condition_files[count_cond].append(filename)

            count_cond += 1

        # Compute directional metrics per condition & per position
        plot_data = []
        any_borders = False

        for cond_idx, condition_name in enumerate(conditions):
            files = condition_files[cond_idx]
            if len(files) == 0:
                print(f"Warning: No files found for {condition_name}")
                continue

            # Which positions exist for this condition?
            positions = sorted(list({int(f.split("_")[-4][3:]) for f in files}))

            for pos in positions:
                files_pos = [f for f in files if int(f.split("_")[-4][3:]) == pos]
                df_dir = extract_motile_directions(files_pos)
                axis, has_borders = axis_for_pos(pos)
                any_borders = any_borders or has_borders
                index, frac_right, frac_left = compute_direction_metrics(df_dir, axis)

                plot_data.append({
                    "condition": condition_name,
                    "position": pos,
                    "index": index,
                    "frac_right": frac_right,
                    "frac_left": frac_left,
                })

        df_plot = pd.DataFrame(plot_data)

        # Sort by custom plotting order
        df_plot["condition"] = pd.Categorical(df_plot["condition"], categories=custom_order, ordered=True)
        df_plot = df_plot.sort_values("condition")

        # Persist the per-position values alongside the plots
        df_plot.to_csv(os.path.join(path, "directional_metrics.csv"), index=False)

        # Statistics per condition (mean +/- sem across positions)
        stats = df_plot.groupby("condition").agg(
            index_mean=("index", "mean"), index_sem=("index", "sem"),
            right_mean=("frac_right", "mean"), right_sem=("frac_right", "sem"),
            left_mean=("frac_left", "mean"), left_sem=("frac_left", "sem"),
        ).reset_index()

        # "+axis" is toward the right border when borders were used, otherwise the
        # explicit cardinal chemokine direction.
        pos_label = "right border" if any_borders else f"{chemokine_direction}"
        neg_label = "left border" if any_borders else "opposite"
        axis_desc = "perpendicular to borders" if any_borders else f"cardinal {chemokine_direction}"

        conds = stats["condition"].astype(str).tolist()
        x = np.arange(len(conds))

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(max(1.2 * len(conds), 6), 4.2)
        )

        # (1) signed chemotaxis index
        ax1.bar(x, stats["index_mean"], yerr=stats["index_sem"], capsize=5, edgecolor="black")
        ax1.axhline(0.0, linestyle="-", linewidth=0.8, color="black")
        ax1.set_ylim(-1, 1)
        ax1.set_ylabel(f"Chemotaxis index\n(+ = toward {pos_label})")
        ax1.set_title(f"Signed index ({axis_desc})")
        ax1.set_xticks(x)
        ax1.set_xticklabels(conds, rotation=45, ha="right")

        # (2) fraction toward each border (grouped bars)
        w = 0.4
        ax2.bar(x - w / 2, stats["right_mean"], width=w, yerr=stats["right_sem"],
                capsize=4, edgecolor="black", label=f"toward {pos_label}")
        ax2.bar(x + w / 2, stats["left_mean"], width=w, yerr=stats["left_sem"],
                capsize=4, edgecolor="black", label=f"toward {neg_label}")
        ax2.set_ylim(0, 1)
        ax2.axhline(0.33, linestyle="--", linewidth=1, color="black", alpha=0.7)
        ax2.set_ylabel("Fraction of motile cells (+/-60 deg)")
        ax2.set_title("Fraction toward each border")
        ax2.set_xticks(x)
        ax2.set_xticklabels(conds, rotation=45, ha="right")
        ax2.legend(fontsize=8)

        outpath = os.path.join(path, "plot_directional_thresh05.png")
        plt.tight_layout()
        plt.savefig(outpath, dpi=300)
        plt.close()

        print(f"Saved: {outpath}")


def _hour_key(h):
    m = re.match(r"(\d+)", str(h))
    return int(m.group(1)) if m else 0


def plot_cell_distribution_along_axis(
    celltype,
    path_list,
    conditions,
    custom_order,
    acquisition_mode,
    pos_num,
    n_bins=20,
    motile_only=False,
):
    """
    Spatial distribution of cells along the chemokine axis over time.

    For every timepoint folder in ``path_list`` each cell is projected onto the
    left->right (perpendicular-to-borders) axis and normalized to u in [0, 1]
    (0 = left border, 1 = right border). Per condition we plot how that
    distribution changes across the hours - a shift of mass toward one border
    over time indicates a directional (chemotactic) response.

    One figure per condition is written to the parent data folder: an overlay of
    per-hour histograms plus a hour x position heatmap. Pooled counts are saved
    to ``cell_distribution_along_axis.csv``.
    """
    thresh_motile = MOTILITY_DEFINITION[celltype]
    acq_sequential = ACQUISITION_MODE[acquisition_mode]
    num_conditions = len(conditions)

    # (condition, hour) -> list of normalized positions u
    collected = defaultdict(list)
    hours = set()

    for path, _ in path_list:
        hour = os.path.basename(os.path.normpath(path)).split("_")[0]  # e.g. '3h'
        hours.add(hour)
        border_cache = {}

        for f in glob.glob(os.path.join(path, "*" + str(thresh_motile) + "umin*.csv")):
            pos = int(f.split("_")[-4][3:])
            cond_idx = pos // pos_num if acq_sequential else pos % num_conditions
            if cond_idx >= num_conditions:
                continue
            condition = conditions[cond_idx]

            if pos not in border_cache:
                ref = border_utils.reference_cdb_for_pos(path, pos)
                border_cache[pos] = border_utils.load_borders_from_path(ref)
            borders = border_cache[pos]
            if borders is None:
                continue

            df = pd.read_csv(f, index_col=0)
            if not {"id", "x", "y"}.issubset(df.columns):
                continue
            if motile_only and "motile" in df.columns:
                df = df[df["motile"] == True]
            if df.empty:
                continue

            # one point per cell (mean track position)
            g = df.groupby("id").agg(x=("x", "mean"), y=("y", "mean"))
            u = border_utils.normalized_position(borders, g["x"].values, g["y"].values)
            u = u[np.isfinite(u)]
            u = u[(u >= 0.0) & (u <= 1.0)]
            collected[(condition, hour)].extend(u.tolist())

    if not collected:
        print("plot_cell_distribution_along_axis: no data collected")
        return

    hours = sorted(hours, key=_hour_key)
    out_dir = os.path.dirname(os.path.normpath(path_list[0][0]))
    order = [c for c in custom_order if any((c, h) in collected for h in hours)]

    # remap (condition, hour_label) -> (condition, hour_index) for the renderer
    collected_idx = defaultdict(list)
    for (condition, hour), vals in collected.items():
        collected_idx[(condition, hours.index(hour))].extend(vals)

    _axis_distribution.render_distribution(
        collected_idx, order, labels=hours, out_dir=out_dir,
        prefix="cell_distribution_along_axis", n_bins=n_bins, time_title="hour",
    )

    # tidy per-bin counts for downstream stats
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    rows = []
    for condition in order:
        for hour in hours:
            h_counts, _ = np.histogram(np.asarray(collected.get((condition, hour), [])), bins=bins)
            for bi in range(n_bins):
                rows.append({"condition": condition, "hour": hour,
                             "u_center": centers[bi], "n_cells": int(h_counts[bi])})
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "cell_distribution_along_axis.csv"), index=False
    )


"""
example usage

celltype = "NK"
pathlist = [(r"Y:\nhuhn\Microscopy\mic2_mic3\cf_migration\20250527_elexa_teza_iva_nk92_12well\data\0h_corrected", None)]
conditions = ["dmso", "eti", "iva1_5", "iva1", "iva3", "iva5"]
order = ["dmso", "eti", "iva1", "iva1_5", "iva3", "iva5"]
chem_dir = "up"
acq_mode = "skip"
pos_num = 5


plot_fraction_toward_chemokine(
        celltype=celltype,
        path_list=pathlist,
        conditions=conditions,
        custom_order=order,
        chemokine_direction=chem_dir,
        acquisition_mode=acq_mode,
        pos_num=pos_num
    )
"""
