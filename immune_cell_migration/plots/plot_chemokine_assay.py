import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

MOTILITY_DEFINITION = {"NK": 6.5, "pigPBMCs": 6.0, "Jurkat": 4.0, "NK_day14": 13, "Treg": 13}
ACQUISITION_MODE = {"skip": 0, "sequential": 1}
CHEMOTACTIC_GRADIENT = {"up": 0, "down": 1, "right": 2, "left": 3}


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


def compute_fraction_toward(df, chemokine_direction):
    """Compute fraction of cells moving toward chemokine."""

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

        # Compute fraction toward chemokine per condition & per position
        plot_data = []

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
                fraction = compute_fraction_toward(df_dir, chemokine_direction)

                plot_data.append({
                    "condition": condition_name,
                    "position": pos,
                    "fraction": fraction
                })

        df_plot = pd.DataFrame(plot_data)

        # Sort by custom plotting order
        df_plot["condition"] = pd.Categorical(df_plot["condition"], categories=custom_order, ordered=True)
        df_plot = df_plot.sort_values("condition")

        # Statistics per condition
        stats = df_plot.groupby("condition")["fraction"].agg(["mean", "sem"]).reset_index()

        # Barplot
        plt.figure(figsize=(0.6 * len(custom_order), 4))
        plt.bar(stats["condition"], stats["mean"], yerr=stats["sem"], capsize=5, edgecolor="black")
        plt.ylim(0, 1)
        plt.axhline(0.5, linestyle="--", linewidth=1, color="black", alpha=0.7)
        plt.ylabel("Fraction toward chemokine")
        plt.title(f"Chemokine direction: {chemokine_direction}")
        plt.xticks(rotation=45, ha="right")

        outpath = os.path.join(path, "plot_directional_fraction.png")
        plt.tight_layout()
        plt.savefig(outpath, dpi=300)
        plt.close()

        print(f"Saved: {outpath}")

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
