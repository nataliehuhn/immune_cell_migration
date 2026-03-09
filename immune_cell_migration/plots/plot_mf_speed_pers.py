import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

MOTILITY_DEFINITION = {"NK": 6.5, "pigPBMCs": 6.0, "Jurkat": 4.0, "NK_day14": 13, "Treg": 13}
ACQUISITION_MODE = {"skip": 0, "sequential": 1}
BAR_COLOR = "#4C72B0"


# -----------------------------
# utilities
# -----------------------------

def extract_timepoint(folder_name):
    return os.path.basename(folder_name).split("_")[0]


def compute_figsize(num_conditions, base_height=6.5, width_per_condition=0.85):
    fig_width = max(3.5, width_per_condition * num_conditions)
    return (fig_width, base_height)


def load_mean_data(file_path):

    df = pd.read_excel(file_path, sheet_name="mean data")
    df.columns = df.columns.str.strip()

    # normalize decimals
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(",", ".", regex=False)

    df = df.apply(pd.to_numeric, errors="ignore")

    if "condition" in df.columns:
        df["condition"] = df["condition"].astype(str).str.strip()

    return df


# -----------------------------
# core plotting function
# -----------------------------

def _plot_metric(parent_folder,
                 custom_order,
                 value_col,
                 error_col,
                 ylabel,
                 title_prefix,
                 filename_prefix):

    sns.set(style="whitegrid")
    data_by_timepoint = {}

    # collect data
    for root, _, files in os.walk(parent_folder):
        for file in files:
            if file.endswith(".xlsx"):

                file_path = os.path.join(root, file)
                timepoint = extract_timepoint(root)
                df = load_mean_data(file_path)

                if all(col in df.columns for col in ["condition", value_col, error_col]):

                    subset = df[["condition", value_col, error_col]].copy()
                    subset["timepoint"] = timepoint
                    data_by_timepoint.setdefault(timepoint, []).append(subset)

    # plotting loop
    for timepoint, data_list in data_by_timepoint.items():

        combined_data = pd.concat(data_list, ignore_index=True)

        # ---- normalize conditions safely
        combined_data["condition"] = (
            combined_data["condition"]
            .astype(str)
            .str.strip()
        )

        # keep only requested conditions
        combined_data = combined_data[
            combined_data["condition"].isin(custom_order)
        ]

        if combined_data.empty:
            print(f"No valid data for {timepoint}")
            continue

        # ---- aggregate explicitly
        aggregated = (
            combined_data
            .groupby("condition", as_index=False)
            .agg({
                value_col: "mean",
                error_col: "mean"
            })
        )

        aggregated["condition"] = pd.Categorical(
            aggregated["condition"],
            categories=custom_order,
            ordered=True
        )

        aggregated = aggregated.sort_values("condition")

        fig_size = compute_figsize(len(custom_order))
        plt.figure(figsize=fig_size)

        ax = sns.barplot(
            data=aggregated,
            x="condition",
            y=value_col,
            order=custom_order,
            color=BAR_COLOR,
            edgecolor="black",
            errorbar=None,
            width=0.6
        )

        # ---- error bars aligned to bar centers
        bar_centers = [
            patch.get_x() + patch.get_width() / 2
            for patch in ax.patches
        ]

        ax.errorbar(
            x=bar_centers,
            y=aggregated[value_col],
            yerr=aggregated[error_col],
            fmt="none",
            ecolor="black",
            capsize=4,
            lw=1.2
        )

        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_xlabel("Condition", fontsize=16)
        ax.set_title(f"{title_prefix} at {timepoint}", fontsize=18, weight="bold")

        plt.xticks(rotation=45, ha="right", fontsize=14)
        plt.tight_layout()

        plot_path = os.path.join(
            parent_folder,
            timepoint + "_corrected",
            f"{filename_prefix}_{timepoint}.png"
        )

        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300)
        plt.close()

    print(f"{filename_prefix} plots saved.")


# -----------------------------
# public wrappers
# -----------------------------

def plot_motile_fractions(parent_folder, custom_order):
    _plot_metric(
        parent_folder,
        custom_order,
        value_col="motile fraction calculated from tracks",
        error_col="mf_std",
        ylabel="Motile Fraction (%)",
        title_prefix="Motile Fractions",
        filename_prefix="motile_fraction"
    )


def plot_speed(parent_folder, custom_order):
    _plot_metric(
        parent_folder,
        custom_order,
        value_col="speed [µm/min]",
        error_col="speed_std",
        ylabel="Speed [µm/min]",
        title_prefix="Speed per Condition",
        filename_prefix="speed"
    )


def plot_persistence(parent_folder, custom_order):
    _plot_metric(
        parent_folder,
        custom_order,
        value_col="persistence",
        error_col="persistence_std",
        ylabel="Persistence",
        title_prefix="Persistence per Condition",
        filename_prefix="persistence"
    )
