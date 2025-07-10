import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

MOTILITY_DEFINITION = {"NK": 6.5, "pigPBMCs": 6.0, "Jurkat": 4.0, "NK_day14": 13}
ACQUISITION_MODE = {"skip": 0, "sequential": 1}
BAR_COLOR = "#4C72B0"  # Consistent single color


def extract_timepoint(folder_name):
    return os.path.basename(folder_name).split('_')[0]


def compute_figsize(num_conditions, base_height=6.5, width_per_condition=0.85):
    fig_width = max(3.5, width_per_condition * num_conditions)
    return (fig_width, base_height)


def plot_motile_fractions(parent_folder, custom_order):
    data_by_timepoint = {}

    for root, _, files in os.walk(parent_folder):
        for file in files:
            if file.endswith(".xlsx") or file.endswith(".csv"):
                file_path = os.path.join(root, file)
                timepoint = extract_timepoint(root)

                df = pd.read_excel(file_path) if file.endswith(".xlsx") else pd.read_csv(file_path)
                df.columns = df.columns.str.strip()
                df.replace(',', '.', regex=True, inplace=True)
                df = df.apply(pd.to_numeric, errors='ignore')

                if "condition" in df.columns and "motile fraction calculated from tracks" in df.columns and "mf_std" in df.columns:
                    motile_data = df[["condition", "motile fraction calculated from tracks", "mf_std"]]
                    motile_data["timepoint"] = timepoint
                    data_by_timepoint.setdefault(timepoint, []).append(motile_data)

    for timepoint, data_list in data_by_timepoint.items():
        combined_data = pd.concat(data_list, ignore_index=True)
        combined_data["condition"] = pd.Categorical(combined_data["condition"], categories=custom_order, ordered=True)
        combined_data = combined_data.sort_values("condition").reset_index(drop=True)

        fig_size = compute_figsize(len(custom_order))
        plt.figure(figsize=fig_size)
        ax = sns.barplot(
            data=combined_data,
            x="condition",
            y="motile fraction calculated from tracks",
            color=BAR_COLOR,
            edgecolor="black",
            ci=None  # Important to disable auto error bars
        )

        # Set thinner bar width and reposition
        bar_width = 0.5
        for bar in ax.patches:
            bar.set_width(bar_width)
            bar.set_x(bar.get_x() + (1 - bar_width) / 2)

        # Compute new tick positions
        tick_positions = [bar.get_x() + bar.get_width() / 2 for bar in ax.patches]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(custom_order)

        # Add manual error bars
        for bar, (_, row) in zip(ax.patches, combined_data.iterrows()):
            ax.errorbar(
                bar.get_x() + bar.get_width() / 2,
                row["motile fraction calculated from tracks"],
                yerr=row["mf_std"],
                fmt='none',
                c='black',
                capsize=4,
                lw=1.2
            )

        plt.xticks(rotation=45, ha="right", fontsize=14)
        plt.ylabel("Motile Fraction (%)", fontsize=16)
        plt.xlabel("Condition", fontsize=16)
        plt.title(f"Motile Fractions at {timepoint}", fontsize=18, weight='bold')
        plt.tight_layout()
        plot_path = os.path.join(parent_folder, timepoint + "_corrected", f"motile_fraction_{timepoint}.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300)
        plt.close()
    print("Motile fraction plots saved.")


def plot_speed(parent_folder, custom_order):
    sns.set(style="whitegrid")
    data_by_timepoint = {}

    for root, _, files in os.walk(parent_folder):
        for file in files:
            if file.endswith(".xlsx") or file.endswith(".csv"):
                file_path = os.path.join(root, file)
                timepoint = extract_timepoint(root)

                df = pd.read_excel(file_path) if file.endswith(".xlsx") else pd.read_csv(file_path)

                if all(col in df.columns for col in ["condition", "speed [µm/min]", "speed_std"]):
                    speed_data = df[["condition", "speed [µm/min]", "speed_std"]].copy()
                    speed_data["timepoint"] = timepoint
                    data_by_timepoint.setdefault(timepoint, []).append(speed_data)

    for timepoint, data_list in data_by_timepoint.items():
        combined_data = pd.concat(data_list, ignore_index=True)
        combined_data["condition"] = pd.Categorical(combined_data["condition"], categories=custom_order, ordered=True)
        combined_data = combined_data.sort_values("condition").reset_index(drop=True)

        fig_size = compute_figsize(len(custom_order))
        plt.figure(figsize=fig_size)
        ax = sns.barplot(
            data=combined_data,
            x="condition",
            y="speed [µm/min]",
            color=BAR_COLOR,
            edgecolor="black",
            ci=None
        )

        bar_width = 0.5
        for bar in ax.patches:
            bar.set_width(bar_width)
            bar.set_x(bar.get_x() + (1 - bar_width) / 2)

        tick_positions = [bar.get_x() + bar.get_width() / 2 for bar in ax.patches]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(custom_order)

        for bar, (_, row) in zip(ax.patches, combined_data.iterrows()):
            ax.errorbar(
                bar.get_x() + bar.get_width() / 2,
                row["speed [µm/min]"],
                yerr=row["speed_std"],
                fmt='none',
                c='black',
                capsize=4,
                lw=1.2
            )

        plt.xticks(rotation=45, ha="right", fontsize=14)
        plt.ylabel("Speed [µm/min]", fontsize=16)
        plt.xlabel("Condition", fontsize=16)
        plt.title(f"Speed per Condition at {timepoint}", fontsize=18, weight='bold')
        plt.tight_layout()
        plot_path = os.path.join(parent_folder, timepoint + "_corrected", f"speed_{timepoint}.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300)
        plt.close()
    print("Speed plots saved.")


def plot_persistence(parent_folder, custom_order):
    sns.set(style="whitegrid")
    data_by_timepoint = {}

    for root, _, files in os.walk(parent_folder):
        for file in files:
            if file.endswith(".xlsx") or file.endswith(".csv"):
                file_path = os.path.join(root, file)
                timepoint = extract_timepoint(root)

                df = pd.read_excel(file_path) if file.endswith(".xlsx") else pd.read_csv(file_path)

                if all(col in df.columns for col in ["condition", "persistence", "persistence_std"]):
                    persistence_data = df[["condition", "persistence", "persistence_std"]].copy()
                    persistence_data["timepoint"] = timepoint
                    data_by_timepoint.setdefault(timepoint, []).append(persistence_data)

    for timepoint, data_list in data_by_timepoint.items():
        combined_data = pd.concat(data_list, ignore_index=True)
        combined_data["condition"] = pd.Categorical(combined_data["condition"], categories=custom_order, ordered=True)
        combined_data = combined_data.sort_values("condition").reset_index(drop=True)

        fig_size = compute_figsize(len(custom_order))
        plt.figure(figsize=fig_size)
        ax = sns.barplot(
            data=combined_data,
            x="condition",
            y="persistence",
            color=BAR_COLOR,
            edgecolor="black",
            ci=None
        )

        bar_width = 0.5
        for bar in ax.patches:
            bar.set_width(bar_width)
            bar.set_x(bar.get_x() + (1 - bar_width) / 2)

        tick_positions = [bar.get_x() + bar.get_width() / 2 for bar in ax.patches]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(custom_order)

        for bar, (_, row) in zip(ax.patches, combined_data.iterrows()):
            ax.errorbar(
                bar.get_x() + bar.get_width() / 2,
                row["persistence"],
                yerr=row["persistence_std"],
                fmt='none',
                c='black',
                capsize=4,
                lw=1.2
            )

        plt.xticks(rotation=45, ha="right", fontsize=14)
        plt.ylabel("Persistence", fontsize=16)
        plt.xlabel("Condition", fontsize=16)
        plt.title(f"Persistence per Condition at {timepoint}", fontsize=18, weight='bold')
        plt.tight_layout()
        plot_path = os.path.join(parent_folder, timepoint + "_corrected", f"persistence_{timepoint}.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300)
        plt.close()
    print("Persistence plots saved.")
