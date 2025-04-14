import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

MOTILITY_DEFINITION = {"NK": 6.5, "pigPBMCs": 6.0, "Jurkat": 4.0, "NK_day14": 13}
ACQUISITION_MODE = {"skip": 0, "sequential": 1}


def extract_timepoint(folder_name):
    """Extracts the time point (e.g., '0h', '6h') from the folder name."""
    return os.path.basename(folder_name).split('_')[0]


def plot_motile_fractions(parent_folder, custom_order):
    """Generates motile fraction plots for each time step from Excel files in a parent folder."""

    # Dictionary to store data for different time points
    data_by_timepoint = {}

    # Traverse the parent folder
    for root, _, files in os.walk(parent_folder):
        for file in files:
            if file.endswith("results*.xlsx") or file.endswith("results*.csv"):
                file_path = os.path.join(root, file)
                timepoint = extract_timepoint(root)

                # Read the data
                df = pd.read_excel(file_path) if file.endswith(".xlsx") else pd.read_csv(file_path)

                # Ensure the relevant columns exist
                if "condition" in df.columns and "motile fraction calculated from tracks" in df.columns and "mf_std" in df.columns:
                    motile_data = df[["condition", "motile fraction calculated from tracks", "mf_std"]]
                    motile_data["timepoint"] = timepoint

                    # Store data by timepoint
                    if timepoint not in data_by_timepoint:
                        data_by_timepoint[timepoint] = []
                    data_by_timepoint[timepoint].append(motile_data)

        # Generate improved plots
    for timepoint, data_list in data_by_timepoint.items():
        combined_data = pd.concat(data_list, ignore_index=True)
        combined_data["condition"] = pd.Categorical(
            combined_data["condition"],
            categories=custom_order,
            ordered=True
        )
        combined_data = combined_data.sort_values("condition").reset_index(drop=True)

        plt.figure(figsize=(8, 6))
        ax = sns.barplot(
            data=combined_data,
            x="condition",
            y="motile fraction calculated from tracks",
            palette="Set2",
            edgecolor="black"
        )

        # Thinner bars manually
        bar_width = 0.5
        for i, bar in enumerate(ax.patches):
            bar.set_width(bar_width)
            bar.set_x(bar.get_x() + (1 - bar_width) / 2)

        # Center x-ticks under the bars
        tick_positions = [bar.get_x() + bar.get_width() / 2 for bar in ax.patches]
        ax.set_xticks(tick_positions)
        grouped = combined_data.groupby("condition")
        ax.set_xticklabels(grouped.indices)

        # Add error bars manually
        for bar, (_, row) in zip(ax.patches, combined_data.iterrows()):
            bar_center = bar.get_x() + bar.get_width() / 2
            ax.errorbar(
                x=bar_center,
                y=row["motile fraction calculated from tracks"],  # or "persistence"
                yerr=row["mf_std"],  # or "persistence_std"
                fmt='none',
                c='black',
                capsize=4,
                lw=1.2
            )

        # Add labels on bars
        for p in ax.patches:
            height = p.get_height()
            # ax.annotate(f'{height:.1f}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom', fontsize=9, color='black')

        plt.xticks(rotation=45, ha="right", fontsize=16)
        plt.ylabel("Motile Fraction (%)", fontsize=16)
        plt.xlabel("Condition", fontsize=16)
        plt.title(f"Motile Fractions at {timepoint}", fontsize=18, weight='bold')
        plt.tight_layout()

        plot_path = os.path.join(parent_folder, timepoint + "_corrected", f"motile_fraction_{timepoint}.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300)
        plt.close()
    print(f"Plots saved in respective timepoint folders")

def plot_speed(parent_folder, custom_order):
    """Generates speed plots for each time step from Excel/CSV files."""
    sns.set(style="whitegrid")
    data_by_timepoint = {}

    # Traverse the parent folder
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

    # Generate plots
    for timepoint, data_list in data_by_timepoint.items():
        combined_data = pd.concat(data_list, ignore_index=True)
        combined_data["condition"] = pd.Categorical(
            combined_data["condition"],
            categories=custom_order,
            ordered=True
        )
        combined_data = combined_data.sort_values("condition").reset_index(drop=True)

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            data=combined_data,
            x="condition",
            y="speed [µm/min]",
            palette="Set2",
            edgecolor="black"
        )

        # Thinner bars manually
        bar_width = 0.6
        for i, bar in enumerate(ax.patches):
            bar.set_width(bar_width)
            bar.set_x(bar.get_x() + (1 - bar_width) / 2)

        # Center x-ticks under the bars
        tick_positions = [bar.get_x() + bar.get_width() / 2 for bar in ax.patches]
        ax.set_xticks(tick_positions)
        grouped = combined_data.groupby("condition")
        ax.set_xticklabels(grouped.indices)

        # Add error bars manually
        for bar, (_, row) in zip(ax.patches, combined_data.iterrows()):
            bar_center = bar.get_x() + bar.get_width() / 2
            ax.errorbar(
                x=bar_center,
                y=row["speed [µm/min]"],  # or "persistence"
                yerr=row["speed_std"],  # or "persistence_std"
                fmt='none',
                c='black',
                capsize=4,
                lw=1.2
            )

        # Add data labels on bars
        for p in ax.patches:
            height = p.get_height()
            # ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom', fontsize=9, color='black')

        plt.xticks(rotation=45, ha="right", fontsize=16)
        plt.ylabel("Speed [µm/min]", fontsize=16)
        plt.xlabel("Condition", fontsize=16)
        plt.title(f"Speed per Condition at {timepoint}", fontsize=18, weight='bold')
        plt.tight_layout()

        plot_path = os.path.join(parent_folder, timepoint + "_corrected", f"speed_{timepoint}.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300)
        plt.close()

    print("Speed plots saved in respective timepoint folders.")

def plot_persistence(parent_folder, custom_order):
    """Generates persistence plots for each time step from Excel/CSV files."""
    sns.set(style="whitegrid")
    data_by_timepoint = {}

    # Traverse the parent folder
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

    # Generate plots
    for timepoint, data_list in data_by_timepoint.items():
        combined_data = pd.concat(data_list, ignore_index=True)
        combined_data["condition"] = pd.Categorical(
            combined_data["condition"],
            categories=custom_order,
            ordered=True
        )
        combined_data = combined_data.sort_values("condition").reset_index(drop=True)

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            data=combined_data,
            x="condition",
            y="persistence",
            palette="Set2",
            edgecolor="black"
        )

        # Thinner bars manually
        bar_width = 0.6
        for i, bar in enumerate(ax.patches):
            bar.set_width(bar_width)
            bar.set_x(bar.get_x() + (1 - bar_width) / 2)

        # Center x-ticks under the bars
        tick_positions = [bar.get_x() + bar.get_width() / 2 for bar in ax.patches]
        ax.set_xticks(tick_positions)
        grouped = combined_data.groupby("condition")
        ax.set_xticklabels(grouped.indices)

        # Add error bars manually
        for bar, (_, row) in zip(ax.patches, combined_data.iterrows()):
            bar_center = bar.get_x() + bar.get_width() / 2
            ax.errorbar(
                x=bar_center,
                y=row["persistence"],  # or "persistence"
                yerr=row["persistence_std"],  # or "persistence_std"
                fmt='none',
                c='black',
                capsize=4,
                lw=1.2
            )

        # Add data labels
        for p in ax.patches:
            height = p.get_height()
            # ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom', fontsize=9, color='black')

        plt.xticks(rotation=45, ha="right", fontsize=16)
        plt.ylabel("Persistence", fontsize=16)
        plt.xlabel("Condition", fontsize=16)
        plt.title(f"Persistence per Condition at {timepoint}", fontsize=18, weight='bold')
        plt.tight_layout()

        plot_path = os.path.join(parent_folder, timepoint + "_corrected", f"persistence_{timepoint}.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300)
        plt.close()

    print("Persistence plots saved in respective timepoint folders.")
