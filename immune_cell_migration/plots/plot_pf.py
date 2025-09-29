import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


ACQUISITION_MODE = {"skip": 0, "sequential": 1}
BAR_COLOR = "#4C72B0"  # Consistent single color

def extract_timepoint(folder_name):
    return os.path.basename(folder_name).split('_')[0]


def compute_figsize(num_conditions, base_height=6.5, width_per_condition=0.85):
    fig_width = max(3.5, width_per_condition * num_conditions)
    return (fig_width, base_height)


def plot_persistent_fraction(parent_folder, custom_order):
    sns.set(style="whitegrid")
    data_by_timepoint = {}

    for root, _, files in os.walk(parent_folder):
        for file in files:
            if (file.endswith(".xlsx") or file.endswith(".csv")) and file.startswith("results"):
                file_path = Path(root) / file
                timepoint = extract_timepoint(root)

                df = pd.read_excel(file_path) if file.endswith(".xlsx") else pd.read_csv(file_path)
                df.columns = df.columns.str.strip()

                if all(col in df.columns for col in ["condition", "persistent fraction", "persistent fraction std"]):
                    persistence_data = df[["condition", "persistent fraction", "persistent fraction std"]].copy()
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
            y="persistent fraction",
            color=BAR_COLOR,       # unified solid color like plot_speed
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
                row["persistent fraction"],
                yerr=row["persistent fraction std"],
                fmt='none',
                c='black',
                capsize=4,
                lw=1.2
            )

        plt.xticks(rotation=45, ha="right", fontsize=14)
        plt.ylabel("Persistent Fraction (%)", fontsize=16)
        plt.xlabel("Condition", fontsize=16)
        plt.title(f"Persistent Fractions at {timepoint}", fontsize=18, weight='bold')
        plt.tight_layout()

        output_folder = os.path.join(parent_folder, f"{timepoint}_corrected")
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f"persistent_fraction_{timepoint}.png")
        plt.savefig(output_path, dpi=300)
        plt.close()

    print("Persistent fraction plots saved for each timepoint.")