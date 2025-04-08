import numpy as np
import pandas as pd
from glob import glob
import os
import glob
from scipy.stats import shapiro, wilcoxon, ttest_rel, ttest_ind, mannwhitneyu, ranksums
import matplotlib.pyplot as plt
from ..utils import get_data
import seaborn as sns

MOTILITY_DEFINITION = {"NK": 6.5, "pigPBMCs": 6.0, "Jurkat": 4.0, "NK_day14": 13}
ACQUISITION_MODE = {"skip": 0, "sequential": 1}

def extract_timepoint(folder_name):
    """Extracts the time point (e.g., '0h', '6h') from the folder name."""
    return os.path.basename(folder_name).split('_')[0]


def plot_motile_fractions(parent_folder):
    """Generates motile fraction plots for each time step from Excel files in a parent folder."""

    # Dictionary to store data for different time points
    data_by_timepoint = {}

    # Traverse the parent folder
    for root, _, files in os.walk(parent_folder):
        for file in files:
            if file.endswith(".xlsx") or file.endswith(".csv"):
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

    # Generate plots
    for timepoint, data_list in data_by_timepoint.items():
        combined_data = pd.concat(data_list, ignore_index=True)

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            data=combined_data,
            x="condition",
            y="motile fraction calculated from tracks",
            yerr=combined_data["mf_std"],
            capsize=0.2,
            color="white",
            edgecolor="black"
        )

        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Motile Fraction (%)")
        plt.xlabel("Condition")
        plt.title(f"Motile Fractions at {timepoint}")
        plt.tight_layout()

        # Save plot in the corresponding timepoint folder
        plot_path = os.path.join(parent_folder, timepoint + "_corrected", f"motile_fraction_{timepoint}.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        plt.close()

    print(f"Plots saved in respective timepoint folders")
