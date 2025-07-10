import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path


def extract_timepoint(folder_name):
    return os.path.basename(folder_name).split('_')[0]

def plot_quadrant_percentages(parent_folder, custom_order):
    """Generates stacked bar plots for quadrant percentages for each time step."""

    for root, _, files in os.walk(parent_folder):
        for file in files:
            if file == "quadrant_percentages.csv":
                print(f"Found: {file} in {root}")
                file_path = Path(root) / file
                timepoint = extract_timepoint(root)

                # Read the CSV
                df = pd.read_csv(file_path, index_col=0)

                # Sort according to custom order
                df.index = pd.Categorical(df.index, categories=custom_order, ordered=True)
                df = df.sort_index()
                labels = ["fast & persistent", "fast & not persistent", "slow & not persistent", "slow & persistent"]
                # Plot
                plt.figure(figsize=(10, 6))
                ax = df.plot(kind='bar', stacked=True, colormap='tab20', edgecolor="black", width=0.5)

                plt.ylabel('Percentage (%)', fontsize=16)
                plt.xlabel('Condition', fontsize=16)
                plt.title(f"Quadrant Percentages at {timepoint}", fontsize=18, weight='bold')
                plt.xticks(rotation=45, ha="right", fontsize=14)
                plt.legend(title="Quadrant", bbox_to_anchor=(1.05, 1), loc='upper left', labels=labels)
                plt.tight_layout()

                # Save the plot
                output_folder = os.path.join(parent_folder, f"{timepoint}_corrected")
                os.makedirs(output_folder, exist_ok=True)

                output_path = os.path.join(output_folder, f"quadrant_percentages_{timepoint}.png")
                plt.savefig(output_path, dpi=300)
                plt.close()

                print(f"Saved quadrant plot for {timepoint}.")

    print("Quadrant percentage plots saved for each timepoint.")