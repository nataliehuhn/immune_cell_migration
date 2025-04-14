import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def extract_timepoint(folder_name):
    return os.path.basename(folder_name).split('_')[0]

def plot_persistent_fraction(parent_folder, custom_order):
    """Generates persistent fraction plots for each time step from Excel or CSV files."""

    data_by_timepoint = {}

    # Traverse the parent folder
    for root, _, files in os.walk(parent_folder):
        for file in files:
            if file.endswith(".xlsx") or file.endswith(".csv") and file.startswith("results"):
                print("file: ", file)
                file_path = Path(root)/file
                timepoint = extract_timepoint(root)

                # Read the data
                df = pd.read_excel(file_path) if file.endswith(".xlsx") else pd.read_csv(file_path)  # Handle commas in the CSV

                # Remove leading and trailing spaces from column names
                df.columns = df.columns.str.strip()
                print("Columns in DataFrame:", df.columns.tolist())

                # Ensure the relevant columns exist
                if "condition" in df.columns and "persistent fraction" in df.columns and "persistent fraction std" in df.columns:
                    print("column found!")
                    persistence_data = df[["condition", "persistent fraction", "persistent fraction std"]]
                    persistence_data["timepoint"] = timepoint

                    # Store data by timepoint
                    if timepoint not in data_by_timepoint:
                        data_by_timepoint[timepoint] = []
                    data_by_timepoint[timepoint].append(persistence_data)
                else:
                    print("persistent fraction column not found!")

    # Generate improved plots
    for timepoint, data_list in data_by_timepoint.items():
        combined_data = pd.concat(data_list, ignore_index=True)
        combined_data["condition"] = pd.Categorical(combined_data["condition"], categories=custom_order, ordered=True)
        combined_data = combined_data.sort_values("condition").reset_index(drop=True)

        plt.figure(figsize=(8, 6))
        ax = sns.barplot(
            data=combined_data,
            x="condition",
            y="persistent fraction",
            palette="Set2",
            edgecolor="black"
        )

        # Adjust bar width
        bar_width = 0.5
        for i, bar in enumerate(ax.patches):
            bar.set_width(bar_width)
            bar.set_x(bar.get_x() + (1 - bar_width) / 2)

        # Center x-ticks under the bars
        tick_positions = [bar.get_x() + bar.get_width() / 2 for bar in ax.patches]
        ax.set_xticks(tick_positions)

        # Set x-tick labels correctly
        unique_conditions = combined_data["condition"].unique()
        ax.set_xticklabels(unique_conditions)

        # Add error bars manually
        for bar, (_, row) in zip(ax.patches, combined_data.iterrows()):
            bar_center = bar.get_x() + bar.get_width() / 2
            ax.errorbar(
                x=bar_center,
                y=row["persistent fraction"],
                yerr=row["persistent fraction std"],
                fmt='none',
                c='black',
                capsize=4,
                lw=1.2
            )

        plt.xticks(rotation=45, ha="right", fontsize=16)
        plt.ylabel("Persistent Fraction (%)", fontsize=16)
        plt.xlabel("Condition", fontsize=16)
        plt.title(f"Persistent Fractions at {timepoint}", fontsize=18, weight='bold')
        plt.tight_layout()

        # Save the plot
        output_folder = os.path.join(parent_folder, f"{timepoint}_corrected")
        os.makedirs(output_folder, exist_ok=True)

        output_path = os.path.join(output_folder, f"persistent_fraction_{timepoint}.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(timepoint)
        print("persistence plot!")

    print("Persistent fraction plots saved for each timepoint.")