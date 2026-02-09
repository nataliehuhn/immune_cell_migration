import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np


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

    # -----------------------------
    # data collection
    # -----------------------------
    for root, _, files in os.walk(parent_folder):
        for file in files:

            if (file.endswith(".xlsx") or file.endswith(".csv")) and file.startswith("results"):

                file_path = Path(root) / file
                timepoint = extract_timepoint(root)

                if file.endswith(".xlsx"):
                    df = pd.read_excel(file_path)
                else:
                    df = pd.read_csv(file_path)

                df.columns = df.columns.str.strip()

                # convert decimal commas
                for col in df.columns:
                    if df[col].dtype == object:
                        df[col] = df[col].astype(str).str.replace(",", ".", regex=False)

                df = df.apply(pd.to_numeric, errors="ignore")

                # normalize condition column
                if "condition" in df.columns:
                    df["condition"] = df["condition"].astype(str).str.strip()

                required_cols = [
                    "condition",
                    "persistent fraction",
                    "persistent fraction std"
                ]

                if all(col in df.columns for col in required_cols):
                    subset = df[required_cols].copy()
                    subset["timepoint"] = timepoint
                    data_by_timepoint.setdefault(timepoint, []).append(subset)

    # -----------------------------
    # plotting
    # -----------------------------
    for timepoint, data_list in data_by_timepoint.items():

        combined_data = pd.concat(data_list, ignore_index=True)

        combined_data["condition"] = combined_data["condition"].astype(str).str.strip()

        # handle empty condition column
        if combined_data["condition"].isin(["nan", "", "None"]).all():

            print(f"condition column empty at {timepoint}, assigning order")

            if len(combined_data) != len(custom_order):
                print(f"No valid data for {timepoint}")
                continue

            combined_data["condition"] = custom_order

        combined_data = combined_data[
            combined_data["condition"].isin(custom_order)
        ]

        if combined_data.empty:
            print(f"No valid data for {timepoint}")
            continue

        combined_data["condition"] = pd.Categorical(
            combined_data["condition"],
            categories=custom_order,
            ordered=True
        )

        combined_data = (
            combined_data
            .sort_values("condition")
            .reset_index(drop=True)
        )

        fig_size = compute_figsize(len(custom_order))
        plt.figure(figsize=fig_size)

        ax = sns.barplot(
            data=combined_data,
            x="condition",
            y="persistent fraction",
            order=custom_order,
            color=BAR_COLOR,
            edgecolor="black",
            errorbar=None,
            width=0.6
        )

        ax.errorbar(
            x=np.arange(len(combined_data)),
            y=combined_data["persistent fraction"],
            yerr=combined_data["persistent fraction std"],
            fmt="none",
            ecolor="black",
            capsize=4,
            lw=1.2
        )

        ax.set_ylabel("Persistent Fraction (%)", fontsize=16)
        ax.set_xlabel("Condition", fontsize=16)
        ax.set_title(f"Persistent Fractions at {timepoint}", fontsize=18, weight="bold")

        plt.xticks(rotation=45, ha="right", fontsize=14)
        plt.tight_layout()

        output_folder = os.path.join(parent_folder, f"{timepoint}_corrected")
        os.makedirs(output_folder, exist_ok=True)

        output_path = os.path.join(
            output_folder,
            f"persistent_fraction_{timepoint}.png"
        )

        plt.savefig(output_path, dpi=300)
        plt.close()

    print("Persistent fraction plots saved for each timepoint.")
