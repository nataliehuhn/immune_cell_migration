import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from scipy.stats import ttest_ind  # or use mannwhitneyu
from itertools import combinations


def extract_timepoint(folder_name):
    """Extracts the time point (e.g., '0h', '6h') from the folder name."""
    return os.path.basename(folder_name).split('_')[0]


def pooled_data_by_timepoint(folders, expected_columns, column_renames=None):
    """
    Collects and pools data from multiple parent folders grouped by timepoint.
    Only processes files that start with 'results_file_' and end with '.xlsx'.
    """
    data_by_timepoint = {}

    for parent_folder in folders:
        for root, _, files in os.walk(parent_folder):
            for file in files:
                # Only allow results_file_*.xlsx
                if not file.startswith("results_file_") or not file.endswith(".xlsx"):
                    #print("skipping here, no results_file found!", file)
                    continue  # Skip unrelated files

                file_path = os.path.join(root, file)
                print(file)
                timepoint = extract_timepoint(root)

                try:
                    df = pd.read_excel(file_path)
                except Exception as e:
                    print(f"Failed to read {file_path}: {e}")
                    continue

                df.columns = df.columns.str.strip()
                if column_renames:
                    df.rename(columns=column_renames, inplace=True)

                if all(col in df.columns for col in expected_columns):
                    df = df[expected_columns].copy()
                    df["timepoint"] = timepoint
                    data_by_timepoint.setdefault(timepoint, []).append(df)
                else:
                    print(f"Skipping {file_path}, missing required columns.")

    return data_by_timepoint


def add_stat_annotations(ax, data, group_col, value_col, pairs, custom_order):
    """Draw significance bars and stars for condition pairs with p < 0.05."""
    y_max = data[value_col].max()
    y_offset = y_max * 0.05  # small vertical spacing
    height = y_max + y_offset * 0.5  # starting point below the top of the plot
    used_heights = []  # to avoid overlapping bars

    for group1, group2 in pairs:
        vals1 = data[data[group_col] == group1][value_col]
        vals2 = data[data[group_col] == group2][value_col]

        # Perform statistical test
        stat, pval = ttest_ind(vals1, vals2)

        if pval < 0.05:  # Only annotate if significant
            if pval < 0.001:
                star = '***'
            elif pval < 0.01:
                star = '**'
            else:
                star = '*'

            x1 = custom_order.index(group1)
            x2 = custom_order.index(group2)
            x_coords = [x1, x1, x2, x2]

            # Find next available height
            while height in used_heights:
                height -= y_offset
            used_heights.append(height)

            # Draw horizontal bracket line
            ax.plot(x_coords, [height, height + y_offset, height + y_offset, height],
                    lw=1.2, c='black')

            ax.text((x1 + x2) / 2, height + y_offset * 1.1, star,
                    ha='center', va='bottom', color='black', fontsize=14)


def plot_pooled_metric(folders, custom_order, value_col, error_col, ylabel, output_base):
    data_by_timepoint = pooled_data_by_timepoint(folders, [value_col, error_col, "condition"])

    for timepoint, data_list in data_by_timepoint.items():
        combined_data = pd.concat(data_list, ignore_index=True)
        combined_data["condition"] = pd.Categorical(combined_data["condition"], categories=custom_order, ordered=True)
        combined_data = combined_data.sort_values("condition").reset_index(drop=True)

        # Assign experiment ID
        combined_data["experiment"] = combined_data.groupby("condition").cumcount()

        width_per_condition = 0.85  # Adjust this to your liking
        base_height = 7
        num_conditions = len(custom_order)
        fig_width = width_per_condition * num_conditions
        plt.figure(figsize=(fig_width, base_height))

        ax = plt.gca()

        # Summary per condition
        summary_df = combined_data.groupby("condition").agg(
            mean_value=(value_col, "mean"),
            sem_value=(value_col, "sem")  # or "std" if you want standard deviation
        ).reindex(custom_order).reset_index()

        # Bar plot
        bar_width = 0.4
        bars = sns.barplot(
            data=summary_df,
            x="condition",
            y="mean_value",
            order=custom_order,
            color="steelblue",
            edgecolor="black",
            zorder=1
        )

        # Adjust bar widths and collect centers
        bar_centers = []
        for bar in bars.patches:
            bar.set_width(bar_width)
            new_x = bar.get_x() + (1 - bar_width) / 2
            bar.set_x(new_x)
            bar_centers.append(new_x + bar_width / 2)

        # Error bars on top of bars
        for i, (_, row) in enumerate(summary_df.iterrows()):
            ax.errorbar(
                x=bar_centers[i],
                y=row["mean_value"],
                yerr=row["sem_value"],
                fmt='none',
                c='black',
                capsize=4,
                lw=1.2,
                zorder=2
            )

        # Strip plot for single points
        sns.stripplot(
            data=combined_data,
            x="condition",
            y=value_col,
            order=custom_order,
            color='black',
            size=6,
            jitter=False,
            ax=ax,
            zorder=3
        )

        # Dotted lines per experiment
        for exp_id in combined_data["experiment"].unique():
            exp_data = combined_data[combined_data["experiment"] == exp_id]
            if len(exp_data) > 1:
                x_vals = [custom_order.index(c) for c in exp_data["condition"]]
                y_vals = exp_data[value_col].values
                ax.plot(
                    x_vals,
                    y_vals,
                    linestyle='dotted',
                    color='black',
                    alpha=0.6,
                    zorder=4
                )

        # Significance stars
        from itertools import combinations
        condition_pairs = list(combinations(custom_order, 2))
        add_stat_annotations(
            ax=ax,
            data=combined_data,
            group_col="condition",
            value_col=value_col,
            pairs=condition_pairs,
            custom_order=custom_order
        )

        # Final styling
        ax.set_xlabel("Condition", fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_title(f"{ylabel} at {timepoint}", fontsize=18, weight='bold')
        plt.xticks(rotation=45, ha="right", fontsize=14)
        plt.tight_layout()

        # Save with appropriate filename
        if "motile fraction" in value_col.lower():
            prefix = "motile_fraction_plot"
        elif "speed" in value_col.lower():
            prefix = "speed_plot"
        elif "persistence" in value_col.lower():
            prefix = "persistence_plot"
        else:
            prefix = "metric_plot"

        filename = f"{prefix}_{timepoint}.png"
        plot_path = os.path.join(output_base, filename)
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300)
        plt.close()


def plot_pooled_motile_fraction(folders, custom_order, output_base):
    plot_pooled_metric(
        folders=folders,
        custom_order=custom_order,
        value_col="motile fraction calculated from tracks",
        error_col="mf_std",
        ylabel="Motile Fraction [%]",
        output_base=output_base
    )


def plot_pooled_speed(folders, custom_order, output_base):
    plot_pooled_metric(
        folders=folders,
        custom_order=custom_order,
        value_col="speed [µm/min]",
        error_col="speed_std",
        ylabel="Speed [µm/min]",
        output_base=output_base
    )


def plot_pooled_persistence(folders, custom_order, output_base):
    plot_pooled_metric(
        folders=folders,
        custom_order=custom_order,
        value_col="persistence",
        error_col="persistence_std",
        ylabel="Persistence",
        output_base=output_base
    )
