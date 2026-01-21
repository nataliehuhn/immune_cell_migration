import os
import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import glob
from matplotlib.lines import Line2D

# speed_stepwidth_um_min

MOTILITY_DEFINITION = {"NK": 6.5, "pigPBMCs": 6.0, "Jurkat": 4.0, "NK_day14": 13, "Treg": 13}
# if measurement takes images for saved positions: 0, 4, 8, 12, 1, 5, 9, 13: skip
# if measurement takes images for saved positions:
ACQUISITION_MODE = {"skip": 0, "sequential": 1}
UPPER_LIMIT_KDE = {"NK": 30, "NK_day14": 30, "Jurkat": 15, "pigPBMCs": 25, "Treg": 30}


def generate_kde_plot(celltype, path_list, conditions, acquisition_mode, pos_num, custom_order):
    thresh_motile = MOTILITY_DEFINITION[celltype]
    acq_sequential = ACQUISITION_MODE[acquisition_mode]
    upper_limit = UPPER_LIMIT_KDE[celltype]

    cond_sets = [[d] for d in conditions]
    print(cond_sets)

    quadrant_summary = {}
    for path, _ in path_list:
        print(f"Processing path: {path}")
        # Set up matplotlib configurations
        plt.rcParams.update({
            'axes.linewidth': 0.5,
            'grid.linewidth': 0.25,
            'font.sans-serif': 'Arial',
            'font.size': 10,
            'axes.labelsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 8,
            'axes.axisbelow': True,
        })
        plt.rc("axes.spines", top=False, right=False)

        """Main function to handle data processing and plotting."""
        num_conditions = len(conditions)
        print("num_conditions: ", num_conditions)
        fig, axes = plt.subplots(1, num_conditions, figsize=(3*num_conditions, 3))

        # Ensure axes is iterable even if it's a single Axes object
        if num_conditions == 1:
            axes = [axes]

        data_per_condition = []

        count_cond = 0
        condition_files = [[] for _ in range(num_conditions)]

        for d in cond_sets:
            print("d in cond_sets:", d)

            filenames = glob.glob(os.path.join(path, "*" + str(thresh_motile) + "umin*.csv"))
            print(f"Found files: {filenames}")

            if len(filenames) == 0:
                print('Warning: No files found')
                continue  # Skip this condition if no files are found

            for filename in filenames:
                position_from_file = int(filename.split("_")[-4][3:])
                print(position_from_file)

                if acq_sequential:
                    if position_from_file // pos_num != count_cond:
                        print("sequential and position not right!")
                        continue

                if not acq_sequential:
                    if position_from_file % len(cond_sets) != count_cond:
                        print("skip and position not right!")
                        continue

                condition_files[count_cond].append(filename)

            print(condition_files)
            try:
                # Load and aggregate all data for this condition into a single DataFrame
                data = load_data(condition_files[count_cond])
                print("len data: ", len(data))
                data_per_condition.append(data)
            except ValueError as e:
                print(e)
                continue

            if not data_per_condition:
                raise RuntimeError("No data available for any condition.")

            count_cond += 1
            """
            print("counting cond: ", count_cond)
            try:
                data = load_data(condition_files[count_cond])
                ax = axes[count_cond]
                plt.sca(ax)
                kde_plot(data, title=conditions[count_cond])
                ax.set_ylim([0.1, upper_limit])
                ax.set_ylabel('Speed [µm/min]')
            except ValueError as e:
                print(e)
                continue
            count_cond += 1

        plt.tight_layout()
        plt.savefig(os.path.join(path, 'figure-kde-plot' + '.png'), dpi=200)
        plt.show(block=False)
        plt.close()
        """
        for plot_idx, condition_name in enumerate(custom_order):
            if condition_name not in conditions:
                print(f"Warning: '{condition_name}' not found in loaded conditions.")
                continue

            original_idx = conditions.index(condition_name)
            print("original index: ", original_idx)
            print("index now:", plot_idx)
            print("condition_name: ", condition_name)
            print("len data_per_condition: ", len(data_per_condition))
            data = data_per_condition[original_idx]
            percentages = compute_quadrant_percentages(data)
            quadrant_summary[condition_name] = percentages

            ax = axes[plot_idx]
            plt.sca(ax)
            kde_plot(data, title=condition_name)
            ax.set_ylim([0.1, upper_limit])
            ax.set_ylabel('Speed [µm/min]')

        plt.tight_layout()
        plt.savefig(os.path.join(path, 'figure-kde-plot' + '.png'), dpi=200)
        plt.show(block=False)
        plt.close()
        save_quadrant_summary(path, quadrant_summary)


def load_data(files):
    """Load and preprocess data from CSV files."""
    data_frames = []
    for file in files:
        df = pd.read_csv(file, index_col=0)

        # Ensure necessary columns exist
        required_cols = {'cos_angle', 'speed_stepwidth_um_min', 'id'}
        if not required_cols.issubset(df.columns):
            missing_cols = required_cols - set(df.columns)
            print(f"File {file} is missing columns: {missing_cols}. Skipping this file.")
            continue

        # Directly use relevant columns without averaging
        # processed = df[['cos_angle', 'speed_stepwidth_um_min', 'id']].dropna().reset_index(drop=True)
        processed = df[['cos_angle', 'speed_stepwidth_um_min', 'id']].groupby('id').mean().dropna().reset_index(drop=True)
        data_frames.append(processed)

    if not data_frames:
        raise ValueError("No valid data found.")

    # Concatenate all data from all files into a single DataFrame for this condition
    combined_data = pd.concat(data_frames, ignore_index=True)
    combined_data['speed_stepwidth_um_min'] = np.log10(combined_data['speed_stepwidth_um_min'])

    # Remove NaNs and Infs resulting from the log transformation
    combined_data = combined_data.replace([np.inf, -np.inf], np.nan).dropna()

    return combined_data


def kde_plot(data, title, pers_thres=0., speed_thres=0., plot_mf=True):
    if data.empty:
        raise ValueError("Data for plotting is empty.")

    xy = data[['cos_angle', 'speed_stepwidth_um_min']].values.T
    kde = ss.gaussian_kde(xy)

    x_min, x_max = data['cos_angle'].min(), data['cos_angle'].max()
    y_min, y_max = data['speed_stepwidth_um_min'].min(), data['speed_stepwidth_um_min'].max()

    X, Y = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )

    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(positions).reshape(X.shape)

    # Define contour levels for typical cumulative probabilities
    contour_probs = [0.1, 0.5, 0.9]
    contour_levels = [find_contour_level(kde, X, Y, p) for p in contour_probs]

    # Plot scatter points colored by density
    z_values = kde(xy)
    order = z_values.argsort()
    x_sorted, y_sorted, z_sorted = xy[0, order], xy[1, order], z_values[order]

    color = (z_sorted - np.min(z_sorted)) / np.ptp(z_sorted)
    cmap = plt.cm.get_cmap('turbo')

    plt.scatter(x_sorted, 10 ** y_sorted, c=cmap(color), s=50, alpha=0.25, lw=0)

    # Define styles and colors for each contour
    linestyles = ['--', '-', ':']
    colors = ['black', 'black', 'black']

    # Plot contours
    for lvl, ls, c, prob in zip(contour_levels, linestyles, colors, contour_probs):
        plt.contour(X, 10 ** Y, Z, levels=[lvl], colors=c, linestyles=ls, linewidths=1.5,
                    alpha=0.8)

    # Add legend for contours
    legend_lines = [Line2D([0], [0], color=c, ls=ls, lw=1.5) for c, ls in zip(colors, linestyles)]
    legend_labels = [f"{int(p * 100)}% contour" for p in contour_probs]
    # plt.legend(legend_lines, legend_labels, fontsize=8)

    if plot_mf:
        plt.axvline(pers_thres, lw=0.75, c='w', ls='--')
        plt.axhline(speed_thres, lw=0.75, c='w', ls='--')

    plt.yscale('log')
    plt.xlabel('Persistence')
    plt.ylabel('Cell speed (µm/min)')
    plt.title(title)
    plt.tight_layout()
    plt.gca().set_facecolor('#E1E1E1')


def find_contour_level(kde, X, Y, prob=0.5):
    """Find the KDE contour level that encloses the specified cumulative probability."""
    Z = kde(np.vstack([X.ravel(), Y.ravel()]))
    Z = Z.reshape(X.shape)

    Z_flat_sorted = np.sort(Z.ravel())[::-1]
    cumsum = np.cumsum(Z_flat_sorted)
    cumsum /= cumsum[-1]

    idx = np.searchsorted(cumsum, prob)
    level = Z_flat_sorted[idx]
    return level


def compute_quadrant_percentages(data, speed_thres=1.3, pers_thres=0.0):
    log_speed_thres = np.log10(speed_thres)
    q1 = data[(data['cos_angle'] >= pers_thres) & (data['speed_stepwidth_um_min'] >= log_speed_thres)]
    q2 = data[(data['cos_angle'] < pers_thres) & (data['speed_stepwidth_um_min'] >= log_speed_thres)]
    q3 = data[(data['cos_angle'] < pers_thres) & (data['speed_stepwidth_um_min'] < log_speed_thres)]
    q4 = data[(data['cos_angle'] >= pers_thres) & (data['speed_stepwidth_um_min'] < log_speed_thres)]
    total = len(data)
    return {
        "Q1": len(q1) / total * 100 if total else 0,
        "Q2": len(q2) / total * 100 if total else 0,
        "Q3": len(q3) / total * 100 if total else 0,
        "Q4": len(q4) / total * 100 if total else 0,
    }


def save_quadrant_summary(output_path, summary_dict, filename="quadrant_percentages_pooled.csv"):
    df = pd.DataFrame.from_dict(summary_dict, orient='index')
    df.index.name = 'Condition'
    df.to_csv(os.path.join(output_path, filename))
