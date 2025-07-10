import os
import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import glob
import re
from matplotlib.lines import Line2D
# speed_stepwidth_um_min

MOTILITY_DEFINITION = {"NK": 6.5, "pigPBMCs": 6.0, "Jurkat": 4.0, "NK_day14": 13}
# if measurement takes images for saved positions: 0, 4, 8, 12, 1, 5, 9, 13: skip
# if measurement takes images for saved positions:
ACQUISITION_MODE = {"skip": 0, "sequential": 1}
UPPER_LIMIT_KDE = {"NK": 30, "NK_day14": 30, "Jurkat": 15, "pigPBMCs": 25}


def find_timepoints(base_folder):
    """Scan base folder for subdirectories matching *h_corrected pattern."""
    # print(f"Scanning: {base_folder}")
    subfolders = [f.name for f in os.scandir(base_folder) if f.is_dir()]
    pattern = re.compile(r'\d+h_corrected')
    timepoints = sorted([f for f in subfolders if pattern.fullmatch(f)])
    # print(timepoints)
    return timepoints


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


def generate_kde_plot(celltype, folders, conditions, acquisition_mode, pos_num, custom_order, output_base):
    thresh_motile = MOTILITY_DEFINITION[celltype]
    acq_sequential = ACQUISITION_MODE[acquisition_mode]
    upper_limit = UPPER_LIMIT_KDE[celltype]

    cond_sets = [[d] for d in conditions]

    # Detect timepoints from the first folder automatically
    timepoints = find_timepoints(folders[0])
    # print(f"Detected timepoints: {timepoints}")

    for tp in timepoints:
        print(f"\nPooling data for timepoint: {tp}")
        quadrant_summary = {}

        data_per_condition = []

        for cond_idx, cond in enumerate(conditions):
            all_files_for_condition = []

            for folder in folders:
                # print("folder in folders: ", folder)
                tp_folder = os.path.join(folder, tp)
                if not os.path.exists(tp_folder):
                    print(f"Warning: Timepoint folder {tp_folder} does not exist.")
                    continue

                pattern = f"*{thresh_motile}umin*.csv"
                files = glob.glob(os.path.join(tp_folder, pattern))

                filtered_files = []
                for f in files:
                    try:
                        position_from_file = int(f.split("_")[-4][3:])
                    except Exception as e:
                        print(f"Skipping file {f}, can't parse position: {e}")
                        continue

                    if acq_sequential:
                        if position_from_file // pos_num != cond_idx:
                            continue
                    else:
                        if position_from_file % len(cond_sets) != cond_idx:
                            continue
                    filtered_files.append(f)

                all_files_for_condition.extend(filtered_files)

            if not all_files_for_condition:
                print(f"No files found for condition '{cond}' at timepoint '{tp}'.")
                data_per_condition.append(pd.DataFrame())
                continue

            try:
                data = load_data(all_files_for_condition)
                data_per_condition.append(data)
                print(f"Loaded {len(data)} rows for condition '{cond}' at timepoint '{tp}'.")
            except ValueError as e:
                print(f"Error loading data for condition '{cond}' at timepoint '{tp}': {e}")
                data_per_condition.append(pd.DataFrame())

        # Plot the KDE for all conditions for this timepoint
        num_conditions = len(conditions)
        fig, axes = plt.subplots(1, num_conditions, figsize=(3*num_conditions, 3))
        if num_conditions == 1:
            axes = [axes]

        for plot_idx, condition_name in enumerate(custom_order):
            if condition_name not in conditions:
                print(f"Warning: '{condition_name}' not found in conditions.")
                continue

            original_idx = conditions.index(condition_name)
            data = data_per_condition[original_idx]

            if data.empty:
                print(f"No data for plotting condition '{condition_name}' at timepoint '{tp}'.")
                continue

            percentages = compute_quadrant_percentages(data)
            quadrant_summary[condition_name] = percentages

            ax = axes[plot_idx]
            plt.sca(ax)
            kde_plot(data, title=f"{condition_name} ({tp})")
            ax.set_ylim([0.1, upper_limit])
            ax.set_ylabel('Speed [µm/min]')

        plt.tight_layout()
        save_path = os.path.join(output_base, f'figure-kde-plot-pooled-{tp}.png')
        # print("savepath: ", save_path)
        plt.savefig(save_path, dpi=200)
        plt.show(block=False)
        plt.close()

        save_quadrant_summary(folders[0], quadrant_summary, filename=f"quadrant_percentages_pooled_{tp}.csv")


def kde_plot(data, title, pers_thres=0., speed_thres=0., plot_mf=True):
    """Create a KDE plot with contours."""
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

    # Define contour levels for desired cumulative probabilities
    contour_probs = [0.1, 0.5, 0.9]
    contour_levels = [find_contour_level(kde, X, Y, p) for p in contour_probs]

    z_values = kde(xy)
    order = z_values.argsort()
    x_sorted, y_sorted, z_sorted = xy[0, order], xy[1, order], z_values[order]

    color = (z_sorted - np.min(z_sorted)) / np.ptp(z_sorted)
    cmap = plt.cm.get_cmap('turbo')

    plt.scatter(x_sorted, 10**y_sorted, c=cmap(color), s=50, alpha=0.25, lw=0)

    # Define contour styles
    linestyles = ['--', '-', ':']
    colors = ['black', 'black', 'black']

    for lvl, ls, c, prob in zip(contour_levels, linestyles, colors, contour_probs):
        plt.contour(X, 10**Y, Z, levels=[lvl], colors=c, linestyles=ls, linewidths=1.5, alpha=0.8)

    # Add legend
    legend_lines = [Line2D([0], [0], color=c, ls=ls, lw=1.5) for c, ls in zip(colors, linestyles)]
    legend_labels = [f"{int(p*100)}% contour" for p in contour_probs]
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
