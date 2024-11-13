import os
import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import glob

# speed_stepwidth_um_min

MOTILITY_DEFINITION = {"NK": 6.5, "pigPBMCs": 6.0, "Jurkat": 4.0, "NK_day14": 13}
ACQUISITION_MODE = {"skip": 0, "sequential": 1}

def generate_kde_plot(celltype, path_list, savename, conditions, acquisition_mode, pos_num=10):

    thresh_motile = MOTILITY_DEFINITION[celltype]
    acq_sequential = ACQUISITION_MODE[acquisition_mode]
    cond_sets = [[d] for d in conditions]

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
        fig, axes = plt.subplots(1, num_conditions, figsize=(3*num_conditions, 3))

        # Ensure axes is iterable even if it's a single Axes object
        if num_conditions == 1:
            axes = [axes]

        data_per_condition = []

        count_cond = 0
        condition_files = []

        for d in cond_sets:
            # Process each CSV file found
            filenames = glob.glob(os.path.join(path, "*" + str(thresh_motile) + "umin*.csv"))
            print(f"Found files: {filenames}")

            if len(filenames) == 0:
                print('Warning: No files found')
                continue  # Skip this condition if no files are found

            for filename in filenames:
                # Z:\\nhuhn\\Microscopy\\Mic2_mic3\\test_run\\0h_corrected\\20240903-004004_pos00_x00_mode0_6.5umin5min.csv
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

                condition_files.append(filename)

            print(condition_files)
            try:
                # Load and aggregate all data for this condition into a single DataFrame
                data = load_data(condition_files)
                print("len data: ", len(data))
                # print(len(data[0]))
                data_per_condition.append(data)
                print("len of data_per_condition: ", len(data_per_condition))
            except ValueError as e:
                print(e)
                continue
            # data_per_condition =
            count_cond += 1

            if not data_per_condition:
                raise RuntimeError("No data available for any condition.")

            for ax, data, condition in zip(axes, data_per_condition, conditions):
                plt.sca(ax)
                kde_plot(data, title=condition)
                ax.set_ylim([0.1, 30])
                circle = plt.Circle((0.65, 0.65), 0.175, color='C3', fill=False, lw=1, ls='--', transform=ax.transAxes)
                # ax.add_patch(circle)
                ax.set_ylabel('')

        plt.tight_layout()
        plt.savefig(os.path.join(path, 'figure-kde-plot' + '.png'), dpi=200)
        plt.show(block=False)
        plt.close()


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


def kde_plot(data, title='', pers_thres=0., speed_thres=0., plot_mf=True):
    """Create a KDE plot."""
    if data.empty:
        raise ValueError("Data for plotting is empty.")

    z = ss.gaussian_kde(data[['cos_angle', 'speed_stepwidth_um_min']].values.T)(
        data[['cos_angle', 'speed_stepwidth_um_min']].values.T)
    x, y = data['cos_angle'].values, data['speed_stepwidth_um_min'].values

    # Order scatter points for smooth lines
    ordering = np.argsort(z)
    x, y, z = x[ordering], y[ordering], z[ordering]

    # Normalize colors for consistent colormap
    color = (z - np.min(z)) / np.ptp(z)
    cmap = plt.cm.get_cmap('turbo')

    plt.scatter(x, 10 ** y, c=cmap(color), s=50, alpha=0.25, lw=0)

    if plot_mf:
        plt.axvline(pers_thres, lw=0.75, c='w', ls='--')
        plt.axhline(speed_thres, lw=0.75, c='w', ls='--')

    plt.yscale('log')
    plt.xlabel('Persistence')
    plt.ylabel('Cell speed (µm/min)')
    plt.title(title)
    plt.gca().set_facecolor('#E1E1E1')
