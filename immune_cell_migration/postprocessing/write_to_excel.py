import pandas as pd
import numpy as np
import os
import glob
import re
import xlsxwriter

# Configuration dictionaries
MOTILITY_DEFINITION = {"NK": 6.5, "pigPBMCs": 6.0, "Jurkat": 4.0, "NK_day14": 13}
ACQUISITION_MODE = {"skip": 0, "sequential": 1}


def excel_writer(celltype, path_list, savename, conditions, acquisition_mode, pos_num=10):
    """
    Function to write data to an Excel file.

    :param celltype: The type of cell being analyzed (e.g., NK, Jurkat)
    :param path_list: List of tuples (path, some_value)
    :param savename: Name of the Excel file to save, usually date in format 20241118
    :param conditions: List of conditions to process
    :param acquisition_mode: Acquisition mode ("skip" or "sequential")
    :param pos_num: Number of positions to process
    :param is_sorted: Boolean flag to indicate whether positions are sorted (True) or dynamically allocated (False)
    """
    thresh_motile = MOTILITY_DEFINITION[celltype]
    acq_sequential = ACQUISITION_MODE[acquisition_mode]
    cond_sets = [[d] for d in conditions]

    for path, _ in path_list:
        print(f"Processing path: {path}")
        wb = xlsxwriter.Workbook(os.path.join(path, savename + "_" + str(thresh_motile) + '.xlsx'))
        sheet1 = wb.add_worksheet('mean data')
        sheet2 = wb.add_worksheet('all positions data')

        # Set Excel formatting parameters
        bold = wb.add_format({'bold': True})
        bold.set_bold()
        format1 = wb.add_format({'bottom': 5})

        # Column headers
        xlwt_col = ['condition', 'mf [%]', 'mf_std', 'speed [µm/min]', 'speed_std', 'persistence',
                    'persistence_std', 'all_tracks', 'motile_tracks', 'motile fraction calculated from tracks',
                    '# positions', 'speed stepwidth [µm/min]', 'speed stp std']
        [sheet1.write(0, i, t, bold) for i, t in enumerate(xlwt_col)]
        [sheet2.write(0, i, t, bold) for i, t in enumerate(xlwt_col)]

        count_cond = 0
        for d in cond_sets:
            print(f"Processing condition: {d[0]}")

            # Initialize data collections for each condition
            speed1, speed_std1, speedstp1, speedstp_std1 = [], [], [], []
            direction1, dir_std1 = [], []
            mf1, mf_std1 = [], []
            alltracks1, mottracks1 = [], []
            all_pos = []

            

            filenames = glob.glob(os.path.join(path, "*" + str(thresh_motile) + "umin*.csv"))
            print(f"Found files: {filenames}")

            if len(filenames) == 0:
                print('Warning: No files found')
                continue  # Skip this condition if no files are found

            # Process each CSV file found
            for filename in filenames:
                # Z:\\nhuhn\\Microscopy\\Mic2_mic3\\test_run\\0h_corrected\\20240903-004004_pos00_x00_mode0_6.5umin5min.csv
                position_from_file = int(filename.split("_")[-4][3:])
                print(position_from_file)

                if acq_sequential:
                    if position_from_file//pos_num != count_cond:
                        print("sequential and position not right!")
                        continue

                if not acq_sequential:
                    if position_from_file % len(cond_sets) != count_cond:
                        print("skip and position not right!")
                        continue
                    
                data = pd.read_csv(filename, index_col=0)
                data_old = data
                data["id2"] = data["id"]
                data = data.groupby(["id"]).mean()
                data["id"] = data["id2"]
                data.drop(columns="id2", inplace=True)
                data["frame"] = data_old.groupby(["id"]).count().frame
                data["file"] = filename
                all_pos.append(re.findall("pos\d.", os.path.basename(filename))[0][3:])
                print("all_pos:", all_pos)

                motileCells1 = data.loc[data.motile == True]
                alltracks1.append(len(data))
                mottracks1.append(len(motileCells1))

                # Speed calculations
                sp = motileCells1[['speed_boundingbox_um_min']]
                spstep = motileCells1[['speed_stepwidth_um_min']]
                speed1.append(sp.mean()[0])
                print(speed1)
                speed_std1.append(sp.sem()[0])
                speedstp1.append(spstep.mean()[0])
                speedstp_std1.append(spstep.sem()[0])

                # Direction and persistence calculations
                dir1 = motileCells1[['cos_angle']]
                direction1.append(dir1.mean()[0])
                dir_std1.append(dir1.sem()[0])

                # Motile fraction
                # motile1 = [data[['motile', 'file']].groupby(['file']).mean().motile[i] for i in range(len(data))]
                motile1 = [data[['motile', 'file']].groupby(['file']).mean().motile[i] for i in range(0, len(data[['motile', 'file']].groupby(['file']).mean().motile))]
                mf1.append(np.nanmean(motile1) * 100)
                print(len(mf1))

            # Handle writing data based on whether the positions are sorted or dynamically allocated
            i = count_cond * pos_num
            if not acq_sequential:
                # If positions are sorted, process them sequentially for each condition
                for j in range(pos_num):
                    sheet2.write(1 + i + j, 0, d[0])  # Condition
                    sheet2.write(1 + i + j, 1, mf1[j] if not np.isnan(mf1[j]) else 0)  # Motile Fraction
                    sheet2.write(1 + i + j, 2, np.nanstd(mf1[j]))  # Motile Fraction std
                    sheet2.write(1 + i + j, 3, speed1[j] if not np.isnan(speed1[j]) else 0)  # Speed
                    sheet2.write(1 + i + j, 4, speed_std1[j] if not np.isnan(speed_std1[j]) else 0)  # Speed std
                    sheet2.write(1 + i + j, 5, direction1[j] if not np.isnan(direction1[j]) else 0)  # Direction
                    sheet2.write(1 + i + j, 6, dir_std1[j] if not np.isnan(dir_std1[j]) else 0)  # Direction std
                    sheet2.write(1 + i + j, 7, alltracks1[j] if not np.isnan(alltracks1[j]) else 0)  # All tracks
                    sheet2.write(1 + i + j, 8, mottracks1[j] if not np.isnan(mottracks1[j]) else 0)  # Motile tracks
                    sheet2.write(1 + i + j, 9, mottracks1[j]/alltracks1[j])  # Motile fraction
                    sheet2.write(1 + i + j, 10, all_pos[j])  # Position
                    sheet2.write(1 + i + j, 11, speedstp1[j]if not np.isnan(speedstp1[j]) else 0)  # Speed step width
                    sheet2.write(1 + i + j, 12,
                                 speedstp_std1[j] if not np.isnan(speedstp_std1[j]) else 0)  # Speed step std
            if acq_sequential:
                # If positions are dynamically allocated across conditions, distribute them based on number of conditions
                total_positions = pos_num * len(conditions)
                pos_range = np.arange(total_positions)
                condition_pos_start = count_cond * pos_num
                condition_pos_end = (count_cond + 1) * pos_num
                condition_positions = pos_range[condition_pos_start:condition_pos_end]

                for j, pos in enumerate(condition_positions):
                    sheet2.write(1 + i + j, 0, d[0])  # Condition
                    sheet2.write(1 + i + j, 1, mf1[j] if not np.isnan(mf1[j]) else 0)  # Motile Fraction
                    sheet2.write(1 + i + j, 2, np.nanstd(mf1[j]))  # Motile Fraction std
                    sheet2.write(1 + i + j, 3, speed1[j] if not np.isnan(speed[j]) else 0)  # Speed
                    sheet2.write(1 + i + j, 4, speed_std1[j] if not np.isnan(speed_std1[j]) else 0)  # Speed std
                    sheet2.write(1 + i + j, 5, direction1[j] if not np.isnan(direction1[j]) else 0)  # Direction
                    sheet2.write(1 + i + j, 6, dir_std1[j] if not np.isnan(dir_std1[j]) else 0)  # Direction std
                    sheet2.write(1 + i + j, 7, alltracks1[j] if not np.isnan(alltracks1[j]) else 0)  # All tracks
                    sheet2.write(1 + i + j, 8, mottracks1[j] if not np.isnan(mottracks1[j]) else 0)  # Motile tracks
                    sheet2.write(1 + i + j, 9, mottracks1[j] / alltracks1[j])  # Motile fraction
                    sheet2.write(1 + i + j, 10, all_pos[j])  # Position
                    sheet2.write(1 + i + j, 11, speedstp1[j] if not np.isnan(speedstp1[j]) else 0)  # Speed step width
                    sheet2.write(1 + i + j, 12,
                                 speedstp_std1[j] if not np.isnan(speedstp_std1[j]) else 0)  # Speed step std


                # Set row format for better visibility
                sheet2.set_row(1 + i, 15, format1)

            count_cond += 1
            sheet1.write(count_cond, 0, d[0])
            sheet1.write(count_cond, 1, np.nanmean(mf1))
            sheet1.write(count_cond, 2, (np.nanstd(mf1) / (np.sqrt(len(mf1)))))
            sheet1.write(count_cond, 3, (np.nanmean(np.where(np.isnan(speed1), 0, speed1))))
            sheet1.write(count_cond, 4, (np.nanmean(np.where(np.isnan(speed_std1), 0, speed_std1))))
            sheet1.write(count_cond, 5, (np.nanmean(np.where(np.isnan(direction1), 0, direction1))))
            sheet1.write(count_cond, 6, (np.nanmean(np.where(np.isnan(dir_std1), 0, dir_std1))))
            sheet1.write(count_cond, 7, str(np.sum(alltracks1)))
            sheet1.write(count_cond, 8, str(np.sum(mottracks1)))
            sheet1.write(count_cond, 9, str((np.sum(mottracks1) / np.sum(alltracks1)) * 100))
            sheet1.write(count_cond, 10, str(len(filename)))
            sheet1.write(count_cond, 11, (np.nanmean(np.where(np.isnan(speedstp1), 0, speedstp1))))
            sheet1.write(count_cond, 12, (np.nanmean(np.where(np.isnan(speedstp_std1), 0, speedstp_std1))))

        wb.close()
