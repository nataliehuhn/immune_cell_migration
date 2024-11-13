import numpy as np
import pandas as pd
from glob import glob
import os
import glob
from scipy.stats import shapiro, wilcoxon, ttest_rel, ttest_ind, mannwhitneyu, ranksums
import matplotlib.pyplot as plt
from ..utils import get_data


# glob(os.path.join(path, "*rep000_pos*zMaxProj.tif"))

def plot_all(file_paths, output_directory, time_points, savename, conditions):
    motile_fractions = []
    for condition in conditions:
        for time_point in time_points:
            for file_path in file_paths:
                path = os.path.join(file_path + r"\\" + time_point, savename + ".xlsx")
                print(path)

                data = pd.read_excel(path, 0)
                print(data)
                motile_fractions.append()

            mf = np.mean(motile_fractions)
            mf_std = np.std(motile_fractions)


def load_all_data(paths, average=True):
    data_list = []
    for index, file in enumerate(paths):
        # load the data and the config
        data = get_data(file, average)
        data.reset_index(drop=True, inplace=True)
        data_list.append(data)
    data = pd.concat(data_list)
    data.reset_index(drop=True, inplace=True)
    return data

