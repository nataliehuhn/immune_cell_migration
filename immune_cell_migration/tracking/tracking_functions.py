"""
This script takes folders with clickpoints databases within them. Every clickpoints database requires a mask segmenting
the cells in every frame. This script creates tracks for the specified objects and saves them to the clickpoints
database. Those tracks are based on the overlap from an object in one frame to the next frame and are subjected to
stitching with the provided parameters.
"""
import os
import numpy as np
from scipy import ndimage
from skimage.morphology import remove_small_objects
import copy
from skimage.measure import regionprops

""" Tracking parameters """
start_frame = 1  # frame to start the tracking at .. 1
track_name = "nk_tracks_greedy_stitched_without_short_high_z"  # Name of the tracks within clickpoints
track_color = "#00ffff"  # Color of the tracks
# maximal value for the distance matrix for fast k562 cells (bigger distances can't be connected)
k562_fast_max_value = 15
nk_fast_max_value = 32  # maximal value for the distance matrix for fast nk cells
minimal_cell_size = 30  # all segmentations below this size will be removed
nk_cell_mask_number = 1  # 2 for claus or for killing assay  # 1 normally,only this number will be used in the segmentation mask for tracking nk cells
# only those numbers will be used in the segmentation mask for tracking k562 cells

k562_cell_mask_number_living = 2

k562_cell_mask_number_dead = 5
z_lim = 100  # z difference limit for overlap tracking (for overlapping cells)(bigger z distances can't be connected)
z_lim_fast = 4  # z difference limit for nearest neighbour after overlap (bigger z distances can't be connected)
path_to_drift_folder = "E:\LucasScricpts\calculatedDrifts"  # folder for the drift files
use_drift = False  # if true use a drift correction (must be calculated beforehand (calculate_drift.py))
stitch = True  # if true the tracks will be subjected to stitching
block_k562_and_dead_areas = False  # if true, remove all segmentations within half a cell diameter of K562 cells
track_k562 = False  # track k562 instead of nk cells (will use all tracking parameters for k562 cells)
# if true, each track will be stitched to it best matching track if they're cost is below the maximal cost
# if false only tracks which can be stitched unambiguously will be stitched
use_best_fitting_tracks_for_stitching = True #Greedy Algorithm is used, if False: Save Stitcher
interpolate_missing_positions = False  # interpolate all missing positions
remove_all_short_tracks = True  # if true remove all tracks below the minimal track length
min_track_len = 4  # remove all tracks below this minimal track len
stitch_again_after_removing_short_tracks = True  # If true, try to stitch tracks again after short tracks were removed
# if true, all tracks which doesn't start at the starting frame will be removed (will be done after stitching)
remove_tracks_starting_after_the_start_frame = False  # (only useful for k562 tracks since they don't move)
# remove all tracks which doesn't start at frame 1 and are below this length (only for k562 cells)
min_len_after_interpolation = 999
bias_tracks_starting_from_frame_1 = False  # give tracks starting at the starting frame a cost reduction for stitching
bias_reduced_cost = 120
""" General stitching parameters """
step_limiter = 20  # limits the number of stitching steps (usually 4-5 will be enough and it will stop automatically)
max_cost = 50  # limits the maximal accepted cost for stitching (tracks with bigger stitching costs can't be stitched)
""" Stitching parameters for k562 tracks """
# multiplicative factor for the xy distance for stitching k562 tracks within the same frame
xy_fac_k562_for_tracks_in_same_frame = 3
xy_fac_k562 = 3  # multiplicative xy factor for stitching k562 tracks
z_fac_k562 = 2  # multiplicative z factor for stitching k562 tracks
frame_fac_pos_k562 = 6  # multiplicative frame difference factor for stitching k562 tracks which don't overlap
frame_fac_neg_k562 = 10  # multiplicative frame difference factor for stitching k562 tracks that overlap
k562_initial_stitches = 4  # number of steps, in which only stitches with a certain frame difference are allowed
""" Stitching parameters for nk tracks"""
# multiplicative factor for the xy distance for stitching NK tracks within the same frame
xy_fac_for_tracks_in_same_frame = 5
z_fac = 3  # multiplicative z factor for stitching nk tracks
frame_fac_pos = 10  # multiplicative frame difference factor for stitching k562 tracks which don't overlap
frame_fac_neg = 20  # multiplicative frame difference factor for stitching k562 tracks that overlap
nk_initial_stitches = 0  # for save stitcher, number of steps, in which only stitches with a frame difference of one are allowed


def calculate_nk_diameter(cdb, cur_frame):
    """
    Calculates the mean half major axis length of all nk cells within the specified frame
    :param cdb: (object) clickpoints database
    :param cur_frame: (int) number of a single frame
    :return: mean half major axis length of all nk cells in the specified frame
    """
    nk_min_ind = cdb.getImage(frame=cur_frame, layer="MinIndices")
    mask = cdb.getMask(image=nk_min_ind).data
    nk_mask = (mask == nk_cell_mask_number)  # only consider nk cells
    nk_mask = remove_small_objects(nk_mask, min_size=minimal_cell_size) + 0  # remove segmentations below a minimal size
    nk_mask_l, nk_mask_num = ndimage.label(nk_mask)
    regions = regionprops(nk_mask_l)
    diameters = [region.major_axis_length for region in regions]  # calculates the major axis length for all cells
    mean_radius = int(np.mean(diameters)/2.)
    print("Mean_diam: ", mean_radius)
    return mean_radius


def get_nk_label(cdb, cur_frame, drift=None, max_drifts=None, mean_radius=None):
    """
    Creates a labeled mask for the specified frame, and returns the xyz-values for all labeled objects
    :param cdb: (object) clickpoints database
    :param cur_frame: (int) number of a single frame
    :param drift: (array) array containing the drift for every frame in respect to the first frame
    :param max_drifts: (array) maximal drift in x and y direction
    :param mean_radius: (float) mean half major axis length of nk cells
    :return: labeled nk mask, number of nk cells, z,xy positions for the specified frame
    """
    nk_ind = cdb.getImage(frame=cur_frame, layer="MaxIndices")  # MinIndices image for z positions
    mask = cdb.getMask(image=nk_ind).data
    nk_ind = nk_ind.data
    if track_k562:  # if yes, use the number in the mask which denotes K562 cells
        nk_mask = (mask == k562_cell_mask_number_living)  # denotes living K562 cells
        dead_mask = (mask == k562_cell_mask_number_dead)  # denotes dead k562 cells
        # Combine any cells which are partially segmented as living and partially as dead, to one living cell
        dead_mask_dil = ndimage.binary_dilation(dead_mask)  # dilate dead cells to detect any overlap with a living one
        dead_mask_l, dead_mask_num = ndimage.label(dead_mask_dil)
        regions_dead = regionprops(dead_mask_l)
        comb_mask = dead_mask_l * nk_mask
        for idx, region in enumerate(regions_dead):
            coordinates = region.coords
            y = coordinates[:, 0]
            x = coordinates[:, 1]
            if np.sum(comb_mask[y, x]) != 0:  # Check if a dead cell overlaps with a living cell
                nk_mask[(dead_mask_l == idx+1) * dead_mask] = k562_cell_mask_number_living
    else:  # if no, use the number in the mask which denotes nk cells
        nk_mask = (mask == nk_cell_mask_number)
    # remove any segmentations below minimal size
    nk_mask = remove_small_objects(nk_mask, min_size=minimal_cell_size) + 0
    if drift is not None:  # if yes, use the drift values within the drift array to correct for a drift
        # Obtains the field of view, which can be seen in every frame
        nk_mask = nk_mask[
                  int(max_drifts[0] - drift[cur_frame, 0]):int(max_drifts[0] - drift[cur_frame, 0] + max_drifts[2]),
                  int(max_drifts[1] - drift[cur_frame, 1]):int(max_drifts[1] - drift[cur_frame, 1] + max_drifts[3])]
        nk_ind = nk_ind[
                     int(max_drifts[0] - drift[cur_frame, 0]):int(max_drifts[0] - drift[cur_frame, 0] + max_drifts[2]),
                     int(max_drifts[1] - drift[cur_frame, 1]):int(max_drifts[1] - drift[cur_frame, 1] + max_drifts[3])]
    if block_k562_and_dead_areas:  # if yes, any nk cell segments within a certain radius of a k562 cell are removed
        k562_mask_dead = ((mask == k562_cell_mask_number_living) + (mask == k562_cell_mask_number_dead))
        if drift is not None:
            k562_mask_dead = k562_mask_dead[int(max_drifts[0] - drift[cur_frame, 0]):int(
                max_drifts[0] - drift[cur_frame, 0] + max_drifts[2]), int(max_drifts[1] - drift[cur_frame, 1]):int(
                max_drifts[1] - drift[cur_frame, 1] + max_drifts[3])]
        for rad in range(mean_radius):  # blow up any k562 cell segmentation by the mean nk cell radius
            k562_mask_dead = ndimage.binary_dilation(k562_mask_dead).astype(k562_mask_dead.dtype)
        edge_mask = np.zeros(k562_mask_dead.shape)  # also remove any nk cell segments at the border of the image
        edge_mask[:, :mean_radius] = 1
        edge_mask[:, -mean_radius:] = 1
        edge_mask[:mean_radius, :] = 1
        edge_mask[-mean_radius:, :] = 1
        blocked_mask = ((k562_mask_dead + edge_mask) > 0) + 0  # mask containing all blocked areas
        nk_mask = (nk_mask - nk_mask * blocked_mask).astype("bool")
        nk_mask = remove_small_objects(nk_mask, min_size=minimal_cell_size) + 0
        nk_mask_l, nk_mask_num = ndimage.label(nk_mask)
        regions = regionprops(nk_mask_l, nk_ind)
    else:  # if no, simply take the nk cell mask as is
        nk_mask_l, nk_mask_num = ndimage.label(nk_mask)
        regions = regionprops(nk_mask_l, nk_ind)
    z_positions = []
    pixel_nums = []
    for region in regions:
        coor = region.coords.transpose()
        z_positions.append(np.median(nk_ind[coor[0], coor[1]]))
        pixel_nums.append(region.area)
    # z_positions = [region.mean_intensity for region in regions]
    yx_positions = [list(region.centroid) for region in regions]
    return nk_mask_l, nk_mask_num, z_positions, yx_positions, pixel_nums


def remove_short_tracks(tracks, min_len=2, remove_after_interpolation=False):
    """
    Removes all tracks with a length below the minimum required length
    :param tracks: (dict) all tracks
    :param min_len: (int) minimal required track length
    :param remove_after_interpolation: (bool) if true, only tracks which don't start at frame 1 will be affected
    :return: all tracks with a length bigger than or equal to the minimum required length
    """
    to_remove = []
    for track_id, values in tracks.items():
        if np.sum(~np.isnan(values["pos"]).any(axis=1)) < min_len:  # only keep tracks with the minimum required length
            if remove_after_interpolation:
                if values["start_frame"] != start_frame:
                    to_remove.append(track_id)
            else:
                to_remove.append(track_id)
    for rem in to_remove:
        tracks.pop(rem, None)
    print("Removed %d tracks" % len(to_remove))
    return tracks


def get_pixel_overlap(mask1, mask2, mask1_num, mask2_num):
    """
    Calculates the overlap between each label of two labeled mask
    :param mask1: (array) labeled mask
    :param mask2: (array) labeled mask to be compared with mask1
    :param mask1_num: (int) number of labels in mask1
    :param mask2_num: (int) number of labels in mask2
    :return: 2-dimensional array containing the pixel overlap for each label from mask1 and mask2
    """
    overlap_matrix = np.zeros((mask1_num+1, mask2_num+1))
    regions = regionprops(mask1)
    combined_mask = mask2 * ((mask1 > 0) * (mask2 > 0))  # Mask containing only overlapping pixels
    for idx, region in enumerate(regions):  # iterate through mask1 regions
        coordinates = region.coords
        y = coordinates[:, 0]
        x = coordinates[:, 1]
        occ = np.unique(combined_mask[y, x])  # obtain all mask2 values at a mask1 region
        for num in occ:
            if num != 0:
                overlap_matrix[idx+1, num] = np.sum(combined_mask[y, x] == num)  # count the pixel overlap
    return overlap_matrix


def merge_tracks(tracks1, tracks2):
    """
    Merge tracks according to their starting frame and label and their ending frame and label
    (merge start_frame 4, label[5,6] with start_frame 5 label[6,7])
    :param tracks1: (dict) tracks1
    :param tracks2: (dict) tracks to merge with tracks1
    :return: Dictionary containing merged tracks and list containing the ids for non merged tracks
    """
    # keep the ids of tracks1 and merge tracks2 to them
    tracks3 = copy.deepcopy(tracks1)
    merged_add_after = []
    merged_add_before = []
    tracks_not_merged = []
    for track_id, values in tracks2.items():  # iterate through tracks2
        merged = 0
        start_pos = [values["start_frame"], values["label"][0]]  # obtain starting frame and label
        end_pos = [values["start_frame"] + len(values["label"]) - 1, values["label"][-1]]
        for track_id2, values2 in tracks1.items():  # iterate through tracks1
            sf = values2["start_frame"]  # obtain starting frame
            if sf <= start_pos[0]:  # in case track1 doesn't start after track2
                try:
                    la = values2["label"][start_pos[0] - sf]  # obtain the track1 label at the starting frame of track2
                    if la == start_pos[1] and len(values2["label"]) == start_pos[0] - sf + 1:
                        # if the tracks can be merged, merge them within tracks3
                        tracks3[track_id2]["label"].extend(values["label"][1:])
                        merged_add_after.append([track_id2, track_id])  # save the ids of the merged tracks
                        merged = 1
                except:
                    pass
            elif sf == end_pos[0]:  # in case track1 starts at the end of track2
                try:
                    la = values2["label"][0]
                    if la == end_pos[1]:
                        # if the tracks can be merged, merge them within tracks3
                        tracks3[track_id2]["start_frame"] = start_pos[0]
                        tracks3[track_id2]["label"] = tracks2[track_id]["label"][:-1] + tracks1[track_id2]["label"]
                        merged_add_before.append([track_id, track_id2])  # save the ids of the merged tracks
                        merged = 1
                except:
                    pass
        if merged == 0:  # save the id for all tracks, that couldn't be merged
            tracks_not_merged.append(track_id)
    to_merge = []
    # check if two merged tracks can be merged together (example: merged tracks [5,6] and merged tracks [6,7] can be
    # merged to [5,6,7])
    # Here the form is [track1_id1, track2_id],[track2_id,track1_id2] to be merged to [track1_id1, track1_id2]
    for m in merged_add_before:
        for m2 in merged_add_after:
            if m[0] == m2[1]:
                to_merge.append([m2[0], m[1]])
    to_merge = np.array(to_merge)
    fin_merges = []
    already_checked = []
    for item in to_merge:  # check if different merged tracks can be merged together
        # here the form is [track1_id1, track1_id2],[track1_id2, track1_id3]
        do_append = 1
        if item[0] not in already_checked:  # check if the current tracks has already been merged to another one
            already_checked.append(item[0])
            in_question = item[1]  # track to check for further merges
            fin_merge = item.tolist()  # initialise final merge
            while len(np.where(to_merge[:, 0] == in_question)[0]) != 0:  # continue till the track has no further merges
                new = to_merge[np.where(to_merge[:, 0] == in_question)[0][0]]  # new track to check for merges
                if new[0] in already_checked:  # if yes, track has already been merged after
                    pos_to_insert = np.where([pos[0] for pos in fin_merges] == new[0])[0][0]
                    for l, k in enumerate(fin_merge[:-1]):  # add the current track in front of the new track
                        fin_merges[pos_to_insert].insert(l, k)
                    do_append = 0
                    break
                else:  # if no, simply add the new track after the current
                    fin_merge.append(new[-1])
                    in_question = copy.deepcopy(new[-1])
                    already_checked.append(new[0])
            if do_append:  # if yes, add the current merged track to the final merges
                fin_merges.append(fin_merge)
    for merge in fin_merges:  # finally merge all tracks, which can be merged, together
        label = []
        first = 1
        frame1 = tracks3[merge[0]]["start_frame"]
        for m in merge:
            if first:  # if yes, add the labels of the current track to the list
                label.extend(tracks3[m]["label"])
            else:  # if no, add the labels of the current track to the list, and delete the current track
                frame_cur = tracks3[m]["start_frame"]
                label.extend(tracks3[m]["label"][frame1 + len(label) - frame_cur:])
                tracks3.pop(m, None)
            first = 0
        tracks3[merge[0]]["label"] = label
    return tracks3, tracks_not_merged


def get_tracks_for_stitching(tracks):
    """
    Returns the tracks in a format, which is easy to use for stitching
    :param tracks: (dict) tracks to reformat
    :return: tracks to be used for stitching
    """
    # initialise array [track_id, start/end, xyz pix start/end frame]
    tracks_for_stitching = np.zeros((len(tracks), 2, 5))
    for k, track in enumerate(tracks):
        t_start = tracks[track]["start_frame"]  # start frame of the current track
        t_end = t_start + len(tracks[track]["pos"]) - 1  # end frame of the current track
        tracks_for_stitching[k, 0, :] = tracks[track]["pos"][0] + [t_start]
        tracks_for_stitching[k, 1, :] = tracks[track]["pos"][-1] + [t_end]
    return tracks_for_stitching


def add_drift_for_cdb(tracks, max_drift, drift):
    """
    Re adds the drift to the tracks, before writing them in the clickpoints database
    :param tracks: (dict) tracks
    :param max_drift: (list) maximal drift in x and y direction
    :param drift: (array) contains xy drift for every frame
    :return: tracks with added drift
    """
    for track_id, values in tracks.items():
        fframe = values["start_frame"]
        new_positions = []
        for pos in values["pos"]:
            new_positions.append(
                [max_drift[0] - drift[fframe, 0] + pos[0], max_drift[1] - drift[fframe, 1] + pos[1], pos[2], pos[3]])
            fframe += 1
        tracks[track_id]["pos"] = copy.deepcopy(new_positions)
    return tracks


def find_matches(tracks, ids, first_time=0, find_best_stitches=False):
    """
    Find tracks to stitch, by using a cost matrix. Only allow stitches which are unambiguously
    :param tracks: (array) tracks to use for stitching
    :param ids: (list) track ids
    :param first_time: (int) allow only stitches of tracks which are this number of frames apart
    :param find_best_stitches: (bool) if true, stitch the best fitting tracks, else stitch only those tracks which can
     be stitched unambiguously
    :return: list of id pairs to be stitched (example [2,3],[5,6])
    """
    distance_matrix = np.zeros((len(tracks), len(tracks)))
    print("Distance matrix:", np.shape(distance_matrix))
    dt = (tracks[None, :, 0, 4] - tracks[:, None, 1, 4])  # calculate frame difference
    dis_xy = np.linalg.norm((tracks[None, :, 0, :2] - tracks[:, None, 1, :2]), axis=-1)  # calculate xy distance
    dis_z = np.abs(tracks[None, :, 0, 2] - tracks[:, None, 1, 2])  # calculate z distance
    e2e = ((tracks[None, :, 1, 4] - tracks[:, None, 1, 4]) > 0) + 0  # check for end to end distance
    s2s = ((tracks[None, :, 0, 4] - tracks[:, None, 0, 4]) > 0) + 0  # check for start to start distance
    if first_time == 1:  # allow only stitches of tracks which are one frame apart
        dt_mask_neg = dt == -1
        dt_mask_pos = dt == 1
        dt_mask_0 = dt == 0
    elif first_time > 1:  # allow only stitches which are first_time frames apart
        dt_mask_neg = dt == -1  # for overlapping tracks, allow only one frame
        dt_mask_pos = dt == first_time
        dt_mask_0 = dt == 0
    else:  # no frame limit for stitches
        dt_mask_neg = dt < 0
        dt_mask_pos = dt > 0
        dt_mask_0 = dt == 0
    if track_k562:  # if yes, use the parameters for tracking k562 cells
        # From (xy_dist + z-fac*z-dist/dt) + c*dt
        distance_matrix[dt_mask_0] = (xy_fac_k562_for_tracks_in_same_frame * dis_xy[dt_mask_0] + z_fac_k562 * dis_z[
            dt_mask_0]) * e2e[dt_mask_0] * s2s[dt_mask_0]
        distance_matrix[dt_mask_neg] = (xy_fac_k562 * dis_xy[dt_mask_neg] + z_fac_k562 * dis_z[
            dt_mask_neg] + frame_fac_neg_k562 * -dt[dt_mask_neg]) * e2e[dt_mask_neg] * s2s[dt_mask_neg]
        distance_matrix[dt_mask_pos] = (xy_fac_k562 * dis_xy[dt_mask_pos] + z_fac_k562 * dis_z[
            dt_mask_pos] + frame_fac_pos_k562 * dt[dt_mask_pos]) * e2e[dt_mask_pos] * s2s[dt_mask_pos]
    else:  # use the parameters for tracking nk cells
        # From (xy_dist + z-fac*z-dist/dt) + c*dt
        distance_matrix[dt_mask_0] = (xy_fac_for_tracks_in_same_frame * dis_xy[dt_mask_0] + z_fac * dis_z[dt_mask_0]) *\
                                     e2e[dt_mask_0] * s2s[dt_mask_0]
        distance_matrix[dt_mask_neg] = (((dis_xy[dt_mask_neg] + z_fac * dis_z[dt_mask_neg]) / -dt[
            dt_mask_neg]) + frame_fac_neg * -dt[dt_mask_neg]) * e2e[dt_mask_neg] * s2s[dt_mask_neg]
        distance_matrix[dt_mask_pos] = (((dis_xy[dt_mask_pos] + z_fac * dis_z[dt_mask_pos]) / dt[
            dt_mask_pos]) + frame_fac_pos * dt[dt_mask_pos]) * e2e[dt_mask_pos] * s2s[dt_mask_pos]
    print("Distance matrix calculated")
    if find_best_stitches:
        matches_1 = []
        distance_matrix[distance_matrix == 0] = np.nan  # Replace all values of zero with nan
        if bias_tracks_starting_from_frame_1:
            biased_tracks = np.where(tracks[:, 0, 4] == start_frame)[0]
            distance_matrix[biased_tracks, :] -= bias_reduced_cost
        while (np.nanmin(distance_matrix)) < max_cost:  # find matches starting from the best one
            min_posi = list(np.unravel_index(np.nanargmin(distance_matrix), distance_matrix.shape))
            distance_matrix[min_posi[0], :] = np.nan
            distance_matrix[:, min_posi[1]] = np.nan
            matches_1.append(min_posi)
        matches_1 = np.array(matches_1).transpose()
        if len(matches_1) == 0:
            matches_1 = np.array([[], []]).astype("int")
        ids = np.array(ids)
    else:
        distance_matrix[distance_matrix == 0] = max_cost + 1  # Replace all values of zero with the maximal costs
        distance_matrix = distance_matrix <= max_cost  # only take entries below or equal to max_cost
        track1_matches = np.sum(distance_matrix, axis=1)  # Count the number of matches each track1 got
        track2_matches = np.sum(distance_matrix, axis=0)  # Count the number of matches each track2 got
        # Only allow stitching, where a single match was found
        distance_matrix[np.where(track1_matches != 1)[0], :] = False
        # Only allow stitching, where a single match was found
        distance_matrix[:, np.where(track2_matches != 1)[0]] = False
        matches_1 = np.where(distance_matrix == True)  # The remaining entries full fill all conditions
        ids = np.array(ids)
    return [[x, y] for x, y in zip(ids[matches_1[0]], ids[matches_1[1]])]  # return a list of tracks to stitch


def average_measurements(meas1, meas2):
    """
    Take the mean value for two measurements
    :param meas1: (list) measurement1
    :param meas2: (list) measurement2
    :return: mean of two measurements
    """
    meas1 = np.array(meas1)
    meas2 = np.array(meas2)
    return list(np.nanmean([meas1, meas2], axis=0))


def stitch_tracks(tracks, find_best_fitting_stitches=False):
    """
    Stitches tracks, which can only be stitched to one track
    :param tracks: (dict) tracks to be stitched
    :param find_best_fitting_stitches: (bool) if true, stitch the best fitting tracks, else stitch only those tracks
     which can be stitched unambiguously
    :return: dictionary containing track ids and their positions
    """
    intern_tracks = copy.deepcopy(tracks)
    len_test = 0
    len_test2 = 1
    limit = 1
    while limit < step_limiter and (len_test - len_test2) != 0:  # As long as something gets stitched keep going
        len_test = len(intern_tracks)
        print("Tracks:", len_test)
        tracks_for_stitching = get_tracks_for_stitching(intern_tracks)  # Gets the tracks in a better fromat
        ids = [x for x in intern_tracks]
        if track_k562:
            if limit <= k562_initial_stitches:  # the first few steps only allow for a specific frame distance
                len_test = 0
                # finds all possible matches
                found_matches = find_matches(tracks_for_stitching, ids, first_time=limit,
                                             find_best_stitches=find_best_fitting_stitches)
            else:
                found_matches = find_matches(tracks_for_stitching, ids, find_best_stitches=find_best_fitting_stitches)
        else:
            if limit <= nk_initial_stitches:  # the first few steps only allow for a frame distance of one
                len_test = 0
                found_matches = find_matches(tracks_for_stitching, ids, first_time=limit,
                                             find_best_stitches=find_best_fitting_stitches)
            else:
                found_matches = find_matches(tracks_for_stitching, ids, find_best_stitches=find_best_fitting_stitches)
        print("Matches:", len(found_matches))
        translate_ids = {old_id: old_id for old_id in intern_tracks}  # keeps track of stitched tracks
        for group in found_matches:  # go through all found matches and stitch them
            start = tracks_for_stitching[ids.index(group[1])][0][-1]
            end = tracks_for_stitching[ids.index(group[0])][1][-1]
            track1_id = translate_ids[group[0]]
            while track1_id != translate_ids[track1_id]:
                track1_id = translate_ids[track1_id]  # finds the right track id, if previous tracks have been stitched
            track2_id = translate_ids[group[1]]
            while track2_id != translate_ids[track2_id]:
                track2_id = translate_ids[track2_id]  # finds the right track id, if previous tracks have been stitched
            track1 = intern_tracks[track1_id]
            track2 = intern_tracks[track2_id]
            if start == end:  # check if the first track ends in the same frame like the second starts
                # Reposition Marker
                meas1 = track1["pos"][-1]
                meas2 = track2["pos"][0]
                track1["pos"][-1] = average_measurements(meas1, meas2)  # average their positions
                for meas in track2["pos"][1:]:
                    track1["pos"].append(meas)
            elif start < end:  # if the second track starts before the first ends, reposition their marker
                frame_diff = end - start
                # Reposition Marker
                for p2, p1 in enumerate(np.arange(-int(frame_diff)-1, 0, 1)):
                    meas1 = track1["pos"][p1]
                    meas2 = track2["pos"][p2]
                    track1["pos"][p1] = average_measurements(meas1, meas2)
                for meas in track2["pos"][int(frame_diff) + 1:]:
                    track1["pos"].append(meas)
            else:  # if the second track starts after the first ends, fill the wholes with nan
                frame_diff = start - end
                for p1 in range(int(frame_diff) - 1):
                    track1["pos"].append([np.nan, np.nan, np.nan, np.nan])
                for meas in track2["pos"]:
                    track1["pos"].append(meas)
            intern_tracks[track1_id] = copy.deepcopy(track1)  # save the stitched track
            intern_tracks.pop(track2_id)  # remove the now, redundant track
            translate_ids[group[1]] = translate_ids[group[0]]
        len_test2 = len(intern_tracks)
        limit += 1
    return intern_tracks


def write_to_cdb(cdb, tracks):
    """
    Writes all tracks to a clickpoints database
    :param cdb: (object) clickpoints database
    :param tracks: (dict) tracks ({track_id:{"start_frame":int, "pos":list}})
    :return: None
    """
    try:  # the name of the tracks and their colour
        cdb.deleteMarkerTypes(name=track_name)
        cdb.setMarkerType(name=track_name, color=track_color, mode=4)
    except:
        pass
    print("Number of Tracks:", len(tracks))
    print("Max Track:", np.max(list(tracks.keys())))
    for track_id, values in tracks.items():
        print("Track:", track_id)
        new_track = cdb.setTrack(type=track_name)  # Initialise a new track
        f_frame = values["start_frame"]
        frames = []
        for f1, f2 in enumerate(range(f_frame, f_frame + len(values["pos"]))):  # checks all frames for nans
            if not np.any(np.isnan(values["pos"][f1])):
                frames.append(f2)
        try:
            y, x, z, pix = zip(*values["pos"])  # extract all x and y values
        except TypeError:
            y, x, z, pix = values["pos"]
            frames = [frames[0]]
        y = np.array(y)
        y = y[~np.isnan(y)].tolist()
        x = np.array(x)
        x = x[~np.isnan(x)].tolist()
        z = np.array(z)
        z = z[~np.isnan(z)].tolist()
        pix = np.array(pix)
        pix = pix[~np.isnan(pix)].tolist()
        marker_text = [str(s) + "    " + str(j) for s, j in zip(z, pix)]
        # cdb.setMarkers(frame=frames, x=x, y=y, track=new_track, type=track_name, text=marker_text, layer="MinProj") #with text marker (z median, pixel size)
        cdb.setMarkers(frame=frames, x=x, y=y, track=new_track, type=track_name, layer="MinProj")


def interpolate_positions(tracks):
    """
    Fills up all wholes for each track, by interpolating the previous and the next position
    :param tracks: (dict) all tracks {track: {"start_frame": int, "pos": [[x, y, z], ...]}, ...}
    :return: (dict) tracks without nans in them
    """
    for track_id, values in tracks.items():
        pos_array = np.array(values["pos"])  # extracts positions from a track
        nan_array = np.where(np.isnan(pos_array[:, 0]))[0]  # extract all indices with a nan
        nan_places = []
        count = 0
        max_count = len(nan_array)
        while count < max_count:  # finds all connected nan areas
            cu_pos = nan_array[count]
            cu_nan = [cu_pos]
            count += 1
            while cu_pos + 1 in nan_array:
                cu_nan.append(cu_pos + 1)
                cu_pos += 1
                count += 1
            nan_places.append(cu_nan)
        for p1 in nan_places:
            empty_spaces = p1[-1] - p1[0] + 2  # number of nans
            first_p = pos_array[p1[0] - 1]  # previous position
            last_p = pos_array[p1[-1] + 1]  # next position
            # fills all empty spaces by linear interpolation
            fill_pos = np.array([np.linspace(first_p[pos], last_p[pos], empty_spaces, endpoint=False) for pos in
                                 range(len(first_p))]).transpose().tolist()
            tracks[track_id]["pos"][p1[0]:(p1[-1] + 1)] = fill_pos[1:]
    return tracks


def run_tracking(cdb_path, cdb, frames, start_frame = 1 ):
    end_frame = frames - 2
    available_labels = {}
    overlaps = []
    image_shape = cdb.getImage(frame=0).data.shape
    if use_drift:  # if yes, calculate the field of view, which occurs in all frames
        drift_name = "_".join(cdb_path.split("/")[-7:]).split(".")[0] + ".txt"
        drift_path = os.path.join(path_to_drift_folder, drift_name)
        drift_array = np.loadtxt(drift_path, delimiter=',')
        drift_sum = np.cumsum(drift_array, axis=0)  # calculate drift for every frame in respect to the first frame
        drift_sum = np.insert(drift_sum, 0, np.array([0, 0]), axis=0)
        max_x_dr = np.max(drift_sum[:, 1])
        max_y_dr = np.max(drift_sum[:, 0])
        min_x_dr = np.min(drift_sum[:, 1])
        min_y_dr = np.min(drift_sum[:, 0])
        # define the shape of the field of view
        drift_shape = (int(image_shape[0] - max_y_dr + min_y_dr), int(image_shape[1] - max_x_dr + min_x_dr))
        drift_values = [max_y_dr, max_x_dr, drift_shape[0], drift_shape[1]]
    else:  # if no, simply ignore any drift
        drift_array = None
        drift_sum = None
        drift_values = None
    if block_k562_and_dead_areas:  # if yes, a certain radius around any k562 cell will be blocked for nk cells
        radius = calculate_nk_diameter(cdb, start_frame)  # calculate the average half major axis length
    else:
        radius = None
    z_positions_all = []  # list containing the z-values for all labels in every frame
    yx_positions_all = []  # list containing the yx-values for all labels in every frame
    pixel_nums_all = []  # list containing the pixel sizes for all labels in every frame
    # get a labeled mask
    cur_mask, cur_mask_num, cur_z, cur_yx, cur_pix_num = get_nk_label(cdb, start_frame, drift=drift_sum,
                                                                      max_drifts=drift_values, mean_radius=radius)
    z_positions_all.append(cur_z)
    yx_positions_all.append(cur_yx)
    pixel_nums_all.append(cur_pix_num)
    available_labels[start_frame] = (np.arange(1, cur_mask_num + 1).tolist())  # start with all labels available
    for frame in range(start_frame+1, end_frame+1):  # iterate through all frames
        print("Frame:", frame)
        next_mask, next_mask_num, next_z, next_yx, next_pix_num = get_nk_label(cdb, frame, drift=drift_sum,
                                                                               max_drifts=drift_values,
                                                                               mean_radius=radius)
        # calculate the overlap between the current and the next mask
        overlaps.append(get_pixel_overlap(cur_mask, next_mask, cur_mask_num, next_mask_num))
        cur_mask = copy.deepcopy(next_mask)
        cur_mask_num = copy.deepcopy(next_mask_num)
        z_positions_all.append(next_z)
        yx_positions_all.append(next_yx)
        pixel_nums_all.append(next_pix_num)
        available_labels[frame] = (np.arange(1, cur_mask_num + 1).tolist())
    save_tracks = {}
    track_num = 1
    initialise_tracks = 1
    old_track_ids = []
    tracks_without_any_overlap = {}
    for fr, ov_mat in enumerate(overlaps):  # iterate through all overlaps
        bool_mat = ov_mat != 0  # transform the overlap matrix into a boolean matrix
        cur_tracks = np.sum(bool_mat, axis=1)  # sum up the number of overlaps for the current tracks
        next_tracks = np.sum(bool_mat, axis=0)  # sum up the number of overlaps for the next tracks
        new_track_ids = np.zeros(len(next_tracks))
        qualified = np.where(cur_tracks == 1)[0]  # current tracks with a single overlap
        tracks_without_any_overlap[fr+start_frame] = [np.where(cur_tracks == 0)[0], np.where(next_tracks == 0)[0]]
        if initialise_tracks:
            for t in qualified:  # iterate through all qualified tracks
                to_check = np.where(bool_mat[t, :] == True)[0][0]  # check these next tracks for multiple overlaps
                # if the next tracks got multiple overlaps or the z-distance to the current track is too large,
                # don't connect them
                if next_tracks[to_check] == 1 and np.abs(
                        z_positions_all[fr][t - 1] - z_positions_all[fr + 1][to_check - 1]) <= z_lim:
                    save_tracks[track_num] = {"start_frame": fr + start_frame, "label": [t, to_check]}
                    new_track_ids[to_check] = copy.deepcopy(track_num)  # keep track of the id
                    track_num += 1
                    initialise_tracks = 0
        else:  # here the track already exists so simply add any further connections
            for t in qualified:
                to_check = np.where(bool_mat[t, :] == True)[0][0]
                if next_tracks[to_check] == 1 and np.abs(
                        z_positions_all[fr][t - 1] - z_positions_all[fr + 1][to_check - 1]) <= z_lim:
                    if old_track_ids[t] in save_tracks.keys():
                        save_tracks[old_track_ids[t]]["label"].append(to_check)
                        new_track_ids[to_check] = old_track_ids[t]
                    else:
                        save_tracks[track_num] = {"start_frame": fr + start_frame, "label": [t, to_check]}
                        new_track_ids[to_check] = copy.deepcopy(track_num)
                        track_num += 1
        old_track_ids = copy.deepcopy(new_track_ids)
    save_tracks2 = {}
    track_num2 = 1
    c_track_ids = np.zeros(len(yx_positions_all[0]) + 1)
    next_track_ids = []
    # check all labels without any overlap for possible connections
    for key, value in tracks_without_any_overlap.items():
        distance_matrix_yx = np.zeros((len(value[0]) - 1, len(value[1]) - 1))
        distance_matrix_z = np.zeros((len(value[0]) - 1, len(value[1]) - 1))
        next_track_ids = np.zeros(len(yx_positions_all[key - start_frame + 1]) + 1)
        for k1, val1 in enumerate(value[0][1:]):  # fill up the distance matrix for all possible connections
            for k2, val2 in enumerate(value[1][1:]):
                # Form: sqrt((x1-x2)**2+(y1-y2)**2)
                distance_matrix_yx[k1, k2] = np.sqrt(np.sum((np.array(
                    yx_positions_all[key - start_frame][val1 - 1]) - np.array(
                    yx_positions_all[key - start_frame + 1][val2 - 1])) ** 2))
                distance_matrix_z[k1, k2] = np.abs(
                    z_positions_all[key - start_frame][val1 - 1] - z_positions_all[key - start_frame + 1][val2 - 1])
        matches = []
        shape = distance_matrix_yx.shape
        if shape[0] == 0 or shape[1] == 0:  # if there aren't any possible matches continue
            c_track_ids = copy.deepcopy(next_track_ids)
            continue
        if track_k562:  # depending on the track takes the fitting max value
            max_value = k562_fast_max_value
        else:
            max_value = nk_fast_max_value
        while (np.nanmin(distance_matrix_yx)) < max_value:  # find matches starting from the best one
            min_pos = list(np.unravel_index(np.nanargmin(distance_matrix_yx), shape))
            if distance_matrix_z[min_pos[0], min_pos[1]] <= z_lim_fast:
                matches.append(min_pos)
                distance_matrix_yx[min_pos[0], :] = np.nan
                distance_matrix_yx[:, min_pos[1]] = np.nan
            else:
                distance_matrix_yx[min_pos[0], min_pos[1]] = np.nan
        for match in matches:  # connect the found matches
            if c_track_ids[value[0][match[0] + 1]] == 0:
                save_tracks2[track_num2] = {"start_frame": key,
                                            "label": [value[0][match[0] + 1], value[1][match[1] + 1]]}
                next_track_ids[value[1][match[1] + 1]] = track_num2
                track_num2 += 1
            else:
                save_tracks2[c_track_ids[value[0][match[0] + 1]]]["label"].append(value[1][match[1] + 1])
                next_track_ids[value[1][match[1] + 1]] = c_track_ids[value[0][match[0] + 1]]
        c_track_ids = copy.deepcopy(next_track_ids)
    """ Merge save_tracks with save_tracks2 here """
    # merge overlap tracks and fast tracks together
    save_tracks3, not_merged_tracks = merge_tracks(save_tracks, save_tracks2)
    fin_tracks = {}
    counter = 1
    used_labels = {frame: [] for frame in range(start_frame, end_frame + 1)}
    for key, value in save_tracks3.items():  # replace the labels with their positions for all tracks
        first_frame = value["start_frame"]
        positions = []
        for i, lab in enumerate(value["label"]):
            positions.append(yx_positions_all[first_frame - start_frame + i][lab - 1] + [
                z_positions_all[first_frame - start_frame + i][lab - 1]] + [
                                 pixel_nums_all[first_frame - start_frame + i][lab - 1]])
            used_labels[first_frame + i].append(lab)  # keep track of all used labels
        fin_tracks[counter] = {"start_frame": first_frame, "pos": positions}
        counter += 1
    for tr in not_merged_tracks:  # add all non merged fast tracks to the final tracks
        first_frame = save_tracks2[tr]["start_frame"]
        positions = []
        for i, lab in enumerate(save_tracks2[tr]["label"]):
            positions.append(yx_positions_all[first_frame - start_frame + i][lab - 1] + [
                z_positions_all[first_frame - start_frame + i][lab - 1]] + [
                                 pixel_nums_all[first_frame - start_frame + i][lab - 1]])
            used_labels[first_frame + i].append(lab)  # keep track of all used labels
        fin_tracks[counter] = {"start_frame": first_frame, "pos": positions}
        counter += 1
    for key, value in used_labels.items():
        # calculate available labels for every frame
        available_labels[key] = list(set(available_labels[key]) - set(value))
    for key, value in available_labels.items():  # add all still available labels as single tracks
        for v in value:
            fin_tracks[counter] = {"start_frame": key, "pos": [
                yx_positions_all[key - start_frame][v - 1] + [z_positions_all[key - start_frame][v - 1]] + [
                    pixel_nums_all[key - start_frame][v - 1]]]}
            counter += 1
    if stitch:
        stitched_tracks = stitch_tracks(fin_tracks)  # stitch all available tracks where possible
        if use_best_fitting_tracks_for_stitching:
            stitched_tracks = stitch_tracks(stitched_tracks, find_best_fitting_stitches=True)
        if remove_all_short_tracks:
            stitched_tracks = remove_short_tracks(stitched_tracks, min_len=min_track_len)
            if stitch_again_after_removing_short_tracks:
                stitched_tracks = stitch_tracks(stitched_tracks, find_best_fitting_stitches=False)
                if use_best_fitting_tracks_for_stitching:
                    stitched_tracks = stitch_tracks(stitched_tracks, find_best_fitting_stitches=True)
    else:
        stitched_tracks = fin_tracks
    if interpolate_missing_positions:  # if yes, fill up all wholes with interpolated values
        stitched_tracks = interpolate_positions(stitched_tracks)
        if remove_tracks_starting_after_the_start_frame:
            stitched_tracks = remove_short_tracks(stitched_tracks, min_len=min_len_after_interpolation,
                                                  remove_after_interpolation=True)
    if use_drift:
        stitched_tracks = add_drift_for_cdb(stitched_tracks, max_drift=drift_values, drift=drift_sum)
    write_to_cdb(cdb, stitched_tracks)  # write all resulting tracks to a clickpoints database
    cdb.db.close()
    print('------done tracking------')
    return track_name


