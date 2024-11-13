import numpy as np
import clickpoints


def get_drift(nanpadded_trackarray, frames):
    ptTracks = nanpadded_trackarray.astype(np.float)
    # Drift correction. If commented out, turn Drift_True to False and drift=drift to None
    drift = calc_drift(frames, nanpadded_trackarray, 5)
    # subtract the drift from the track marker positions
    nanpadded_trackarray_withoutdrift = ptTracks - drift
    return nanpadded_trackarray_withoutdrift


def get_drift_wo_track_array(db_path, track_type):
    db = clickpoints.DataFile(db_path)
    #calculates frames
    frames = db.getImages(layer=1).count()
    # create a nanpadded array of tracks
    ptTracks = db.getTracksNanPadded(type=track_type)[:]
    ptTracks = ptTracks.astype(np.float)
    # Drift correction. If commented out, turn Drift_True to False and drift=drift to None
    drift = calc_drift(frames, ptTracks, 5)
    # subtract the drift from the track marker positions
    nanpadded_trackarray_withoutdrift = ptTracks - drift
    return nanpadded_trackarray_withoutdrift


def get_list_of_track_with_nframes(list2, frames):
    # list2 has the dimensions of TRACKS x IMAGE x XY
    # calculate the difference between the max and min of each track
    distances = np.linalg.norm(np.nanmax(list2, axis=1)-np.nanmin(list2, axis=1), axis=1)
    # check for each track if the first frame and the "Frames"th frame is not nan
    valid_tracks = np.isfinite(list2[:, 2, 0]) & np.isfinite(list2[:, frames - 2, 0])

    # get the indices for the valid tracks
    drift_list = np.where(valid_tracks)[0]
    # get only the distances for the valid tracks
    drift_distance = distances[valid_tracks]

    return drift_list, drift_distance


def calc_drift(frames, list2, percentile):

    drift_list, drift_distance = get_list_of_track_with_nframes(list2, frames)

    if len(drift_list) == 0:
        #print("Warning: no tracks found for drift correction. Skipping drift correction", file=sys.stderr)
        return np.zeros(list2.shape[1:])

    # if we found less than 25 such tracks, increase the percentile by 10
    if len(drift_distance) <= 25:
        percentile += 10
    # NEW TINA - fuer Chemokineassay by Ben:
    # percentile = 50
    # get the 5% or 15% of tracks with the least movement
    p = np.percentile(drift_distance, percentile)
    # add these slowest tracks to a new list
    indices_of_slowest = drift_distance < p
    drift_list = drift_list[indices_of_slowest]

    # calculate the mean displacement of the tracks for each image
    mean_drift = np.nanmean(list2[drift_list, 1:, :] - list2[drift_list, :-1, :], axis=0)
    # add a 0,0 for the drift in the first frame
    mean_drift = np.vstack((np.zeros(2), mean_drift))
    # set nan values to 0
    mean_drift[np.isnan(mean_drift)] = 0.0

    # calculate the cumulative sum of all drifts
    sum_drift = np.cumsum(mean_drift, axis=0)
    # return the cumulative sum and the list of tracks, as well, as the flag if the last frame was missing
    return sum_drift

