import numpy as np
import clickpoints
import os
import glob
import pandas as pd
import tqdm
import skimage.measure
from ..utils import name_glob
import sys
from .drift_correction import calc_drift

np.set_printoptions(suppress=True)

MOTILITY_DEFINITION = {"NK": 6.5, "pigPBMCs": 6.0, "Jurkat": 4.0, "NK_day14": 13}
track_type = 'nk_tracks_greedy_stitched_without_short_high_z'
thres_speed_umpromin = 20  # Zellen, die oberhalb des Geles (=Medium) schwimmen, ignorieren
thres_distance_pxl = 20  # 20 bei 15sec&5min

def filter_cdb(celltype, time_step, path_list, pixelsize_ccd=4.0954, objective=10):
    thresh_motile = MOTILITY_DEFINITION[celltype]
    res = pixelsize_ccd/objective  # Lumenera ; 6.45/10  Hamamatsu
    save_name_csv = '_' + str(thresh_motile) + 'umin5min'

    for path, _ in path_list:
        # find all databases in the folder
        database_names = glob.glob(os.path.join(path, "*-*_pos*.cdb"))
        print(database_names)
        # iterate over all found databases
        # for path in database_names[1:]:
        for path in database_names[0:]:
            # print status
            print("Processing database:", path)
            # load database
            db = clickpoints.DataFile(path)
            # get number of frames
            database_to_pandas(db, time_step, res, celltype)


def get_track_distance(nanpadded_trackarray):
    # calculate euclidean distance sqrt(( x max - x min)^2 + (y max - y min)^2)
    return np.linalg.norm(np.nanmax(nanpadded_trackarray, axis=1) - np.nanmin(nanpadded_trackarray, axis=1), axis=1)


def get_speed_boundingbox(nanpadded_trackarray, res, time_step):
    """calculate the speed over bounding box (maximum coordinates - minimum coordinates of each track)"""
    # get number of track points of each track
    len_trackpoints = np.sum(~np.isnan(nanpadded_trackarray), axis=1)[:, 0]

    track_distance_in_pxl = get_track_distance(nanpadded_trackarray)

    return track_distance_in_pxl * res / ((len_trackpoints - 1) * (time_step / 60.))


def get_speed_stepwidth(nanpadded_trackarray, time_step, res):
    """calculate speed of cells with step width"""
    # step width of cells
    stepwidth_pixel = nanpadded_trackarray[:, 1:, :] - nanpadded_trackarray[:, :-1, :]
    # get the absolute value of these vectors
    stepwidth_pixel = np.linalg.norm(stepwidth_pixel, axis=2)
    # calculate speed in um / min
    speed_stepwidth_um_min = ((stepwidth_pixel * res) / (time_step / 60.))
    return speed_stepwidth_um_min


def get_directionality(nanpadded_trackarray):
    """calculate directionality"""
    # get the differences between subsequent positions
    differences_vector = nanpadded_trackarray[:, 1:, :] - nanpadded_trackarray[:, :-1, :]
    # create two lists of such vectors
    vector1 = differences_vector[:, :-1, :]
    vector2 = differences_vector[:, 1:, :]
    # calculate the scale product of each pair
    scalar_product = np.sum(vector1 * vector2, axis=2)
    # and calculate the cos of the angle between each vector pair
    denom = (np.linalg.norm(vector1, axis=2) * np.linalg.norm(vector2, axis=2))
    cos_angle = scalar_product / denom
    cos_angle[denom == 0] = 0
    # left or right turn isn't intereseting for us --> absolute value of cos is possible  #np.nanmean(cos_angle, axis = 1)
    return cos_angle


def measure_tracks(nanpadded_trackarray, time_step, res):
    """
    calculate speed and direction of tracks
    :param nanpadded_trackarray: nan padded track array, shape (tracks, frames, xy)
    :param time_step:  Time in seconds between two pictures
    :param res: Resolution correction px in mum
    :return: result_dict

    turning_angle = turning angles of each track
    speed_boundingbox = speed of each track (calculated with bounding box)
    speed_stepwidth  = speed of each track (calculated with step width)
    speed_stepwidth_overtime_eachtrack  =  mean speed of each track (calculated with step width)
    direction = mean direction of each track
    speed_boundingbox_nanmean = mean speed (calculated with bounding box) of all tracks
    speed_stepwidth_nanmean = mean speed (calculated with step width) of all tracks
    direction_nanmean = mean direction of all tracks
    """

    track_distance_in_pxl = get_track_distance(nanpadded_trackarray)
    speed_boundingbox_um_min = get_speed_boundingbox(nanpadded_trackarray, res, time_step)
    speed_stepwidth_um_min = get_speed_stepwidth(nanpadded_trackarray, time_step, res)
    cos_angle = get_directionality(nanpadded_trackarray)

    return track_distance_in_pxl, speed_boundingbox_um_min, speed_stepwidth_um_min, cos_angle


def extract(db, data, time_step, res):
    frames = db.getImages(layer=1).count()
    # get the nan padded track over time array - get a list with all positions of the marker 'PT_Track_marker'
    nanpadded_trackarray = get_tracks_nan_padded(db, type=track_type, layer=1)

    drift = calc_drift(frames, nanpadded_trackarray, 5)
    # print(drift)

    for i, d in enumerate(drift):
        # print(i)
        # print(d)
        im = db.getImage(frame=i, layer=1)  #layer=1
        db.setOffset(im, -d[0], -d[1])

    def adjust_length(array, index):
        return array[~np.isnan(nptrack[index, 1:, 0])]

    def adjust_length2(array, index):
        return array[~np.isnan(nptrack[index, 1:-1, 0])]

    for id in data.id.unique():
        nptrack = get_tracks_nan_padded(db, id=id, type=track_type, layer=1)
        track_distance_in_pxl, speed_boundingbox_um_min, speed_stepwidth_um_min, cos_angle = measure_tracks(nptrack,
                                                                                                           time_step,
                                                                                                           res)

        data.loc[data.id == id, "distance_um"] = track_distance_in_pxl[0] * res
        data.loc[data.id == id, "speed_boundingbox_um_min"] = speed_boundingbox_um_min[0]
        try:
            data.loc[data.id == id, "speed_stepwidth_um_min"] = adjust_length(speed_stepwidth_um_min[0], 0)
        except ValueError:
            pass
        try:
            data.loc[data.id == id, "cos_angle"] = adjust_length2(cos_angle[0], 0)
        except ValueError:
            pass
    # return data


def get_tracks_nan_padded(self, type=None, id=None, start_frame=None, end_frame=None, skip=None, layer=0, apply_offset=True):
    """
    Return an array of all track points with the given filters. The array has the shape of [n_tracks, n_images, pos],
    where pos is the 2D position of the markers.

    See also: :py:meth:`~.DataFile.getTrack`, :py:meth:`~.DataFile.setTrack`, :py:meth:`~.DataFile.deleteTracks`, :py:meth:`~.DataFile.getTracks`.

    Parameters
    ----------
    type: :py:class:`MarkerType`, str, array_like, optional
        the marker type/types or name of the marker type for the track.
    id : int, array_like, optional
        the  :py:class:`Track` ID
    start_frame : int, optional
        the frame where to begin the array. Default: first frame.
    end_frame : int, optional
        the frame where to end the array. Default: last frame.
    skip : int, optional
        skip every nth frame. Default: don't skip frames.
    layer : int, optional
        which layer to use for the images.
    apply_offset : bool, optional
        whether to apply the image offsets to the marker positions. Default: False.

    Returns
    -------
    nan_padded_array : ndarray
        the array which contains all the track marker positions.
    """

    layer_count = self.table_layer.select().count()

    """ image conditions """
    where_condition_image = []

    # get the filter condition (only filter if it is necessary, e.g. if we have more than one layer)
    if layer is not None and layer_count != 1:
        if layer == 0:
            layer = self.table_layer.select().where(self.table_layer.id == self.table_layer.base_layer).limit(1)[0]
        else:
            layer = self.table_layer.select().where(self.table_layer.id == layer).limit(1)[0]
        where_condition_image.append("layer_id = %d" % layer.id)

    # if a start frame is given, only export marker from images >= the given frame
    if start_frame is not None:
        where_condition_image.append("i.sort_index >= %d" % start_frame)
    # if a end frame is given, only export marker from images < the given frame
    if end_frame is not None:
        where_condition_image.append("i.sort_index < %d" % end_frame)
    # skip every nth frame
    if skip is not None:
        where_condition_image.append("i.sort_index %% %d = 0" % skip)

    # append sorting by sort index
    if len(where_condition_image):
        where_condition_image = " WHERE " + " AND ".join(where_condition_image)
    else:
        where_condition_image = ""

    # get the image ids according to the conditions
    image_ids = self.db.execute_sql("SELECT id FROM image i "+where_condition_image+" ORDER BY sort_index;").fetchall()
    image_count = len(image_ids)

    """ track conditions """
    where_condition_tracks = []

    if type is not None:
        type = self._processesTypeNameField(type, ["TYPE_Track"])
        if not isinstance(type, list):
            where_condition_tracks.append("t.type_id = %d" % type.id)
        else:
            where_condition_tracks.append("t.type_id in " % str([t.id for t in type]))

    if id is not None:
        where_condition_tracks.append("t.id = %d" % id)

    # append sorting by sort index
    if len(where_condition_tracks):
        where_condition_tracks = " WHERE " + " AND ".join(where_condition_tracks)
    else:
        where_condition_tracks = ""

    track_ids = self.db.execute_sql("SELECT id FROM track t "+where_condition_tracks+";").fetchall()
    track_count = len(track_ids)

    # create empty array to be filled by the queries
    pos = np.zeros((track_count, image_count, 2), "float")

    # iterate either over images or over tracks
    # for some reasons it is better to iterate over the images even if the number of tracks is lower
    if image_count < track_count * 100:
        # iterate over all images
        for index, (id,) in enumerate(image_ids):
            # get the tracks for this image
            q = self.db.execute_sql(
                "SELECT x, y FROM track t LEFT JOIN marker m ON m.track_id = t.id AND m.image_id = ? "+where_condition_tracks+" ORDER BY t.id",
                (id,))
            # store the result in the array
            pos[:, index] = q.fetchall()
    else:
        # iterate over all tracks
        for index, (id,) in enumerate(track_ids):
            # get the images for this track
            q = self.db.execute_sql(
                "SELECT x, y FROM image i LEFT JOIN marker m ON m.track_id = ? AND m.image_id = i.id " + where_condition_image + " ORDER BY i.sort_index",
                (id,))
            # store the result in the array
            pos[index] = q.fetchall()

    # if the offset is required, get the offsets for all images and add them to the marker positions
    if apply_offset:
        query_offset = "SELECT IFNULL(o.x, 0) AS x, IFNULL(o.y, 0) AS y FROM image AS i LEFT JOIN offset o ON i.id = o.image_id"
        offsets = np.array(self.db.execute_sql(query_offset + where_condition_image + " ORDER BY sort_index;").fetchall()).astype(float)
        pos += offsets

    return pos


def fix_database(db, track_type):
    nan_padded = get_tracks_nan_padded(db, track_type, layer=4)
    db.setTracksNanPadded(nan_padded[:, :-1, :], track_type=track_type, start_frame=1)


def database_to_pandas(db, time_step, res, celltype):
    thresh_motile = MOTILITY_DEFINITION[celltype]
    save_name_csv = '_' + str(thresh_motile) + 'umin5min'
    data = []
    # get the mask type so we can filter just for the NK mask
    mask_type = db.getMaskType(name="NK") #NK
    # iterate over all frames, (here we direclty iterate over the images in the MinIndices layer)
    for im in tqdm.tqdm(db.getImageIterator()):
        if im.mask is None:
            continue
        # get the pixel data of the image
        im_data = db.getImage(frame=im.sort_index, layer="MinIndices").data
        # get the mask pixel data for the nk mask
        mask = (im.mask.data == mask_type.index)
        # get at labeled version of the mask to be used with regionprops (every region ("cell") is represended by a different number)
        mask_labeled = skimage.measure.label(mask)
        # get all the track markers of this image with the marker type "nk_tracks_greedy_stitched_without_short_high_z"
        markers = db.getMarkers(image=im, type=track_type)

        props = skimage.measure.regionprops(mask_labeled)

        props = {prop.label: prop for prop in props}

        # extract the data of all these markers
        for marker in markers:
            # the image index
            frame = im.sort_index
            # get the id of the corresponding track (e.g. the "cell" id)
            id = marker.track_id
            # get the position
            x = marker.x
            y = marker.y
            # the z position is the pixel value of the minimum projection image
            z = im_data[int(y), int(x)]
            # get the label of the region to which this x,y position belongs
            label = mask_labeled[int(y), int(x)]
            try:
                # find the prop object which has the same label as the cell
                prop = props[label]
                # get the area and eccentricity of this region
                area = prop.area
                eccentricity = prop.eccentricity
                # if area > 300: #hinzugefügt!!!!!!!!!!! TEST 22.10.2024
                #     # add all to the data list
                #     data.append([frame, id, x, y, z, area, eccentricity])
            except KeyError:
                # sometimes the cell is concave and the center is thus not on the cell mask
                # therefore, we cannot find the mask when we just have the center of the cell
                # -> set properties to nan
                area = np.nan
                eccentricity = np.nan
                # data.append([frame, id, x, y, z, area, eccentricity]) #hinzugefügt!!!!!!!!!!! TEST 22.10.2024

            data.append([frame, id, x, y, z, area, eccentricity])

    # convert the data list to a DataFrame
    data = pd.DataFrame(data, columns=["frame", "id", "x", "y", "z", "area", "eccentricity"])
    # print(data['y'])
    extract(db, data, time_step, res)

    # drift = calc_drift(data, )
    # print(data['y'])

    if len(data) == 0:
        print("WARNING: empty file", db._database_filename, file=sys.stderr)
        return data

    data = data[data["speed_boundingbox_um_min"] < thres_speed_umpromin]

    data["motile"] = data["distance_um"] > thresh_motile
    print(np.mean(data.groupby("id").motile.mean()))
    # save the dataframe
    data.to_csv(db._database_filename[:-4] + save_name_csv +".csv")
    #colorize motile tracks
    colorize_tracks(db, data)
    return data

#--------------------------------------------------- TEST 22.10.2024
# import matplotlib.pyplot as plt
# plt.imshow(mask_labeled, vmin=0, vmax=1)
# data_grouped = data.groupby(data.id).mean()
# for id in data_grouped.index:
#     d = data_grouped.loc[id]
#     plt.text(d.x, d.y, d.area, color='red')
#--------------------------------------------------


def colorize_tracks(cdb, data):
    data_grouped = data.groupby(data.id).mean()
    for id in data_grouped.index:
        d = data_grouped.loc[id]
        if d.motile:
            style = '{"color": "#FF0000"}'
        else:
            style = '{"color": "#0000FF"}'# "#00FF00"}'

        text = ""  # f"x {id} {d.distance_um:.1f} {d.speed_boundingbox_um_min:.1f} {d.speed_stepwidth_um_min:.1f} {d.cos_angle:.1f}"
        cdb.table_track.update({cdb.table_track.style: style, cdb.table_track.text: text}).where(
            cdb.table_track.id == id).execute()
