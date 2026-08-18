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

MOTILITY_DEFINITION = {"NK": 6.5, "pigPBMCs": 6.0, "Jurkat": 4.0, "NK_day14": 13, "Treg": 13}
track_type = 'nk_tracks_greedy_stitched_without_short_high_z'
thres_speed_umpromin = 20  # Zellen, die oberhalb des Geles (=Medium) schwimmen, ignorieren
thres_distance_pxl = 20  # 20 bei 15sec&5min


def filter_cdb(celltype, time_step, path_list, pixelsize_ccd, objective=10, motility_window_min=None,
               colorize_direction=False, chemokine_direction=None,
               toward_fraction=0.5, color_per_frame=True, color_window_min=15.0): #pixelsize 3.45 or 4.56
    """
    ``motility_window_min``: if given (e.g. 5.5), a cell counts as motile when it
    moves >= the threshold within *any* sliding window of that many minutes,
    instead of over the whole track. Needed for long acquisitions where a static
    cell would otherwise accumulate enough jitter over hours to be called motile.
    For the short ~5.5-min movies a 5.5-min window equals the whole track, so the
    result is unchanged. ``None`` keeps the original whole-track behavior.
    """
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
            database_to_pandas(db, time_step, res, celltype, motility_window_min=motility_window_min,
                               colorize_direction=colorize_direction, chemokine_direction=chemokine_direction,
                               toward_fraction=toward_fraction, color_per_frame=color_per_frame,
                               color_window_min=color_window_min)


def get_track_distance(nanpadded_trackarray):
    # calculate euclidean distance sqrt(( x max - x min)^2 + (y max - y min)^2)
    return np.linalg.norm(np.nanmax(nanpadded_trackarray, axis=1) - np.nanmin(nanpadded_trackarray, axis=1), axis=1)


def windowed_max_displacement(track_xy, window):
    """Max bounding-box displacement (pixels) over any sliding window of ``window``
    frames within a single track. ``track_xy`` is a (frames, 2) array that may
    contain nans for missing frames."""
    n = len(track_xy)
    if window >= n:
        seg = track_xy[~np.isnan(track_xy[:, 0])]
        if len(seg) < 2:
            return 0.0
        return float(np.linalg.norm(seg.max(axis=0) - seg.min(axis=0)))
    best = 0.0
    for s in range(0, n - 1):
        seg = track_xy[s:s + window]
        seg = seg[~np.isnan(seg[:, 0])]
        if len(seg) >= 2:
            d = float(np.linalg.norm(seg.max(axis=0) - seg.min(axis=0)))
            if d > best:
                best = d
    return best


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


def extract(db, data, time_step, res, window_frames=None):
    frames = db.getImages(layer=1).count()

    # The images are ALREADY drift-corrected in preprocessing (pixels are shifted in
    # correct_drift / correct_drift_longterm). We must therefore NOT re-apply a
    # track-derived drift as per-image offsets: doing so reintroduces visible drift
    # in the cdb, accumulates over long movies (cumsum) so frames slide out of view
    # and only one seems to show, and detaches the tracks from the image. Instead we
    # clear any offsets left from a previous run, so the cdb shows the corrected
    # images as-is with the tracks sitting on their cells (re-runnable).
    for im in db.getImages():
        db.setOffset(im, 0, 0)

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
        if window_frames:
            # max displacement within any sliding window (for long-movie motility)
            data.loc[data.id == id, "window_distance_um"] = \
                windowed_max_displacement(nptrack[0], window_frames) * res
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


def database_to_pandas(db, time_step, res, celltype, motility_window_min=None,
                       colorize_direction=False, chemokine_direction=None,
                       toward_fraction=0.5, color_per_frame=True, color_window_min=15.0):
    thresh_motile = MOTILITY_DEFINITION[celltype]
    save_name_csv = '_' + str(thresh_motile) + 'umin5min'
    # translate the motility window (minutes) into frames for this movie
    window_frames = None
    if motility_window_min:
        window_frames = max(2, int(round(motility_window_min * 60.0 / time_step)))
        print(f"motility window: {motility_window_min} min -> {window_frames} frames "
              f"(time_step={time_step}s)")
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
    extract(db, data, time_step, res, window_frames=window_frames)

    # drift = calc_drift(data, )
    # print(data['y'])

    if len(data) == 0:
        print("WARNING: empty file", db._database_filename, file=sys.stderr)
        return data

    data = data[data["speed_boundingbox_um_min"] < thres_speed_umpromin]

    # motile = moved beyond the threshold. For long movies use the max displacement
    # within any sliding window (so a jittery static cell isn't called motile just
    # from accumulating drift over hours); otherwise use the whole-track distance.
    if window_frames:
        data["motile"] = data["window_distance_um"] > thresh_motile
    else:
        data["motile"] = data["distance_um"] > thresh_motile
    print(np.mean(data.groupby("id").motile.mean()))
    # save the dataframe
    data.to_csv(db._database_filename[:-4] + save_name_csv +".csv")
    # colorize tracks: by migration direction vs the chemokine axis if requested and
    # borders are available, otherwise the plain motile/non-motile coloring.
    colored = False
    if colorize_direction:
        from ..preprocessing import borders as border_utils   # lazy: avoid import cycle
        ref = border_utils.find_reference_cdb(db._database_filename)
        borders = border_utils.load_borders_from_path(ref)
        if borders is not None:
            axis = border_utils.perpendicular_vector(borders, hint=chemokine_direction)
            axis = axis / (np.linalg.norm(axis) or 1.0)
            # colour on the LONG (e.g. 30-min) window, not the short motility window:
            # small direction changes must not flip the colour, but a cell that heads
            # toward the chemokine for 30 min and then away for 30 min gets 2 colours.
            color_frames = max(2, int(round(color_window_min * 60.0 / time_step)))
            colorize_tracks_by_direction(db, data, axis, window_frames=color_frames,
                                         toward_fraction=toward_fraction, per_frame=color_per_frame)
            colored = True
    if not colored:
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


# Magenta/green instead of red/green: distinguishable for red-green colour blindness.
COLOR_TOWARD = "#00CC00"     # green   - toward the chemokine
COLOR_AWAY = "#FF00FF"       # magenta - away from the chemokine
COLOR_SIDEWAYS = "#888888"   # grey    - motile but not directional
COLOR_NONMOTILE = "#0000FF"  # blue    - non-motile


def _direction_color(dx, dy, axis, cos_threshold):
    n = (dx * dx + dy * dy) ** 0.5
    if n == 0:
        return COLOR_SIDEWAYS
    cos = (dx * axis[0] + dy * axis[1]) / n
    if cos >= cos_threshold:
        return COLOR_TOWARD
    if cos <= -cos_threshold:
        return COLOR_AWAY
    return COLOR_SIDEWAYS


def colorize_tracks_by_direction(cdb, data, axis, cos_threshold=0.5, window_frames=None,
                                 toward_fraction=0.5, per_frame=True):
    """Color tracks by migration direction relative to ``axis`` (unit vector toward
    the chemokine, image x,y coords).

        green = toward the chemokine (within +/-60 deg, cos >= cos_threshold)
        red   = away from it
        grey  = motile but sideways / non-directional
        blue  = non-motile

    Direction is evaluated per ``window_frames`` sub-segment, so a cell that changes
    direction over time is handled properly:

    * ``per_frame=True`` colors each track marker by ITS OWN window's direction, so
      the track visibly CHANGES COLOR along its length (green while it heads toward
      the chemokine, grey/red when it does not).
    * The whole-track color is set from the FRACTION of windows heading toward the
      chemokine: green if that fraction >= ``toward_fraction`` (persistently
      chemotactic), red if the away-fraction dominates, else grey.

    With ``window_frames=None`` the whole track is one window (net start->end).
    """
    # frame (sort_index) -> image ids, built once for the per-marker coloring
    frame_to_images = {}
    if per_frame:
        for im in cdb.getImages():
            frame_to_images.setdefault(int(im.sort_index), []).append(im.id)

    for tid, g in data.groupby("id"):
        g = g.sort_values("frame")
        motile = bool(g["motile"].iloc[0]) if "motile" in g.columns else True
        if not motile:
            cdb.table_track.update({cdb.table_track.style: '{"color": "%s"}' % COLOR_NONMOTILE,
                                    cdb.table_track.text: ""}).where(cdb.table_track.id == tid).execute()
            continue

        xy = g[["x", "y"]].values
        frames = g["frame"].values
        w = int(window_frames) if window_frames else len(xy)
        w = max(2, w)

        colors = []          # one per window
        for s in range(0, max(len(xy) - 1, 1), w):
            seg = xy[s:s + w]
            if len(seg) < 2:
                continue
            c = _direction_color(seg[-1, 0] - seg[0, 0], seg[-1, 1] - seg[0, 1], axis, cos_threshold)
            colors.append(c)
            if per_frame:
                # color this window's markers so the track changes color over time
                img_ids = []
                for f in frames[s:s + w]:
                    img_ids.extend(frame_to_images.get(int(f), []))
                if img_ids:
                    cdb.table_marker.update({cdb.table_marker.style: '{"color": "%s"}' % c}).where(
                        (cdb.table_marker.track == tid) &
                        (cdb.table_marker.image << img_ids)).execute()
        if not colors:
            continue
        frac_toward = colors.count(COLOR_TOWARD) / len(colors)
        frac_away = colors.count(COLOR_AWAY) / len(colors)
        if frac_toward >= toward_fraction:
            track_color = COLOR_TOWARD
        elif frac_away >= toward_fraction:
            track_color = COLOR_AWAY
        else:
            track_color = COLOR_SIDEWAYS
        cdb.table_track.update(
            {cdb.table_track.style: '{"color": "%s"}' % track_color,
             cdb.table_track.text: ""}).where(
            cdb.table_track.id == tid).execute()
