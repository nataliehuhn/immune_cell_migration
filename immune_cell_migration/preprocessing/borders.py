"""
Border-line handling for the chemokine assay.

The user draws two roughly-vertical border lines (a clickpoints ``TYPE_Line``
marker, by convention named "boarder") into the *first* timestep of every
``0h_corrected`` database - one per position. These lines mark the left and
right edge of the channel. Everything outside them (channel walls, debris) must
be excluded from detection/tracking, and the chemokine gradient runs
perpendicular to the borders.

This module reads those lines and provides:
    * ``read_borders(cdb)``            - the two border segments, ordered left->right
    * ``region_mask(shape, borders)``  - boolean mask, True between the borders
    * ``perpendicular_vector(...)``    - unit gradient vector perpendicular to them
    * helpers to locate the reference (0h) database for any position

If a database has fewer than two lines the helpers return ``None`` and callers
fall back to their previous, full-frame behaviour.
"""
import os
import re
import glob
import numpy as np

# Sign hints, kept compatible with plots.plot_chemokine_assay.CHEMO_VECTORS.
# Only used to decide which way the perpendicular gradient points.
CHEMO_VECTORS = {
    "up":    np.array([0.0, 1.0]),
    "down":  np.array([0.0, -1.0]),
    "right": np.array([1.0, 0.0]),
    "left":  np.array([-1.0, 0.0]),
}

REFERENCE_FOLDER_NAME = "0h_corrected"


def _pos_from_filename(name):
    """Extract the integer position from a filename, e.g. '..._pos03_...' -> 3."""
    m = re.search(r"pos(\d+)", os.path.basename(name))
    return int(m.group(1)) if m else None


def reference_folder(any_path, reference_folder_name=REFERENCE_FOLDER_NAME):
    """Return the sibling reference folder (e.g. .../data/0h_corrected).

    ``any_path`` may be a timepoint folder (``.../data/1h_corrected``) or a file
    inside one.
    """
    p = os.path.abspath(any_path).rstrip("\\/")
    base_dir = p if os.path.isdir(p) else os.path.dirname(p)
    parent = os.path.dirname(base_dir)
    return os.path.join(parent, reference_folder_name)


def reference_cdb_for_pos(any_path, pos, reference_folder_name=REFERENCE_FOLDER_NAME):
    """Find the reference (0h) cdb for a given position, or None."""
    ref = reference_folder(any_path, reference_folder_name)
    if pos is None or not os.path.isdir(ref):
        return None
    for f in sorted(glob.glob(os.path.join(ref, "*.cdb"))):
        if _pos_from_filename(f) == pos:
            return f
    return None


def find_reference_cdb(cdb_path, reference_folder_name=REFERENCE_FOLDER_NAME):
    """Given any position's cdb, return the matching reference (0h) cdb path.

    For a cdb that already lives in the reference folder this returns the same
    path back.
    """
    pos = _pos_from_filename(cdb_path)
    return reference_cdb_for_pos(cdb_path, pos, reference_folder_name)


# Marker-type names accepted for border lines (case-insensitive substring match).
# Includes the common "boarder" typo and the correct "border" spelling.
BORDER_TYPE_KEYS = ("border", "boarder")


def _is_border_line(line):
    """True if the line's marker type looks like a border (by name)."""
    try:
        name = (line.type.name or "").lower()
    except Exception:
        return False
    return any(key in name for key in BORDER_TYPE_KEYS)


def read_borders(cdb):
    """Read the two border lines from an open clickpoints database.

    Returns ``(left, right)`` where each border is an ``(x1, y1, x2, y2)`` tuple,
    ordered by ascending mean-x (so ``left`` is the left border). Returns
    ``None`` if fewer than two lines are present. If more than two lines exist
    the two outermost (smallest and largest mean-x) are used.

    Lines drawn with a marker type named like a border ("border"/"boarder",
    any case) are preferred; if at least two such lines exist only they are used.
    Otherwise every line marker in the database is considered (so an unnamed or
    differently-named line type still works).
    """
    lines = cdb.getLines()
    if lines is None or len(lines) < 2:
        return None
    lines = list(lines)
    border_lines = [l for l in lines if _is_border_line(l)]
    use = border_lines if len(border_lines) >= 2 else lines
    if len(use) < 2:
        return None
    segs = [(float(l.x1), float(l.y1), float(l.x2), float(l.y2)) for l in use]
    segs.sort(key=lambda s: 0.5 * (s[0] + s[2]))
    if len(segs) > 2:
        segs = [segs[0], segs[-1]]
    return segs[0], segs[1]


def load_borders_from_path(cdb_path):
    """Open ``cdb_path`` read-only and return its borders (or None)."""
    if cdb_path is None or not os.path.exists(cdb_path):
        return None
    import clickpoints
    with clickpoints.DataFile(cdb_path) as db:
        return read_borders(db)


def cdbs_missing_borders(folder_with_cdbs):
    """Return the cdb paths in ``folder_with_cdbs`` that don't have two borders.

    Used to gate the pipeline: detection should not run until the user has drawn
    the border lines into every reference (0h) database.
    """
    missing = []
    for f in sorted(glob.glob(os.path.join(folder_with_cdbs, "*.cdb"))):
        if load_borders_from_path(f) is None:
            missing.append(f)
    return missing


def _x_at_y(seg, ys):
    """x-coordinate of a line segment at given row(s) y (extrapolated)."""
    x1, y1, x2, y2 = seg
    if y2 == y1:  # horizontal line - fall back to its mean x
        return np.full(np.shape(ys), 0.5 * (x1 + x2), dtype=float)
    t = (ys - y1) / (y2 - y1)
    return x1 + (x2 - x1) * t


def region_mask(shape, borders):
    """Boolean array (H, W), True for pixels between the two borders.

    The borders are treated as infinite lines (extrapolated past their drawn
    endpoints) so the whole image height is covered.
    """
    H, W = int(shape[0]), int(shape[1])
    left, right = borders
    ys = np.arange(H)
    xl = _x_at_y(left, ys)
    xr = _x_at_y(right, ys)
    lo = np.minimum(xl, xr)[:, None]
    hi = np.maximum(xl, xr)[:, None]
    xx = np.arange(W)[None, :]
    return (xx >= lo) & (xx <= hi)


def normalized_position(borders, xs, ys):
    """Position of points along the left->right axis, normalized per row.

    Returns ``u`` where 0.0 is on the left border, 1.0 is on the right border
    (measured horizontally at each point's own y, so border tilt and per-position
    channel width are accounted for and values are poolable across positions).
    Points outside the channel fall below 0 or above 1; the caller can clip.
    """
    left, right = borders
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    xl = _x_at_y(left, ys)
    xr = _x_at_y(right, ys)
    denom = xr - xl
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (xs - xl) / denom
    u[denom == 0] = np.nan
    return u


def distance_to_borders(borders, xs, ys):
    """Horizontal distance (pixels) of points to the left and right border.

    Returns ``(dist_left, dist_right)``, each measured at the point's own y so
    border tilt is accounted for. Positive = inside the channel; negative means the
    point lies outside that border.
    """
    left, right = borders
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    xl = _x_at_y(left, ys)
    xr = _x_at_y(right, ys)
    lo = np.minimum(xl, xr)
    hi = np.maximum(xl, xr)
    return xs - lo, hi - xs


def perpendicular_vector(borders, hint=None):
    """Unit vector perpendicular to the borders, in image (x, y) coordinates.

    The sign is chosen so the vector points roughly along ``CHEMO_VECTORS[hint]``
    (the coarse cardinal direction of the chemokine source). If ``hint`` is
    unknown the vector points from the left border toward the right border.
    """
    left, right = borders

    def direction(seg):
        d = np.array([seg[2] - seg[0], seg[3] - seg[1]], dtype=float)
        n = np.linalg.norm(d)
        return d / n if n > 0 else d

    dl = direction(left)
    dr = direction(right)
    if np.dot(dl, dr) < 0:  # borders may be drawn in opposite directions
        dr = -dr
    d = dl + dr
    n = np.linalg.norm(d)
    d = d / n if n > 0 else dl
    perp = np.array([-d[1], d[0]])  # rotate 90 degrees

    ref = CHEMO_VECTORS.get(hint) if hint is not None else None
    if ref is None:
        # default: point from the left border toward the right border
        mid_left = np.array([0.5 * (left[0] + left[2]), 0.5 * (left[1] + left[3])])
        mid_right = np.array([0.5 * (right[0] + right[2]), 0.5 * (right[1] + right[3])])
        ref = mid_right - mid_left
    if np.dot(perp, ref) < 0:
        perp = -perp
    return perp
