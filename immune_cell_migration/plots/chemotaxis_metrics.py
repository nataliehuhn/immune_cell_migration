"""
Chemotaxis metrics on the border-perpendicular gradient axis.

Everything is computed on SHORT MOVING SEGMENTS, not whole tracks: each track is
walked in non-overlapping ``motility_window_min`` windows (e.g. 5.5 min), and a
window is used only if the cell CLEARLY moves in it (bounding-box displacement >=
the motility threshold). This is the crucial filter - it restricts every metric to
cells that are genuinely migrating during the analyzed time, so static/jittery
cells and dead time don't dilute the signal. Each segment is tagged with its start
time, so metrics can be resolved over time (to find on which timescale chemotaxis
happens) and restricted to an early window (``max_minutes``).

Per moving segment (gradient axis points left border -> right border, so positive
"parallel" = toward the right border):
    fmi_par      = net displacement along gradient / path length   (Forward Migration Index)
    fmi_perp     = net displacement perpendicular  / path length
    directedness = cos(angle between net displacement and the axis)
    persistence  = net displacement / path length                  (straightness)
    speed        = path length / time      (how fast the cell moves along its path)
    velocity_par = net displacement along gradient / time          (directed speed)
    angle        = atan2(perp, parallel)   (0 = toward right border)
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..preprocessing import borders as border_utils

MOTILITY_DEFINITION = {"NK": 6.5, "pigPBMCs": 6.0, "Jurkat": 4.0, "NK_day14": 13, "Treg": 13}
ACQUISITION_MODE = {"skip": 0, "sequential": 1}


def _perp(axis):
    return np.array([-axis[1], axis[0]])


def _bbox(xy):
    return float(np.linalg.norm(xy.max(axis=0) - xy.min(axis=0))) if len(xy) >= 2 else 0.0


def _windowed_bbox(xy, window):
    """Max bounding-box displacement over any sliding ``window``-frame sub-window.
    Used to decide whether a cell is MOVING somewhere within a longer segment."""
    n = len(xy)
    if n < 2:
        return 0.0
    if window >= n:
        return _bbox(xy)
    best = 0.0
    for s in range(0, n - 1):
        d = _bbox(xy[s:s + window])
        if d > best:
            best = d
    return best


def _slow_track_ids(df, slow_percentile=25):
    """IDs of the slowest tracks (bottom ``slow_percentile`` by net displacement).

    These cells barely migrate on their own, so their common motion tracks the
    residual drift. Using the slow cells (rather than the median of ALL cells) keeps
    the estimate valid even when the majority of cells are motile - a median over
    everyone would then be biased by the movers.
    """
    def net(g):
        xy = g.values
        return float(np.linalg.norm(xy[-1] - xy[0])) if len(xy) >= 2 else 0.0
    disp = df.sort_values("frame").groupby("id")[["x", "y"]].apply(net)
    if disp.empty:
        return set()
    cutoff = np.percentile(disp.values, slow_percentile)
    return set(disp[disp <= cutoff].index)


def drift_correct_tracks(df, slow_percentile=25, min_tracks=5):
    """Remove residual drift by subtracting the per-frame drift estimated from the
    SLOWEST cells (bottom ``slow_percentile``). Their common frame-to-frame step is
    the drift; subtracting its cumulative sum re-aligns every frame. Robust even when
    most cells are motile. Returns a copy of ``df`` with corrected x, y."""
    df = df.copy().sort_values(["id", "frame"])
    slow = df[df["id"].isin(_slow_track_ids(df, slow_percentile))].copy()
    d = slow.groupby("id")[["x", "y"]].diff()
    slow["_dx"], slow["_dy"] = d["x"], d["y"]
    grp = slow.dropna(subset=["_dx"]).groupby("frame")
    med = grp[["_dx", "_dy"]].median()
    cnt = grp["_dx"].count()
    med.loc[cnt < min_tracks] = 0.0
    med = med.reindex(sorted(df["frame"].unique())).fillna(0.0)
    cum = med.cumsum().rename(columns={"_dx": "_cdx", "_dy": "_cdy"})
    df = df.merge(cum, left_on="frame", right_index=True, how="left")
    df["x"] = df["x"] - df["_cdx"].fillna(0.0)
    df["y"] = df["y"] - df["_cdy"].fillna(0.0)
    return df.drop(columns=["_cdx", "_cdy"])


def _window_drift(df, window_frames, res, slow_percentile=25, min_tracks=5):
    """Per absolute time-window residual-drift magnitude (µm), from the slowest
    cells' median displacement. Returns {window_index: (drift_um, n_slow)}."""
    slow_ids = _slow_track_ids(df, slow_percentile)
    out = {}
    w = (df["frame"] // window_frames).astype(int)
    for k, gw in df.groupby(w):
        disps = []
        for _cid, g in gw[gw["id"].isin(slow_ids)].groupby("id"):
            g = g.sort_values("frame")
            xy = g[["x", "y"]].values
            if len(xy) >= 2:
                disps.append(xy[-1] - xy[0])
        mag = float(np.linalg.norm(np.median(disps, axis=0))) * res if len(disps) >= min_tracks else np.nan
        out[int(k)] = (mag, len(disps))
    return out


def segment_metrics(xy, axis, res, dt_min):
    """Chemotaxis descriptors for one moving segment (time-ordered pixels)."""
    if len(xy) < 2:
        return None
    acc = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))   # path length (px)
    if acc == 0:
        return None
    net = xy[-1] - xy[0]
    net_mag = float(np.linalg.norm(net))
    perp = _perp(axis)
    d_par = float(net @ axis)
    d_perp = float(net @ perp)
    t = (len(xy) - 1) * dt_min
    return {
        "acc_um": acc * res, "net_um": net_mag * res,
        "d_par_um": d_par * res, "d_perp_um": d_perp * res,
        "fmi_par": d_par / acc, "fmi_perp": d_perp / acc,
        "persistence": net_mag / acc,
        "directedness": (d_par / net_mag) if net_mag > 0 else 0.0,
        "speed": (acc * res / t) if t > 0 else np.nan,
        "velocity_par": (d_par * res / t) if t > 0 else np.nan,
        "angle": float(np.arctan2(d_perp, d_par)),
        "x_pos": float(xy[:, 0].mean()), "y_pos": float(xy[:, 1].mean()),
    }


def rayleigh_test(angles):
    """Rayleigh test for a non-uniform (directional) angular distribution.
    Returns (R, p): R = mean resultant length, p = significance."""
    angles = np.asarray(angles, dtype=float)
    n = angles.size
    if n == 0:
        return np.nan, np.nan
    C, S = np.mean(np.cos(angles)), np.mean(np.sin(angles))
    R = float(np.hypot(C, S))
    z = n * R * R
    p = float(np.exp(-z) * (1 + (2 * z - z * z) / (4 * n)))
    return R, min(max(p, 0.0), 1.0)


def collect_segments(path, celltype, conditions, acquisition_mode, pos_num, time_step,
                     pixelsize_ccd, objective, directionality_window_min=15.0,
                     motility_window_min=5.5, max_minutes=None, drift_threshold=None,
                     drift_correct=True, slow_percentile=25, keep_nonmoving=False):
    """Per-segment chemotaxis metrics for one folder (tidy DataFrame), two timescales.

    Direction/FMI/persistence are measured over absolute ``directionality_window_min``
    windows (e.g. 30 min) so that sustained, persistent migration is captured despite
    short-timescale amoeboid wiggling. A window is kept only if the cell is MOVING
    (translocates >= the motility threshold in some ``motility_window_min`` sub-window).

    DRIFT CHECKPOINT: for each position/window a drift score is computed from ALL
    tracks; windows with score >= ``drift_threshold`` (nearly all tracks moving the
    same way = residual drift) are DROPPED, since they would masquerade as perfect
    chemotaxis. ``max_minutes`` restricts to windows within the first N minutes.
    """
    thresh = MOTILITY_DEFINITION[celltype]
    res = pixelsize_ccd / objective
    dt_min = time_step / 60.0
    thresh_px = thresh / res
    window_dir = max(2, int(round(directionality_window_min * 60.0 / time_step)))
    window_mot = max(2, int(round(motility_window_min * 60.0 / time_step)))
    acq_sequential = ACQUISITION_MODE[acquisition_mode]
    num_conditions = len(conditions)

    border_cache = {}
    rows = []
    for f in glob.glob(os.path.join(path, "*" + str(thresh) + "umin*.csv")):
        pos = int(f.split("_")[-4][3:])
        cond_idx = pos // pos_num if acq_sequential else pos % num_conditions
        if cond_idx >= num_conditions:
            continue
        condition = conditions[cond_idx]
        if pos not in border_cache:
            ref = border_utils.reference_cdb_for_pos(path, pos)
            border_cache[pos] = border_utils.load_borders_from_path(ref)
        borders = border_cache[pos]
        if borders is None:
            continue
        axis = border_utils.perpendicular_vector(borders, hint=None)
        axis = axis / (np.linalg.norm(axis) or 1.0)

        df = pd.read_csv(f, index_col=0)
        if not {"id", "x", "y", "frame"}.issubset(df.columns):
            continue
        if max_minutes is not None:
            df = df[df["frame"] * time_step / 60.0 <= max_minutes]

        # drift checkpoint: residual-drift magnitude (µm) per window from the RAW
        # tracks (slowest cells), recorded for the diagnostic.
        drift = _window_drift(df, window_dir, res, slow_percentile)
        drift_cut = drift_threshold if drift_threshold is not None else thresh   # µm
        # correct residual drift (subtract the slow-cell drift) so drift windows keep
        # only the real, differential motion; pure-drift windows collapse to ~0 and
        # are then removed by the motility gate below.
        if drift_correct:
            df = drift_correct_tracks(df, slow_percentile)

        for _cid, g in df.groupby("id"):
            g = g.sort_values("frame")
            xy = g[["x", "y"]].values
            fr = g["frame"].values
            wk = (fr // window_dir).astype(int)             # absolute window index per point
            for k in np.unique(wk):
                ds = drift.get(int(k), (np.nan, 0))[0]       # drift magnitude (µm)
                # if not correcting, drop windows where the slow cells drifted a lot
                if (not drift_correct) and np.isfinite(ds) and ds >= drift_cut:
                    continue
                seg = xy[wk == k]
                if len(seg) < 2:
                    continue
                moving = _windowed_bbox(seg, window_mot) >= thresh_px
                if not moving and not keep_nonmoving:
                    continue
                m = segment_metrics(seg, axis, res, dt_min)     # net over the window
                if m is None:
                    continue
                u = border_utils.normalized_position(borders, [m["x_pos"]], [m["y_pos"]])[0]
                m["condition"] = condition
                m["position"] = pos
                m["u"] = float(u) if np.isfinite(u) else np.nan   # 0 = left border, 1 = right
                m["half"] = "left" if (np.isfinite(u) and u < 0.5) else "right"
                m["time_min"] = float(k * window_dir * time_step / 60.0)
                m["drift_um"] = float(ds) if np.isfinite(ds) else np.nan
                m["moving"] = bool(moving)
                if not moving:
                    m["direction"] = "not moving"   # below the motility threshold in this window
                else:
                    d = m["directedness"]
                    m["direction"] = "toward right" if d >= 0.5 else ("toward left" if d <= -0.5 else "perp")
                rows.append(m)
    return pd.DataFrame(rows)


def _ordered(df, custom_order):
    return [c for c in custom_order if c in set(df["condition"])]


def _agg(df, col, order):
    return df.groupby("condition")[col].agg(["mean", "sem"]).reindex(order)


def plot_drift_check(celltype, path_list, conditions, custom_order, acquisition_mode, pos_num,
                     time_step, pixelsize_ccd=3.45, objective=10,
                     directionality_window_min=15.0, max_minutes=None,
                     drift_threshold=None, slow_percentile=25):
    """Residual-drift checkpoint: how far the SLOWEST cells drift per window.

    The slowest cells barely migrate, so their net displacement in a window is the
    residual drift (µm). If it approaches/exceeds the motility threshold (red line),
    that window has a real drift problem that would fake chemotaxis - the metrics
    correct it (slow-cell subtraction) by default. Writes ``drift_check.csv`` +
    ``plot_drift_check.png`` per folder.
    """
    thresh = MOTILITY_DEFINITION[celltype]
    res = pixelsize_ccd / objective
    cut = drift_threshold if drift_threshold is not None else thresh
    acq_sequential = ACQUISITION_MODE[acquisition_mode]
    num_conditions = len(conditions)
    window_dir = max(2, int(round(directionality_window_min * 60.0 / time_step)))
    for path, _ in path_list:
        rows = []
        for f in glob.glob(os.path.join(path, "*" + str(thresh) + "umin*.csv")):
            pos = int(f.split("_")[-4][3:])
            cond_idx = pos // pos_num if acq_sequential else pos % num_conditions
            if cond_idx >= num_conditions:
                continue
            df = pd.read_csv(f, index_col=0)
            if not {"id", "x", "y", "frame"}.issubset(df.columns):
                continue
            if max_minutes is not None:
                df = df[df["frame"] * time_step / 60.0 <= max_minutes]
            for k, (drift_um, n) in _window_drift(df, window_dir, res, slow_percentile).items():
                rows.append({"position": pos, "condition": conditions[cond_idx],
                             "time_min": k * window_dir * time_step / 60.0,
                             "drift_um": drift_um, "n_slow": n})
        dfp = pd.DataFrame(rows)
        if dfp.empty:
            continue
        dfp.to_csv(os.path.join(path, "drift_check.csv"), index=False)
        fig, ax = plt.subplots(figsize=(max(1.2 * dfp["time_min"].nunique(), 7), 4.2))
        for pos, gp in dfp.groupby("position"):
            gp = gp.sort_values("time_min")
            ax.plot(gp["time_min"], gp["drift_um"], "-o",
                    label=f"pos{int(pos):02d} ({gp['condition'].iloc[0]})")
        ax.axhline(cut, color="red", ls="--", lw=1.5, label=f"motility threshold ({cut:g} µm)")
        ax.set_ylim(bottom=0)
        ax.set_xlabel("time (min)")
        ax.set_ylabel("residual drift of slowest cells (µm / window)")
        ax.set_title(f"Drift checkpoint (slowest {slow_percentile}% of cells)")
        ax.legend(fontsize=7)
        plt.tight_layout()
        out = os.path.join(path, "plot_drift_check.png")
        plt.savefig(out, dpi=200); plt.close()
        print(f"Saved: {out}")


def plot_fmi(celltype, path_list, conditions, custom_order, acquisition_mode, pos_num,
             time_step, pixelsize_ccd=3.45, objective=10, max_minutes=None,
             directionality_window_min=15.0, motility_window_min=5.5, drift_threshold=None, drift_correct=True, slow_percentile=25):
    """Forward Migration Index (parallel & perpendicular) per condition, on 30-min moving windows."""
    for path, _ in path_list:
        df = collect_segments(path, celltype, conditions, acquisition_mode, pos_num, time_step,
                              pixelsize_ccd, objective, directionality_window_min,
                              motility_window_min, max_minutes, drift_threshold, drift_correct, slow_percentile)
        if df.empty:
            continue
        df.to_csv(os.path.join(path, "chemotaxis_segments.csv"), index=False)
        order = _ordered(df, custom_order)
        x = np.arange(len(order)); w = 0.4
        par, perp = _agg(df, "fmi_par", order), _agg(df, "fmi_perp", order)
        fig, ax = plt.subplots(figsize=(max(1.4 * len(order), 6), 4.4))
        ax.bar(x - w / 2, par["mean"], width=w, yerr=par["sem"], capsize=4,
               edgecolor="black", color="#4C72B0", label="FMI∥ (along gradient)")
        ax.bar(x + w / 2, perp["mean"], width=w, yerr=perp["sem"], capsize=4,
               edgecolor="black", color="#C44E52", label="FMI⊥ (perpendicular)")
        ax.axhline(0.0, color="black", lw=0.8); ax.set_ylim(-1, 1)
        ax.set_ylabel("Forward Migration Index")
        ttl = "FMI (+∥ = toward right border)"
        if max_minutes:
            ttl += f"  ·  first {int(max_minutes)} min"
        ax.set_title(ttl)
        ax.set_xticks(x); ax.set_xticklabels(order, rotation=45, ha="right")
        ax.legend(fontsize=8)
        plt.tight_layout()
        out = os.path.join(path, "plot_fmi.png")
        plt.savefig(out, dpi=300); plt.close()
        print(f"Saved: {out}")


def plot_speed_gradient(celltype, path_list, conditions, custom_order, acquisition_mode,
                        pos_num, time_step, pixelsize_ccd=3.45, objective=10,
                        max_minutes=None, directionality_window_min=15.0, motility_window_min=5.5,
                        drift_threshold=None, drift_correct=True, slow_percentile=25):
    """Are cells that head toward the gradient faster AND straighter?

    Among the moving windows, each is classed by its net direction (toward the right
    border / perpendicular-along-channel / toward the left border). Left panel: PATH
    SPEED per class; right panel: PERSISTENCE (straightness = net/path, 1 = perfectly
    straight) per class. A chemotactic population moves faster and straighter when
    heading toward the source.
    """
    # same palette as the cdb track colouring (magenta/green: colour-blind friendly)
    groups = ["toward right", "perp", "toward left"]
    colors = {"toward right": "#00CC00", "perp": "#888888", "toward left": "#FF00FF"}
    labels = {"toward right": "toward", "perp": "sideways", "toward left": "away"}
    for path, _ in path_list:
        df = collect_segments(path, celltype, conditions, acquisition_mode, pos_num, time_step,
                              pixelsize_ccd, objective, directionality_window_min,
                              motility_window_min, max_minutes, drift_threshold, drift_correct,
                              slow_percentile)          # moving segments only
        if df.empty:
            continue
        order = _ordered(df, custom_order)
        x = np.arange(len(order)); w = 0.26
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(2.6 * len(order), 10), 4.6))
        for k, grp in enumerate(groups):
            sub = df[df["direction"] == grp]
            sp = sub.groupby("condition")["speed"].agg(["mean", "sem"]).reindex(order)
            pe = sub.groupby("condition")["persistence"].agg(["mean", "sem"]).reindex(order)
            ax1.bar(x + (k - 1) * w, sp["mean"], width=w, yerr=sp["sem"], capsize=3,
                    edgecolor="black", color=colors[grp], label=labels[grp])
            ax2.bar(x + (k - 1) * w, pe["mean"], width=w, yerr=pe["sem"], capsize=3,
                    edgecolor="black", color=colors[grp], label=labels[grp])
        ax1.set_ylabel("speed (µm/min)")
        ax2.set_ylabel("persistence")
        ax2.set_ylim(0, 1)
        for ax in (ax1, ax2):
            ax.set_xticks(x); ax.set_xticklabels(order, rotation=45, ha="right")
        ax1.legend(fontsize=8, frameon=False)
        plt.tight_layout()
        out = os.path.join(path, "plot_speed_persistence_by_direction.png")
        plt.savefig(out, dpi=300); plt.close()
        print(f"Saved: {out}")


def _wrap(a):
    """Wrap angles to (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def collect_angular_steps(path, celltype, conditions, acquisition_mode, pos_num, time_step,
                          pixelsize_ccd, objective, coarse_frames=3, min_step_um=1.0,
                          max_minutes=None, drift_correct=True, slow_percentile=25):
    """Angle phi to the gradient and its change per step, for every moving step.

    The direction of motion is coarse-grained over ``coarse_frames`` frames (a single
    60 s step is too noisy to define an orientation), giving for each step:
        phi        - angle of motion relative to the gradient (0 = toward chemokine)
        dphi_dt    - turning rate (rad/min) to the next step
        speed      - speed during the step (um/min)
    Steps shorter than ``min_step_um`` are dropped, since their angle is meaningless.
    """
    thresh = MOTILITY_DEFINITION[celltype]
    res = pixelsize_ccd / objective
    acq_sequential = ACQUISITION_MODE[acquisition_mode]
    num_conditions = len(conditions)
    dt_min = coarse_frames * time_step / 60.0          # minutes per coarse step

    rows = []
    for f in glob.glob(os.path.join(path, "*" + str(thresh) + "umin*.csv")):
        pos = int(f.split("_")[-4][3:])
        cond_idx = pos // pos_num if acq_sequential else pos % num_conditions
        if cond_idx >= num_conditions:
            continue
        borders = border_utils.load_borders_from_path(
            border_utils.reference_cdb_for_pos(path, pos))
        if borders is None:
            continue
        axis = border_utils.perpendicular_vector(borders, hint=None)
        axis = axis / (np.linalg.norm(axis) or 1.0)
        perp = _perp(axis)

        df = pd.read_csv(f, index_col=0)
        if not {"id", "x", "y", "frame"}.issubset(df.columns):
            continue
        if max_minutes is not None:
            df = df[df["frame"] * time_step / 60.0 <= max_minutes]
        if df.empty:
            continue
        if drift_correct:
            df = drift_correct_tracks(df, slow_percentile)

        for cid, g in df.groupby("id"):
            g = g.sort_values("frame")
            xy = g[["x", "y"]].values[::coarse_frames]      # coarse-grained sampling
            fr = g["frame"].values[::coarse_frames]
            if len(xy) < 3:
                continue
            steps = np.diff(xy, axis=0)
            lens = np.linalg.norm(steps, axis=1) * res      # um
            # angle of each step relative to the gradient axis
            phi = np.arctan2(steps @ perp, steps @ axis)
            for i in range(len(steps) - 1):
                if lens[i] < min_step_um or lens[i + 1] < min_step_um:
                    continue                                # angle undefined for tiny steps
                rows.append({
                    "condition": conditions[cond_idx], "position": pos, "id": int(cid),
                    "phi": float(phi[i]),
                    "dphi_dt": float(_wrap(phi[i + 1] - phi[i]) / dt_min),   # rad/min
                    "speed": float(lens[i] / dt_min),                        # um/min
                    "time_min": float(fr[i] * time_step / 60.0),
                })
    return pd.DataFrame(rows)


def plot_angular_dynamics(celltype, path_list, conditions, custom_order, acquisition_mode, pos_num,
                          time_step, pixelsize_ccd=3.45, objective=10, coarse_frames=3,
                          min_step_um=1.0, n_angle_bins=12, max_minutes=None,
                          drift_correct=True, slow_percentile=25, min_steps=10):
    """Fokker-Planck style decomposition of the angular dynamics (torque vs noise).

    Following the framework of Jakuszeit et al. (bioRxiv 2025, "Torque-based immune
    cell chemotaxis"), the turning dynamics are split into a deterministic and a
    stochastic part, both as a function of the angle phi to the gradient
    (phi = 0 means heading toward the chemokine):

        A(phi) = < dphi/dt | phi >                 deterministic drift = TORQUE
        D(phi) = var(dphi/dt | phi) * dt / 2       angular noise (rotational diffusion)
        v(phi) = < speed | phi >                   speed modulation

    Interpretation:
      * A(phi) ~ -Omega*sin(phi) with Omega > 0  -> cells actively STEER up-gradient
        (torque-based, DC-like). Omega is fitted and reported.
      * A(phi) ~ 0 but D(phi) or v(phi) lower/higher when aligned -> the bias comes
        from modulating angular NOISE and/or SPEED (neutrophil-like).

    Writes ``angular_dynamics.csv`` (binned A, D, v) + ``plot_angular_dynamics.png``.
    """
    for path, _ in path_list:
        st = collect_angular_steps(path, celltype, conditions, acquisition_mode, pos_num,
                                   time_step, pixelsize_ccd, objective, coarse_frames,
                                   min_step_um, max_minutes, drift_correct, slow_percentile)
        if st.empty:
            print(f"  no angular steps collected in {path}")
            continue
        order = _ordered(st, custom_order)
        edges = np.linspace(-np.pi, np.pi, n_angle_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        dt_min = coarse_frames * time_step / 60.0

        rows = []
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        cmap = plt.get_cmap("tab10")
        for ci, cond in enumerate(order):
            sub = st[st["condition"] == cond]
            b = np.clip(np.digitize(sub["phi"], edges) - 1, 0, n_angle_bins - 1)
            A = np.full(n_angle_bins, np.nan)
            D = np.full(n_angle_bins, np.nan)
            V = np.full(n_angle_bins, np.nan)
            Ase = np.full(n_angle_bins, np.nan)
            Dse = np.full(n_angle_bins, np.nan)
            Vse = np.full(n_angle_bins, np.nan)
            for j in range(n_angle_bins):
                s = sub[b == j]
                n_j = len(s)
                if n_j < min_steps:
                    continue
                A[j] = s["dphi_dt"].mean()
                Ase[j] = s["dphi_dt"].sem()
                D[j] = s["dphi_dt"].var() * dt_min / 2.0
                # SE of a variance estimate: var * sqrt(2/(n-1))
                Dse[j] = D[j] * np.sqrt(2.0 / max(n_j - 1, 1))
                V[j] = s["speed"].mean()
                Vse[j] = s["speed"].sem()
                rows.append({"condition": cond, "phi": centers[j],
                             "A_rad_per_min": A[j], "A_sem": Ase[j],
                             "D_rad2_per_min": D[j], "D_sem": Dse[j],
                             "speed_um_per_min": V[j], "speed_sem": Vse[j], "n": int(n_j)})
            # fit A(phi) = -Omega sin(phi): Omega = -sum(A*sin)/sum(sin^2)
            ok = np.isfinite(A)
            omega = (-np.sum(A[ok] * np.sin(centers[ok])) / np.sum(np.sin(centers[ok]) ** 2)
                     if ok.sum() > 2 else np.nan)
            col = cmap(ci % 10)
            axes[0].errorbar(centers[ok], A[ok], yerr=Ase[ok], fmt="-o", color=col,
                             capsize=2, label=f"{cond} (Ω={omega:.3f})")
            if np.isfinite(omega):
                axes[0].plot(centers, -omega * np.sin(centers), ":", color=col, lw=1)
            okD, okV = np.isfinite(D), np.isfinite(V)
            axes[1].errorbar(centers[okD], D[okD], yerr=Dse[okD], fmt="-o", color=col,
                             capsize=2, label=cond)
            axes[2].errorbar(centers[okV], V[okV], yerr=Vse[okV], fmt="-o", color=col,
                             capsize=2, label=cond)

        for ax, ylab, ttl in zip(
                axes,
                ["A(φ)  (rad/min)", "D(φ)  (rad²/min)", "speed (µm/min)"],
                ["torque / drift", "angular noise", "speed modulation"]):
            ax.axvline(0, color="grey", ls=":", lw=1)
            ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
            ax.set_xticklabels(["away", "⊥", "toward", "⊥", "away"], fontsize=8)
            ax.set_ylabel(ylab)
            ax.set_title(ttl, fontsize=10)
        axes[0].axhline(0, color="black", lw=0.8)
        axes[0].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        pd.DataFrame(rows).to_csv(os.path.join(path, "angular_dynamics.csv"), index=False)
        out = os.path.join(path, "plot_angular_dynamics.png")
        fig.savefig(out, dpi=200); plt.close()
        print(f"Saved: {out}")


def plot_chemotaxis_map(celltype, path_list, conditions, custom_order, acquisition_mode, pos_num,
                        time_step, pixelsize_ccd=3.45, objective=10, metric="fmi_par",
                        n_space_bins=6, max_minutes=None, directionality_window_min=30.0,
                        motility_window_min=5.5, drift_threshold=None, drift_correct=True,
                        slow_percentile=25, min_segments=5):
    """WHERE in the channel does chemotaxis happen, and does that change over time?

    Each moving segment is placed in a space bin (its position along the gradient
    axis, 0 = left border .. 1 = right border) and a time bin, and ``metric``
    (default FMI parallel; e.g. "directedness", "speed", "persistence") is averaged.

    Per condition: a space x time heatmap (top) showing where the directional bias
    sits and whether it drifts/decays, and a spatial profile pooled over time
    (bottom) with SEM. Bins with fewer than ``min_segments`` segments are left blank.
    Writes ``chemotaxis_map.csv`` + ``plot_chemotaxis_map.png``.
    """
    for path, _ in path_list:
        df = collect_segments(path, celltype, conditions, acquisition_mode, pos_num, time_step,
                              pixelsize_ccd, objective, directionality_window_min,
                              motility_window_min, max_minutes, drift_threshold, drift_correct,
                              slow_percentile)
        if df.empty or "u" not in df.columns:
            continue
        df = df[np.isfinite(df["u"])]
        if df.empty:
            continue
        order = _ordered(df, custom_order)
        edges = np.linspace(0, 1, n_space_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        df = df.assign(sbin=np.clip(np.digitize(df["u"], edges) - 1, 0, n_space_bins - 1),
                       tbin=(df["time_min"] // directionality_window_min).astype(int))
        t_bins = sorted(df["tbin"].unique())
        t_labels = [f"{int(t * directionality_window_min)}-"
                    f"{int((t + 1) * directionality_window_min)}" for t in t_bins]

        df.groupby(["condition", "tbin", "sbin"])[metric].agg(["mean", "sem", "count"]).to_csv(
            os.path.join(path, "chemotaxis_map.csv"))

        signed = metric in ("fmi_par", "fmi_perp", "directedness", "velocity_par")
        vals = df.groupby(["condition", "tbin", "sbin"])[metric].mean()
        vmax = float(np.nanmax(np.abs(vals.values))) if len(vals) else 1.0
        ncols = len(order)
        fig, axes = plt.subplots(2, ncols, figsize=(3.6 * ncols, 6.4), squeeze=False,
                                 gridspec_kw={"height_ratios": [1.25, 1]})
        for c, cond in enumerate(order):
            sub = df[df["condition"] == cond]
            grid = np.full((len(t_bins), n_space_bins), np.nan)
            for i, t in enumerate(t_bins):
                for j in range(n_space_bins):
                    v = sub[(sub["tbin"] == t) & (sub["sbin"] == j)][metric]
                    if len(v) >= min_segments:
                        grid[i, j] = v.mean()
            ax = axes[0][c]
            im = ax.imshow(grid, aspect="auto", origin="upper", extent=[0, 1, len(t_bins) - 0.5, -0.5],
                           cmap="PiYG" if signed else "viridis",
                           vmin=-vmax if signed else None, vmax=vmax)
            ax.set_yticks(range(len(t_bins))); ax.set_yticklabels(t_labels, fontsize=7)
            ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["left", "mid", "right"], fontsize=8)
            ax.set_title(cond, fontsize=9)
            if c == 0:
                ax.set_ylabel("time (min)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

            ax2 = axes[1][c]
            prof = sub.groupby("sbin")[metric].agg(["mean", "sem", "count"]).reindex(range(n_space_bins))
            keep = prof["count"] >= min_segments
            ax2.errorbar(centers[keep.values], prof["mean"][keep], yerr=prof["sem"][keep],
                         fmt="-o", color="#00CC00", capsize=3)
            if signed:
                ax2.axhline(0.0, color="black", lw=0.8)
                ax2.set_ylim(-vmax * 1.2, vmax * 1.2)
            ax2.set_xlim(0, 1)
            ax2.set_xticks([0, 0.5, 1]); ax2.set_xticklabels(["left", "mid", "right"], fontsize=8)
            if c == 0:
                ax2.set_ylabel(metric)
        fig.tight_layout()
        out = os.path.join(path, "plot_chemotaxis_map.png")
        fig.savefig(out, dpi=200); plt.close()
        print(f"Saved: {out}")


def plot_border_exits(celltype, path_list, conditions, custom_order, acquisition_mode, pos_num,
                      time_step, pixelsize_ccd=3.45, objective=10, edge_um=15.0,
                      edge_frac=None, max_minutes=None, end_margin_frames=3):
    """How many cells VANISH at the right vs the left drawn border.

    A track counts as a border exit when it ends (disappears) before the movie ends
    and its last position is within ``edge_um`` MICROMETERS of a border. An absolute
    distance (rather than a fraction of the channel) keeps the exit zone the same
    size relative to a cell no matter how wide the channel is, so different channels
    stay comparable. ``edge_frac`` (fraction of channel width) can be given instead
    for the old relative behaviour. Tracks that simply run to the end of the movie,
    or that end in the interior, are not counted.

    Reported per condition: exit counts at each border, and an exit index
    (right-left)/(right+left) which is +1 if every escape is toward the right border.
    Writes ``border_exits.csv`` + ``plot_border_exits.png`` per folder.
    """
    thresh = MOTILITY_DEFINITION[celltype]
    res = pixelsize_ccd / objective          # um per pixel
    acq_sequential = ACQUISITION_MODE[acquisition_mode]
    num_conditions = len(conditions)

    for path, _ in path_list:
        rows = []
        for f in glob.glob(os.path.join(path, "*" + str(thresh) + "umin*.csv")):
            pos = int(f.split("_")[-4][3:])
            cond_idx = pos // pos_num if acq_sequential else pos % num_conditions
            if cond_idx >= num_conditions:
                continue
            ref = border_utils.reference_cdb_for_pos(path, pos)
            borders = border_utils.load_borders_from_path(ref)
            if borders is None:
                continue
            df = pd.read_csv(f, index_col=0)
            if not {"id", "x", "y", "frame"}.issubset(df.columns):
                continue
            if max_minutes is not None:
                df = df[df["frame"] * time_step / 60.0 <= max_minutes]
            if df.empty:
                continue
            last_frame = int(df["frame"].max())
            for cid, g in df.groupby("id"):
                g = g.sort_values("frame")
                end_f = int(g["frame"].iloc[-1])
                if end_f >= last_frame - end_margin_frames:
                    continue                      # movie ended, the cell did not vanish
                xe, ye = g["x"].iloc[-1], g["y"].iloc[-1]
                u = border_utils.normalized_position(borders, [xe], [ye])[0]
                if not np.isfinite(u):
                    continue
                if edge_frac is not None:                     # relative (fraction of width)
                    near_left, near_right = u <= edge_frac, u >= 1.0 - edge_frac
                else:                                        # absolute distance in um
                    dl, dr = border_utils.distance_to_borders(borders, [xe], [ye])
                    edge_px = edge_um / res
                    near_left, near_right = dl[0] <= edge_px, dr[0] <= edge_px
                if near_left:
                    side = "left"
                elif near_right:
                    side = "right"
                else:
                    continue                      # vanished in the interior: not a border exit
                rows.append({"condition": conditions[cond_idx], "position": pos,
                             "id": int(cid), "side": side, "u_end": float(u),
                             "exit_min": end_f * time_step / 60.0})
        dfe = pd.DataFrame(rows)
        if dfe.empty:
            print(f"  no border exits found in {path}")
            continue
        dfe.to_csv(os.path.join(path, "border_exits.csv"), index=False)

        order = [c for c in custom_order if c in set(dfe["condition"])]
        counts = dfe.groupby(["condition", "side"]).size().unstack(fill_value=0).reindex(order).fillna(0)
        for s in ("left", "right"):
            if s not in counts.columns:
                counts[s] = 0
        x = np.arange(len(order)); w = 0.38
        idx = (counts["right"] - counts["left"]) / (counts["right"] + counts["left"]).replace(0, np.nan)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(2.4 * len(order), 9), 4.4))
        ax1.bar(x - w / 2, counts["left"], width=w, edgecolor="black", color="#DD8452", label="left border")
        ax1.bar(x + w / 2, counts["right"], width=w, edgecolor="black", color="#4C72B0", label="right border")
        ax1.set_ylabel("cells vanishing at the border")
        ttl = "Cells leaving the field at each border"
        if max_minutes:
            ttl += f"  ·  first {int(max_minutes)} min"
        ax1.set_title(ttl)
        ax1.set_xticks(x); ax1.set_xticklabels(order, rotation=45, ha="right")
        ax1.legend(fontsize=8)

        ax2.bar(x, idx.values, width=0.5, edgecolor="black", color="#55A868")
        ax2.axhline(0.0, color="black", lw=0.8)
        ax2.set_ylim(-1, 1)
        ax2.set_ylabel("exit index  (right-left)/(right+left)")
        ax2.set_title("Exit asymmetry  (+1 = all leave right)")
        ax2.set_xticks(x); ax2.set_xticklabels(order, rotation=45, ha="right")
        plt.tight_layout()
        out = os.path.join(path, "plot_border_exits.png")
        plt.savefig(out, dpi=300); plt.close()
        print(f"Saved: {out}")


def plot_rose_over_time(celltype, path_list, conditions, custom_order, acquisition_mode, pos_num,
                        time_step, pixelsize_ccd=3.45, objective=10, n_bins=24,
                        max_minutes=None, directionality_window_min=15.0, motility_window_min=5.5,
                        drift_threshold=None, drift_correct=True, slow_percentile=25,
                        rose_window_min=None):
    """Grid of angle roses: rows = conditions, columns = time windows.

    Each cell contributes its net migration direction over a window (gated by 6-min
    motility). Set ``rose_window_min`` to zoom the rose to a finer timescale (columns
    and direction window); it defaults to ``directionality_window_min``.

    All panels share the SAME radial scale (max cell count across the grid), so the
    circle sizes are comparable on the cell-number level between time steps. Theta
    labels are degrees (0 = toward right border, 180 = left, +/-90 = along channel).
    """
    win = rose_window_min if rose_window_min else directionality_window_min
    for path, _ in path_list:
        df = collect_segments(path, celltype, conditions, acquisition_mode, pos_num, time_step,
                              pixelsize_ccd, objective, win,
                              motility_window_min, max_minutes, drift_threshold, drift_correct, slow_percentile)
        if df.empty:
            continue
        order = _ordered(df, custom_order)
        t_end = max_minutes if max_minutes else (df["time_min"].max() + win)
        edges = np.arange(0, t_end + win, win)
        ncols = max(1, len(edges) - 1)
        nrows = len(order)
        bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        centers = 0.5 * (bins[:-1] + bins[1:]); width = bins[1] - bins[0]

        # first pass: histogram every cell so all panels can share one radial scale
        grid = {}
        rmax = 1
        for r, cond in enumerate(order):
            for c in range(ncols):
                sel = df[(df["condition"] == cond) &
                         (df["time_min"] >= edges[c]) & (df["time_min"] < edges[c + 1])]
                ang = sel["angle"].values
                counts, _ = np.histogram(ang, bins=bins)
                grid[(r, c)] = (ang, counts)
                rmax = max(rmax, int(counts.max()) if counts.size else 0)

        fig, axes = plt.subplots(nrows, ncols, figsize=(2.7 * ncols, 2.9 * nrows),
                                 subplot_kw={"projection": "polar"}, squeeze=False)
        for r, cond in enumerate(order):
            for c in range(ncols):
                ax = axes[r][c]
                ang, counts = grid[(r, c)]
                R, p = rayleigh_test(ang)
                ax.bar(centers, counts, width=width, color="#4C72B0", edgecolor="white", alpha=0.85)
                if ang.size:
                    mean_ang = np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang)))
                    ax.plot([mean_ang, mean_ang], [0, R * rmax], color="crimson", lw=2)
                ax.set_ylim(0, rmax)              # common radial scale -> comparable circle sizes
                ax.set_theta_zero_location("E")   # 0 = toward right border
                ax.set_theta_direction(1)
                ax.tick_params(labelsize=5)       # keep default degree labels (custom labels break polar)
                sig = "*" if (np.isfinite(p) and p < 0.05) else ""
                ax.set_title(f"n={ang.size} p={p:.0e}{sig}", fontsize=6, pad=2)
                if r == 0:
                    ax.annotate(f"{int(edges[c])}-{int(edges[c + 1])} min",
                                xy=(0.5, 1.28), xycoords="axes fraction",
                                ha="center", fontsize=8, weight="bold")
                if c == 0:
                    ax.annotate(cond, xy=(-0.45, 0.5), xycoords="axes fraction",
                                va="center", ha="right", fontsize=9, weight="bold", rotation=90)
        fig.suptitle(f"Migration angle over time (common scale, max n={rmax}; "
                     "0°=right border, ±90°=along channel; red=mean)", fontsize=9)
        fig.tight_layout(rect=(0.03, 0, 1, 0.95))
        out = os.path.join(path, "plot_rose_over_time.png")
        fig.savefig(out, dpi=200); plt.close()
        print(f"Saved: {out}")
