"""
Shared renderer for "cell distribution along the chemokine axis over time".

Used by both the standard chemokine assay (time = hourly folders) and the
long-term assay (time = frame bins within one database). Given, per condition,
the normalized axis positions (u in [0, 1], 0 = left border, 1 = right border)
grouped by time bin, it draws one figure per condition with two clear views:

* a ridgeline (one filled, smoothed distribution per time bin, stacked top=early
  to bottom=late) so the *shape* and drift of the population is obvious, and
* a density heatmap with the per-bin mean position overlaid as a trend line, so a
  net shift toward a border reads as a single moving curve.
"""
import os
import numpy as np
import matplotlib.pyplot as plt


def _smooth(y, k=3):
    if k <= 1 or y.size < 3:
        return y
    kernel = np.ones(k) / k
    return np.convolve(y, kernel, mode="same")


def render_distribution(collected, order, labels, out_dir, prefix,
                        n_bins=24, time_title="time"):
    """
    collected : dict {(condition, time_bin_index): [u, ...]}
    order     : list of condition names, in plot order
    labels    : list of time-bin labels (index == time_bin_index)
    out_dir   : folder to write PNGs into
    prefix    : filename prefix -> ``<prefix>_<condition>.png``
    Returns the list of written paths.
    """
    n_time = len(labels)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    written = []

    present = [c for c in order if any((c, b) in collected for b in range(n_time))]
    for condition in present:
        dens = np.zeros((n_time, n_bins))
        mean_u = np.full(n_time, np.nan)
        median_u = np.full(n_time, np.nan)
        n_cells = np.zeros(n_time, dtype=int)
        for b in range(n_time):
            vals = np.asarray(collected.get((condition, b), []), dtype=float)
            n_cells[b] = vals.size
            if vals.size:
                h, _ = np.histogram(vals, bins=bins)
                s = h.sum()
                if s:
                    dens[b] = _smooth(h / s)
                mean_u[b] = vals.mean()
                median_u[b] = np.median(vals)

        cmap = plt.get_cmap("viridis")
        colors = [cmap(i / max(n_time - 1, 1)) for i in range(n_time)]

        fig, (axr, axh) = plt.subplots(
            1, 2, figsize=(12, max(4.5, 0.55 * n_time + 1.5)),
            gridspec_kw={"width_ratios": [1.15, 1.0]},
        )

        # ---- ridgeline: earliest on top, latest at the bottom ----
        peak = dens.max() if dens.max() > 0 else 1.0
        scale = 1.7 / peak                      # vertical exaggeration per row
        for i, b in enumerate(range(n_time)):
            base = n_time - 1 - i               # top = first time bin
            y = base + dens[b] * scale
            axr.fill_between(centers, base, y, color=colors[b], alpha=0.85,
                             linewidth=0.8, edgecolor="white", zorder=n_time - i)
            tick_h = 0.9   # fixed marker height (row spacing is 1.0)
            if np.isfinite(mean_u[b]):
                axr.plot([mean_u[b], mean_u[b]], [base, base + tick_h], color="black",
                         lw=1.3, zorder=n_time + 1)       # mean position tick
            if np.isfinite(median_u[b]):
                axr.plot([median_u[b], median_u[b]], [base, base + tick_h], color="red",
                         lw=1.3, ls=(0, (2, 1)), zorder=n_time + 2)  # median position tick
        # legend proxies for the two position markers
        axr.plot([], [], color="black", lw=1.3, label="mean")
        axr.plot([], [], color="red", lw=1.3, ls=(0, (2, 1)), label="median")
        axr.legend(loc="upper right", fontsize=7, framealpha=0.7)
        axr.set_yticks([n_time - 1 - i for i in range(n_time)])
        axr.set_yticklabels(labels, fontsize=8)
        axr.set_ylim(-0.3, n_time + 1.2)
        axr.set_xlim(0, 1)
        axr.set_xlabel("Position along axis  (0 = left border, 1 = right border)")
        axr.set_ylabel(time_title)
        axr.set_title(f"{condition}: distribution per {time_title}")
        axr.axvline(0.5, color="grey", ls=":", lw=1, alpha=0.7)

        # ---- heatmap + mean-position trend line ----
        im = axh.imshow(dens, aspect="auto", cmap="magma", origin="upper",
                        extent=[0, 1, n_time - 0.5, -0.5], interpolation="nearest")
        good = np.isfinite(mean_u)
        axh.plot(mean_u[good], np.arange(n_time)[good], "-o", color="cyan",
                 lw=2, markersize=5, markeredgecolor="black",
                 label="mean position")
        gmed = np.isfinite(median_u)
        axh.plot(median_u[gmed], np.arange(n_time)[gmed], "--s", color="red",
                 lw=2, markersize=5, markeredgecolor="black",
                 label="median position")
        axh.axvline(0.5, color="white", ls=":", lw=1, alpha=0.6)
        axh.set_yticks(range(n_time))
        axh.set_yticklabels(labels, fontsize=8)
        axh.set_xlim(0, 1)
        axh.set_xlabel("Position along axis (0=left, 1=right)")
        axh.set_title("Density + mean/median position")
        axh.legend(loc="upper right", fontsize=8, framealpha=0.7)
        fig.colorbar(im, ax=axh, fraction=0.046, pad=0.04, label="fraction of cells")

        outpath = os.path.join(out_dir, f"{prefix}_{condition}.png")
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()
        written.append(outpath)
        print(f"Saved: {outpath}")
    return written
