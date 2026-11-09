"""Tools to inspect FoF groups: select a group, scatter its members, colour groups.

Everything here works on plain arrays so it can be used for any grouping:
    pos     (n, 3) positions (any length unit; comoving kpc in the scripts)
    labels  (n,)   group label per particle, `sentinel` = ungrouped
    boxsize None, float or (3,) -> periodic box; None = no wrapping

Groups that straddle the periodic boundary are unwrapped first, so they appear
as one contiguous object. This assumes a group is smaller than half the box.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.spatial import cKDTree

AXES = "xyz"


# ───────────────────────────── selection ─────────────────────────────
def min_image(delta, boxsize=None):
    """Wrap separations to the nearest periodic image, in all 3 dimensions."""
    if boxsize is None:
        return delta
    box = np.asarray(boxsize, dtype=float)
    return delta - box * np.round(delta / box)


def largest_groups(labels, sentinel, n=20):
    """Return (group_ids, sizes) of the n largest groups, largest first."""
    labels = np.asarray(labels)
    ids, counts = np.unique(labels[labels != sentinel], return_counts=True)
    order = np.argsort(-counts, kind="stable")[:n]
    return ids[order], counts[order]


def select_group(labels, group_id):
    """Indices of the particles belonging to `group_id`."""
    idx = np.flatnonzero(np.asarray(labels) == group_id)
    if idx.size == 0:
        raise ValueError(f"Group {group_id} not found")
    return idx


def unwrap_group(pos_members, boxsize=None):
    """Members' positions made contiguous across the periodic boundary."""
    ref = pos_members[0]
    unwrapped = ref + min_image(pos_members - ref, boxsize)
    if boxsize is not None:
        extent = np.ptp(unwrapped, axis=0)
        if (extent > 0.5 * np.asarray(boxsize)).any():
            print(
                f"Warning: group extent {extent} exceeds half the box; unwrap unreliable"
            )
    return unwrapped


def find_close_groups(
    pos, labels, sentinel, boxsize=None, max_sep=1.0, min_size=2, n=None
):
    """Merger candidates: pairs of *different* groups with a member pair closer than max_sep.

    Since FoF links everything within l_link, any such pair has min_sep > l_link;
    choose max_sep = a few times l_link. Returns a list of dicts sorted by the
    combined size of the two groups (largest first):
        group_a, group_b, size_a, size_b, min_sep, n_close_pairs
    """
    labels = np.asarray(labels)
    g = np.flatnonzero(labels != sentinel)
    p = np.asarray(pos, dtype=float)[g]
    box = None if boxsize is None else np.broadcast_to(np.asarray(boxsize, float), (3,))
    if box is not None:
        p = np.mod(p, box)
        p[p >= box] = 0.0
    pairs = cKDTree(p, boxsize=box).query_pairs(max_sep, output_type="ndarray")
    if len(pairs) == 0:
        return []
    la, lb = labels[g[pairs[:, 0]]], labels[g[pairs[:, 1]]]
    d = np.linalg.norm(min_image(p[pairs[:, 0]] - p[pairs[:, 1]], box), axis=1)
    diff = la != lb
    la, lb, d = la[diff], lb[diff], d[diff]
    if la.size == 0:
        return []
    lo, hi = np.minimum(la, lb), np.maximum(la, lb)
    order = np.lexsort((hi, lo))
    lo, hi, d = lo[order], hi[order], d[order]
    start = np.flatnonzero(np.r_[True, (lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1])])
    dmin = np.minimum.reduceat(d, start)
    npairs = np.diff(np.r_[start, d.size])

    ids, counts = np.unique(labels[labels != sentinel], return_counts=True)
    size = dict(zip(ids.tolist(), counts.tolist()))
    rows = [
        {
            "group_a": int(lo[k]),
            "group_b": int(hi[k]),
            "size_a": size[int(lo[k])],
            "size_b": size[int(hi[k])],
            "min_sep": float(dm),
            "n_close_pairs": int(npr),
        }
        for k, dm, npr in zip(start, dmin, npairs)
        if size[int(lo[k])] >= min_size and size[int(hi[k])] >= min_size
    ]
    rows.sort(key=lambda r: -(r["size_a"] + r["size_b"]))
    return rows[:n]


def print_close_groups(rows, unit="ckpc"):
    print(
        f"{'group A':>8} {'N_A':>6} {'group B':>8} {'N_B':>6} {'min sep[' + unit + ']':>14} {'n pairs':>8}"
    )
    for r in rows:
        print(
            f"{r['group_a']:>8} {r['size_a']:>6} {r['group_b']:>8} {r['size_b']:>6} "
            f"{r['min_sep']:>14.3f} {r['n_close_pairs']:>8}"
        )


def group_summary(pos, labels, sentinel, boxsize=None, masses=None, n=20):
    """Table (list of dicts) for the n largest groups: size, mass, centre, max radius."""
    rows = []
    for rank, (gid, size) in enumerate(zip(*largest_groups(labels, sentinel, n)), 1):
        idx = select_group(labels, gid)
        up = unwrap_group(pos[idx], boxsize)
        w = None if masses is None else masses[idx]
        centre = np.average(up, axis=0, weights=w)
        rows.append(
            {
                "rank": rank,
                "group_id": int(gid),
                "n_stars": int(size),
                "mass": None if w is None else float(w.sum()),
                "centre": np.mod(centre, boxsize) if boxsize is not None else centre,
                "r_max": float(np.linalg.norm(up - centre, axis=1).max()),
            }
        )
    return rows


def print_group_table(rows, unit="ckpc"):
    has_mass = rows and rows[0]["mass"] is not None
    head = f"{'rank':>4} {'group':>8} {'N':>7} " + (
        f"{'mass[Msun]':>11} " if has_mass else ""
    )
    print(head + f"{'r_max[' + unit + ']':>11}   centre")
    for r in rows:
        m = f"{r['mass']:>11.3e} " if has_mass else ""
        c = ", ".join(f"{x:.1f}" for x in r["centre"])
        print(
            f"{r['rank']:>4} {r['group_id']:>8} {r['n_stars']:>7} {m}{r['r_max']:>11.2f}   ({c})"
        )


# ───────────────────────────── plotting ──────────────────────────────
SELECTED_COLOURS = [
    "tab:red",
    "tab:blue",
    "tab:green",
    "tab:orange",
    "tab:purple",
    "tab:brown",
]
NEIGHBOUR_COLOURS = [
    "tab:cyan",
    "gold",
    "tab:pink",
    "tab:olive",
    "lightgreen",
    "violet",
    "salmon",
]


def plot_groups(
    ax,
    pos,
    labels,
    group_ids,
    sentinel,
    boxsize=None,
    l_link=None,
    projection=(0, 1),
    context=True,
    show_links=True,
    show_gaps=True,
    label_groups=True,
    pad=0.3,
    unit="ckpc",
    legend=True,
    max_neighbours=6,
    legend_fontsize=8,
):
    """Scatter one or more selected groups, each in its own colour, with a legend.

    - Selected groups (`group_ids`) get strong colours and are numbered in the plot.
    - Neighbouring groups in the window get their own lighter colours (legend lists
      the largest `max_neighbours`; any further ones are "other nearby groups").
    - Ungrouped stars are light grey.
    - Lines join pairs closer than l_link, in the colour of their group. A link
      between two *different* groups would be drawn black and flagged in the
      legend: that would mean the FoF is wrong.
    - With 2-4 selected groups (a merger candidate), a dashed line marks the
      closest pair of stars between each two groups, labelled with the 3D gap.
    The window is a cube around the selected groups, so grey/coloured neighbours
    are real neighbours, not line-of-sight projections.
    """
    labels = np.asarray(labels)
    group_ids = [int(g) for g in np.atleast_1d(group_ids)]
    i, j = projection
    members = {g: select_group(labels, g) for g in group_ids}
    sel_idx = np.concatenate(list(members.values()))

    # one common frame: nearest image relative to the first selected star, then centred
    rel = min_image(pos - pos[members[group_ids[0]][0]], boxsize)
    rel = rel - rel[sel_idx].mean(axis=0)

    half = (1 + pad) * 0.5 * np.ptp(rel[sel_idx], axis=0).max()
    if l_link is not None:
        half = max(half, 2 * l_link)
    half = max(half, 1e-9)
    in_win = (np.abs(rel) < half).all(axis=1)
    if not context:
        in_win &= np.isin(labels, group_ids)
    win = np.flatnonzero(in_win)
    wl = labels[win]

    # colours
    colour = {
        g: SELECTED_COLOURS[k % len(SELECTED_COLOURS)] for k, g in enumerate(group_ids)
    }
    oids, ocnt = np.unique(
        wl[(wl != sentinel) & ~np.isin(wl, group_ids)], return_counts=True
    )
    oids = oids[np.argsort(-ocnt, kind="stable")]
    named, rest = oids[:max_neighbours], oids[max_neighbours:]
    for k, g in enumerate(named):
        colour[int(g)] = NEIGHBOUR_COLOURS[k % len(NEIGHBOUR_COLOURS)]
    rest_colour = "0.4"

    def scatter(idx, col, s, z, lab=None):
        if len(idx):
            ax.scatter(
                rel[idx, i], rel[idx, j], s=s, color=col, lw=0, zorder=z, label=lab
            )

    # ungrouped, other groups, named neighbours, selected (back to front)
    scatter(win[wl == sentinel], "0.8", 3, 1, "ungrouped stars" if context else None)
    if len(rest):
        scatter(
            win[np.isin(wl, rest)],
            rest_colour,
            5,
            2,
            f"{len(rest)} other nearby groups",
        )
    for g in named:
        scatter(
            win[wl == g],
            colour[int(g)],
            8,
            3,
            f"Group {g} (N={int((labels == g).sum())})",
        )
    for g in group_ids:
        scatter(
            members[g], colour[g], 16, 5, f"Group {g} (N={len(members[g])}) [selected]"
        )
        if label_groups:
            c = rel[members[g]].mean(axis=0)
            ax.annotate(
                str(g),
                (c[i], c[j]),
                fontsize=9,
                weight="bold",
                color="k",
                zorder=7,
                xytext=(5, 5),
                textcoords="offset points",
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "fc": "w",
                    "ec": colour[g],
                    "alpha": 0.85,
                },
            )

    # link lines
    if show_links and l_link is not None:
        gi = win[wl != sentinel]
        if len(gi) > 1:
            pr = cKDTree(rel[gi]).query_pairs(l_link, output_type="ndarray")
            if len(pr):
                lab_a, lab_b = labels[gi[pr[:, 0]]], labels[gi[pr[:, 1]]]
                bad = lab_a != lab_b
                cols = [colour.get(int(a), rest_colour) for a in lab_a]
                cols = np.where(
                    bad[:, None], np.array([[0, 0, 0, 1.0]]), matplotlib_colors(cols)
                )
                segs = rel[gi][pr][:, :, [i, j]]
                ax.add_collection(
                    LineCollection(
                        segs,
                        colors=cols,
                        linewidths=np.where(bad, 1.8, 0.5),
                        alpha=0.7,
                        zorder=4,
                    )
                )
                if bad.any():
                    ax.plot(
                        [],
                        [],
                        color="k",
                        lw=1.8,
                        label="linked but different groups (!)",
                    )

    # gaps between selected groups (merger candidates)
    if show_gaps and 2 <= len(group_ids) <= 4:
        for a_k, a in enumerate(group_ids):
            for b in group_ids[a_k + 1 :]:
                d, nn = cKDTree(rel[members[b]]).query(rel[members[a]])
                m = int(np.argmin(d))
                pa, pb = rel[members[a][m]], rel[members[b][nn[m]]]
                ax.plot([pa[i], pb[i]], [pa[j], pb[j]], "k--", lw=1, zorder=6)
                mid = 0.5 * (pa + pb)
                ax.annotate(
                    f"gap {d[m]:.2f} {unit}",
                    (mid[i], mid[j]),
                    fontsize=8,
                    zorder=7,
                    xytext=(0, -12),
                    textcoords="offset points",
                    ha="center",
                    bbox={"boxstyle": "round,pad=0.15", "fc": "w", "alpha": 0.85},
                )

    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal")
    ax.set_xlabel(rf"$\Delta${AXES[i]} [{unit}]")
    ax.set_ylabel(rf"$\Delta${AXES[j]} [{unit}]")
    ax.set_title(
        f"Group {group_ids[0]}: {len(members[group_ids[0]])} stars"
        if len(group_ids) == 1
        else "Groups " + ", ".join(map(str, group_ids)),
        fontsize=11,
    )
    ax.grid(True, color="0.5", alpha=0.35, lw=0.5)
    ax.set_axisbelow(True)
    if l_link is not None:
        add_linking_length_bar(ax, l_link, unit, fontsize=legend_fontsize)
    if legend:
        ax.legend(
            loc="upper right",
            fontsize=legend_fontsize,
            framealpha=0.85,
            markerscale=1.5,
        )
    return ax


def add_linking_length_bar(ax, l_link, unit="ckpc", loc="lower left", fontsize=8):
    """Scale bar of length l_link, in the axes' data units, to compare with point separations.

    Two stars closer than the bar are linked; farther apart they are not (in 3D, so a
    pair that looks closer than the bar in a projection can still be farther apart).
    """
    ymin, ymax = ax.get_ylim()
    bar = AnchoredSizeBar(
        ax.transData,
        l_link,
        f"l = {l_link:g} {unit}",
        loc=loc,
        pad=0.5,
        sep=4,
        frameon=True,
        size_vertical=0.012 * (ymax - ymin),
        fill_bar=True,
        color="k",
        fontproperties={"size": fontsize},
    )
    bar.patch.set_alpha(0.85)
    ax.add_artist(bar)
    return bar


def matplotlib_colors(cols):
    from matplotlib.colors import to_rgba_array

    return to_rgba_array(cols)


def plot_group(
    ax,
    pos,
    labels,
    group_id,
    sentinel,
    boxsize=None,
    l_link=None,
    projection=(0, 1),
    **kw,
):
    """Single-group convenience wrapper around plot_groups."""
    return plot_groups(
        ax, pos, labels, [group_id], sentinel, boxsize, l_link, projection, **kw
    )


def plot_group_projections(
    pos, labels, group_ids, sentinel, boxsize=None, l_link=None, **kw
):
    """xy, xz and yz projections of one group (int) or several groups (list), legend on the first."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3))
    for k, (ax, proj) in enumerate(zip(axes, [(0, 1), (0, 2), (1, 2)])):
        plot_groups(
            ax,
            pos,
            labels,
            group_ids,
            sentinel,
            boxsize,
            l_link,
            projection=proj,
            legend=(k == 0),
            **kw,
        )
    fig.tight_layout()
    return fig


def plot_largest_groups_grid(
    pos,
    labels,
    sentinel,
    n=20,
    ncols=5,
    boxsize=None,
    l_link=None,
    projection=(0, 1),
    **kw,
):
    """Small multiples of the n largest groups; each panel has a small legend of what is in view."""
    ids, _ = largest_groups(labels, sentinel, n)
    nrows = int(np.ceil(len(ids) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows), squeeze=False
    )
    kw.setdefault("legend_fontsize", 6)
    kw.setdefault("max_neighbours", 3)
    for k, ax in enumerate(axes.ravel()):
        if k >= len(ids):
            ax.axis("off")
            continue
        plot_group(
            ax,
            pos,
            labels,
            ids[k],
            sentinel,
            boxsize,
            l_link,
            projection=projection,
            **kw,
        )
        ax.set_title(
            f"#{k + 1}  id {ids[k]}  N={int((labels == ids[k]).sum())}", fontsize=9
        )
        ax.tick_params(labelsize=7)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.tight_layout()
    return fig


def plot_groups_overview(
    pos,
    labels,
    sentinel,
    boxsize=None,
    n_highlight=20,
    ax=None,
    projection=(0, 1),
    unit="ckpc",
):
    """All stars in one panel: the n largest groups each get their own colour,
    numbered by rank and listed in a legend; other groups light grey, ungrouped dark grey."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 9))
    i, j = projection
    labels = np.asarray(labels)
    grouped = labels != sentinel
    ids, sizes = largest_groups(labels, sentinel, n_highlight)
    top = np.isin(labels, ids)

    ax.scatter(
        pos[~grouped, i],
        pos[~grouped, j],
        s=0.5,
        color="0.25",
        lw=0,
        rasterized=True,
        label="ungrouped stars",
    )
    ax.scatter(
        pos[grouped & ~top, i],
        pos[grouped & ~top, j],
        s=2,
        color="0.75",
        lw=0,
        rasterized=True,
        label="other groups",
    )

    cmap = plt.get_cmap("tab20" if len(ids) <= 20 else "turbo")
    for rank, (gid, size) in enumerate(zip(ids, sizes)):
        col = cmap(rank % 20) if len(ids) <= 20 else cmap(rank / max(len(ids) - 1, 1))
        idx = select_group(labels, gid)
        ax.scatter(
            pos[idx, i],
            pos[idx, j],
            s=8,
            color=col,
            lw=0,
            zorder=3,
            rasterized=True,
            label=f"#{rank + 1}  id {gid}  (N={size})",
        )
        c = (
            np.mod(unwrap_group(pos[idx], boxsize).mean(axis=0), boxsize)
            if boxsize is not None
            else pos[idx].mean(0)
        )
        ax.annotate(
            str(rank + 1),
            (c[i], c[j]),
            color=col,
            fontsize=9,
            weight="bold",
            xytext=(4, 4),
            textcoords="offset points",
            zorder=4,
        )
    ax.set_aspect("equal")
    ax.set_xlabel(f"{AXES[i]} [{unit}]")
    ax.set_ylabel(f"{AXES[j]} [{unit}]")
    ax.set_title(f"{len(ids)} largest groups coloured (numbers = rank)", fontsize=11)
    ax.grid(True, color="0.5", alpha=0.35, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=7,
        markerscale=2.5,
        framealpha=0.9,
        title="group",
    )
    return ax
