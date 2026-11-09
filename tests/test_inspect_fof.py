import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mrp.fof_core import fof_labels_kdtree
from mrp.plot.fof_inspect import (
    group_summary,
    largest_groups,
    plot_group,
    plot_group_projections,
    plot_groups_overview,
    plot_largest_groups_grid,
    select_group,
    unwrap_group,
)

S = -1
L = 10.0


def make_data(seed=0):
    rng = np.random.default_rng(seed)
    bg = rng.uniform(0, L, (300, 3))
    clusters = [
        (np.array([5.0, 5.0, 5.0]), 80),
        (np.array([0.0, 5.0, 5.0]), 60),  # straddles x edge
        (np.array([2.0, 2.0, 9.9]), 40),
        (np.array([7.0, 8.0, 3.0]), 25),
    ]
    pts = [bg] + [c + rng.normal(0, 0.25, (n, 3)) for c, n in clusters]
    pos = np.mod(np.vstack(pts), L)
    return pos, fof_labels_kdtree(pos, 0.15, S, boxsize=L)


def test_unwrap_boundary_group():
    p = np.array([[9.8, 5, 5], [0.1, 5, 5], [0.3, 5, 5]])
    assert np.ptp(unwrap_group(p, L), axis=0).max() < 0.6


def test_largest_groups_sorted():
    _, lab = make_data()
    ids, sizes = largest_groups(lab, S, 5)
    assert (np.diff(sizes) <= 0).all() and len(ids) == 5


def test_summary_and_select():
    pos, lab = make_data()
    rows = group_summary(pos, lab, S, L, n=3)
    assert rows[0]["n_stars"] >= rows[1]["n_stars"]
    assert len(select_group(lab, rows[0]["group_id"])) == rows[0]["n_stars"]
    assert max(r["r_max"] for r in rows) < 0.5 * L  # boundary group not torn apart


def test_plots_run():
    pos, lab = make_data()
    ids, _ = largest_groups(lab, S, 4)
    _fig, ax = plt.subplots()
    plot_group(ax, pos, lab, ids[0], S, L, 0.15)
    plot_group_projections(pos, lab, ids[0], S, L, 0.15)
    plot_largest_groups_grid(pos, lab, S, 4, ncols=2, boxsize=L, l_link=0.15)
    plot_groups_overview(pos, lab, S, L, 4)
    plt.close("all")
