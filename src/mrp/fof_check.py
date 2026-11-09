"""Independent correctness checks for FoF labels (uses scipy's cKDTree as reference)."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def check_fof(pos, labels, l_link, sentinel, boxsize=None):
    """Raise AssertionError if `labels` are not the FoF groups of `pos`.

    pos: (n, 3); l_link and boxsize in the same units as pos. Periodic wrapping in
    all 3 dimensions if boxsize is given. Linking rule: distance <= l_link.

    Checks:
      1. every grouped particle has a neighbour within l_link
      2. no ungrouped particle has a neighbour within l_link      (supervisor's two)
      3. no pair within l_link carries different labels           (split groups)
      4. groups are exactly the connected components of the links (over-merging)
    Returns a summary dict when everything passes.
    """
    pos = np.asarray(pos, dtype=float)
    labels = np.asarray(labels)
    n = len(pos)
    assert labels.shape == (n,), (
        f"labels {labels.shape} do not match positions {pos.shape}"
    )
    box = None if boxsize is None else np.broadcast_to(np.asarray(boxsize, float), (3,))
    if box is not None:
        pos = np.mod(pos, box)
        pos[pos >= box] = 0.0  # cKDTree needs [0, L)

    pairs = cKDTree(pos, boxsize=box).query_pairs(l_link, output_type="ndarray")
    has_nb = np.zeros(n, bool)
    has_nb[pairs.ravel()] = True
    grouped = labels != sentinel

    bad = int((~has_nb[grouped]).sum())
    assert bad == 0, f"{bad} grouped particle(s) with no neighbour within l"
    bad = int(has_nb[~grouped].sum())
    assert bad == 0, f"{bad} ungrouped particle(s) with a neighbour within l"
    bad = int((labels[pairs[:, 0]] != labels[pairs[:, 1]]).sum())
    assert bad == 0, f"{bad} linked pair(s) split across groups"

    adj = coo_matrix(
        (np.ones(len(pairs), bool), (pairs[:, 0], pairs[:, 1])), shape=(n, n)
    )
    _, ref = connected_components(adj, directed=False)
    a, b = labels[grouped].astype(np.int64), ref[grouped].astype(np.int64)
    n_a, n_b = np.unique(a).size, np.unique(b).size
    n_ab = np.unique(a * (b.max() + 1 if b.size else 1) + b).size
    assert n_a == n_b == n_ab, (
        f"partition differs from connected components ({n_a} vs {n_b} groups)"
    )

    return {
        "n_particles": n,
        "n_grouped": int(grouped.sum()),
        "n_groups": n_a,
        "n_links": len(pairs),
    }
