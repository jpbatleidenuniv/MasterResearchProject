"""Pure-numpy naive Friends-of-Friends (no SWIFT dependency, so it is easy to test)."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from tqdm import tqdm


def fof_labels(pos, l_link, sentinel=-1, boxsize=None, progress=False):
    """Naive O(n^2) FoF with optional periodic boundaries.

    Two particles are linked if their (minimum-image) distance is <= l_link.
    Convention is `<=` so it matches scipy's cKDTree.

    Args:
        pos: (n, 3) array of positions, same units as l_link.
        l_link: linking length.
        sentinel: label for ungrouped particles.
        boxsize: None (no wrapping), a float, or a length-3 array. Wrapping is
            applied in all three dimensions: any separation larger than half
            the box is replaced by the separation through the boundary.
    Returns:
        (n,) int64 array of labels; ungrouped particles hold `sentinel`.
    """
    pos = np.asarray(pos, dtype=float)
    n = len(pos)
    l2 = float(l_link) ** 2
    box = None if boxsize is None else np.asarray(boxsize, dtype=float)

    labels = np.full(n, sentinel, dtype=np.int64)
    next_id = 0

    for i in tqdm(range(n - 1), disable=not progress, desc="Naive FoF"):
        delta = pos[i + 1 :] - pos[i]
        if box is not None:
            delta -= box * np.round(delta / box)  # minimum image, all 3 axes
        r2 = np.einsum("ij,ij->i", delta, delta)

        for j in np.nonzero(r2 <= l2)[0] + (i + 1):
            id_i, id_j = labels[i], labels[j]
            if id_i == sentinel and id_j == sentinel:
                if sentinel >= 0 and next_id >= sentinel:
                    raise OverflowError("More groups than the sentinel can distinguish")
                labels[i] = labels[j] = next_id
                next_id += 1
            elif id_i == sentinel:
                labels[i] = id_j
            elif id_j == sentinel:
                labels[j] = id_i
            elif id_i != id_j:
                labels[labels == id_j] = id_i  # merge the two groups
    return labels


def fof_labels_kdtree(pos, l_link, sentinel=-1, boxsize=None):
    """Fast FoF (~O(n log n)): cKDTree pairs + connected components.

    Same linking rule as `fof_labels` (distance <= l_link, minimum image in all
    3 dimensions), so the partition is identical; only the label numbers differ.
    """
    pos = np.asarray(pos, dtype=float)
    n = len(pos)
    box = None if boxsize is None else np.broadcast_to(np.asarray(boxsize, float), (3,))
    if box is not None:
        pos = np.mod(pos, box)
        pos[pos >= box] = 0.0  # np.mod can round up to exactly L; cKDTree needs [0, L)

    pairs = cKDTree(pos, boxsize=box).query_pairs(l_link, output_type="ndarray")
    adj = coo_matrix(
        (np.ones(len(pairs), bool), (pairs[:, 0], pairs[:, 1])), shape=(n, n)
    )
    _, comp = connected_components(adj, directed=False)
    grouped = np.bincount(comp)[comp] > 1  # singletons are ungrouped

    labels = np.full(n, sentinel, dtype=np.int64)
    _, consecutive = np.unique(comp[grouped], return_inverse=True)
    if sentinel >= 0 and consecutive.size and consecutive.max() >= sentinel:
        raise OverflowError("More groups than the sentinel can distinguish")
    labels[grouped] = consecutive
    return labels


def fof_naive(data, l_link, sentinel=-1):
    """Thin SWIFT wrapper. Sets data.stars.fof_ids_naive and returns the labels."""
    from swiftsimio import cosmo_array

    stars = data.stars
    boxsize = data.metadata.boxsize.to("kpc").value
    pos = np.mod(stars.coordinates.to("kpc").value, boxsize)
    labels = fof_labels(
        pos, l_link.to("kpc").value, sentinel=sentinel, boxsize=boxsize, progress=True
    )
    stars.fof_ids_naive = cosmo_array(
        labels,
        units="dimensionless",
        comoving=True,
        scale_factor=data.metadata.scale_factor,
        scale_exponent=0,
    )
    return labels
