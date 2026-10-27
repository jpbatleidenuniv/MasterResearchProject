import numpy as np

from swiftsimio import SWIFTDataset
from swiftsimio.objects import cosmo_quantity
from tqdm import tqdm


def fof_naive(data: SWIFTDataset, l_link: cosmo_quantity, sentinel: int = -1):
    """Runs the Naive Friends-of-Friends algorithm, sets the ungrouped particles to the sentinel value. In most cases this is the largest unsigned integer datatype of int16/32/64

    Args:
        data (SWIFTDataset) : Particle dataset
        l_link (cosmo_quantity): the linking length
        sentinel (float): the value for ungrouped particles

    Returns:
        The FoF ids (ndarray), also in-place adds the data.stars.fof_ids_naive group in the SWIFTDataset"""

    positions = data.stars.coordinates.to("kpc").value
    l_link2 = l_link.to("kpc").value ** 2
    n = len(positions)
    fof_ids = np.full(n, sentinel, dtype=np.int64)

    next_id = 0

    for i in tqdm(range(n), desc=f"Naive FoF algorithm on {n} particles"):
        for j in range(i + 1, n):
            # Loop over all pairs
            delta = positions[i] - positions[j]
            r2 = np.dot(delta, delta)

            # Check if outside linking-length
            if r2 >= l_link2:
                continue

            # Get their FoF ids
            id_i = fof_ids[i]
            id_j = fof_ids[j]

            # If both ungrouped, we group them
            if id_i == sentinel and id_j == sentinel:
                fof_ids[i] = next_id
                fof_ids[j] = next_id
                next_id += 1

            # If 1 is ungrouped we adopt him to 2's group, viceversa.
            elif id_i == sentinel:
                fof_ids[i] = id_j
            elif id_j == sentinel:
                fof_ids[j] = id_i

            # Both are different group. we set all friends of 2 into 1s group
            elif id_i != id_j:
                fof_ids[fof_ids == id_j] = id_i

    ids, counts = np.unique(fof_ids, return_counts=True)

    n_groups = np.count_nonzero(ids != sentinel)
    n_ungrouped = counts[ids == sentinel].sum()

    print("Naive FoF algorithm completed.")
    print(f"Number of groups found: {n_groups}")
    print(f"Ungrouped particles: {n_ungrouped} / {n}")
    data.stars.fof_ids_naive = fof_ids

    return fof_ids
