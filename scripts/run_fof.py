from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import swiftsimio as sw
from icecream import ic

from mrp.fof import fof_naive
from mrp.swift_utils import load_tile_data

if __name__ == "__main__":
    sim_name = "L25m7"
    snapshot_nr = 127
    TILE_INDEX = 4
    data = load_tile_data(sim_name, snapshot=snapshot_nr, tile_index=TILE_INDEX)
    coordinates = data.stars.coordinates

    sentinel = np.iinfo(np.int16).max
    print(f"Using sentinel: {sentinel}")

    boxsize = data.metadata.boxsize
    l_emp = sw.cosmo_quantity(
        0.78, units="kpc", cosmo_factor=boxsize.cosmo_factor, comoving=True
    )

    fof_ids = fof_naive(data, l_link=l_emp, sentinel=sentinel)
    ids, counts = np.unique(fof_ids, return_counts=True)
    second_largest = ids[-2]
    ic(second_largest)

    # Group-size distribution, excluding ungrouped particles
    ids, counts = np.unique(fof_ids, return_counts=True)
    group_counts = counts[ids != sentinel]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(group_counts, bins=30, edgecolor="black", color="red")
    ax.set_xlabel("Number of particles per FoF group")
    ax.set_ylabel("Number of groups")
    ax.set_yscale("log")

    output_dir = Path("images")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_dir / f"fof_group_sizes_T{TILE_INDEX}_{sim_name}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)
