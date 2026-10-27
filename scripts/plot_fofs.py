from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import swiftsimio as sw
from icecream import ic
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from swiftsimio.visualisation.projection import project_pixel_grid

from mrp.swift_utils import load_region, sim_files_from_name, subvolume_regions

"""Plot one region of the simulation with FOF groups marked"""

IMAGE_RESOLUTION = 4096
IMAGE_DPI = 800
TILE_INDEX = 4
sim_name = "L25m7"
snapshot_nr = 127
save_dir = Path("images")


def fof_locs(data_group):
    """For every FoF group, return its ID, the index of one (the first) star in it, and its size. Currently uses int32 as bit range for ungrouped FoFs. Check manually for largest fofgroup_id"""
    fof_ids = np.asarray(data_group.fofgroup_ids)

    FOF_UNGROUPED = np.iinfo(np.int32).max  # check with np.unique(fof_ids)[-3:]

    grouped = np.flatnonzero(fof_ids < FOF_UNGROUPED)  # indices of grouped stars
    group_ids, first, counts = np.unique(
        fof_ids[grouped], return_index=True, return_counts=True
    )
    first_idx = grouped[first]  # map back to indices into the full dataset

    print(
        f"Out of {len(fof_ids)} particles, {len(grouped)} are in a FoF group, "
        f"making {len(group_ids)} groups"
    )
    ic(group_ids[:5], first_idx[:5], counts[:5])
    return group_ids, first_idx, counts


if __name__ == "__main__":
    snapshot_file, catalogue_file = sim_files_from_name(
        sim_name=sim_name, snapshot=snapshot_nr, data_dir=Path("data/colibre")
    )
    #  ─────── load the same region from snapshot and catalogue ───────

    boxsize = sw.mask(snapshot_file).metadata.boxsize
    region = subvolume_regions(boxsize)[TILE_INDEX]  # (3, 2)

    #  ───────────── load the catalogue and snapshot data ─────────────
    data = load_region(snapshot_file, region)
    catalogue = load_region(catalogue_file, region)
    stars = data.stars
    fof_cat = catalogue.input_halos_fof

    #  ─────────────────── Plotting the simulation ────────────────────
    region = region.ravel()
    extent = region[:-2]

    group_ids, first_idx, counts = fof_locs(stars)

    img = (
        project_pixel_grid(
            stars,
            IMAGE_RESOLUTION,
            project="masses",
            region=region,
            parallel=True,
            periodic=True,
        )
        .to_physical("Msun/kpc**2")
        .T
    )

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(cmap(0.0))

    fig, ax = plt.subplots(figsize=(8, 8))

    pos_vals = img.value[img.value > 0]
    vmin = 1e1
    vmax = 5e4
    norm = LogNorm(vmin=vmin, vmax=vmax)
    im = ax.imshow(img.value, norm=norm, cmap=cmap, origin="lower", extent=extent)

    # colour bar so the image has a scale
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(r"$\Sigma_\bigstar$ [M$_\odot$ kpc$^{-2}$]", fontsize=14)

    #  ───── one star per FoF group (white) ─────
    pos = stars.coordinates.to_comoving().to_value("Mpc")
    sizes = 10 + 25 * np.log10(
        counts
    )  # counts has one entry per group, same order as first_idx
    ax.scatter(
        pos[first_idx, 0],
        pos[first_idx, 1],
        s=sizes,
        facecolors="none",
        edgecolors="white",
        linewidths=0.6,
        zorder=3,
    )

    #  ───── halo catalogue FoF groups: centre + radius (red) ─────
    cat_centres = fof_cat.centres.to_value("Mpc")
    cat_radii = fof_cat.radii.to_comoving().to_value("Mpc")
    ax.add_collection(
        PatchCollection(
            [
                Circle((cx, cy), r)
                for cx, cy, r in zip(*cat_centres[:, :2].T, cat_radii)
            ],
            facecolors="none",
            edgecolors="red",
            linewidths=0.6,
            alpha=0.2,
            zorder=2,
        )
    )
    ax.scatter(
        cat_centres[:, 0],
        cat_centres[:, 1],
        marker="+",
        s=12,
        alpha=0.2,
        color="red",
        linewidths=0.6,
        zorder=2,
    )

    ax.set_aspect("equal")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel(r"$X$ [cMpc]", fontsize=14)
    ax.set_ylabel(r"$Y$ [cMpc]", fontsize=14)
    ax.tick_params(labelsize=11)

    z = 1 / data.metadata.scale_factor - 1
    fig.suptitle(f"{sim_name.upper()}, z = {z:.1f}", fontsize=20)
    ax.set_title(
        f"Stellar surface density (region {TILE_INDEX}) with FoF groups",
        fontsize=12,
    )

    # explicit legend handles, since the scatter/patch artists aren't labelled
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            mfc="none",
            mec="white",
            label="Star FoF group (one member star)",
        ),
        Line2D([], [], marker="+", ls="", color="red", label="Catalogue FoF centre"),
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            mfc="none",
            mec="red",
            markersize=10,
            label="Catalogue FoF radius",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        fontsize=10,
        facecolor="black",
        edgecolor="white",
        labelcolor="white",
        framealpha=0.8,
    )

    plt.tight_layout()
    save_dir.mkdir(exist_ok=True)
    fig.savefig(save_dir / f"{sim_name}_region{TILE_INDEX}.png", dpi=IMAGE_DPI)
