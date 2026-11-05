from mrp.groups import valid_group_mask
from swiftsimio import cosmo_array
from mrp.utils import check_sentinel
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
from mrp.fof import fof_naive
from mrp.plot.plot_box import plot_box_masses
from mrp.groups import centre_of_mass

"""Plot one region of the simulation with FOF groups marked"""

IMAGE_RESOLUTION = 4096
IMAGE_DPI = 800
TILE_INDEX = 4
sim_name = "L25m7"
snapshot_nr = 127
save_dir = Path("images/fof_locations")
save_dir.mkdir(parents=True, exist_ok=True)


def marker_size(base_point):
    return 10 + 25 * np.log(base_point)


if __name__ == "__main__":
    snapshot_file, catalogue_file = sim_files_from_name(
        sim_name=sim_name, snapshot=snapshot_nr, data_dir=Path("data/colibre")
    )
    #  ───── load the same region from snapshot and catalogue ─────

    boxsize = sw.mask(snapshot_file).metadata.boxsize
    region = subvolume_regions(boxsize)[TILE_INDEX]  # (3, 2)

    #  ─────────── load the catalogue and snapshot data ───────────
    data = load_region(snapshot_file, region)
    catalogue = load_region(catalogue_file, region)
    stars = data.stars
    fof_cat = catalogue.input_halos_fof

    #  ───────────────── Plotting the simulation ──────────────────
    region = region.ravel()
    fig, ax = plt.subplots(figsize=(6, 6))
    im = plot_box_masses(data_field=stars, ax=ax, vminmax=(1e1, 5e4), region=region)
    # colour bar so the image has a scale
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(r"$\Sigma_\bigstar$ [M$_\odot$ kpc$^{-2}$]", fontsize=14)

    #  ───────────── Computing Centres of FoF groups ──────────────
    com_fof = centre_of_mass(data=stars, group="fofgroup_ids")
    # is_grouped = valid_group_mask(group_data=stars.fofgroup_ids)
    # uniques, unq_indices, counts = np.unique(
    #     stars.fofgroup_ids[is_grouped], return_index=True, return_counts=True
    # )
    #
    # #  ──────────────── Plotting SWIFT FoF-groups ─────────────────
    # ax.scatter(
    #     com_fof[is_grouped][:, 0][unq_indices],
    #     com_fof[is_grouped][:, 1][unq_indices],
    #     ec="w",
    #     fc="none",
    #     marker="o",
    #     s=marker_size(counts),
    #     label="SWIFT FoF stars",
    # )
    #
    # # Naive FoF
    # SENTINEL = np.iinfo(np.int32).max
    # l_emp = sw.cosmo_quantity(
    #     0.78, units="kpc", cosmo_factor=boxsize.cosmo_factor, comoving=True
    # )
    # unique_ids, representative, counts_naive = fof_naive(
    #     data=data, l_link=l_emp, sentinel=SENTINEL
    # )
    # ic(stars.fof_ids_naive)
    #
    # keep = unique_ids != SENTINEL  # drops the ungrouped entry
    #
    # com_fof_naive = centre_of_mass(data=stars, group="fof_ids_naive")
    # is_grouped = valid_group_mask(group_data=stars.fof_ids_naive)
    # uniques, unq_indices, counts = np.unique(
    #     stars.fof_ids_naive[is_grouped], return_index=True, return_counts=True
    # )
    #
    # #  ──────────────── Plotting SWIFT FoF-groups ─────────────────
    # ax.scatter(
    #     com_fof_naive[is_grouped][:, 0][unq_indices],
    #     com_fof_naive[is_grouped][:, 1][unq_indices],
    #     ec="magenta",
    #     fc="none",
    #     marker="s",
    #     s=marker_size(counts),
    #     label="Naive FoF stars",
    #     linewidths=0.8,
    #     zorder=4,
    # )
    # plt.legend()
    # plt.show()
    #
    # #  ───── halo catalogue FoF groups: centre + radius (red) ─────
    # cat_centres = fof_cat.centres.to_value("Mpc")
    # cat_radii = fof_cat.radii.to_comoving().to_value("Mpc")
    # ax.add_collection(
    #     PatchCollection(
    #         [
    #             Circle((cx, cy), r)
    #             for cx, cy, r in zip(*cat_centres[:, :2].T, cat_radii)
    #         ],
    #         facecolors="none",
    #         edgecolors="red",
    #         linewidths=0.6,
    #         alpha=0.2,
    #         zorder=2,
    #     )
    # )
    # ax.scatter(
    #     cat_centres[:, 0],
    #     cat_centres[:, 1],
    #     marker="+",
    #     s=12,
    #     alpha=0.2,
    #     color="red",
    #     linewidths=0.6,
    #     zorder=2,
    # )
    #
    #
    # z = 1 / data.metadata.scale_factor - 1
    # fig.suptitle(f"{sim_name.upper()}, z = {z:.1f}", fontsize=20)
    #
    # # explicit legend handles, since the scatter/patch artists aren't labelled
    # handles = [
    #     Line2D(
    #         [],
    #         [],
    #         marker="o",
    #         ls="",
    #         mfc="none",
    #         mec="white",
    #         label="Snapshot FoF group (one member star)",
    #     ),
    #     Line2D(
    #         [],
    #         [],
    #         marker="s",
    #         ls="",
    #         mfc="none",
    #         mec="magenta",
    #         label="Naive star FoF (one member star)",
    #     ),
    #     Line2D([], [], marker="+", ls="", color="red", label="Catalogue FoF centre"),
    #     Line2D(
    #         [],
    #         [],
    #         marker="o",
    #         ls="",
    #         mfc="none",
    #         mec="red",
    #         markersize=10,
    #         label="Catalogue FoF radius",
    #     ),
    # ]
    # ax.legend(
    #     handles=handles,
    #     loc="lower left",
    #     fontsize=10,
    #     facecolor="black",
    #     edgecolor="white",
    #     labelcolor="white",
    #     framealpha=0.8,
    # )
    #
    # plt.tight_layout()
    # save_dir.mkdir(exist_ok=True)
    # fig.savefig(save_dir / f"{sim_name}_region{TILE_INDEX}.png", dpi=IMAGE_DPI)
    #
    # swift_ids = np.asarray(stars.fofgroup_ids)
    # naive_ids = np.asarray(stars.fof_ids_naive)
    # keep = unique_ids != SENTINEL
    #
    # def colour_stars(ax, ids, xy, title, s=4):
    #     """Grouped stars coloured by group label, ungrouped (sentinel) stars grey."""
    #     grouped = ids != SENTINEL
    #     ax.scatter(xy[~grouped, 0], xy[~grouped, 1], s=s / 2, color="0.35", lw=0)
    #     ax.scatter(
    #         xy[grouped, 0], xy[grouped, 1], s=s, c=ids[grouped] % 20, cmap="tab20", lw=0
    #     )
    #     ax.set_title(title, fontsize=12)
    #     ax.set_aspect("equal")
    #
    # # ── Figure A: whole tile, stars coloured by group ───────────────────
    # fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
    # colour_stars(axes[0], swift_ids, pos[:, :2], f"SWIFT FoF ({len(group_ids)} groups)")
    # colour_stars(
    #     axes[1], naive_ids, pos[:, :2], f"Naive FoF, l = 0.78 kpc ({keep.sum()} groups)"
    # )
    # for ax in axes:
    #     ax.set_xlabel("X [cMpc]")
    # axes[0].set_ylabel("Y [cMpc]")
    # axes[0].set_xlim(extent[0], extent[1])
    # axes[0].set_ylim(extent[2], extent[3])
    # fig.tight_layout()
    # fig.savefig(save_dir / f"{sim_name}_region{TILE_INDEX}_colour.png", dpi=300)
    #
    # # ── Figure B: zoom on the N largest SWIFT groups ────────────────────
    # N_ZOOM = 4
    # largest = np.argsort(counts)[::-1][:N_ZOOM]
    # fig, axes = plt.subplots(2, N_ZOOM, figsize=(4 * N_ZOOM, 8.5))
    #
    # for col, g in enumerate(largest):
    #     members = swift_ids == group_ids[g]
    #     centre = np.median(pos[members, :2], axis=0)
    #     half = max(0.6 * np.ptp(pos[members, :2], axis=0).max(), 0.02)  # Mpc
    #     window = (np.abs(pos[:, :2] - centre) < half).all(axis=1)  # all stars in view
    #
    #     # top: the SWIFT group highlighted among its surroundings
    #     ax = axes[0, col]
    #     ax.scatter(
    #         pos[window & ~members, 0], pos[window & ~members, 1], s=3, color="0.4", lw=0
    #     )
    #     ax.scatter(pos[members, 0], pos[members, 1], s=4, color="tab:red", lw=0)
    #     ax.set_title(f"SWIFT group {group_ids[g]}\n{counts[g]} stars", fontsize=11)
    #
    #     # bottom: the same window coloured by naive group
    #     ax = axes[1, col]
    #     colour_stars(ax, naive_ids[window], pos[window, :2], "")
    #     n_in = len(np.unique(naive_ids[members & (naive_ids != SENTINEL)]))
    #     ax.set_title(f"Naive: {n_in} groups in this SWIFT group", fontsize=11)
    #
    #     for ax in axes[:, col]:
    #         ax.set_xlim(centre[0] - half, centre[0] + half)
    #         ax.set_ylim(centre[1] - half, centre[1] + half)
    #         ax.set_aspect("equal")
    # axes[1, 0].set_xlabel("X [cMpc]")
    # axes[1, 0].set_ylabel("Y [cMpc]")
    # fig.tight_layout()
    # fig.savefig(save_dir / f"{sim_name}_region{TILE_INDEX}_zoom.png", dpi=300)
    #
    # # ── Figure C: group size distributions + fraction grouped ───────────
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    # bins = np.logspace(
    #     0, np.log10(max(counts.max(), counts_naive[keep].max())) + 0.1, 25
    # )
    # ax1.hist(counts, bins=bins, histtype="step", lw=1.5, label="SWIFT FoF")
    # ax1.hist(
    #     counts_naive[keep],
    #     bins=bins,
    #     histtype="step",
    #     lw=1.5,
    #     color="magenta",
    #     label="Naive FoF",
    # )
    # ax1.set_xscale("log")
    # ax1.set_yscale("log")
    # ax1.set_xlabel("Stars per group")
    # ax1.set_ylabel("Number of groups")
    # ax1.legend()
    #
    # frac = [(swift_ids != SENTINEL).mean(), (naive_ids != SENTINEL).mean()]
    # ax2.bar(["SWIFT", "Naive"], frac, color=["tab:blue", "magenta"])
    # ax2.set_ylabel("Fraction of stars in a group")
    # ax2.set_ylim(0, 1)
    # fig.tight_layout()
    # fig.savefig(save_dir / f"{sim_name}_region{TILE_INDEX}_stats.png", dpi=300)
