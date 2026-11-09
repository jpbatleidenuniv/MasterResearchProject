"""Inspect the largest FoF groups saved by run_fof.py.

Writes to images/fof_inspect/<file stem>/ (every plot has a legend naming each group):
    overview.png                all stars, the N largest groups coloured
    largest_groups_grid.png     small multiples of the N largest groups
    group_<id>_projections.png  xy / xz / yz view of the top --n-detail groups
    groups_<ids>_projections.png  chosen groups together (--group-id a b ...)
    merger_<a>_<b>.png          nearby unlinked group pairs with the gap marked (--mergers)
and prints a table (size, mass, centre, max radius).

Examples:
    python scripts/inspect_fof.py --tile 4 --n-groups 20
    python scripts/inspect_fof.py --tile 4 --group-id 17            # one group
    python scripts/inspect_fof.py --tile 4 --group-id 17 42         # several groups in one view
    python scripts/inspect_fof.py --tile 4 --mergers                # nearby-but-unlinked group pairs
    python scripts/inspect_fof.py --tile all --method kdtree --mergers
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import swiftsimio as sw

from mrp.fof_io import load_fof_hdf5, match_labels_to_particles
from mrp.plot.fof_inspect import (
    find_close_groups,
    group_summary,
    largest_groups,
    plot_group_projections,
    plot_groups_overview,
    plot_largest_groups_grid,
    print_close_groups,
    print_group_table,
)
from mrp.swift_utils import load_region, sim_files_from_name, subvolume_regions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sim", default="L25m7")
    p.add_argument("--snapshot", type=int, default=127)
    p.add_argument("--tile", default="4", help="tile index or 'all'")
    p.add_argument("--n-side", type=int, default=3)
    p.add_argument("--method", choices=["naive", "kdtree"], default="naive")
    p.add_argument(
        "--l-link-kpc", type=float, default=0.78, help="comoving kpc (for link lines)"
    )
    p.add_argument("--n-groups", type=int, default=20)
    p.add_argument(
        "--n-detail", type=int, default=10, help="per-group projection figures"
    )
    p.add_argument(
        "--group-id",
        type=int,
        nargs="+",
        default=None,
        help="inspect only these group(s); several ids are drawn together",
    )
    p.add_argument(
        "--mergers",
        action="store_true",
        help="plot pairs of nearby but unlinked groups (merger candidates)",
    )
    p.add_argument(
        "--merger-sep-factor",
        type=float,
        default=3.0,
        help="candidates have a gap < factor * l_link",
    )
    p.add_argument(
        "--min-size", type=int, default=10, help="min stars per group for mergers"
    )
    p.add_argument("--n-mergers", type=int, default=10)
    p.add_argument("--data-dir", type=Path, default=Path("data/colibre"))
    args = p.parse_args()

    full_box = args.tile.lower() == "all"
    tag = "fullbox" if full_box else f"tile{int(args.tile)}_n{args.n_side}"
    snapshot_file, _ = sim_files_from_name(args.sim, args.snapshot, args.data_dir)
    fof_file = snapshot_file.parent / f"fof_{args.method}_{args.snapshot:04}_{tag}.hdf5"
    fof = load_fof_hdf5(fof_file)
    sentinel = int(fof["header"]["sentinel"])

    # reload the same stars and attach the saved labels via particle IDs
    boxsize = sw.mask(snapshot_file).metadata.boxsize
    if full_box:
        data = sw.load(snapshot_file)
    else:
        region = subvolume_regions(boxsize, n_side=args.n_side)[int(args.tile)]
        data = load_region(snapshot_file, region)
    stars = data.stars
    pos = stars.coordinates.to("kpc").value
    masses = stars.masses.to("Msun").value
    labels = match_labels_to_particles(stars.particle_ids.value, fof)
    box_kpc = boxsize.to("kpc").value
    pos = np.mod(pos, box_kpc)

    out = Path("images/fof_inspect") / fof_file.stem
    out.mkdir(parents=True, exist_ok=True)

    rows = group_summary(pos, labels, sentinel, box_kpc, masses, n=args.n_groups)
    print_group_table(rows)

    if args.group_id is not None:
        fig = plot_group_projections(
            pos, labels, args.group_id, sentinel, box_kpc, l_link=args.l_link_kpc
        )
        name = "_".join(map(str, args.group_id))
        fig.savefig(
            out / f"groups_{name}_projections.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)
    elif args.mergers:
        cand = find_close_groups(
            pos,
            labels,
            sentinel,
            box_kpc,
            max_sep=args.merger_sep_factor * args.l_link_kpc,
            min_size=args.min_size,
            n=args.n_mergers,
        )
        print_close_groups(cand)
        for c in cand:
            fig = plot_group_projections(
                pos,
                labels,
                [c["group_a"], c["group_b"]],
                sentinel,
                box_kpc,
                l_link=args.l_link_kpc,
            )
            fig.savefig(
                out / f"merger_{c['group_a']}_{c['group_b']}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(11, 9))
        plot_groups_overview(pos, labels, sentinel, box_kpc, args.n_groups, ax=ax)
        fig.savefig(out / "overview.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        fig = plot_largest_groups_grid(
            pos,
            labels,
            sentinel,
            args.n_groups,
            boxsize=box_kpc,
            l_link=args.l_link_kpc,
        )
        fig.savefig(out / "largest_groups_grid.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        for gid in largest_groups(labels, sentinel, args.n_detail)[0]:
            fig = plot_group_projections(
                pos, labels, int(gid), sentinel, box_kpc, l_link=args.l_link_kpc
            )
            fig.savefig(
                out / f"group_{gid}_projections.png", dpi=150, bbox_inches="tight"
            )
            plt.close(fig)
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
