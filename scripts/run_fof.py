"""Run FoF on the stars of one tile (or the entire box) and save labels to HDF5.

Examples:
    python scripts/run_fof.py --sim L25m7 --snapshot 127 --tile 4
    python scripts/run_fof.py --sim L25m7 --snapshot 127 --tile all --method kdtree
"""

import argparse
from pathlib import Path

import numpy as np
import swiftsimio as sw

from mrp.fof_check import check_fof
from mrp.fof_core import fof_labels_kdtree, fof_naive
from mrp.fof_io import save_fof_hdf5
from mrp.groups import GROUP_SENTINEL
from mrp.swift_utils import load_region, sim_files_from_name, subvolume_regions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sim", default="L25m7")
    p.add_argument("--snapshot", type=int, default=127)
    p.add_argument(
        "--tile", default="4", help="tile index, or 'all' for the entire box"
    )
    p.add_argument("--n-side", type=int, default=3)
    p.add_argument("--l-link-kpc", type=float, default=0.78, help="comoving kpc")
    p.add_argument("--data-dir", type=Path, default=Path("data/colibre"))
    p.add_argument("--method", choices=["naive", "kdtree"], default="naive")
    p.add_argument("--no-check", action="store_true", help="skip the correctness check")
    p.add_argument("--force", action="store_true", help="overwrite existing output")
    args = p.parse_args()

    full_box = args.tile.lower() == "all"
    tag = "fullbox" if full_box else f"tile{int(args.tile)}_n{args.n_side}"

    snapshot_file, _ = sim_files_from_name(args.sim, args.snapshot, args.data_dir)
    out_path = snapshot_file.parent / f"fof_{args.method}_{args.snapshot:04}_{tag}.hdf5"
    if out_path.exists() and not args.force:
        print(f"{out_path} exists, use --force to overwrite.")
        return

    boxsize = sw.mask(snapshot_file).metadata.boxsize
    if full_box:
        data = sw.load(snapshot_file)  # no mask: the whole box
        region = np.stack([np.zeros(3), boxsize.to("Mpc").value], axis=-1) * 1.0
        region_mpc = region
    else:
        region = subvolume_regions(boxsize, n_side=args.n_side)[int(args.tile)]
        data = load_region(snapshot_file, region)
        region_mpc = region.to_value("Mpc")
    stars = data.stars
    n = stars.coordinates.shape[0]
    print(f"{n} stars in {tag}; method = {args.method}.")
    if args.method == "naive":
        print(
            "Naive FoF is O(n^2): the full box is ~ (n_side^2)^2 times slower than a tile."
        )

    l_link = sw.cosmo_quantity(
        args.l_link_kpc, units="kpc", cosmo_factor=boxsize.cosmo_factor, comoving=True
    )
    if args.method == "naive":
        labels = fof_naive(data, l_link=l_link, sentinel=GROUP_SENTINEL)
    else:
        box_kpc = boxsize.to("kpc").value
        labels = fof_labels_kdtree(
            stars.coordinates.to("kpc").value,
            l_link.to("kpc").value,
            sentinel=GROUP_SENTINEL,
            boxsize=box_kpc,
        )

    if not args.no_check:
        # independent cKDTree check on the real data; raises (and nothing is saved) on failure
        summary = check_fof(
            stars.coordinates.to("kpc").value,
            labels,
            args.l_link_kpc,
            GROUP_SENTINEL,
            boxsize.to("kpc").value,
        )
        print(f"FoF correctness check PASSED: {summary}")

    save_fof_hdf5(
        out_path,
        particle_ids=stars.particle_ids.value,
        labels=labels,
        sentinel=GROUP_SENTINEL,
        attrs={
            "sim_name": args.sim,
            "snapshot": args.snapshot,
            "tile": args.tile,
            "method": args.method,
            "n_side": args.n_side,
            "l_link_kpc_comoving": args.l_link_kpc,
            "boxsize_kpc_comoving": boxsize.to("kpc").value,
            "region_mpc": region_mpc,
            "scale_factor": float(data.metadata.scale_factor),
            "periodic": True,
            "check_passed": not args.no_check,
        },
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
