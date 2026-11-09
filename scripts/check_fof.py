"""Run the FoF correctness checks on a FoF file that was already saved by run_fof.py.

python scripts/check_fof.py --tile 4 --method naive
"""

import argparse
from pathlib import Path

import numpy as np
import swiftsimio as sw

from mrp.fof_check import check_fof
from mrp.fof_io import load_fof_hdf5, match_labels_to_particles
from mrp.swift_utils import load_region, sim_files_from_name, subvolume_regions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sim", default="L25m7")
    p.add_argument("--snapshot", type=int, default=127)
    p.add_argument("--tile", default="4", help="tile index or 'all'")
    p.add_argument("--n-side", type=int, default=3)
    p.add_argument("--method", choices=["naive", "kdtree"], default="naive")
    p.add_argument("--data-dir", type=Path, default=Path("data/colibre"))
    args = p.parse_args()

    full_box = args.tile.lower() == "all"
    tag = "fullbox" if full_box else f"tile{int(args.tile)}_n{args.n_side}"
    snapshot_file, _ = sim_files_from_name(args.sim, args.snapshot, args.data_dir)
    fof = load_fof_hdf5(
        snapshot_file.parent / f"fof_{args.method}_{args.snapshot:04}_{tag}.hdf5"
    )
    sentinel = int(fof["header"]["sentinel"])
    l_link = float(fof["header"]["l_link_kpc_comoving"])  # the l the file was made with

    boxsize = sw.mask(snapshot_file).metadata.boxsize
    if full_box:
        data = sw.load(snapshot_file)
    else:
        data = load_region(
            snapshot_file, subvolume_regions(boxsize, args.n_side)[int(args.tile)]
        )
    stars = data.stars
    ids = stars.particle_ids.value
    assert np.isin(ids, fof["particle_ids"]).all(), (
        "snapshot stars missing from the FoF file"
    )
    labels = match_labels_to_particles(ids, fof)

    summary = check_fof(
        stars.coordinates.to("kpc").value,
        labels,
        l_link,
        sentinel,
        boxsize.to("kpc").value,
    )
    print(f"FoF correctness check PASSED (l = {l_link} ckpc): {summary}")


if __name__ == "__main__":
    main()
