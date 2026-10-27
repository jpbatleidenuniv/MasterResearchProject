from pathlib import Path

import numpy as np
import swiftsimio as sw


def sim_files_from_name(
    sim_name: str, snapshot: int, data_dir: Path = Path("data/colibre")
) -> tuple[Path, Path]:
    """Extract simulation and halo catalogue files from simulation name, e.g. L100m6. Not case-sensitive.

    TODO: Make it work for LXXXNYYY-format names.

    Args:
        sim_name (str): The simulation name. e.g. L100m6, not case-sensitive
        snapshot (int): The snapshot number, for COLIBRE between 1-128
        data_dir (Path): Path where simulations are are stored, i.e. data_dir/L100m6/...

    Returns:
        tuple of particle and catalogue filepaths, (snapshot_file, catalogue_file)

    """

    available_sim = [
        sim
        for sim in data_dir.iterdir()
        if sim.is_dir() and sim.name.casefold() == sim_name.casefold()
    ]
    if not available_sim:
        raise ValueError(f"No directory has been found for {sim_name} in {data_dir}")

    snapshot_dir = available_sim[0] / "SOAP-HBT"
    available_snaps = {
        int(f.stem.rsplit("_", maxsplit=1)[-1])
        for f in snapshot_dir.glob("colibre_with_SOAP_membership_*.hdf5")
    }

    if snapshot not in available_snaps:
        raise ValueError(
            f"Snapshot {snapshot} is not available in {snapshot_dir}.\n"
            f"Available snaps: {sorted(available_snaps)}"
        )

    snapshot_file = snapshot_dir / f"colibre_with_SOAP_membership_{snapshot:04}.hdf5"
    catalogue_file = snapshot_dir / f"halo_properties_{snapshot:04}.hdf5"

    missing = [f for f in (snapshot_file, catalogue_file) if not f.exists()]
    if missing:
        raise ValueError(
            f"Missing files for {sim_name}, snapshot {snapshot}: {missing}\n"
            f"Files found: {sorted(snapshot_dir.glob(f'*{snapshot:04}.hdf5'))}"
        )
    print(
        f"Extracted simulation files for {sim_name} snapshot {snapshot}:\nParticle data: {snapshot_file}.\nCatalogue data: {catalogue_file}."
    )

    return snapshot_file, catalogue_file


def load_region(path: Path | str, region: sw.cosmo_array):
    """Load a a region of a snapshot by masking and"""
    mask = sw.mask(path)
    mask.constrain_spatial(region)
    return sw.load(path, mask=mask)


def subvolume_regions(boxsize: sw.cosmo_array, n_side: int = 3):
    """Return a cosmo_array of shape (n_side**2, 3, 2) with [lower, upper] bounds per piece."""
    edges = np.linspace(0, 1, n_side + 1)
    lo, hi = edges[:-1], edges[1:]

    xlo, ylo = np.meshgrid(lo, lo, indexing="ij")
    xhi, yhi = np.meshgrid(hi, hi, indexing="ij")
    zlo, zhi = np.zeros_like(xlo), np.ones_like(xlo)

    fractions = np.stack(
        [
            np.stack([xlo, xhi], axis=-1),
            np.stack([ylo, yhi], axis=-1),
            np.stack([zlo, zhi], axis=-1),
        ],
        axis=-2,
    ).reshape(-1, 3, 2)

    return fractions * boxsize.reshape(1, 3, 1)


def load_tile_data(sim_name, snapshot, tile_index):
    snapshot_file, catalogue_file = sim_files_from_name(
        sim_name=sim_name, snapshot=snapshot, data_dir=Path("data/colibre")
    )
    boxsize = sw.mask(snapshot_file).metadata.boxsize
    region = subvolume_regions(boxsize)[tile_index]  # (3, 2)

    data = load_region(snapshot_file, region)
    return data
