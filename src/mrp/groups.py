import numpy as np
from numpy._typing import ArrayLike, NDArray
from swiftsimio import cosmo_array
from swiftsimio.reader import __SWIFTGroupDataset

from mrp.utils import check_sentinel

GROUP_SENTINEL = np.iinfo(np.int32).max


def valid_group_mask(data_gr: ArrayLike) -> NDArray[np.bool_]:
    """Checks group particle data like fofgroud_ids and checks for sentinels and filters.
    Returns mask of valid non-sentinel objects"""
    sentinel = check_sentinel(data_gr)
    return np.asarray(data_gr) != sentinel


def centre_of_mass(
    data: __SWIFTGroupDataset,
    group: str,
) -> cosmo_array:
    """Compute the centre of mass of grouped objects by their group attribute. So centre_of_mass(data.stars, group='fofgroup_ids') gives the centre of mass of each fofgroup.

    Args:
        data (__SWIFTGroupDataset): The particle dataset. E.g. data.stars, data.dark_matter. Not to be confused with the group specification
        group (str): The field particles are grouped by, e.g. fofgroup_ids, hbt track id, etc.
    Returns:
        CoM (cosmo_array) with shape (n_groups, 3)"""

    assert hasattr(data, group)
    data_gr: cosmo_array = getattr(data, group)

    is_grouped = valid_group_mask(data_gr)

    # Note: uniques[inverse] = original
    # inverse is basically another label for the fof_ids
    uniques, inverse, count = np.unique(
        data_gr[is_grouped], return_inverse=True, return_counts=True
    )

    pos = data.coordinates[is_grouped].to("Mpc").to_ndarray()
    masses = data.masses[is_grouped].to("Msun").to_ndarray()
    total_mass = np.bincount(inverse, weights=masses)

    com_grouped = np.zeros((uniques.size, 3))
    for i in range(3):
        com_grouped[:, i] = (
            np.bincount(inverse, weights=pos[:, i] * masses) / total_mass
        )
    com_grouped = cosmo_array(
        com_grouped,
        cosmo_factor=data.coordinates.cosmo_factor,
        scale_exponent=1,
        scale_factor=data.metadata.scale_factor,
    )
    print(f"{type(com_grouped)=}")
    print(f"{com_grouped=}")
    com_full = np.full_like(data.coordinates, np.nan)  # full length, NaN = "no group"
    com_full[is_grouped] = com_grouped[inverse]
    assert com_full.shape == data.coordinates.shape, (
        f"The shapes of the CoMs is not in the right homogeneous SWIFT shape: {com_full.shape} != {data.coordinates.shape}."
    )
    field_name = f"com_{group}"
    setattr(data, field_name, com_full)
    print(f"Set field {field_name} to {type(data)}.")
    return com_grouped
