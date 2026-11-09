"""Save / load FoF results in a standalone HDF5 file."""

from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np


def save_fof_hdf5(path, particle_ids, labels, sentinel, attrs=None):
    """Write per-particle FoF labels (+ a small per-group table) to `path`.

    Layout:
        /Stars/ParticleIDs   (n,)  int64   -> to match back onto the snapshot
        /Stars/FoFNaiveIDs   (n,)  int64   -> group label, `sentinel` = ungrouped
        /Groups/GroupIDs     (g,)  int64
        /Groups/Sizes        (g,)  int64   -> number of stars per group
        /Header              attributes (sim, snapshot, tile, l_link, ...)
    """
    path = Path(path)
    particle_ids = np.asarray(particle_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    assert particle_ids.shape == labels.shape

    grouped = labels != sentinel
    group_ids, sizes = np.unique(labels[grouped], return_counts=True)

    tmp = path.with_suffix(".hdf5.tmp")  # write-then-rename: no half-written files
    with h5py.File(tmp, "w") as f:
        stars = f.create_group("Stars")
        stars.create_dataset("ParticleIDs", data=particle_ids, compression="gzip")
        stars.create_dataset("FoFNaiveIDs", data=labels, compression="gzip")
        grp = f.create_group("Groups")
        grp.create_dataset("GroupIDs", data=group_ids)
        grp.create_dataset("Sizes", data=sizes)

        hdr = f.create_group("Header")
        hdr.attrs["sentinel"] = int(sentinel)
        hdr.attrs["n_particles"] = labels.size
        hdr.attrs["n_groups"] = group_ids.size
        hdr.attrs["n_ungrouped"] = int((~grouped).sum())
        hdr.attrs["created"] = datetime.now(UTC).isoformat()
        for k, v in (attrs or {}).items():
            hdr.attrs[k] = v
    tmp.replace(path)
    return path


def load_fof_hdf5(path):
    """Read a file written by save_fof_hdf5. Returns dict with arrays + header."""
    with h5py.File(path, "r") as f:
        return {
            "particle_ids": f["Stars/ParticleIDs"][:],
            "labels": f["Stars/FoFNaiveIDs"][:],
            "group_ids": f["Groups/GroupIDs"][:],
            "group_sizes": f["Groups/Sizes"][:],
            "header": dict(f["Header"].attrs),
        }


def match_labels_to_particles(snapshot_ids, fof):
    """Map saved labels onto a snapshot's particle IDs (any order / subset).

    Particles not present in the file get the sentinel.
    """
    order = np.argsort(fof["particle_ids"])
    sorted_ids = fof["particle_ids"][order]
    idx = np.searchsorted(sorted_ids, snapshot_ids)
    idx = np.clip(idx, 0, sorted_ids.size - 1)
    found = sorted_ids[idx] == snapshot_ids
    out = np.full(snapshot_ids.shape, fof["header"]["sentinel"], dtype=np.int64)
    out[found] = fof["labels"][order[idx[found]]]
    return out
