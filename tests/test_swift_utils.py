from pathlib import Path
from mrp.swift_utils import sim_files_from_name


def test_sim_from_file_name():
    sim_file = "data/colibre/l25m7/SOAP-HBT/colibre_with_SOAP_membership_0127.hdf5"
    cat_file = "data/colibre/l25m7/SOAP-HBT/halo_properties_0127.hdf5"

    result = (Path(sim_file), Path(cat_file))
    assert sim_files_from_name("L25m7", 127) == result


sim_files_from_name("L25M7", 127)
