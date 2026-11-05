from mrp.swift_utils import load_tile_data
from mrp.groups import group_by


def test_group_by():
    data = load_tile_data(sim_name="l25m7", snapshot=127, tile_index=0)
    stars = data.stars
    group_by(stars, group_name="fofgroup_ids")
