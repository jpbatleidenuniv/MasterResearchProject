from icecream import ic
from mrp.swift_utils import load_tile_data

data = load_tile_data(sim_name="l25m7", snapshot=127, tile_index=0)
data.stars.my_field = [0, 1, 2]
