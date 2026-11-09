import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from mrp.fof_core import fof_labels

S = -1  # sentinel


def check_fof(pos, labels, l, sentinel=S, boxsize=None):
    """Supervisor's checks (1, 2) plus split / over-merge checks (3, 4)."""
    pos = np.mod(pos, boxsize) if boxsize is not None else pos
    n = len(pos)
    pairs = cKDTree(pos, boxsize=boxsize).query_pairs(l, output_type="ndarray")
    has_nb = np.zeros(n, bool)
    has_nb[pairs.ravel()] = True
    grouped = labels != sentinel

    # 1. every grouped particle has a neighbour within l
    assert has_nb[grouped].all(), "grouped particle with no neighbour within l"
    # 2. no ungrouped particle has any neighbour within l
    assert not has_nb[~grouped].any(), "ungrouped particle has a neighbour within l"
    # 3. no linked pair has different labels (no split groups)
    assert (labels[pairs[:, 0]] == labels[pairs[:, 1]]).all(), "linked pair split"
    # 4. groups == true connected components (no over-merging)
    adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, ref = connected_components(adj, directed=False)
    a, b = labels[grouped], ref[grouped]
    assert len(set(zip(a, b))) == len(set(a)) == len(set(b)), "partition differs"


def run(pos, l, boxsize=None):
    pos = np.asarray(pos, float)
    labels = fof_labels(pos, l, sentinel=S, boxsize=boxsize)
    check_fof(pos, labels, l, boxsize=boxsize)
    return labels


def test_two_clusters():
    lab = run([[0, 0, 0], [0.5, 0, 0], [5, 5, 5], [5.5, 5, 5]], 1.0)
    assert lab[0] == lab[1] and lab[2] == lab[3] and lab[0] != lab[2]


def test_chain_links_transitively():
    lab = run([[i * 0.9, 0, 0] for i in range(6)], 1.0)
    assert len(set(lab)) == 1 and lab[0] != S


def test_isolated_particle_is_ungrouped():
    lab = run([[0, 0, 0], [0.5, 0, 0], [8, 8, 8]], 1.0)
    assert lab[2] == S and lab[0] != S


def test_late_bridge_merges_groups():
    # groups A (0,1) and B (3,4) form first, particle 2 bridges them last
    pos = [[0, 0, 0], [0.9, 0, 0], [1.8, 0, 0], [2.7, 0, 0], [3.6, 0, 0]]
    order = [0, 1, 3, 4, 2]
    p = np.array(pos)[order]
    lab = run(p, 1.0)
    assert len(set(lab)) == 1


def test_exactly_at_linking_length_is_linked():
    lab = run([[0, 0, 0], [1.0, 0, 0]], 1.0)
    assert lab[0] == lab[1] != S


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_pair_across_boundary_each_axis(axis):
    a, b = np.full(3, 5.0), np.full(3, 5.0)
    a[axis], b[axis] = 0.2, 9.9
    lab = run([a, b], 0.5, boxsize=10.0)
    assert lab[0] == lab[1] != S
    # without wrapping they must NOT be linked
    assert (fof_labels([a, b], 0.5, S) == S).all()


def test_pair_across_corner_all_three_axes():
    lab = run([[0.1, 0.1, 0.1], [9.9, 9.9, 9.9]], 0.5, boxsize=10.0)
    assert lab[0] == lab[1] != S


def test_group_spanning_boundary_is_one_group():
    pos = [[9.6, 5, 5], [9.9, 5, 5], [0.2, 5, 5], [0.5, 5, 5]]
    lab = run(pos, 0.4, boxsize=10.0)
    assert len(set(lab)) == 1


def test_sentinel_overflow_guard():
    pos = [[3 * i, 0, 0] for i in range(10)]
    pos = [p for q in zip(pos, [[x + 0.1, y, z] for x, y, z in pos]) for p in q]
    with pytest.raises(OverflowError):
        fof_labels(pos, 0.5, sentinel=3)


@pytest.mark.parametrize("boxsize", [None, 10.0])
@pytest.mark.parametrize("seed", range(5))
def test_random_against_ckdtree(seed, boxsize):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 10, size=(400, 3))
    run(pos, 0.5, boxsize=boxsize)
