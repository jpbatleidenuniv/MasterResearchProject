import numpy as np
import pytest

from mrp.utils import check_sentinel


def test_detects_minus_one():
    assert check_sentinel([1, 2, 15000, -1]) == -1


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
def test_detects_signed_max_for_each_dtype(dtype):
    smax = np.iinfo(dtype).max
    x = np.array([1, 2, 100, smax], dtype=dtype)
    assert check_sentinel(x) == smax


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.uint64])
def test_detects_unsigned_max_for_each_dtype(dtype):
    umax = np.iinfo(dtype).max
    x = np.array([1, 2, 100, umax], dtype=dtype)
    assert check_sentinel(x) == umax


def test_python_list_with_int16_max():
    # a list becomes int64, where 32767 is still a candidate
    smax = np.iinfo(np.int16).max
    assert check_sentinel([1, 2, 15000, smax]) == smax


def test_returns_python_int():
    result = check_sentinel(np.array([0, 1, 2147483647], dtype=np.int32))
    assert isinstance(result, int)


def test_repeated_sentinel_is_not_multiple():
    # 15 ungrouped stars share one sentinel: that is one distinct value
    x = np.array([0, 0, 7, 2147483647, 2147483647, 7], dtype=np.int32)
    assert check_sentinel(x) == 2147483647


def test_no_sentinel_returns_none(capsys):
    assert check_sentinel([0, 1, 2, 15000]) is None
    assert "No sentinels" in capsys.readouterr().out


def test_multiple_sentinels_raise():
    smax = np.iinfo(np.int16).max
    with pytest.raises(ValueError):
        check_sentinel([1, 2, 15000, smax, -1])


def test_non_integer_raises():
    with pytest.raises(TypeError):
        check_sentinel(np.array([0.5, 1.5]))


def test_unsigned_dtype_ignores_minus_one_candidate():
    # -1 is not representable in uint8; it must not cause an overflow error
    x = np.array([1, 2, 255], dtype=np.uint8)
    assert check_sentinel(x) == 255
