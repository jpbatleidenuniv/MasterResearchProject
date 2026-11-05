import numpy as np

_BITS = (8, 16, 32, 64)
_CANDIDATES = (
    {-1}
    | {2 ** (b - 1) - 1 for b in _BITS}  # signed max: 127, 32767, 2147483647, ...
    | {2**b - 1 for b in _BITS}  # unsigned max: 255, 65535, ...
)


def check_sentinel(x) -> int | None:
    """Detect a 'no group' sentinel in an integer label array.

    Looks for -1 and the max value of 8/16/32/64-bit signed and unsigned integers.
    Returns the sentinel as an int, None if there isn't one, and raises if
    more than one distinct candidate value is present.
    """
    x = np.asarray(x)
    if not np.issubdtype(x.dtype, np.integer):
        raise TypeError(f"Expected an integer array, got {x.dtype}")

    # only compare against values representable in x's dtype (avoids overflow/casting issues)
    info = np.iinfo(x.dtype)
    candidates = sorted(c for c in _CANDIDATES if info.min <= c <= info.max)

    found = np.unique(x[np.isin(x, candidates)])

    if found.size == 0:
        print("No sentinels found in data")
        return None
    if found.size > 1:
        raise ValueError(f"Multiple sentinel candidates found: {found}")
    return int(found[0])
