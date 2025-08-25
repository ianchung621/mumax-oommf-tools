import numpy as np
import pytest
from pathlib import Path

from mumax_oommf_tools import read_ovf2, read_simulation_result

DATA_DIR = Path(__file__).parent / "data" / "tiny_ovfs"
FILES = [
    "text.ovf",
    "binary4.ovf",
    "binary8.ovf",
]

@pytest.mark.parametrize("fname", FILES)
def test_ovf_shapes_and_values(fname):
    fn = DATA_DIR / fname
    meta, arr = read_ovf2(fn)

    assert arr.shape == (2, 2, 2, 3)

    # Expected values: (i+1, j+1, k+1)
    expected = np.zeros((2, 2, 2, 3), dtype=float)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                expected[i, j, k] = (i+1, j+1, k+1)

    assert np.allclose(arr, expected, rtol=1e-6, atol=1e-6)


def test_out():
    OUT_DIR = Path(__file__).parent / "data" / "test.out"
    metadata, time, magnetization = read_simulation_result(OUT_DIR, overwrite=True)

    assert len(time) == 10
    assert magnetization.shape == (10, 100, 10, 2, 3)