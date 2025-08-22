import pytest
from pathlib import Path

from mumax_oommf_tools import read_ovf2
from mumax_oommf_tools.io.exceptions import OVF2Error

DATA_DIR = Path(__file__).parent / "data" / "bad_ovfs"

BAD_CASES = [
    ("not_ovf2.ovf",           "Invalid OVF2 header"),
    ("no_header_marker.ovf",   "Header markers not found"),
    ("no_data_marker.ovf",     "Data block not found"),
    ("bad_binary4_flag.ovf",   "Binary4 flag mismatch"),
    ("bad_binary8_flag.ovf",   "Binary8 flag mismatch"),
    ("invalid_output_format.ovf", "Unsupported data mode. Expected one of 'Text', 'Binary 4', 'Binary 8"),
    ("not_rectangular.ovf",    "Unsupported mesh type"),
    ("unclosed_datablock.ovf",     "Data block not found"),
    ("valuedim_not_3.ovf",     "Unsupported valuedim"),
    ("wrong_data_number.ovf",  "Node count mismatch"),
]

@pytest.mark.parametrize("fname, msg", BAD_CASES)
def test_bad_cases_with_message(fname, msg):
    fn = DATA_DIR / fname
    with pytest.raises(OVF2Error, match=msg):
        read_ovf2(fn)