"""
reference:
[2] OVF 2.0:
https://math.nist.gov/oommf/doc/userguide20b0/userguide/OVF_2.0_format.html
"""
import numpy as np

OVF2_FIRST_LINE = b"# OOMMF OVF 2.0\n"

HEADER_BEGIN_MARKER = b'# Begin: Header\n'
HEADER_END_MARKER = b'# End: Header\n'

HEADER_READ_BYTES = 64 * 1024 # typically, header will be under 1 kB, but we read 64 kB for safety

# valuemultiplier, boundary, ValueRangeMaxMag and ValueRangeMinMag of the OVF 1.0 format are not supported.
HEADER_DTYPES = {
    "Title": str,
    "meshtype": str,
    "meshunit": str,

    "xmin": float,
    "ymin": float,
    "zmin": float,
    "xmax": float,
    "ymax": float,
    "zmax": float,

    "valuedim": int,
    "valuelabels": str,   # space-separated labels, e.g. "m_x m_y m_z"
    "valueunits": str,    # space-separated units, e.g. "A/m A/m A/m"

    "xbase": float,
    "ybase": float,
    "zbase": float,

    "xnodes": int,
    "ynodes": int,
    "znodes": int,

    "xstepsize": float,
    "ystepsize": float,
    "zstepsize": float,
}


DATA_BEGIN_MARKER = b"# Begin: Data"
DATA_END_MARKER   = b"# End: Data"

# OVF 2.0 writes data in little endian (LSB) order, as compared to the MSB order used in the OVF 1.0 format
# these are initial check value present in data block
BINARY4_FLAG = np.float32(1234567.0).tobytes()
BINARY8_FLAG = np.float64(123456789012345.0).tobytes()