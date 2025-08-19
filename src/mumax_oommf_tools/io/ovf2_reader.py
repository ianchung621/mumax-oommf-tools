import numpy as np

from ..configs import OVF2_FIRST_LINE, \
    HEADER_READ_BYTES, HEADER_DTYPES, HEADER_BEGIN_MARKER, HEADER_END_MARKER, \
    DATA_BEGIN_MARKER, DATA_END_MARKER, \
    BINARY4_FLAG, BINARY8_FLAG

def extract_metadata(content: bytes) -> dict[str, int|float|str]:
    start = content.find(HEADER_BEGIN_MARKER)
    end = content.find(HEADER_END_MARKER)

    if start == -1 or end == -1:
        raise ValueError("Header markers not found.")

    header_bytes = content[start + len(HEADER_BEGIN_MARKER):end]
    header_lines = header_bytes.decode().splitlines()

    metadata = {}
    for line in header_lines:
        if ":" not in line:
            continue
        key, value = line.rsplit(":",1)
        key = key.strip("# ")
        value = value.strip()
        dtype = HEADER_DTYPES.get(key, str)

        if dtype is int:
            metadata[key] = int(value)
        elif dtype is float:
            metadata[key] = float(value)
        else:  # str
            metadata[key] = value
    
    return metadata

def reorder_xyz(m_flat: np.ndarray, X: int, Y: int, Z: int) -> np.ndarray:
    """
    OVF increments x fastest, then y, then z.

    see https://math.nist.gov/oommf/doc/userguide20b0/userguide/Data_block.html

    m_flat: (N,3) with N = X*Y*Z
    -> return (X, Y, Z, 3)
    """
    return np.transpose(m_flat.reshape(Z, Y, X, 3), (2, 1, 0, 3))

def extract_magnetic_data_from_text(content: bytes) -> np.ndarray:
    
    start = content.find(DATA_BEGIN_MARKER)
    end   = content.find(DATA_END_MARKER)
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Data block not found.")

    payload_start = content.find(b"\n", start) + 1

    payload = content[payload_start:end]

    m_flat = np.fromstring(payload.decode(), sep=" ", dtype=np.float32)
    
    if m_flat.size % 3 != 0:
        raise ValueError("Data size not divisible by 3 (valuedim must be 3).")

    return m_flat

def read_ovf2(fn: str) -> tuple[dict[str, int|float|str], np.ndarray]:
    """
    Read an OVF 2.0 file produced by Mumax3 or OOMMF.

    Parameters
    ----------
    fn : str
        Path to the OVF 2.0 file.

    Returns
    -------
    metadata, magnetization

    metadata : dict
        Dictionary of parsed header fields. Keys include 'xnodes', 'ynodes',
        'znodes', 'xstepsize', 'ystepsize', 'zstepsize', 'meshunit', etc.
        Values are converted to the proper Python type (int, float, str).
    magnetization : np.ndarray
        Magnetization field with shape (X, Y, Z, 3), where:
          - X = xnodes
          - Y = ynodes
          - Z = znodes
          - 3 = vector components (mx, my, mz)

    Notes
    -----
    - Only OVF 2.0 rectangular mesh files with valuedim=3 are supported.
    - Data mode may be Text, Binary 4, or Binary 8
    - For Binary 4 or Binary 8, memmap (a subclass of np.ndarray) is returned (efficient)
    - For Text, np.array is returned but require full file reading (not efficient)
    """

    with open(fn, "rb") as f:
        head = f.read(HEADER_READ_BYTES)

    if not head.startswith(OVF2_FIRST_LINE):
        raise ValueError("Not a valid OVF 2.0 file (missing OVF2 header).")

    metadata = extract_metadata(head)

    if metadata.get("meshtype", "").lower() != "rectangular":
        raise ValueError(f"Unsupported mesh type: {metadata.get('meshtype')}, expected 'rectangular'")
    if metadata.get("valuedim") != 3:
        raise ValueError(f"Unsupported valuedim: {metadata.get('valuedim')}, expected 3")
    
    X, Y, Z = metadata["xnodes"], metadata["ynodes"], metadata["znodes"]
    N = X * Y * Z

    data_marker_start = head.find(DATA_BEGIN_MARKER)

    data_marker_end = head.find(b"\n", data_marker_start) + 1
    data_marker_line = head[data_marker_start:data_marker_end]
    mode = data_marker_line.removeprefix(DATA_BEGIN_MARKER).strip() # e.g. "Text", "Binary 4", "Binary 8"

    payload_start = data_marker_end

    # for Binary 4 and Binary 8, return view from memmap, efficient
    if mode == b"Binary 4":
        if head[payload_start:payload_start+4] != BINARY4_FLAG:
            raise ValueError("Binary4 flag mismatch (expected 1234567.0 float32)")
        offset = payload_start + 4
        dtype = "<f4"
    elif mode == b"Binary 8":
        if head[payload_start:payload_start+8] != BINARY8_FLAG:
            raise ValueError("Binary8 flag mismatch (expected 123456789012345.0 float64)")
        offset = payload_start + 8
        dtype = "<f8"

    # for Text, require full file read, not efficient
    elif mode == b"Text":
        with open(fn, 'rb') as f:
            full_content = f.read()
        m_flat = extract_magnetic_data_from_text(full_content)
        if len(m_flat) != N:
            raise ValueError(f"Nodes number mismatch: got {len(m_flat)}, expected xnodes*ynodes*znodes = {N}")
        magnetization = reorder_xyz(m_flat, X, Y, Z)
        return metadata, magnetization
    else:
        raise ValueError(f"Unsupported data mode: {mode}, supported: Text, Binary 4, Binary 8")

    mm = np.memmap(fn, mode="r", dtype=dtype, offset=offset, shape=3 * N)
    magnetization = reorder_xyz(mm, X, Y, Z)
    
    return metadata, magnetization