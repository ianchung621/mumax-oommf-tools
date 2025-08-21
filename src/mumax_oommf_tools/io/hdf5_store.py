import os
import re
import glob
from typing import Literal

import h5py
import numpy as np
from tqdm import tqdm

from .ovf2_reader import read_ovf2
from ..configs import DETECT_OVF_EXTS, HEADER_TIME_KEY_CANDIDATES, \
    DEFAULT_H5NAME, TIME_GROUPKEY, MAGNETIZATION_GROUPKEY, \
    HEADER_DTYPES

def _collect_ovf_files(fdn: str) -> list[str]:
    """Collect all OVF/OMF files under a simulation folder named fdn."""
    if not os.path.isdir(fdn):
        raise FileNotFoundError(f"Folder not found: {fdn}")

    patterns = [os.path.join(fdn, f"*{ext}") for ext in DETECT_OVF_EXTS]

    files: list[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat))

    files = sorted(files)

    if not files:
        raise FileNotFoundError(f"No OVF/OMF files found under: {fdn}")

    return files

def _extract_time_from_header(hdr: dict) -> float | None:
    """Try common header keys for simulation time (in seconds)."""
    for k in HEADER_TIME_KEY_CANDIDATES:
        if k in hdr:
            try:
                value: str = hdr[k]
                value = value.removesuffix("s").strip()
                return float(hdr[k])
            except Exception:
                pass

    return None

def _fallback_time_from_filename(path: str) -> float | None:
    """If header lacks time, try to guess a sortable key from filename digits."""
    base = os.path.basename(path)
    num_pattern = re.compile(r"(\d+(?:\.\d+)?)") # e.g. 123, 45.67, 0009
    # Very weak heuristic: first number in filename -> sort key
    m = num_pattern.search(base)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None

def build_h5_from_ovfs(
    fdn: str,
    h5_name: str = DEFAULT_H5NAME,
    overwrite: bool = False,
    compression: Literal["gzip", "lzf"] | None = None,
    show_progress: bool = False,
) -> None:
    """
    Convert a folder of OVF/OMF files into a single HDF5 container.

    Parameters
    ----------
    fdn : str
        Folder containing the OVF/OMF output files.
        Supports both Mumax3 (*.ovf) and OOMMF (.ovf/.omf) ovf2 layouts.
    h5_name : str, default=DEFAULT_H5NAME
        Name of the HDF5 file to create inside `fdn`.
    overwrite : bool, default=False
        If True, remove existing file at the target path before writing.
    compression : {"gzip", "lzf", None}, default=None
        HDF5 compression filter for the magnetization dataset.
        - None  → no compression (fastest I/O, largest file)
        - "lzf" → very fast, lightweight compression
        - "gzip"→ smaller file size, but slower write/read
    show_progress: bool, default=False
        If True, display progress bar

    Notes
    -----
    - The resulting HDF5 will contain:
        /magnetization  (T, X, Y, Z, 3) float
        /time           (T,) float64
      plus canonical grid/unit metadata stored as file attributes.
    - Times are extracted from OVF headers where possible, or
      from filename digits as a fallback; NaN if unavailable.
    """

    fns = _collect_ovf_files(fdn)

    # Read the first frame to fix grid & dtype
    first_path = fns[0]
    first_hdr, first_mag = read_ovf2(first_path)

    first_t = _extract_time_from_header(first_hdr)
    if first_t is None:
        first_t = _fallback_time_from_filename(first_path)
    if first_t is None:
        first_t = np.nan

    X, Y, Z, _ = first_mag.shape
    T = len(fns)
    dtype = first_mag.dtype

    # Prepare HDF5
    h5_path = os.path.join(fdn, h5_name)
    if os.path.exists(h5_path) and overwrite:
        os.remove(h5_path)

    with h5py.File(h5_path, "w") as f:
        # Chunk per-frame for efficient time slicing; keep last dim uncompressed
        chunks = (1, X, Y, Z, 3)
        dset = f.create_dataset(
            MAGNETIZATION_GROUPKEY,
            shape=(T, X, Y, Z, 3),
            dtype=dtype,
            chunks=chunks,
            compression=compression,
            shuffle=(compression == "gzip"),  # shuffle helps gzip, skip for lzf
        )
        tset = f.create_dataset(TIME_GROUPKEY, shape=(T,), dtype=np.float64)

        # Write frame 0
        dset[0] = first_mag
        tset[0] = first_t

        # Attach key grid metadata as attrs
        for key in HEADER_DTYPES.keys() & first_hdr.keys():
            f.attrs[key] = first_hdr[key]

        # Remaining frames
        iterable = fns[1:]
        if show_progress:
            iterable = tqdm(iterable, desc=f"Parsing OVF files under {fdn}", unit='file')

        for ti, path in enumerate(iterable, start=1):

            hdr, mag = read_ovf2(path)

            for key in HEADER_DTYPES.keys() & first_hdr.keys():
                if key in hdr and hdr[key] != first_hdr[key]:
                    raise ValueError(
                        f"Inconsistent metadata for '{key}': "
                        f"{hdr[key]!r} in {path} vs {first_hdr[key]!r} in {first_path}"
                    )
            
            t = _extract_time_from_header(first_hdr)
            if t is None:
                t = _fallback_time_from_filename(first_path)
            if t is None:
                t = np.nan

            dset[ti] = mag
            tset[ti] = t