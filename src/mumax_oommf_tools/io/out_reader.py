import os
import psutil
from typing import Optional, Literal, Tuple

import numpy as np
import h5py

from .hdf5_store import (
    build_h5_from_ovfs,
    MAGNETIZATION_GROUPKEY,
    TIME_GROUPKEY,
    DEFAULT_H5NAME,
)

def _format_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    f = float(n)
    i = 0
    while f >= 1024 and i < len(units) - 1:
        f /= 1024.0
        i += 1
    return f"{f:.2f} {units[i]}"

def _nbytes(shape: Tuple[int, ...], dtype: np.dtype) -> int:
    n = 1
    for s in shape:
        n *= s
    return n * np.dtype(dtype).itemsize

def _read_metadata(attrs: h5py.AttributeManager):
    meta = {}
    for k, v in attrs.items():
        if isinstance(v, (np.float64, np.int64)):
            meta[k] = v.item()
        else:
            meta[k] = v
    return meta

def read_simulation_result(
    fdn: str,
    h5_name: str = DEFAULT_H5NAME,
    build_if_missing: bool = True,
    overwrite: bool = False,
    compression: Literal["gzip", "lzf"] | None = None,
    show_progress: bool = False
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Reads simulation result folder that contains OVF/OMF files

    Parameters
    ----------
    fdn : str
        Path to the simulation folder containing OVF/OMF files
    h5_name : str, default=DEFAULT_H5NAME
        Name of the HDF5 file inside `fdn`.
    build_if_missing : bool, default=True
        If True, build the HDF5 file from OVF/OMF files when not present.
        If False, raise FileNotFoundError if the HDF5 file is missing.
    overwrite : bool, default=False
        If True and the target HDF5 file exists, it will be removed and rebuilt.
    compression : {"gzip", "lzf", None}, default=None
        Compression filter for the HDF5 magnetization dataset, used only
        when building a new HDF5 file.
    show_progress : bool, default=False
        If True, display a tqdm progress bar when building from OVF/OMF files.

    Returns
    -------
    metadata, time, magnetization

    metadata : dict
        Dictionary of parsed header fields. Keys include 'xnodes', 'ynodes',
        'znodes', 'xstepsize', 'ystepsize', 'zstepsize', 'meshunit', etc.
        Values are converted to the proper Python type (int, float, str).
    time : np.ndarray, dtype float64
        Simulation time values of shape (T,) for each frame.
    magnetization : np.ndarray
        Full magnetization history with shape (X, Y, Z, 3), where:
          - T = time frames (number of ovfs under fdn)
          - X = xnodes
          - Y = ynodes
          - Z = znodes
          - 3 = vector components (mx, my, mz)
    """
    h5_path = os.path.join(fdn, h5_name)

    if os.path.exists(h5_path) and overwrite:
        os.remove(h5_path)

    if not os.path.exists(h5_path):
        if not build_if_missing:
            raise FileNotFoundError(f"No HDF5 at {h5_path}. Set build_if_missing=True.")
        build_h5_from_ovfs(
            fdn,
            h5_name=h5_name,
            overwrite=overwrite,
            compression=compression,
            show_progress=show_progress
        )

    # Inspect without loading the big array
    with h5py.File(h5_path, "r") as f:
        if MAGNETIZATION_GROUPKEY not in f or TIME_GROUPKEY not in f:
            raise ValueError(f"HDF5 missing required datasets '{MAGNETIZATION_GROUPKEY}' or '{TIME_GROUPKEY}'.")
        mset = f[MAGNETIZATION_GROUPKEY]
        tset = f[TIME_GROUPKEY]

        shape = tuple(mset.shape)  # (T,X,Y,Z,3)
        dtype = mset.dtype

        expected_shape = (len(tset), int(f.attrs['xnodes']), int(f.attrs['ynodes']), int(f.attrs['znodes']), 3)
        if shape != expected_shape:
            raise ValueError(f"Expected magnetization shape (T,X,Y,Z,3), got {shape}.")

        needed = _nbytes(shape, dtype)

        if psutil is not None:
            total_ram = psutil.virtual_memory().total
            if needed > total_ram:
                raise MemoryError(
                    f"Magnetization needs {_format_bytes(needed)} but system RAM is "
                    f"{_format_bytes(int(total_ram))}."
                )

        metadata = _read_metadata(f.attrs)
        time = tset[...].astype(np.float64, copy=False)
        magnetization = mset[...]

    return metadata, time, magnetization