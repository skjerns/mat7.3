# -*- coding: utf-8 -*-
"""
savemat — write Python/NumPy data to a MATLAB 7.3 compatible HDF5 .mat file.

Only a subset of types is currently supported; see SAVEMAT_PROGRESS.md for
the implementation status.
"""
import datetime
import h5py
import numpy as np

# Map numpy dtype → MATLAB class attribute string
_DTYPE_TO_MCLASS = {
    np.dtype('float64'):    'double',
    np.dtype('float32'):    'single',
    np.dtype('int8'):       'int8',
    np.dtype('int16'):      'int16',
    np.dtype('int32'):      'int32',
    np.dtype('int64'):      'int64',
    np.dtype('uint8'):      'uint8',
    np.dtype('uint16'):     'uint16',
    np.dtype('uint32'):     'uint32',
    np.dtype('uint64'):     'uint64',
    np.dtype('bool'):       'logical',
    np.dtype('complex64'):  'single',
    np.dtype('complex128'): 'double',
}


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def _to_storage(arr):
    """
    Return arr reshaped/transposed for HDF5 storage so that loadmat's .T
    restores the original shape.

    Rules (matching MATLAB's column-major convention):
      - 0-dim scalar → (1, 1)
      - 1-D (n,)     → (n, 1)   so .T → (1, n) → squeeze → (n,)
      - N-D          → arr.T    so .T → original shape
    """
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)   # (n, 1)
    return arr.T


# ---------------------------------------------------------------------------
# Type writers
# ---------------------------------------------------------------------------

def _write_none(group, name):
    """None → MATLAB empty double."""
    ds = group.create_dataset(name, data=np.array([0, 0], dtype=np.uint64))
    ds.attrs['MATLAB_class'] = np.bytes_('double')
    ds.attrs['MATLAB_empty'] = np.uint8(1)


def _write_logical(group, name, value):
    """bool / np.bool_ → MATLAB logical scalar (stored as uint8 (1,1))."""
    arr = np.array([[int(bool(value))]], dtype=np.uint8)
    ds = group.create_dataset(name, data=arr)
    ds.attrs['MATLAB_class'] = np.bytes_('logical')
    ds.attrs['MATLAB_int_decode'] = np.int32(1)


def _write_double(group, name, value):
    """Python int / float → MATLAB double scalar."""
    arr = np.array([[value]], dtype=np.float64)
    ds = group.create_dataset(name, data=arr)
    ds.attrs['MATLAB_class'] = np.bytes_('double')


def _write_str(group, name, value):
    """
    str → MATLAB char array.

    Stored as uint16 column vector (n, 1) with MATLAB_class='char' and
    MATLAB_int_decode=2.  In MATLAB's transposed convention this is a (1, n)
    row-vector char — the normal representation of a string.
    """
    if len(value) == 0:
        ds = group.create_dataset(name, data=np.array([1, 0], dtype=np.uint64))
        ds.attrs['MATLAB_class'] = np.bytes_('char')
        ds.attrs['MATLAB_empty'] = np.uint8(1)
    else:
        codes = np.array([[ord(c)] for c in value], dtype=np.uint16)  # shape (n, 1)
        ds = group.create_dataset(name, data=codes)
        ds.attrs['MATLAB_class'] = np.bytes_('char')
        ds.attrs['MATLAB_int_decode'] = np.int32(2)


def _write_ndarray(group, name, arr, root):
    """numpy ndarray → MATLAB numeric / logical dataset."""
    if arr.size == 0:
        ds = group.create_dataset(name, data=np.array([0, 0], dtype=np.uint64))
        ds.attrs['MATLAB_class'] = np.bytes_('double')
        ds.attrs['MATLAB_empty'] = np.uint8(1)
        return

    dtype = arr.dtype
    mclass = _DTYPE_TO_MCLASS.get(dtype)
    if mclass is None:
        raise TypeError(f'savemat: unsupported numpy dtype {dtype} for {name!r}')

    stored = _to_storage(arr)

    if np.issubdtype(dtype, np.complexfloating):
        # h5py writes complex128 as plain 'complex128', but mat73 expects a
        # compound dtype with fields named 'real' and 'imag'.  Build it explicitly.
        float_dtype = np.float32 if dtype == np.complex64 else np.float64
        compound = np.dtype([('real', float_dtype), ('imag', float_dtype)])
        data = np.zeros(stored.shape, dtype=compound)
        data['real'] = stored.real
        data['imag'] = stored.imag
        ds = group.create_dataset(name, data=data)
    elif dtype == np.dtype('bool'):
        # MATLAB logical: stored as uint8, class='logical'
        ds = group.create_dataset(name, data=stored.astype(np.uint8))
        ds.attrs['MATLAB_class'] = np.bytes_(mclass)
        ds.attrs['MATLAB_int_decode'] = np.int32(1)
        return
    else:
        ds = group.create_dataset(name, data=stored)

    ds.attrs['MATLAB_class'] = np.bytes_(mclass)


def _write_numpy_scalar(group, name, value):
    """numpy scalar → MATLAB dataset preserving dtype."""
    dtype = value.dtype
    mclass = _DTYPE_TO_MCLASS.get(dtype)
    if mclass is None:
        raise TypeError(f'savemat: unsupported numpy dtype {dtype} for {name!r}')
    if dtype == np.dtype('bool'):
        data = np.array([[int(bool(value))]], dtype=np.uint8)
        ds = group.create_dataset(name, data=data)
        ds.attrs['MATLAB_class'] = np.bytes_(mclass)
        ds.attrs['MATLAB_int_decode'] = np.int32(1)
        return
    elif np.issubdtype(dtype, np.complexfloating):
        float_dtype = np.float32 if dtype == np.complex64 else np.float64
        compound = np.dtype([('real', float_dtype), ('imag', float_dtype)])
        data = np.zeros((1, 1), dtype=compound)
        data['real'] = value.real
        data['imag'] = value.imag
    else:
        data = np.array([[value]], dtype=dtype)
    ds = group.create_dataset(name, data=data)
    ds.attrs['MATLAB_class'] = np.bytes_(mclass)


def _write_list(group, name, lst, root):
    """Python list → numeric array (if homogeneous numbers) or cell array."""
    if len(lst) == 0:
        ds = group.create_dataset(name, data=np.array([0, 0], dtype=np.uint64))
        ds.attrs['MATLAB_class'] = np.bytes_('double')
        ds.attrs['MATLAB_empty'] = np.uint8(1)
        return

    # If every leaf element is a plain number, convert to numpy array
    if _all_numeric(lst):
        try:
            arr = np.asarray(lst, dtype=np.float64)
            _write_ndarray(group, name, arr, root)
            return
        except (ValueError, TypeError):
            pass

    _write_cell(group, name, lst, root)


def _all_numeric(obj):
    """Return True if obj (or every leaf of a nested list) is a number."""
    if isinstance(obj, list):
        return all(_all_numeric(x) for x in obj)
    return isinstance(obj, (int, float, np.number, np.bool_)) and not isinstance(obj, bool)


def _write_cell(group, name, lst, root):
    """
    Write a Python list as a MATLAB 1-D cell array.

    Cell arrays are stored as (n, 1) HDF5 datasets of object references.
    Each reference points to a dataset in the root '#refs#' group.
    mat73's loadmat iterates rows → columns, transposes, then unwraps if length 1.
    """
    refs_group = root.require_group('#refs#')
    refs = []
    for item in lst:
        ref_name = str(len(refs_group))
        _write_variable(refs_group, ref_name, item, root)
        refs.append(refs_group[ref_name].ref)

    ref_arr = np.array(refs, dtype=h5py.ref_dtype).reshape(len(refs), 1)
    ds = group.create_dataset(name, data=ref_arr)
    ds.attrs['MATLAB_class'] = np.bytes_('cell')


def _write_dict(group, name, d, root):
    """Python dict → MATLAB struct (HDF5 Group with MATLAB_class='struct')."""
    subgroup = group.create_group(name)
    subgroup.attrs['MATLAB_class'] = np.bytes_('struct')
    for key, value in d.items():
        _write_variable(subgroup, key, value, root)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def _write_variable(group, name, value, root):
    """Dispatch a single variable to the appropriate writer."""
    if value is None:
        _write_none(group, name)
    elif isinstance(value, (bool, np.bool_)):
        # Must be before int: bool is a subclass of int in Python
        _write_logical(group, name, value)
    elif isinstance(value, str):
        _write_str(group, name, value)
    elif isinstance(value, (int, float)):
        _write_double(group, name, value)
    elif isinstance(value, np.ndarray):
        _write_ndarray(group, name, value, root)
    elif isinstance(value, np.generic):
        # numpy scalar (not ndarray) — preserves dtype
        _write_numpy_scalar(group, name, value)
    elif isinstance(value, list):
        _write_list(group, name, value, root)
    elif isinstance(value, dict):
        _write_dict(group, name, value, root)
    else:
        raise NotImplementedError(
            f'savemat: type {type(value).__name__!r} not yet supported '
            f'(variable {name!r}). See SAVEMAT_PROGRESS.md for roadmap.'
        )


# ---------------------------------------------------------------------------
# MATLAB userblock header
# ---------------------------------------------------------------------------

def _build_matlab_userblock():
    """
    Build the 512-byte userblock that MATLAB prepends to v7.3 .mat files.

    Layout (verified against real MATLAB R2024b output):
      0–115  : descriptive text, space-padded to exactly 116 bytes
      116–123: 8 null bytes
      124–125: version bytes  0x00 0x02
      126–127: endian indicator  b'IM'  (Intel / little-endian)
      128–511: null bytes
    The actual HDF5 data begins at byte 512 (= userblock_size).
    """
    now = datetime.datetime.now()
    date_str = now.strftime('%a %b %d %H:%M:%S %Y')
    text = (f'MATLAB 7.3 MAT-file, Platform: GLNXA64, '
            f'Created on: {date_str} HDF5 schema 1.00 .')

    header = bytearray(512)
    # Text field: exactly 116 bytes, space-padded / truncated
    text_bytes = text.encode('ascii')
    text_field = (text_bytes + b' ' * 116)[:116]
    header[:116] = text_field
    # Version
    header[124] = 0x00
    header[125] = 0x02
    # Endian indicator
    header[126] = 0x49  # 'I'
    header[127] = 0x4D  # 'M'
    return bytes(header)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def savemat(filename, data):
    """
    Save a dictionary to a MATLAB 7.3 compatible HDF5 .mat file.

    The output is readable both by MATLAB (via the 512-byte userblock header)
    and by mat73.loadmat (via h5py, which skips the userblock automatically).

    Currently supported types: None, bool, int, float, str, numpy scalars,
    numpy ndarrays (all numeric dtypes), Python lists of numbers, Python dicts.

    :param filename: path to write (should end in .mat)
    :param data: dict mapping variable names to Python / NumPy values
    """
    if not isinstance(data, dict):
        raise TypeError('savemat: data must be a dict')
    # userblock_size=512 makes h5py reserve 512 bytes before the HDF5 data,
    # which we then overwrite with the MATLAB header.
    with h5py.File(filename, 'w', userblock_size=512) as f:
        for name, value in data.items():
            _write_variable(f, name, value, f)
    # Overwrite the reserved userblock with the MATLAB header.
    with open(filename, 'r+b') as f:
        f.write(_build_matlab_userblock())
