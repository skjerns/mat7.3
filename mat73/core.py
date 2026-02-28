# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 12:20:29 2019

Load MATLAB 7.3 files into Python

@author: Simon
"""
import os
import numpy as np
import h5py
from datetime import datetime, timedelta
import logging
from typing import Iterable
from mat73.version import __version__

logger = logging.getLogger('mat73')

def empty(*dims):
    if len(dims)==1:
        return [[] for x in range(dims[0])]
    else:
        return [empty(*dims[1:]) for _ in range(dims[0])]


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        """this prevents that built-in functions are overwritten"""
        if key=='__getstate__':
            raise AttributeError(key)
        if key in dir(dict):
            return dict.__getattr__(self, key)
        else:
            return self.__getitem__(key)

    def __setattr__(self, key, value):
        return self.__setitem__(key, value)

def print_tree(node):
    if node.startswith('#refs#/') or node.startswith('#subsystem#/'):
        return
    print(' ', node)


class HDF5Decoder():
    def __init__(self, verbose=True, use_attrdict=False,
                 only_include=None):

        # if only_include is a string, convert into a list
        if isinstance(only_include, str):
            only_include = [only_include]

        # make sure all paths start with '/' and are not ending in '/'
        if only_include is not None:
            only_include = [s if s[0]=='/' else f'/{s}' for s in only_include]
            only_include = [s[:-1] if s[-1]=='/' else s for s in only_include]

        self.verbose = verbose
        self._dict_class = AttrDict if use_attrdict else dict
        self.refs = {} # this is used in case of matlab matrices
        self.only_include = only_include

        # set a check if requested include_only var was actually found
        if only_include is not None:
            _vardict = dict(zip(only_include, [False]*len(only_include)))
            self._found_include_var = _vardict

    def is_included(self, hdf5):
        # if only_include is not specified, we always return True
        # because we load everything
        if self.only_include is None:
            return True
        # see if the current name is in any of the included variables
        for s in self.only_include:
            if s in hdf5.name:
                self._found_include_var[s] = True
            if s in hdf5.name or hdf5.name in s:
                return True
        return False

    def mat2dict(self, hdf5):

        if '#refs#' in hdf5:
            self.refs = hdf5['#refs#']
        d = self._dict_class()
        for var in hdf5:
            # this first loop is just here to catch the refs and subsystem vars
            if var in ['#refs#','#subsystem#']:
                continue

            if not self.is_included(hdf5[var]):
                continue
            d[var] = self.unpack_mat(hdf5[var])

        if self.only_include is not None:
            for var, found in self._found_include_var.items():
                if not found:
                    logger.warning(f'Variable "{var}" was specified to be loaded'\
                                  ' but could not be found.')
            if not any(list(self._found_include_var.values())):
                print(hdf5.filename, 'contains the following vars:')
                hdf5.visit(print_tree)
        return d

    # @profile
    def unpack_mat(self, hdf5, depth=0, MATLAB_class=None, force=False):
        """
        unpack a h5py entry: if it's a group expand,
        if it's a dataset convert

        for safety reasons, the depth cannot be larger than 99
        """
        if depth==99:
            raise RecursionError("Maximum number of 99 recursions reached.")

        # sparse elements need to be loaded separately (recursion end)
        # see https://github.com/skjerns/mat7.3/issues/28
        if 'MATLAB_sparse' in hdf5.attrs:
            try:
                from scipy.sparse import csc_matrix
                if 'data' in hdf5:
                    data = hdf5['data']
                    row_ind = hdf5['ir']
                else:
                    data = []
                    row_ind = []

                col_ind = hdf5['jc']
                n_rows = hdf5.attrs['MATLAB_sparse']
                n_cols = len(col_ind) - 1
                return csc_matrix((data, row_ind, col_ind), shape=(n_rows, n_cols))
            except ModuleNotFoundError:
                logger.error(f'`scipy` not installed. To load the sparse matrix'
                                f' `{hdf5.name}`,'
                                ' you need to have scipy installed. Please install'
                                ' via `pip install scipy`')
            except DeprecationWarning:
                logger.error(f'Tried loading the sparse matrix `{hdf5.name}`'
                                ' with scipy, but'
                                ' the interface has been deprecated. Please'
                                ' raise this error as an issue on GitHub:'
                                ' https://github.com/skjerns/mat7.3/issues')
            except KeyError as e:
                logger.error(f'Tried loading the sparse matrix `{hdf5.name}`'
                                ' but something went wrong:\n{e}', exc_info=True)
                raise e

        if isinstance(hdf5, (h5py._hl.group.Group)):
            d = self._dict_class()

            for key in hdf5:
                elem   = hdf5[key]
                if not self.is_included(elem) and not force:
                    continue
                if 'MATLAB_class' in elem.attrs:
                    MATLAB_class = elem.attrs.get('MATLAB_class')
                    if MATLAB_class is not None:
                        MATLAB_class = MATLAB_class.decode()
                unpacked = self.unpack_mat(elem, depth=depth+1,
                                           MATLAB_class=MATLAB_class)


                if MATLAB_class=='struct' and len(elem)>1 and \
                isinstance(unpacked, dict):
                    values = unpacked.values()
                    # we can only pack them together in MATLAB style if
                    # all subitems are the same lengths.
                    # MATLAB is a bit confusing here, and I hope this is
                    # correct. see https://github.com/skjerns/mat7.3/issues/6
                    allist = all([isinstance(item, list) for item in values])
                    if allist:
                        same_len = len(set([len(item) for item in values]))==1
                    else:
                        same_len = False

                    # convert struct to its proper form as in MATLAB
                    # i.e. struct[0]['key'] will access the elements
                    # we only recreate the MATLAB style struct
                    # if all the subelements have the same length
                    # and are of type list
                    if allist and same_len:
                        items = list(zip(*[v for v in values]))

                        keys = unpacked.keys()
                        struct = [{k:v for v,k in zip(row, keys)} for row in items]
                        struct = [self._dict_class(d) for d in struct]
                        unpacked = struct
                d[key] = unpacked

            return d
        elif isinstance(hdf5, h5py._hl.dataset.Dataset):
            if self.is_included(hdf5) or force:
                return self.convert_mat(hdf5, depth, MATLAB_class=MATLAB_class)
        else:
            raise Exception(f'Unknown hdf5 type: {key}:{type(hdf5)}')

    # @profile
    def _has_refs(self, dataset):
        if len(dataset.shape)<2: return False
        # dataset[0].
        dataset[0][0]
        if isinstance(dataset[0][0], h5py.h5r.Reference):
            return True
        return False

    # @profile
    def convert_mat(self, dataset, depth, MATLAB_class=None):
        """
        Converts h5py.dataset into python native datatypes
        according to the matlab class annotation
        """
        # all MATLAB variables have the attribute MATLAB_class
        # if this is not present, it is not convertible
        if MATLAB_class is None and 'MATLAB_class' in dataset.attrs:
            MATLAB_class = dataset.attrs['MATLAB_class'].decode()


        if not MATLAB_class and not self._has_refs(dataset):
            if self.verbose:
                message = 'ERROR: not a MATLAB datatype: ' + \
                          '{}, ({})'.format(dataset, dataset.dtype)
                logger.error(message)
            return None

        known_cls = ['cell', 'char', 'bool', 'logical', 'double', 'single',
                     'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16',
                     'uint32', 'uint64']

        if 'MATLAB_empty' in dataset.attrs.keys():
            mtype = 'empty'
        elif MATLAB_class in known_cls:
                mtype = MATLAB_class
        elif self._has_refs(dataset):
            mtype='cell'
        else:
            mtype = MATLAB_class

        if mtype=='cell':
            cell = []
            for ref in dataset:
                row = []
                # some weird style MATLAB have no refs, but direct floats or int
                if isinstance(ref, Iterable):
                    for r in ref:
                        # force=True because we want to load cell contents
                        entry = self.unpack_mat(self.refs.get(r), depth+1, force=True)
                        row.append(entry)
                else:
                    row = [ref]
                cell.append(row)
            cell = list(map(list, zip(*cell))) # transpose cell
            if len(cell)==1: # singular cells are interpreted as int/float
                cell = cell[0]
            return cell

        elif mtype=='empty':
            if MATLAB_class in ['cell', 'struct']:
                dims = [x for x in dataset]
                return empty(*dims)
            elif MATLAB_class=='char':
                return ''
            else:

                return None

        elif mtype=='char':
            codes = np.asarray(dataset, dtype=np.uint16)

            # object dtype → keeps '\x00'
            # see https://github.com/numpy/numpy/issues/28964
            to_char = np.vectorize(chr, otypes=[object])
            arr = to_char(codes)

            char_axis = 0 if arr.ndim < 3 else -2
            char_arr = np.apply_along_axis(lambda x: ''.join(x), axis=char_axis, arr=arr)

            string_list = char_arr.tolist()

            if arr.ndim==2 and arr.shape[1]==1:
                string_list = string_list[0]

            if arr.ndim>2:
                # print warning to be sure. I haven't encountered any char
                # arrays with ndim>2 in the wild yet so can't be sure that
                # they are actually the way I synthesized them
                logging.warning(f"Loading char array '{dataset.name}' with {arr.ndim} dimensions "
                                f"might be wrong stacked (i.e. dimensions scrambled). "
                                f"please check variable is correct and report errors "
                                f"on github.com/skjerns/mat7.3")

            return string_list

        elif mtype=='bool':
            return bool(dataset)

        elif mtype=='logical':
            arr = squeeze(np.array(dataset, dtype=bool).T)
            if arr.size==1: arr=bool(arr)
            return arr

        elif MATLAB_class == 'datetime':
            # MATLAB datetime objects can be encoded in different ways:
            # 1. With MATLAB_object_decode=3: MCOS objects (object-oriented encoding)
            # 2. Without it: Simple numeric date arrays

            if 'MATLAB_object_decode' in dataset.attrs and dataset.attrs['MATLAB_object_decode'] == 3:
                # This is an MCOS-encoded datetime object
                # The dataset contains reference metadata, not the actual datetime values
                # We need to navigate to the MCOS subsystem to get the actual timestamps

                # Get the file handle from the dataset
                file_handle = dataset.file

                # Check if MCOS subsystem exists
                if '#subsystem#' in file_handle and 'MCOS' in file_handle['#subsystem#']:
                    mcos_refs = file_handle['#subsystem#/MCOS'][0]

                    # The 5th element (index 4) in the dataset appears to be the MCOS reference index
                    # Extract reference indices from the dataset
                    raw_data = np.array(dataset)

                    # Handle different dataset shapes
                    if raw_data.ndim == 2:
                        # Shape is (n, 6) where n is the number of datetime values
                        # The 5th column (index 4) contains a 0-based index into datetime objects
                        # The actual datetime data starts at MCOS[2] (first two are metadata)
                        # So the MCOS index is field[4] + 2
                        mcos_indices = raw_data[:, 4] + 1
                    else:
                        # Fallback for unexpected shapes
                        logger.warning(f"Unexpected shape for MCOS datetime: {raw_data.shape}")
                        return None

                    # Decode each datetime value from MCOS
                    # Each index can point to either a scalar or array of datetime values
                    all_datetimes = []
                    for idx in mcos_indices:
                        try:
                            idx = int(idx)
                            if 0 <= idx < len(mcos_refs):
                                mcos_ref = mcos_refs[idx]
                                mcos_obj = file_handle[mcos_ref]

                                if isinstance(mcos_obj, h5py.Dataset) and mcos_obj.dtype == np.float64:
                                    # The MCOS object contains Unix timestamp(s) in milliseconds
                                    # It can be a scalar or an array of values
                                    timestamps_ms = np.array(mcos_obj)

                                    # Preserve the original shape
                                    original_shape = timestamps_ms.shape

                                    # Flatten to 1D array for processing
                                    flat_timestamps = timestamps_ms.flatten()

                                    # Convert each timestamp
                                    datetime_values = []
                                    for ts_ms in flat_timestamps:
                                        if np.isnan(ts_ms):
                                            datetime_values.append(None)  # NaT (Not-a-Time)
                                        else:
                                            # Convert milliseconds to seconds
                                            timestamp_sec = ts_ms / 1000.0
                                            # Convert to datetime
                                            dt = datetime(1970, 1, 1) + timedelta(seconds=timestamp_sec)
                                            datetime_values.append(dt)

                                    # Reshape back to original shape and apply squeeze
                                    if len(datetime_values) == 1:
                                        all_datetimes.append(datetime_values[0])
                                    else:
                                        result = np.array(datetime_values, dtype=object).reshape(original_shape)
                                        # Apply MATLAB-style transpose and squeeze
                                        result = result.T
                                        all_datetimes.append(squeeze(result))
                                else:
                                    logger.warning(f"Unexpected MCOS object type for datetime at index {idx}")
                                    all_datetimes.append(None)
                            else:
                                logger.warning(f"MCOS index {idx} out of range")
                                all_datetimes.append(None)
                        except Exception as e:
                            logger.warning(f"Error decoding MCOS datetime at index {idx}: {e}")
                            all_datetimes.append(None)

                    # Return the result appropriately
                    if len(all_datetimes) == 1:
                        # Single datetime (scalar or array)
                        return all_datetimes[0]
                    else:
                        # Multiple datetime values/arrays
                        return np.array(all_datetimes, dtype=object)
                else:
                    logger.warning("MCOS subsystem not found for MATLAB_object_decode=3 datetime")
                    return None
            else:
                # Legacy format: numeric date arrays (datenums)
                datenums_raw = np.array(dataset, dtype=np.float64)
                datenums_transposed = datenums_raw.T

                original_shape = datenums_transposed.shape
                flat_datenums = datenums_transposed.flatten()
                py_datetimes = []

                # MATLAB uses days since January 0, 0000, Python uses days since January 1, 0001
                MATLAB_TO_PYTHON_OFFSET = 366

                for mdn in flat_datenums:
                    if np.isnan(mdn):
                        py_datetimes.append(None)  # Represent MATLAB NaT (Not-a-Time) as None
                        continue

                    # Convert MATLAB datenum to Python ordinal
                    python_ordinal = mdn - MATLAB_TO_PYTHON_OFFSET
                    dt_ordinal_day = int(np.floor(python_ordinal))
                    time_fraction = python_ordinal - np.floor(python_ordinal)

                    if dt_ordinal_day < 1:
                        logging.warning(
                            f"MATLAB datetime value {mdn} is before 0001-01-01. Storing as None."
                        )
                        py_datetimes.append(None)
                        continue

                    try:
                        current_dt = datetime.fromordinal(dt_ordinal_day) + timedelta(days=time_fraction)
                        py_datetimes.append(current_dt)
                    except (ValueError, OverflowError) as e:
                        logging.warning(
                            f"Could not convert MATLAB datetime value {mdn}: {e}. Storing as None."
                        )
                        py_datetimes.append(None)

                result_array = np.array(py_datetimes, dtype=object).reshape(original_shape)
                return squeeze(result_array)

        elif mtype=='canonical empty':
            return None

        # complex numbers need to be filtered out separately
        elif 'imag' in str(dataset.dtype):
            if dataset.attrs['MATLAB_class']==b'single':
                dtype = np.complex64
            else:
                dtype = np.complex128
            arr = np.array(dataset)
            arr = (arr['real'] + arr['imag']*1j).astype(dtype)
            return squeeze(arr.T)

        # if it is none of the above, we can convert to numpy array
        elif mtype in ('double', 'single', 'int8', 'int16', 'int32', 'int64',
                       'uint8', 'uint16', 'uint32', 'uint64'):
            arr = np.array(dataset, dtype=dataset.dtype)
            return squeeze(arr.T)
        elif mtype=='missing':
            arr = None
        else:
            if self.verbose:
                message = 'ERROR: MATLAB type not supported: ' + \
                          '{}, ({})'.format(mtype, dataset.dtype)
                logger.error(message)
            return None


def squeeze(arr):
    """Vectors are saved as 2D matrices in MATLAB, however in numpy
    there is no distinction between a column and a row vector.
    Therefore, remove superfluous dimensions if the array is 2D and
    one of the dimensions is singular"""
    if arr.ndim==2 and 1 in arr.shape:
        return arr.reshape([x for x in arr.shape if x>1])
    return arr


def loadmat(file, use_attrdict=False, only_include=None, verbose=True):
    """
    Loads a MATLAB 7.3 .mat file, which is actually a
    HDF5 file with some custom matlab annotations inside

    :param file: filename or file-like object to load from
    :param use_attrdict: make it possible to access structs like in MATLAB
                         using struct.varname instead of struct['varname']
                         WARNING: builtin dict functions cannot be overwritten,
                         e.g. keys(), pop(), ...
                         these will still be available by struct['keys']
    :param verbose: print warnings
    :param only_include: A list of HDF5 paths that should be loaded.
                         this can greatly reduce loading times. If a path
                         contains further sub datasets, these will be loaded
                         as well, e.g. 'struct/' will load all subvars of
                         struct, 'struct/var' will load only ['struct']['var']
    :returns: A dictionary with the matlab variables loaded
    """
    decoder = HDF5Decoder(verbose=verbose, use_attrdict=use_attrdict,
                          only_include=only_include)

    if isinstance(file, str):
        ext = os.path.splitext(file)[1].lower()
        if ext!='.mat':
            logger.warning('Can only load MATLAB .mat file, this file type might '
                            f'be unsupported: {file}')

    try:
        with h5py.File(file, 'r') as hdf5:
            dictionary = decoder.mat2dict(hdf5)
        return dictionary
    except FileNotFoundError:
        raise
    except OSError:
        raise TypeError(f'{file} is not a MATLAB 7.3 file. '
                        'Load with scipy.io.loadmat() instead.')


def savemat(filename, verbose=True):
    raise NotImplementedError


if __name__=='__main__':
    # for testing / debugging
    d = loadmat('../tests/testfile16.mat')
    print(d)
