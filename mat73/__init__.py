# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 12:20:29 2019

Load MATLAB 7.3 files into Python

@author: Simon
"""
import os
import numpy as np
import h5py
import logging
from typing import Iterable


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




class HDF5Decoder():
    def __init__(self, verbose=True, use_attrdict=False):
        self.verbose = verbose
        self._dict_class = AttrDict if use_attrdict else dict
        self.refs = {} # this is used in case of matlab matrices

    def mat2dict(self, hdf5, only_load=None):
        if '#refs#' in hdf5: 
            self.refs = hdf5['#refs#']
        d = self._dict_class()
        for var in hdf5:
            if var in ['#refs#','#subsystem#']:
                continue
            ext = os.path.splitext(hdf5.filename)[1].lower()
            if ext.lower()=='.mat':
                # if hdf5
                d[var] = self.unpack_mat(hdf5[var])
            elif ext=='.h5' or ext=='.hdf5':
                err = 'Can only load .mat. Please use package hdfdict instead'\
                      '\npip install hdfdict\n' \
                      'https://github.com/SiggiGue/hdfdict'
                raise NotImplementedError(err)
            else:
                raise ValueError('can only unpack .mat')
        return d
    
   
    def unpack_mat(self, hdf5, depth=0):
        """
        unpack a h5py entry: if it's a group expand,
        if it's a dataset convert
        
        for safety reasons, the depth cannot be larger than 99
        """
        if depth==99:
            raise RecursionError("Maximum number of 99 recursions reached.")
        if isinstance(hdf5, (h5py._hl.group.Group)):
            d = self._dict_class()

            for key in hdf5:
                matlab_class = hdf5[key].attrs.get('MATLAB_class')
                elem   = hdf5[key]
                unpacked = self.unpack_mat(elem, depth=depth+1)
                if matlab_class==b'struct' and len(elem)>1 and \
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
            return self.convert_mat(hdf5, depth)
        else:
            raise Exception(f'Unknown hdf5 type: {key}:{type(hdf5)}')


    def _has_refs(self, dataset):
        if len(dataset)==0: return False
        if not isinstance(dataset[0], np.ndarray): return False
        if isinstance(dataset[0][0], h5py.h5r.Reference):  
            return True
        return False


    def convert_mat(self, dataset, depth):
        """
        Converts h5py.dataset into python native datatypes
        according to the matlab class annotation
        """      
        # all MATLAB variables have the attribute MATLAB_class
        # if this is not present, it is not convertible
        if not 'MATLAB_class' in dataset.attrs and not self._has_refs(dataset):
            if self.verbose:
                message = 'ERROR: not a MATLAB datatype: ' + \
                          '{}, ({})'.format(dataset, dataset.dtype)
                logging.error(message)
            return None

        if self._has_refs(dataset):
            mtype='cell'
        elif 'MATLAB_empty' in dataset.attrs.keys() and \
            dataset.attrs['MATLAB_class'].decode()in ['cell', 'struct']:
            mtype = 'empty'
        else:
            mtype = dataset.attrs['MATLAB_class'].decode()


        if mtype=='cell':
            cell = []
            for ref in dataset:
                row = []
                # some weird style MATLAB have no refs, but direct floats or int
                if isinstance(ref, Iterable):
                    for r in ref:
                        entry = self.unpack_mat(self.refs.get(r), depth+1)
                        row.append(entry)
                else:
                    row = [ref]
                cell.append(row)
            cell = list(map(list, zip(*cell))) # transpose cell
            if len(cell)==1: # singular cells are interpreted as int/float
                cell = cell[0]
            return cell

        elif mtype=='empty':
            dims = [x for x in dataset]
            return empty(*dims)

        elif mtype=='char': 
            string_array = np.array(dataset).ravel()
            string_array = ''.join([chr(x) for x in string_array])
            string_array = string_array.replace('\x00', '')
            return string_array

        elif mtype=='bool':
            return bool(dataset)

        elif mtype=='logical': 
            arr = np.array(dataset, dtype=bool).T.squeeze()
            if arr.size==1: arr=bool(arr)
            return arr

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
            return arr.T.squeeze()

        # if it is none of the above, we can convert to numpy array
        elif mtype in ('double', 'single', 'int8', 'int16', 'int32', 'int64', 
                       'uint8', 'uint16', 'uint32', 'uint64'):
            arr = np.array(dataset, dtype=dataset.dtype)
            return arr.T.squeeze()
        elif mtype=='missing':
            arr = None
        else:
            if self.verbose:
                message = 'ERROR: MATLAB type not supported: ' + \
                          '{}, ({})'.format(mtype, dataset.dtype)
                logging.error(message)
            return None
        
            
def loadmat(filename, use_attrdict=False, only_load=None, verbose=True):
    """
    Loads a MATLAB 7.3 .mat file, which is actually a
    HDF5 file with some custom matlab annotations inside
    
    :param filename: A string pointing to the file
    :param use_attrdict: make it possible to access structs like in MATLAB
                         using struct.varname instead of struct['varname']
                         WARNING: builtin dict functions cannot be overwritten,
                         e.g. keys(), pop(), ...
                         these will still be available by struct['keys']
    :param verbose: print warnings
    :param only_load: A list of HDF5 paths that should be loaded
    :returns: A dictionary with the matlab variables loaded
    """
    assert os.path.isfile(filename), '{} does not exist'.format(filename)
    decoder = HDF5Decoder(verbose=verbose, use_attrdict=use_attrdict)
    try:
        with h5py.File(filename, 'r') as hdf5:
            dictionary = decoder.mat2dict(hdf5, only_load=only_load)
        return dictionary
    except OSError:
        raise TypeError('{} is not a MATLAB 7.3 file. '\
                        'Load with scipy.io.loadmat() instead.'.format(filename))
            
            
def savemat(filename, verbose=True):
    raise NotImplementedError

    
if __name__=='__main__':
    # d = loadmat('../tests/testfile5.mat', use_attrdict=True)


    file = '../tests/testfile7.mat'
    data = loadmat(file)
