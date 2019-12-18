# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 12:20:29 2019

Load MATLAB 7.3 files into Python

@author: Simon
"""
import os
import numpy as np
import h5py

class HDF5Decoder():
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.refs = {} # this is used in case of matlab matrices

    def mat2dict(self, hdf5):
        if '#refs#' in hdf5: 
            self.refs = hdf5['#refs#']
        d = {}
        for var in hdf5:
            if var in ['#refs#','#subsystem#']:
                continue
            ext = os.path.splitext(hdf5.filename)[1]
            if ext=='.mat':
                d[var] = self.unpack_mat(hdf5[var])
            elif ext=='.h5' or ext=='.hdf5':
                err = 'Can only load .mat. Please use package hdfdict instead'\
                      '\npip install hdfdict\n' \
                      'https://github.com/SiggiGue/hdfdict'
                raise NotImplementedError(err)
            else:
                raise ValueError('can only unpack .h5, .hdf5 or .mat')
        return d
    
   
    def unpack_mat(self, hdf5, depth=0):
        """
        unpack a h5py entry: if it's a group expand,
        if it's a dataset convert
        
        for safety reasons, the depth cannot be larger than 99
        """
        if depth==99:raise Exception
        if isinstance(hdf5, (h5py._hl.group.Group)):
            d = {}
            for key in hdf5:
                elem   = hdf5[key]
                self.d[key] = hdf5
                d[key] = self.unpack_mat(elem, depth=depth+1)
            return d
        elif isinstance(hdf5, h5py._hl.dataset.Dataset):
            
            return self.convert_mat(hdf5)

    def convert_mat(self, dataset):
        """
        Converts h5py.dataset into python native datatypes
        according to the matlab class annotation
        """
        # all MATLAB variables have the attribute MATLAB_class
        # if this is not present, it is not convertible
        if not 'MATLAB_class' in dataset.attrs:
            if verbose:
                print(str(dataset), 'is not a matlab type')
            return None
        
        mtype = dataset.attrs['MATLAB_class'].decode()
       
        if mtype=='cell':
            cell = []
            for ref in dataset:
                for r in ref:
                    entry = self.unpack_mat(self.refs.get(r))
                    cell.append(entry)
            return cell
        elif mtype=='char': 
            return ''.join([chr(x) for x in dataset])
        elif mtype=='bool':
            return bool(dataset)
        elif mtype=='logical': 
            arr = np.array(dataset, dtype=bool)
            if arr.size==1: arr=bool(arr)
            return arr
        elif mtype=='canonical empty': 
            return None
        # complex numbers need to be filtered out separately
        elif 'imag' in str(dataset.dtype):
            return np.array(dataset, np.complex)
        # if it is none of the above, we can convert to numpy array
        elif mtype in ('double', 'single', 'int8', 'int16', 'int32', 'int64', 
                       'uint8', 'uint16', 'uint32', 'uint64'): 
            arr = np.array(dataset, dtype=dataset.dtype)
            # if size is 1, we usually have a single value, not an array
            if arr.size==1: arr=arr.squeeze()
            return arr
        else:
            if not self.verbose: return
            print('data type not supported: {}, {}'.format(mtype, dataset.dtype))
            
def loadmat(filename, verbose=True):
    """
    Loads a MATLAB 7.3 .mat file, which is actually a
    HDF5 file with some custom matlab annotations inside
    
    :param filename: A string pointing to the file
    :returns: A dictionary with the matlab variables loaded
    """
    decoder = HDF5Decoder(verbose=verbose)
    try:
        with h5py.File(filename, 'r') as hdf5:
            dictionary = decoder.mat2dict(hdf5)
        return dictionary
    except OSError:
        raise TypeError('Not a MATLAB 7.3 file. '\
                        'Load with scipy.io.loadmat() instead.')

    
        
