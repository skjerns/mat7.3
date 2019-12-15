# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 12:20:29 2019

@author: Simon
"""
import os
import numpy as np
import h5py
import datetime
fname = 'C:/Users/Simon/dropbox/nt1-hrv-share/A9318_hrv.mat'

# scipy.io.loadmat(fname)

#TODO: Add attrs?
a=[]


class HDF5Decoder():
    def __init__(self):
        self.x = []
        self.d = {}
        self.refs = {} # this is used in case of matlab matrices

    def hdf52dict(self, hdf5):
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
                d[var] = self.unpack_mat(hdf5[var])
            else:
                raise ValueError('can only unpack .h5, .hdf5 or .mat')
        return d
    
    def unpack_hdf5(self, hdf5, depth=0):
        pass
    
    def unpack_mat(self, hdf5, depth=0):
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
        
        # all MATLAB variables have the attribute MATLAB_class
        # if this is not present, it is not convertable
        if not 'MATLAB_class' in dataset.attrs:
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
        elif mtype=='embedded.fi':
            print("ERROR: {} (fixed point) can't be loaded".format(dataset.name))
            self.x.append(dataset)
            return 'fixed point'
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
            self.d[dataset.dtype] = dataset
            # if size is 1, we usually have a single value, not an array
            if arr.size==1: arr=arr.squeeze()
            return arr
        else:
            print('data type not supported: {}, {}'.format(mtype, dataset.dtype))
            
def load_hdf5(filename):
    decoder = HDF5Decoder()
    with h5py.File(filename, 'r') as hdf5:
        dictionary = decoder.hdf52dict(hdf5)
    return dictionary
    
fname = 'data.mat'
fname = 'hdf.h5'

hdf5 = h5py.File(fname, 'r') 
    
self = HDF5Decoder()
data = self.hdf52dict(hdf5)
x = self.x
d = self.d
        
