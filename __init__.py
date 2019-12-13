# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 12:20:29 2019

@author: Simon
"""
import scipy.io
import numpy as np
import h5py
from collections import OrderedDict
fname = 'C:/Users/Simon/dropbox/nt1-hrv-share/A9318_hrv.mat'
fname = 'C:/Users/Simon/desktop/data.mat'
# scipy.io.loadmat(fname)

xxx=[]
def h5py2dict(hdf5):
    def unpack(hdf5, depth=0):
        if depth==99:raise Exception
        if isinstance(hdf5, (h5py._hl.group.Group)):
            d = {}
            for key in hdf5:
                elem   = hdf5[key]
                d[key] = unpack(elem, depth=depth+1)
            return d
        elif isinstance(hdf5, h5py._hl.dataset.Dataset):
            if not 'MATLAB_class' in hdf5.attrs: return None
            mtype = hdf5.attrs['MATLAB_class'].decode()
            npdtypes = [x for x in np.__dict__ if isinstance(np.__dict__[x],type)]
            print(npdtypes)
            if mtype in npdtypes:
                xxx.append(hdf5)
                dtype = np.__dict__[mtype]
                arr = np.array(hdf5, dtype=dtype).T
                return arr.squeeze() if arr.size==1 else arr
            if mtype=='char': return ''.join([chr(x) for x in hdf5])
            if mtype=='logical': 
                arr = np.array(hdf5, dtype=bool)
                return arr.squeeze() if arr.size==1 else arr
            if mtype=='canonical empty': return 'None'
            if mtype=='cell':
                cell = []
                for ref in hdf5:
                    for r in ref:
                        entry = unpack(refs[r])
                        cell.append(entry)
                return cell
            print('Unknown datadtype ', mtype)
            return 'Unknown type {}'.format(mtype)
        
    if '#refs#' in hdf5:
        refs = hdf5['#refs#']
        
    d = {}
    for var in hdf5:
        if var=='#refs#':continue
        d[var] = unpack(hdf5[var])
    return d
   
f =  h5py.File(fname, 'r') 
a = h5py2dict(f)
a = h5py.dataset.Dataset.__repr__
    # b = a['data']['cell_char']
