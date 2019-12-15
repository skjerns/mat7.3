# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 09:45:32 2019

@author: skjerns
"""
import numpy as np
import hdfdict

d = {'dict_' : {'entry_': np.array([1,2,3])},
     'int8' : np.array([1,2,3],dtype=np.int8)
     }

hdfdict.dump(d, 'hdf.h5')
