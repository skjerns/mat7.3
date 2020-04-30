# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 18:03:26 2020

@author: skjerns
"""
import numpy as np
import mat73
import unittest



class Testing(unittest.TestCase):
    
    
    def test_file1(self):
        d = mat73.loadmat('testfile1.mat')
        data = d['data']
        
        assert len(d)==2
        assert len(data)==26
        assert data.arr_two_three.shape==(3,2)
        np.testing.assert_allclose(d['secondvar'], [1,2,3,4])
        np.testing.assert_array_equal(data['arr_bool'], np.array([True,True,False]))
        assert data['arr_bool'].dtype==bool
        assert data['arr_char']=='test'
        np.testing.assert_array_equal(data['arr_double'], np.array([1.1,1.2,0.3],dtype=np.float64))
        assert data['arr_double'].dtype==np.float64
        np.testing.assert_array_equal(data['arr_float'], np.array([[1.1,2],[1.2,3],[0.3,4]],dtype=np.float32).T)
        np.testing.assert_array_equal(data['arr_nan'], np.array([np.nan,np.nan]))
        assert data['bool_']== False
        np.testing.assert_array_equal(data['cell_'][0], np.array([1.1,2.2]))
        np.testing.assert_array_equal(data['cell_'][1], False)
        np.testing.assert_array_equal(data['cell_'][2], [False, True])
        np.testing.assert_array_equal(data['cell_'][3], [1.1])
        np.testing.assert_array_equal(data['cell_'][4], [0])
        np.testing.assert_array_equal(data['cell_'][5], 'test')
        np.testing.assert_array_equal(data['cell_'][6][0], 'subcell')
        np.testing.assert_array_equal(data['cell_'][6][1], 0)
        np.testing.assert_array_equal(data['cell_char_'], [['Smith', 'Chung', 'Morales'], ['Sanchez','Peterson','Adams']])
        assert data['char_']== 'x'
        np.testing.assert_array_equal(data['complex_'], np.array([2+3j]))
        np.testing.assert_array_equal(data['complex2_'], np.array([123456789.123456789+987654321.987654321j]))
        assert data['double_']==0.1
        assert data['double_'].dtype==np.float64
        assert data['int16_']==16
        assert data['int16_'].dtype==np.int16
        assert data['int32_']==1115
        assert data['int32_'].dtype==np.int32  
        assert data['int64_']==65243
        assert data['int64_'].dtype==np.int64 
        assert data['int8_']==2
        assert data['int8_'].dtype==np.int8 
        np.testing.assert_array_equal(data['nan_'], np.nan)
        assert data['nan_'].dtype==np.float64
        assert data['single_']==np.array(0.1, dtype=np.float32)
        assert data['single_'].dtype==np.float32
        assert data['string_']=='tasdfasdf'
        assert data['uint8_']==2
        assert data['uint8_'].dtype==np.uint8
        assert data['uint16_']==12
        assert data['uint16_'].dtype==np.uint16
        assert data['uint32_']==5452
        assert data['uint32_'].dtype==np.uint32       
        assert data['uint64_']==32563
        assert data['uint64_'].dtype==np.uint64
        assert len(data['struct_'])==1
        np.testing.assert_array_equal(data['struct_']['test'],[1,2,3,4])
        assert len(data['struct2_'])==3
        
        
    def test_file2(self):
        d = mat73.loadmat('testfile2.mat')
        raw1 = d['raw1']
        assert raw1.label == ['']*5
        assert raw1.speakerType == ['main']*5
        np.testing.assert_array_equal(raw1.channel,[1,2,3,4,5])
        np.testing.assert_allclose(raw1.measGain,[-1.0160217,-0.70729065,-1.2158508,0.68839645,2.464653])
        for i in range(5):
            assert np.isclose(np.sum(raw1.h[i]),-0.019355850366449)
        for i in range(5):
            assert np.isclose(np.sum(raw1.HSmooth[i]),-0.019355850366449)
        
        
if __name__ == '__main__':  
    unittest.main()





