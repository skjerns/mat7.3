# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 18:03:26 2020

@author: skjerns
"""
import os
import numpy as np
import mat73
import unittest
import time
try:
    from pygit2 import Repository
    import pkg_resources
    version = pkg_resources.get_distribution('mat73').version
except:
    version = '0.00'
try:
    repo = Repository('.')
    head = repo.head
    branch = head.shorthand
    name = head.name
    message = head.peel().message
except:
    branch = 'unknown'
    name = 'unknown'
    message = 'no msg'

print(f'#### Installed version: mat73-{version} on {branch}({name}) "{message}" ####')

class Testing(unittest.TestCase):

    def setUp(self):
        for i in range(1,8):
            file = 'testfile{}.mat'.format(i)
            if not os.path.exists(file):
                file = os.path.join('./tests', file)
            self.__setattr__ ('testfile{}'.format(i), file)

    def test_file1_noattr(self):
        """
        Test each default MATLAB type loads correctl
        """
        d = mat73.loadmat(self.testfile1, use_attrdict=False)
        data = d['data']
        
        assert len(d)==3
        assert len(d.keys())==3
        assert len(data)==29
        assert data['arr_two_three'].shape==(3,2)
        np.testing.assert_allclose(d['secondvar'], [1,2,3,4])
        np.testing.assert_array_equal(data['arr_bool'], np.array([True,True,False]))
        assert data['arr_bool'].dtype==bool
        assert data['arr_char']=='test'
        assert data['missing_']==None
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
        np.testing.assert_array_equal(data['complex3_'], np.array([8.909089035006170e-04 + 0.000000000000000e+00j]))

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
        assert len(data['struct2_'])==2
        assert len(data['structarr_'])==3
        assert data['structarr_'][0] == {'f1': ['some text'], 'f2': ['v1']}
        assert data['structarr_'][1]['f2'] == ['v2']
        assert data['structarr_'][2]['f2'] == ['v3']
        assert d['keys'] == 'must_not_overwrite'

        with self.assertRaises(AttributeError):
            d.structarr_


    def test_file1_withattr(self):
        """
        Test each default MATLAB type loads correctl
        """
        d = mat73.loadmat(self.testfile1, use_attrdict=True)
        data = d['data']
        
        assert len(d)==3
        assert len(d.keys())==3
        assert len(data)==29
        assert data['arr_two_three'].shape==(3,2)
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
        np.testing.assert_array_equal(data['complex3_'], np.array([8.909089035006170e-04 + 0.000000000000000e+00j]))

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
        assert len(data['struct2_'])==2
        assert len(data['structarr_'])==3
        assert data['structarr_'][0] == {'f1': ['some text'], 'f2': ['v1']}
        assert data['structarr_'][1]['f2'] == ['v2']
        assert data['structarr_'][2]['f2'] == ['v3']


        ## now do the same with attrdict
        data = d.data

        assert len(d)==3
        assert len(data)==29
        assert data.arr_two_three.shape==(3,2)
        np.testing.assert_allclose(d.secondvar, [1,2,3,4])
        np.testing.assert_array_equal(data.arr_bool, np.array([True,True,False]))
        assert data.arr_bool.dtype==bool
        assert data.arr_char=='test'
        np.testing.assert_array_equal(data.arr_double, np.array([1.1,1.2,0.3],dtype=np.float64))
        assert data.arr_double.dtype==np.float64
        np.testing.assert_array_equal(data.arr_float, np.array([[1.1,2],[1.2,3],[0.3,4]],dtype=np.float32).T)
        np.testing.assert_array_equal(data.arr_nan, np.array([np.nan,np.nan]))
        assert data.bool_== False
        np.testing.assert_array_equal(data.cell_[0], np.array([1.1,2.2]))
        np.testing.assert_array_equal(data.cell_[1], False)
        np.testing.assert_array_equal(data.cell_[2], [False, True])
        np.testing.assert_array_equal(data.cell_[3], [1.1])
        np.testing.assert_array_equal(data.cell_[4], [0])
        np.testing.assert_array_equal(data.cell_[5], 'test')
        np.testing.assert_array_equal(data.cell_[6][0], 'subcell')
        np.testing.assert_array_equal(data.cell_[6][1], 0)
        np.testing.assert_array_equal(data.cell_char_, [['Smith', 'Chung', 'Morales'], ['Sanchez','Peterson','Adams']])
        assert data.char_== 'x'
        np.testing.assert_array_equal(data.complex_, np.array([2+3j]))
        np.testing.assert_array_equal(data.complex2_, np.array([123456789.123456789+987654321.987654321j]))
        np.testing.assert_array_equal(data.complex3_, np.array([8.909089035006170e-04 + 0.000000000000000e+00j]))

        assert data.double_==0.1
        assert data.double_.dtype==np.float64
        assert data.int16_==16
        assert data.int16_.dtype==np.int16
        assert data.int32_==1115
        assert data.int32_.dtype==np.int32  
        assert data.int64_==65243
        assert data.int64_.dtype==np.int64 
        assert data.int8_==2
        assert data.int8_.dtype==np.int8 
        np.testing.assert_array_equal(data.nan_, np.nan)
        assert data.nan_.dtype==np.float64
        assert data.single_==np.array(0.1, dtype=np.float32)
        assert data.single_.dtype==np.float32
        assert data.string_=='tasdfasdf'
        assert data.uint8_==2
        assert data.uint8_.dtype==np.uint8
        assert data.uint16_==12
        assert data.uint16_.dtype==np.uint16
        assert data.uint32_==5452
        assert data.uint32_.dtype==np.uint32       
        assert data.uint64_==32563
        assert data.uint64_.dtype==np.uint64
        assert len(data.struct_)==1
        np.testing.assert_array_equal(data.struct_.test,[1,2,3,4])
        assert len(data.struct2_)==2
        assert len(data.structarr_)==3
        assert data.structarr_[0] == {'f1': ['some text'], 'f2': ['v1']}
        assert data.structarr_[1].f2 == ['v2']
        assert data.structarr_[2].f2 == ['v3']
        assert d['keys'] == 'must_not_overwrite'
        assert d.keys!='must_not_overwrite'



    def test_file2(self):
        """
        Test that complex numbers are loaded correctly
        """
        d = mat73.loadmat(self.testfile2)
        raw1 = d['raw1']
        assert raw1['label'] == ['']*5
        assert raw1['speakerType'] == ['main']*5
        np.testing.assert_array_equal(raw1['channel'],[1,2,3,4,5])
        np.testing.assert_allclose(raw1['measGain'],[-1.0160217,-0.70729065,-1.2158508,0.68839645,2.464653])
        for i in range(5):
            assert np.isclose(np.sum(raw1['h'][i]),-0.0007341067459898744)
    
        np.testing.assert_array_almost_equal(raw1['HSmooth'][0][2], [ 0.001139-4.233492e-04j,  0.00068 +8.927040e-06j,
        0.002382-7.647651e-04j, -0.012677+3.767829e-03j])


    def test_file3(self):
        """
        Test larger complex numbers are also loaded
        """
        d = mat73.loadmat(self.testfile3)
        raw1 = d['raw1']
        assert raw1['label'] == ['']*5
        assert raw1['speakerType'] == ['main']*5
        np.testing.assert_array_equal(raw1['channel'],[1,2,3,4,5])
        np.testing.assert_allclose(raw1['measGain'],[-1.0160217,-0.70729065,-1.2158508,0.68839645,2.464653])
        for i in range(5):
            assert np.isclose(np.sum(raw1['h'][i]),-0.019355850366449)
        for i in range(5):
            assert np.isclose(np.sum(raw1['HSmooth'][i]),-0.019355850366449)


    def test_file4(self):
        """
        Test a file created by Kubios HRV that created lots of problems before
        """
        d = mat73.loadmat(self.testfile4, use_attrdict=False)
        assert len(d)==1
        res = d['Res']
        assert res['f_name'] == '219_92051.edf'
        assert res['f_path'] == 'C:\\Users\\sleep\\Desktop\\set2\\ECG_I\\'
        assert res['isPremium'] == True
        assert len(res)==4
        self.assertEqual(sorted(res.keys()), ['HRV', 'f_name', 'f_path', 'isPremium'])
        hrv = res['HRV']
        exp_key = ['Data', 'Frequency', 'NonLinear', 'Param', 'Statistics', 'Summary', 'TimeVar', 'timevaranalOK']
        assert sorted(exp_key)==sorted(hrv.keys())
        assert len(hrv)==8
        assert hrv['timevaranalOK']==1.0
        data = hrv['Data']
        assert len(data)==18
        assert len(data['Artifacts'])==1
        assert data['RR'].shape == (4564,)
        assert data['RRcorrtimes'].shape == (51,)
        assert data['RRdt'].shape == (4564,)
        assert data['RRdti'].shape == (20778,)
        assert data['RRi'].shape == (20778,)
        assert data['RRi'].shape == (20778,)
        assert data['T_RR'].shape  == (4565,)
        assert data['T_RRi'].shape  == (20778,)
        assert data['T_RRorig'].shape == (4572,)


        islist = ['RRs', 'RRsdt', 'RRsdti', 'RRsi', 'T_RRs', 'T_RRsi', 'tmp', 'Artifacts']
        for lname in data:
            if lname in islist:
                assert isinstance(data[lname], list)
                assert len(data[lname])==1
            else:
                assert not isinstance(data[lname], list)

        assert data['RRs'][0].shape == (276, )
        assert data['RRsdt'][0].shape == (276, )
        assert data['RRsdti'][0].shape == (1190, )
        assert data['RRsi'][0].shape == (1190, )
        assert data['T_RRs'][0].shape == (276, )
        assert data['T_RRsi'][0].shape == (1190, )
        assert len(data['tmp'][0]) == 3
        for arr in data['tmp'][0].values():
            assert isinstance(arr, np.ndarray)

        np.testing.assert_allclose(data['Artifacts'][0], 2.17391304)

    def test_file5(self):
        """
        Test a file created by Kubios HRV that created lots of problems before
        """
        d = mat73.loadmat(self.testfile5, use_attrdict=False)

    def test_file6_empty_cell_array(self):

        data = mat73.loadmat(self.testfile6)
        np.testing.assert_array_almost_equal(data['A'], [])
        np.testing.assert_array_almost_equal(data['B'], np.array([1,2,3], dtype=float))

    def test_file7_empty_cell_array(self):
        data = mat73.loadmat(self.testfile7)
        
    


if __name__ == '__main__':

    unittest.main()





