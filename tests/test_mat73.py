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
from datetime import datetime
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

EXPECTED_VARS_FILE1 = 30

print(f'#### Installed version: mat73-{version} on {branch}({name}) "{message}" ####')

class Testing(unittest.TestCase):

    def setUp(self):
        """make links to test files and make sure they are present"""
        for i in range(1, 17):
            file = 'testfile{}.mat'.format(i)
            if not os.path.exists(file):
                file = os.path.join('./tests', file)
            self.__setattr__ ('testfile{}'.format(i), file)

        file_npt = 'testfile9.npt'
        if not os.path.exists(file_npt):
            file_npt = os.path.join('./tests', file_npt)
        self.testfile_npt = file_npt

    def test_file_obj_loading(self):
        """test for loading as file object and string filename """
        d = mat73.loadmat(self.testfile1, use_attrdict=False)
        data = d['data']
        assert len(d)==3
        assert len(d.keys())==3
        with open(self.testfile1, 'rb') as f:
            d = mat73.loadmat(f, use_attrdict=False)
            data = d['data']
            assert len(d)==3
            assert len(d.keys())==3


    def test_file1_noattr(self):
        """Test each default MATLAB type loads correctly"""
        d = mat73.loadmat(self.testfile1, use_attrdict=False)
        data = d['data']

        assert len(d)==3
        assert len(d.keys())==3
        assert len(data)==EXPECTED_VARS_FILE1
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
        sparse = data['sparse_'].toarray()
        assert sparse[1,4] == 6
        assert sparse[3,7] == 7
        assert (sparse!=0).sum()==2


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
        assert len(data)==EXPECTED_VARS_FILE1
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
        assert len(data)==EXPECTED_VARS_FILE1
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
        self.assertEqual(raw1['label'], ['']*5)
        self.assertEqual(raw1['speakerType'],['main']*5)
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
        self.assertEqual(raw1['label'], ['']*5)
        self.assertEqual(raw1['speakerType'], ['main']*5)
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

    def test_can_load_other_extension(self):
        with self.assertLogs(level='WARNING'):
            data = mat73.loadmat(self.testfile_npt)


    def test_load_specific_vars(self):
        for key in ['keys', ['keys']]:
            data = mat73.loadmat(self.testfile1, only_include=key)
            assert len(data)==1
            assert data['keys']=='must_not_overwrite'

        with self.assertLogs(level='WARNING'):
            data = mat73.loadmat(self.testfile1, only_include='notpresent')

        data = mat73.loadmat(self.testfile1, only_include=['data', 'keys'])
        assert len(data)==2
        assert len(data['data'])==EXPECTED_VARS_FILE1
        assert len(data['data']['cell_'])==7

        # check if loading times are faster, should be the case.
        start = time.time()
        data = mat73.loadmat(self.testfile4)
        elapsed1 = time.time()-start
        start = time.time()
        data = mat73.loadmat(self.testfile4, only_include='Res/HRV/Param/use_custom_print_command')
        elapsed2 = time.time()-start
        assert elapsed2<elapsed1, 'loading specific var was not faster'

    def test_file8_nullchars(self):
        """test if null chars are retained in char arrays"""
        data = mat73.loadmat(self.testfile8)
        self.assertEqual(len(data['char_array']), 7, 'not all elements loaded')
        self.assertEqual(len(data['char_string']), 11, 'not all elements loaded')
        self.assertEqual(data['char_array'], '\x01\x02\x03\x00\x04\x05\x06')

    def test_file11_specificvars_cells(self):
        """see if contents of cells are also loaded when using only_include"""
        # check regular loading works
        data = mat73.loadmat(self.testfile11)
        assert len(data)==1
        assert data['foo'][0]==1
        assert data['foo'][1]==2

        # load cells correctly
        data = mat73.loadmat(self.testfile11, only_include=['foo'])
        assert len(data)==1
        assert data['foo'][0]==1
        assert data['foo'][1]==2

        # loading should be empty for non-existend var
        data = mat73.loadmat(self.testfile11, only_include=['bar'])
        assert len(data)==0

    def test_file12_sparse_matrix_fix(self):
        """some sparse matrices could not be loaded. check if it now works"""
        # check regular loading works
        import scipy
        data = mat73.loadmat(self.testfile12)
        struct_sparse = data['rec_img']['fwd_model']['stimulation']
        sparse_type = scipy.sparse.csc_matrix
        assert isinstance(struct_sparse[0]['meas_pattern'], sparse_type)
        assert isinstance(struct_sparse[0]['stim_pattern'], sparse_type)
        assert isinstance(struct_sparse[0]['stimulation'], str)

    def test_file13_empty_sparse_matrix(self):
        """some sparse matrices could not be loaded. check if it now works"""
        # check regular loading works
        import scipy
        data = mat73.loadmat(self.testfile13)
        struct_sparse = data['A']
        sparse_type = scipy.sparse.csc_matrix
        assert isinstance(struct_sparse, sparse_type)
        assert struct_sparse.getnnz()==0
        assert struct_sparse.sum()==0
        assert (struct_sparse.toarray()==np.zeros([2,3])).all()
        assert struct_sparse.shape==(2, 3)


    def test_file14_n_d_array_with_singular_axis(self):
        """Test loading of n-D array with one dimension of size 1"""

        # the array was created in Matlab like so:
        # data = reshape(1:24, [3, 1, 4, 2]);

        data = mat73.loadmat(self.testfile14)

        # Assert that there's only one variable in the file
        self.assertEqual(len(data), 1)
        self.assertIn('data', data)

        # Get the data array
        arr = data['data']

        # Assert the shape is correct (3 x 1 x 4 x 2)
        self.assertEqual(arr.shape, (3, 1, 4, 2))

        # Assert the data type
        self.assertEqual(arr.dtype, np.float64)

        # Check the contents of the array
        # Create the expected data array in the correct structure
        expected = np.zeros((3, 1, 4, 2))
        expected[:, 0, 0, 0] = [1, 2, 3]
        expected[:, 0, 1, 0] = [4, 5, 6]
        expected[:, 0, 2, 0] = [7, 8, 9]
        expected[:, 0, 3, 0] = [10, 11, 12]
        expected[:, 0, 0, 1] = [13, 14, 15]
        expected[:, 0, 1, 1] = [16, 17, 18]
        expected[:, 0, 2, 1] = [19, 20, 21]
        expected[:, 0, 3, 1] = [22, 23, 24]
        np.testing.assert_array_equal(arr, expected)

        # Test that the singular dimension is preserved
        self.assertEqual(arr.shape[1], 1)



    def test_file15_strip(self):
        """Test loading of n-D array with one dimension of size 1"""

        # the array was created in Matlab like so:
        # data = reshape(1:24, [3, 1, 4, 2]);

        data = mat73.loadmat(self.testfile15)


        self.assertEqual(data['x_0'], None)
        self.assertEqual(data['x_1_0'], None)
        self.assertEqual(data['x_0_1'], None)
        self.assertEqual(data['x_0_10'], None)
        self.assertEqual(data['x_10_0'], None)

        expected = {'x_1' : (),
                    'x_10' : (10,),
                    'x_1_1': (),
                    'x_1_10': (10,),
                    'x_10_1': (10,),
                    'x_10_10': (10, 10),
                    'x_1_1_10_1_1': (1, 1, 10),
                    'x_10_1_1_10': (10, 1, 1, 10),
                    }


        for var, shape in expected.items():
            self.assertEqual(data[var].shape, shape)
            self.assertEqual(data[var].ndim, len(shape))

    def test_file16_2d_char_array(self):
        """Test loading of 2D char array"""
        # the matlab array has shape (6 X 57)

        data = mat73.loadmat(self.testfile16)

        expected = ['PSTH tensor for image sequences (averaged across frames):',
                    'dimension 1: 2 scales (zoom1x, zoom2x)                   ',
                    'dimension 2: 3 category (natural, synthetic, contrast)   ',
                    'dimension 3: 10 movies                                   ',
                    'dimension 4: sorted units                                ',
                    'dimension 5: PSTH time bins                              ']
        self.assertEqual(data['char_arr_1d'], 'abcd')
        self.assertEqual(data['char_arr_2d'], expected)
        self.assertEqual(data['char_arr_3d'], [['abcd', 'defg'], ['ghij', 'jklm'], ['mnöp', 'pqrs']])

    def test_datetime_loading(self):
        """Test loading of MATLAB datetime objects."""
        test_file_path = os.path.join('tests', 'testfile_datetime.mat')
        if not os.path.exists(test_file_path):
            self.skipTest(f"{test_file_path} not found. User needs to provide this file. Skipping datetime tests.")

        d = mat73.loadmat(test_file_path, use_attrdict=True)

        # Helper function for datetime comparison
        def assert_datetime_equal(dt_obj, year, month, day, hour=0, minute=0, second=0, microsecond=0):
            self.assertIsInstance(dt_obj, datetime)
            self.assertEqual(dt_obj.year, year)
            self.assertEqual(dt_obj.month, month)
            self.assertEqual(dt_obj.day, day)
            self.assertEqual(dt_obj.hour, hour)
            self.assertEqual(dt_obj.minute, minute)
            self.assertEqual(dt_obj.second, second)
            self.assertEqual(dt_obj.microsecond, microsecond)

        # dt_scalar: datetime(2023, 10, 26)
        self.assertIn('dt_scalar', d)
        assert_datetime_equal(d.dt_scalar, 2023, 10, 26)

        # dt_row_vector: [datetime(2023, 11, 1), datetime(2023, 11, 15), datetime(2023, 12, 25)]
        self.assertIn('dt_row_vector', d)
        dt_row_vector = d.dt_row_vector
        self.assertTrue(isinstance(dt_row_vector, (list, np.ndarray)))
        self.assertEqual(len(dt_row_vector), 3)
        assert_datetime_equal(dt_row_vector[0], 2023, 11, 1)
        assert_datetime_equal(dt_row_vector[1], 2023, 11, 15)
        assert_datetime_equal(dt_row_vector[2], 2023, 12, 25)

        # dt_column_vector: [datetime(2024, 1, 10), datetime(2024, 2, 20), datetime(2024, 3, 30)]
        # Expecting a flat list or 1D array due to squeeze
        self.assertIn('dt_column_vector', d)
        dt_column_vector = d.dt_column_vector
        self.assertTrue(isinstance(dt_column_vector, (list, np.ndarray)))
        self.assertEqual(len(dt_column_vector), 3)
        assert_datetime_equal(dt_column_vector[0], 2024, 1, 10)
        assert_datetime_equal(dt_column_vector[1], 2024, 2, 20)
        assert_datetime_equal(dt_column_vector[2], 2024, 3, 30)

        # dt_matrix: [[datetime(2025, 1, 1), datetime(2025, 2, 1)], [datetime(2025, 3, 1), datetime(2025, 4, 1)]]
        # Note: MATLAB is column-major, Python (NumPy) is row-major.
        # mat73 transposes arrays, so d.dt_matrix[row_idx, col_idx] should correspond to MATLAB(row_idx+1, col_idx+1)
        self.assertIn('dt_matrix', d)
        dt_matrix = d.dt_matrix
        self.assertIsInstance(dt_matrix, np.ndarray) # Usually loaded as numpy array
        self.assertEqual(dt_matrix.shape, (2, 2))
        assert_datetime_equal(dt_matrix[0, 0], 2025, 1, 1)
        assert_datetime_equal(dt_matrix[0, 1], 2025, 2, 1) # MATLAB: (1,2)
        assert_datetime_equal(dt_matrix[1, 0], 2025, 3, 1) # MATLAB: (2,1)
        assert_datetime_equal(dt_matrix[1, 1], 2025, 4, 1) # MATLAB: (2,2)
        
        # dt_specific_time: datetime(2023, 3, 15, 14, 30, 45, 678000)
        self.assertIn('dt_specific_time', d)
        assert_datetime_equal(d.dt_specific_time, 2023, 3, 15, 14, 30, 45, 678000)

        # dt_nat: None
        self.assertIn('dt_nat', d)
        self.assertIsNone(d.dt_nat)

        # dt_nat_array: [datetime(2023, 1, 1), None, datetime(2023, 1, 3)]
        self.assertIn('dt_nat_array', d)
        dt_nat_array = d.dt_nat_array
        self.assertTrue(isinstance(dt_nat_array, (list, np.ndarray)))
        self.assertEqual(len(dt_nat_array), 3)
        assert_datetime_equal(dt_nat_array[0], 2023, 1, 1)
        self.assertIsNone(dt_nat_array[1])
        assert_datetime_equal(dt_nat_array[2], 2023, 1, 3)

        # dt_with_timezone: datetime(2023, 10, 26, 10, 0, 0) (naive)
        self.assertIn('dt_with_timezone', d)
        # Timezone information is expected to be lost, testing for naive datetime
        assert_datetime_equal(d.dt_with_timezone, 2023, 10, 26, 10, 0, 0)
        
        # dt_past_scalar: datetime(1500, 1, 1)
        self.assertIn('dt_past_scalar', d)
        assert_datetime_equal(d.dt_past_scalar, 1500, 1, 1)

        # dt_future_scalar: datetime(2500, 1, 1)
        self.assertIn('dt_future_scalar', d)
        assert_datetime_equal(d.dt_future_scalar, 2500, 1, 1)

        # dt_struct:
        self.assertIn('dt_struct', d)
        dt_struct = d.dt_struct
        self.assertTrue(hasattr(dt_struct, 'scalar')) # AttrDict access
        assert_datetime_equal(dt_struct.scalar, 2026, 7, 4)
        
        self.assertTrue(hasattr(dt_struct, 'array'))
        struct_array_field = dt_struct.array
        self.assertTrue(isinstance(struct_array_field, (list, np.ndarray)))
        self.assertEqual(len(struct_array_field), 2)
        assert_datetime_equal(struct_array_field[0], 2026, 8, 1)
        assert_datetime_equal(struct_array_field[1], 2026, 9, 15)

        self.assertTrue(hasattr(dt_struct, 'mixed_datetime_in_struct_array'))
        mixed_struct_array = dt_struct.mixed_datetime_in_struct_array
        self.assertIsInstance(mixed_struct_array, list) # Struct arrays become lists of AttrDicts
        self.assertEqual(len(mixed_struct_array), 2)
        self.assertTrue(hasattr(mixed_struct_array[0], 'd'))
        assert_datetime_equal(mixed_struct_array[0].d, 2026, 10, 1)
        self.assertTrue(hasattr(mixed_struct_array[1], 'd'))
        assert_datetime_equal(mixed_struct_array[1].d, 2026, 11, 1)

        # dt_cell_array:
        # d['dt_cell_array'][0]: datetime(2027, 5, 18)
        # d['dt_cell_array'][1][0]: datetime(2027, 6, 1)
        # d['dt_cell_array'][1][1]: datetime(2027, 7, 1)
        self.assertIn('dt_cell_array', d)
        dt_cell_array = d.dt_cell_array # This will be a list
        self.assertIsInstance(dt_cell_array, list)
        self.assertEqual(len(dt_cell_array), 2)
        assert_datetime_equal(dt_cell_array[0], 2027, 5, 18)
        
        cell_inner_array = dt_cell_array[1]
        self.assertTrue(isinstance(cell_inner_array, (list, np.ndarray)))
        self.assertEqual(len(cell_inner_array), 2)
        assert_datetime_equal(cell_inner_array[0], 2027, 6, 1)
        assert_datetime_equal(cell_inner_array[1], 2027, 7, 1)

        # dt_cell_array_column:
        # d['dt_cell_array_column'][0]: datetime(2027, 8, 1)
        # d['dt_cell_array_column'][1]: datetime(2027, 9, 1)
        self.assertIn('dt_cell_array_column', d)
        dt_cell_array_column = d.dt_cell_array_column # This will be a list
        self.assertIsInstance(dt_cell_array_column, list)
        self.assertEqual(len(dt_cell_array_column), 2) # MATLAB {{dt1};{dt2}} results in a 2x1 cell array
                                                       # which mat73 typically converts to a list of 2 elements.
        assert_datetime_equal(dt_cell_array_column[0], 2027, 8, 1)
        assert_datetime_equal(dt_cell_array_column[1], 2027, 9, 1)


if __name__ == '__main__':

    unittest.main()
