# -*- coding: utf-8 -*-
"""
Tests for mat73.savemat — round-trip: save then reload with mat73.loadmat.

Run with:  pytest -s -vv tests/test_savemat.py
"""
import os
import tempfile
import numpy as np
import pytest
import mat73


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def roundtrip(data: dict) -> dict:
    """Save *data* to a temp .mat file and reload it with mat73.loadmat."""
    with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
        path = f.name
    try:
        mat73.savemat(path, data)
        return mat73.loadmat(path, use_attrdict=False)
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ---------------------------------------------------------------------------
# Python scalars
# ---------------------------------------------------------------------------

class TestPythonScalars:

    def test_int(self):
        result = roundtrip({'x': 42})
        assert result['x'] == 42

    def test_float(self):
        result = roundtrip({'x': 3.14})
        assert np.isclose(result['x'], 3.14)

    def test_negative_float(self):
        result = roundtrip({'x': -1.5})
        assert np.isclose(result['x'], -1.5)

    def test_bool_true(self):
        result = roundtrip({'x': True})
        assert result['x'] == True

    def test_bool_false(self):
        result = roundtrip({'x': False})
        assert result['x'] == False

    def test_string(self):
        result = roundtrip({'x': 'hello'})
        assert result['x'] == 'hello'

    def test_empty_string(self):
        result = roundtrip({'x': ''})
        assert result['x'] == '' or result['x'] is None  # empty strings may come back as None

    def test_none(self):
        result = roundtrip({'x': None})
        assert result['x'] is None

    def test_nan(self):
        result = roundtrip({'x': float('nan')})
        assert np.isnan(result['x'])

    def test_inf(self):
        result = roundtrip({'x': float('inf')})
        assert np.isinf(result['x'])


# ---------------------------------------------------------------------------
# Multiple variables in one file
# ---------------------------------------------------------------------------

class TestMultipleVariables:

    def test_two_scalars(self):
        result = roundtrip({'a': 1, 'b': 2.0})
        assert result['a'] == 1
        assert result['b'] == 2.0

    def test_many_keys(self):
        data = {f'var{i}': float(i) for i in range(10)}
        result = roundtrip(data)
        for i in range(10):
            assert np.isclose(result[f'var{i}'], float(i))


# ---------------------------------------------------------------------------
# Python lists
# ---------------------------------------------------------------------------

class TestPythonLists:

    def test_list_of_ints(self):
        result = roundtrip({'x': [1, 2, 3]})
        np.testing.assert_array_equal(result['x'], [1, 2, 3])

    def test_list_of_floats(self):
        result = roundtrip({'x': [1.1, 2.2, 3.3]})
        np.testing.assert_allclose(result['x'], [1.1, 2.2, 3.3])

    def test_list_of_strings(self):
        result = roundtrip({'x': ['foo', 'bar', 'baz']})
        assert result['x'] == ['foo', 'bar', 'baz']

    def test_nested_list(self):
        result = roundtrip({'x': [[1, 2], [3, 4]]})
        np.testing.assert_array_equal(result['x'], [[1, 2], [3, 4]])

    def test_empty_list(self):
        result = roundtrip({'x': []})
        # empty arrays/lists may come back as empty array or None
        assert result['x'] is None or (hasattr(result['x'], '__len__') and len(result['x']) == 0)


# ---------------------------------------------------------------------------
# NumPy arrays — dtypes
# ---------------------------------------------------------------------------

class TestNumpyDtypes:

    def test_float64_1d(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = roundtrip({'x': arr})
        np.testing.assert_array_equal(result['x'], arr)
        assert result['x'].dtype == np.float64

    def test_float32_1d(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = roundtrip({'x': arr})
        np.testing.assert_allclose(result['x'], arr)
        assert result['x'].dtype == np.float32

    def test_int8(self):
        arr = np.array([1, 2, 3], dtype=np.int8)
        result = roundtrip({'x': arr})
        np.testing.assert_array_equal(result['x'], arr)
        assert result['x'].dtype == np.int8

    def test_int16(self):
        arr = np.array([100, 200, 300], dtype=np.int16)
        result = roundtrip({'x': arr})
        np.testing.assert_array_equal(result['x'], arr)
        assert result['x'].dtype == np.int16

    def test_int32(self):
        arr = np.array([1000, 2000, 3000], dtype=np.int32)
        result = roundtrip({'x': arr})
        np.testing.assert_array_equal(result['x'], arr)
        assert result['x'].dtype == np.int32

    def test_int64(self):
        arr = np.array([10**9, 2 * 10**9], dtype=np.int64)
        result = roundtrip({'x': arr})
        np.testing.assert_array_equal(result['x'], arr)
        assert result['x'].dtype == np.int64

    def test_uint8(self):
        arr = np.array([0, 128, 255], dtype=np.uint8)
        result = roundtrip({'x': arr})
        np.testing.assert_array_equal(result['x'], arr)
        assert result['x'].dtype == np.uint8

    def test_uint16(self):
        arr = np.array([0, 1000, 65535], dtype=np.uint16)
        result = roundtrip({'x': arr})
        np.testing.assert_array_equal(result['x'], arr)
        assert result['x'].dtype == np.uint16

    def test_uint32(self):
        arr = np.array([0, 100000], dtype=np.uint32)
        result = roundtrip({'x': arr})
        np.testing.assert_array_equal(result['x'], arr)
        assert result['x'].dtype == np.uint32

    def test_uint64(self):
        arr = np.array([0, 10**12], dtype=np.uint64)
        result = roundtrip({'x': arr})
        np.testing.assert_array_equal(result['x'], arr)
        assert result['x'].dtype == np.uint64

    def test_bool_array(self):
        arr = np.array([True, False, True], dtype=bool)
        result = roundtrip({'x': arr})
        np.testing.assert_array_equal(result['x'], arr)
        assert result['x'].dtype == bool

    def test_complex128(self):
        arr = np.array([1 + 2j, 3 + 4j], dtype=np.complex128)
        result = roundtrip({'x': arr})
        np.testing.assert_allclose(result['x'], arr)

    def test_nan_in_array(self):
        arr = np.array([1.0, np.nan, 3.0])
        result = roundtrip({'x': arr})
        assert np.isnan(result['x'][1])
        assert result['x'][0] == 1.0
        assert result['x'][2] == 3.0


# ---------------------------------------------------------------------------
# NumPy array shapes
# ---------------------------------------------------------------------------

class TestNumpyShapes:

    def test_scalar_array(self):
        arr = np.array(3.14)
        result = roundtrip({'x': arr})
        assert np.isclose(result['x'], 3.14)

    def test_1d_array(self):
        arr = np.arange(5, dtype=np.float64)
        result = roundtrip({'x': arr})
        np.testing.assert_array_equal(result['x'], arr)

    def test_2d_array(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = roundtrip({'x': arr})
        np.testing.assert_array_equal(result['x'], arr)
        assert result['x'].shape == arr.shape

    def test_3d_array(self):
        arr = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        result = roundtrip({'x': arr})
        np.testing.assert_array_equal(result['x'], arr)
        assert result['x'].shape == arr.shape

    def test_row_vector(self):
        arr = np.array([[1.0, 2.0, 3.0]])  # shape (1, 3)
        result = roundtrip({'x': arr})
        np.testing.assert_array_equal(result['x'].ravel(), arr.ravel())

    def test_column_vector(self):
        arr = np.array([[1.0], [2.0], [3.0]])  # shape (3, 1)
        result = roundtrip({'x': arr})
        np.testing.assert_array_equal(result['x'].ravel(), arr.ravel())

    def test_empty_array(self):
        arr = np.array([])
        result = roundtrip({'x': arr})
        assert result['x'] is None or len(result['x']) == 0

    def test_large_array(self):
        arr = np.random.default_rng(0).random((100, 50))
        result = roundtrip({'x': arr})
        np.testing.assert_allclose(result['x'], arr)


# ---------------------------------------------------------------------------
# Dicts (MATLAB structs)
# ---------------------------------------------------------------------------

class TestDicts:

    def test_simple_struct(self):
        result = roundtrip({'s': {'a': 1.0, 'b': 2.0}})
        assert result['s']['a'] == 1.0
        assert result['s']['b'] == 2.0

    def test_nested_struct(self):
        result = roundtrip({'s': {'inner': {'val': 42.0}}})
        assert result['s']['inner']['val'] == 42.0

    def test_struct_with_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = roundtrip({'s': {'data': arr}})
        np.testing.assert_array_equal(result['s']['data'], arr)

    def test_struct_with_string(self):
        result = roundtrip({'s': {'name': 'hello'}})
        assert result['s']['name'] == 'hello'

    def test_struct_mixed_types(self):
        result = roundtrip({'s': {
            'int_val': np.int32(5),
            'float_val': 2.71828,
            'arr': np.array([10, 20, 30], dtype=np.float64),
            'label': 'test',
        }})
        assert result['s']['int_val'] == 5
        assert np.isclose(result['s']['float_val'], 2.71828)
        np.testing.assert_array_equal(result['s']['arr'], [10, 20, 30])
        assert result['s']['label'] == 'test'


# ---------------------------------------------------------------------------
# NumPy scalar types
# ---------------------------------------------------------------------------

class TestNumpyScalars:

    def test_np_float64_scalar(self):
        result = roundtrip({'x': np.float64(1.23)})
        assert np.isclose(result['x'], 1.23)
        assert result['x'].dtype == np.float64

    def test_np_float32_scalar(self):
        result = roundtrip({'x': np.float32(1.23)})
        assert np.isclose(result['x'], np.float32(1.23))
        assert result['x'].dtype == np.float32

    def test_np_int32_scalar(self):
        result = roundtrip({'x': np.int32(99)})
        assert result['x'] == 99
        assert result['x'].dtype == np.int32

    def test_np_int64_scalar(self):
        result = roundtrip({'x': np.int64(123456789)})
        assert result['x'] == 123456789
        assert result['x'].dtype == np.int64

    def test_np_bool_scalar(self):
        result = roundtrip({'x': np.bool_(True)})
        assert result['x'] == True
