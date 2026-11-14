import os
import tempfile
import unittest

import h5py
import mat73
import numpy as np


class TestOnlyInclude(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.filename = os.path.join(directory.name, 'selection.mat')
        # Synthetic MATLAB-style HDF5 datasets isolate path selection.
        self.values = {
            'variable': 1,
            'variable_1': 2,
            'var': 3,
            'other/variable': 4,
            'tree/variable': 5,
            'tree/variable_1': 6,
            'tree/branch/value': 7,
            'tree/branch_extra/value': 8,
            'tree_1/variable': 9,
        }
        with h5py.File(self.filename, 'w') as file:
            for path, value in self.values.items():
                dataset = file.create_dataset(path, data=[[float(value)]])
                dataset.attrs['MATLAB_class'] = np.bytes_('double')

    def test_exact_variable(self):
        for selection in ('variable', '/variable', 'variable/', ['variable']):
            with self.subTest(selection=selection):
                data = mat73.loadmat(self.filename, only_include=selection)
                self.assertEqual(set(data), {'variable'})
                self.assertEqual(data['variable'], 1)

    def test_exact_nested_path(self):
        for selection, expected in (
                ('tree/variable', {'variable': 5}),
                ('/tree/variable/', {'variable': 5}),
                ('tree/branch/value', {'branch': {'value': 7}})):
            with self.subTest(selection=selection):
                data = mat73.loadmat(self.filename, only_include=selection)
                self.assertEqual(data, {'tree': expected})

    def test_subtree(self):
        for selection in ('tree', 'tree/'):
            with self.subTest(selection=selection):
                data = mat73.loadmat(self.filename, only_include=selection)
                self.assertEqual(data, {'tree': {
                    'variable': 5, 'variable_1': 6,
                    'branch': {'value': 7},
                    'branch_extra': {'value': 8}}})

    def test_multiple_paths_and_attrdict(self):
        data = mat73.loadmat(
            self.filename, use_attrdict=True,
            only_include=['variable', 'tree/branch/value'])
        self.assertEqual(set(data), {'variable', 'tree'})
        self.assertEqual(data.variable, 1)
        self.assertEqual(data.tree.branch.value, 7)
        self.assertEqual(set(data.tree), {'branch'})

    def test_missing_variable(self):
        for selection in ('variab', 'variable_missing'):
            with self.subTest(selection=selection):
                with self.assertLogs('mat73', level='WARNING') as captured:
                    data = mat73.loadmat(self.filename, only_include=selection)
                self.assertEqual(data, {})
                self.assertIn('could not be found', captured.output[0])

    def test_root_and_no_selection(self):
        full = mat73.loadmat(self.filename)
        self.assertEqual(
            set(full), {'variable', 'variable_1', 'var', 'other', 'tree', 'tree_1'})
        self.assertEqual(mat73.loadmat(self.filename, only_include='/'), full)
        self.assertEqual(mat73.loadmat(self.filename, only_include=[]), {})
