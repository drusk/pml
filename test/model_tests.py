# Copyright (C) 2012 David Rusk
#
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the "Software"), to 
# deal in the Software without restriction, including without limitation the 
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
# sell copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS 
# IN THE SOFTWARE.
"""
Unit tests for the model module.

@author: drusk
"""

import unittest
from model import DataSet, as_dataset
from hamcrest import assert_that, contains
from matchers import equals_dataset
import numpy as np
import pandas as pd

class DataSetTest(unittest.TestCase):

    def test_reduce_rows(self):
        dataset = DataSet([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        reduced = dataset.reduce_rows(sum)
        assert_that(reduced.values, contains(6, 15, 24))
        
    def test_drop_column(self):
        original = DataSet([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertEqual(original.num_features(), 3)
        filtered = original.drop_column(1)
        self.assertEqual(filtered.num_features(), 2)
        assert_that(filtered, equals_dataset([[1, 3], [4, 6], [7, 9]]))
        # make sure original unchanged
        self.assertEqual(original.num_features(), 3)
        assert_that(original, equals_dataset([[1, 2, 3], [4, 5, 6], 
                                              [7, 8, 9]]))
        
    def test_get_column(self):
        dataset = DataSet([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        column1 = dataset.get_column(1)
        assert_that(column1.values, contains(2, 5, 8))
    
    def test_get_column_set_value(self):
        dataset = DataSet([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        dataset.get_column(1)[:] = 1
        assert_that(dataset.get_column(1), contains(1, 1, 1))
        
    def test_get_rows(self):
        dataset = DataSet([[1, 2], [3, 4], [5, 6], [7, 8]])
        selection = dataset.get_rows([1, 3])
        
        self.assertEqual(selection.num_samples(), 2)
        assert_that(selection, equals_dataset([[3, 4], [7, 8]]))
        
    def test_split(self):
        dataset = DataSet([[1, 2], [3, 4], [5, 6], [7, 8]])
        first, second = dataset.split(0.5)
        self.assertEqual(first.num_samples(), 2)
        assert_that(first, equals_dataset([[1, 2], [3, 4]]))
        self.assertEqual(second.num_samples(), 2)
        assert_that(second, equals_dataset([[5, 6], [7, 8]]))
        
    def test_unequal_split(self):
        dataset = DataSet([[1, 2], [3, 4], [5, 6], [7, 8]])
        first, second = dataset.split(0.3)
        self.assertEqual(first.num_samples(), 1)
        assert_that(first, equals_dataset([[1, 2]]))
        self.assertEqual(second.num_samples(), 3)
        assert_that(second, equals_dataset([[3, 4], [5, 6], [7, 8]]))

    def test_split_0(self):
        dataset = DataSet([[1, 2], [3, 4], [5, 6], [7, 8]])
        first, second = dataset.split(0)
        self.assertEqual(first.num_samples(), 0)
        assert_that(first, equals_dataset([]))
        self.assertEqual(second.num_samples(), 4)
        assert_that(second, equals_dataset([[1, 2], [3, 4], [5, 6], [7, 8]]))

    def test_split_invalid_percent(self):
        dataset = DataSet([[1, 2], [3, 4], [5, 6], [7, 8]])
        self.assertRaises(ValueError, dataset.split, 50)
        
    def test_fill_missing(self):
        dataset = DataSet([[1, np.NaN, 3], [np.NaN, 5, np.NaN]])
        dataset.fill_missing(0)
        assert_that(dataset, equals_dataset([[1, 0, 3], [0, 5, 0]]))
        
    def test_get_row(self):
        dataset = DataSet([[1, 2], [3, 4], [5, 6], [7, 8]])
        row = dataset.get_row(1)
        assert_that(row.values, contains(3, 4))
        # check that changes made to selected row are reflected in original
        row[:] = 1
        assert_that(dataset.get_row(1), contains(1, 1))
        
    def test_get_row_by_id(self):
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 
                          index=["V01", "V02", "V03"])
        dataset = DataSet(df)
        sample = dataset.get_row("V02")
        assert_that(sample, contains(4, 5, 6))
        # make sure position based index is still usable
        sample = dataset.get_row(1)
        assert_that(sample, contains(4, 5, 6))
        
    def test_get_last_row(self):
        dataset = DataSet([[1, 2], [3, 4], [5, 6], [7, 8]])
        assert_that(dataset.get_row(dataset.num_samples() - 1), contains(7, 8))

    def test_contruct_dataset_from_dataset(self):
        original = DataSet([[1, 2], [3, 4], [5, 6]])
        new = DataSet(original)
        self.assertFalse(new is original)
        new._dataframe.ix[1] = 1
        assert_that(new, equals_dataset([[1, 2], [1, 1], [5, 6]]))
        assert_that(original, equals_dataset([[1, 2], [3, 4], [5, 6]]))
        
    def test_as_dataset(self):
        original = DataSet([[1, 2], [3, 4], [5, 6]])
        dataset = as_dataset(original)
        self.assertTrue(dataset is original)
        dataset._dataframe.ix[1] = 1
        assert_that(dataset, equals_dataset([[1, 2], [1, 1], [5, 6]]))
        assert_that(original, equals_dataset([[1, 2], [1, 1], [5, 6]]))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    