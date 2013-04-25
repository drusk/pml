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
Unit tests for pandas_util module.

@author: drusk
"""

import unittest

import pandas as pd
from hamcrest import assert_that, contains

from pml.utils import pandas_util

class PandasUtilTest(unittest.TestCase):

    def test_df_same_index_same_elements(self):
        df1 = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        df2 = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        self.assertTrue(pandas_util.are_dataframes_equal(df1, df2))
        
    def test_df_same_index_different_elements(self):
        df1 = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        df2 = pd.DataFrame([[1, 3, 3], [4, 5, 6]])
        self.assertFalse(pandas_util.are_dataframes_equal(df1, df2))
        
    def test_df_different_shape(self):
        df1 = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        df2 = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
        self.assertFalse(pandas_util.are_dataframes_equal(df1, df2))

    def test_find_one_value(self):
        series = pd.Series(["friendly", "friendly", "not_friendly"], 
                           index=["cat", "dog", "mouse"])
        indices = pandas_util.find(series, "friendly")
        assert_that(indices, contains("cat", "dog"))

    def test_is_series_numeric_actual_is_int(self):
        series = pd.Series([0, 1, 2])
        self.assertTrue(pandas_util.is_series_numeric(series))
    
    def test_is_series_numeric_actual_is_float(self):
        series = pd.Series([0.1, 1.2, 2.3])
        self.assertTrue(pandas_util.is_series_numeric(series))
    
    def test_is_series_numeric_actual_is_string(self):
        series = pd.Series(["low", "mid", "high"])
        self.assertFalse(pandas_util.is_series_numeric(series))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()