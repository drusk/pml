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
Unit tests for the loader module.

@author: drusk
"""

import unittest
import base_tests
import loader
from hamcrest import assert_that
from matchers import equals_series

class LoaderTest(base_tests.BaseFileLoadingTest):

    def test_load_csv(self):
        data_set = loader.load(self.relative("datasets/3f_header.csv"), 
                               has_ids=False)
        self.assertEqual(data_set.num_features(), 3)
        self.assertEqual(data_set.num_samples(), 4)

    def test_load_csv_no_header(self):
        data_set = loader.load(self.relative("datasets/3f_no_header.csv"), 
                               has_header=False, has_ids=False)
        self.assertEqual(data_set.num_features(), 3)
        self.assertEqual(data_set.num_samples(), 4)
    
    def test_load_tsv(self):
        data_set = loader.load(self.relative("datasets/3f_header.tsv"), 
                               delimiter="\t", has_ids=False)
        self.assertEqual(data_set.num_features(), 3)
        self.assertEqual(data_set.num_samples(), 4)
    
    @unittest.skip("Filtering unimplemented")
    def test_load_and_filter_to_single_value(self):
        data_set = loader.load(self.relative("datasets/3f_header.csv"))
        filtered = data_set.filter("y", 3)
        self.assertEqual(filtered.num_samples(), 1)
        self.assertEqual(filtered.get("x"), 1)
        self.assertEqual(filtered.get("y"), 3)
        self.assertEqual(filtered.get("label"), "b")
    
    @unittest.skip("Filtering unimplemented")
    def test_load_and_filter_to_multiple_values(self):
        data_set = loader.load(self.relative("datasets/3f_header.csv"))
        filtered = data_set.filter("x", 1)
        self.assertEqual(filtered.num_samples(), 2)
        self.assertEqual(filtered.num_features(), 3)

    def test_load_has_ids(self):
        dataset = loader.load(self.relative("datasets/3f_ids_header.csv"))
        self.assertEqual(dataset.num_features(), 3)
        self.assertEqual(dataset.num_samples(), 4)
        
    def test_load_labelled(self):
        dataset = loader.load(self.relative("datasets/3f_ids_header.csv"))
        self.assertTrue(dataset.is_labelled())
        labels = dataset.get_labels()
        assert_that(labels, equals_series({"V01": "c", "V02": "b", "V03": "b", 
                                           "V04": "a"}))
        
    def test_load_unlabelled(self):
        dataset = loader.load(self.relative("datasets/3f_ids_header_no_labels.csv"), 
                              has_labels=False)
        self.assertFalse(dataset.is_labelled())
        self.assertEqual(dataset.num_features(), 3)
        self.assertTrue(dataset.get_labels() is None)
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    