"""
Unit tests for the loader module.

@author: drusk
"""

import unittest
import base_tests
import loader

class LoaderTest(base_tests.BaseFileLoadingTest):

    def testLoadCsv(self):
        data_set = loader.load(self.relative("datasets/3f_header.csv"))
        self.assertEqual(data_set.num_features(), 3)
        self.assertEqual(data_set.num_samples(), 4)

    def testLoadCsvNoHeader(self):
        data_set = loader.load(self.relative("datasets/3f_no_header.csv"))
        self.assertEqual(data_set.num_features(), 3)
        self.assertEqual(data_set.num_samples(), 4)
    
    def testLoadAndFilterToSingleValue(self):
        data_set = loader.load(self.relative("datasets/3f_header.csv"))
        filtered = data_set.filter("y", 3)
        self.assertEqual(filtered.num_samples(), 1)
        self.assertEqual(filtered.get("x"), 1)
        self.assertEqual(filtered.get("y"), 3)
        self.assertEqual(filtered.get("label"), "b")
    
    def testLoadAndFilterToMultipleValues(self):
        data_set = loader.load(self.relative("datasets/3f_header.csv"))
        filtered = data_set.filter("x", 1)
        self.assertEqual(filtered.num_samples(), 2)
        self.assertEqual(filtered.num_features(), 3)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    