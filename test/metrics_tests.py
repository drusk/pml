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
Unit tests for metrics module.

@author: drusk
"""

import unittest

import pandas as pd

from pml.data.model import DataSet
from pml.supervised.classifiers import ClassifiedDataSet
from pml.utils.errors import UnlabelledDataSetError

class MetricsTest(unittest.TestCase):

    def create_dataset(self, labels=None, sample_ids=None):
        """
        Used to create a DataSet for testing purposes with optional labels 
        and sample_ids.
        """
        raw_data = [[1, 2, 3] for _ in xrange(len(labels))]
        df = pd.DataFrame(raw_data, index=sample_ids)
        return DataSet(df, labels=labels)

    def test_accuracy_integer_index(self):
        dataset = self.create_dataset(labels=["a", "b", "c", "b"])
        classified = ClassifiedDataSet(dataset, 
                                       pd.Series(["a", "b", "c", "a"]))
        self.assertAlmostEqual(classified.compute_accuracy(), 0.75, places=2)
    
    def test_accuracy_string_index(self):
        sample_ids = ["V01", "V02", "V03", "V04"]
        results = pd.Series(["a", "b", "c", "a"], index=sample_ids)
        labels = pd.Series(["a", "b", "c", "b"], index=sample_ids)
        dataset = self.create_dataset(labels, sample_ids=sample_ids)
        classified = ClassifiedDataSet(dataset, results)
        self.assertAlmostEqual(classified.compute_accuracy(), 0.75, places=2)
    
    @unittest.skip("This test will no longer be needed, leaving as a reminder"
                   " to add index checks in DataSet constructors (and test)")
    def test_accuracy_unequal_lengths(self):
        results = pd.Series(["a", "b", "c", "a"], 
                            index=["V01", "V02", "V03", "V04"])
        labels = pd.Series(["a", "b", "c"], 
                           index=["V01", "V02", "V03"])
        dataset = self.create_dataset(labels=labels)
        classified = ClassifiedDataSet(dataset, results)
        self.assertRaises(ValueError, classified.compute_accuracy)
    
    def test_accuracy_dataset_unlabelled(self):
        results = pd.Series(["a", "b", "c", "a"], 
                            index=["V01", "V02", "V03", "V04"])
        dataset = DataSet([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        classified = ClassifiedDataSet(dataset, results)
        self.assertRaises(UnlabelledDataSetError, classified.compute_accuracy)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    