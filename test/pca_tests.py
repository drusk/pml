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
Unit tests for pca module.

@author: drusk
"""

import unittest

import pandas as pd
from hamcrest import assert_that, contains

import pca
from model import DataSet
from matchers import equals_dataset

class PCATest(unittest.TestCase):

    def test_remove_means(self):
        dataset = DataSet([[4, 1, 9], [2, 3, 0], [5, 1, 3]])
        # column means are: 3.6667, 1.6667, 4
        pca.remove_means(dataset)
        assert_that(dataset, equals_dataset([[0.33, -0.67, 5], 
                                             [-1.67, 1.33, -4], 
                                             [1.33, -0.67, -1]], 
                                            places=2))

    def test_otago_example(self):
        # example data from Otago university tutorial
        dataset = DataSet([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], 
                           [3.1, 3.0], [2.3, 2.7], [2, 1.6], [1, 1.1], 
                           [1.5, 1.6], [1.1, 0.9]])
        transformed = [[-0.828, -0.175], [1.778, 0.143], [-0.992, 0.384], 
                       [-0.274, 0.130], [-1.676, -0.209], [-0.913, 0.175], 
                       [0.099, -0.350], [1.145, 0.046], [0.4380, 0.018], 
                       [1.224, -0.163]]
        
        principal_components = pca.pca(dataset, 2)
        assert_that(principal_components, equals_dataset(transformed, 
                                                         places=2))
        
    def test_reduced_dataset_has_row_indices(self):
        dataset = DataSet(pd.DataFrame([[1, 2], [3, 4], [5, 6]], 
                          index=["Cat", "Dog", "Rat"]))
        reduced = pca.pca(dataset, 2)
        assert_that(reduced.get_sample_ids(), contains("Cat", "Dog", "Rat"))
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    