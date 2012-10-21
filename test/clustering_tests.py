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
Unit tests for the clustering module.

@author: drusk
"""

import unittest

import pandas as pd
from hamcrest import assert_that

import clustering
from model import DataSet
from matchers import in_range

class ClusteringTest(unittest.TestCase):

    def test_create_random_centroids(self):
        dataset = DataSet(pd.DataFrame([[4, 1, 2], [5, 9, 8], [4, 3, 6]], 
                          columns=["a", "b", "c"]))
        k = 2
        centroids = clustering.create_random_centroids(dataset, k)

        self.assertEqual(len(centroids), k)
        for centroid in centroids:
            assert_that(centroid["a"], in_range(4, 5))
            assert_that(centroid["b"], in_range(1, 9))
            assert_that(centroid["c"], in_range(2, 8))

    def test_create_random_centroids_0_to_1(self):
        dataset = DataSet(pd.DataFrame([[0.4, 0.8], [0.75, 0.25], 
                                        [0.99, 0.15]], 
                                       columns=["a", "b"]))
        k = 2
        centroids = clustering.create_random_centroids(dataset, k)

        self.assertEqual(len(centroids), k)
        for centroid in centroids:
            assert_that(centroid["a"], in_range(0.4, 0.99))
            assert_that(centroid["b"], in_range(0.15, 0.8))

    def test_create_random_centroids_negative(self):
        dataset = DataSet(pd.DataFrame([[-4, 1, -2], [5, 9, -8], [4, -3, 6]], 
                          columns=["a", "b", "c"]))
        k = 2
        centroids = clustering.create_random_centroids(dataset, k)
        
        self.assertEqual(len(centroids), k)
        for centroid in centroids:
            assert_that(centroid["a"], in_range(-4, 5))
            assert_that(centroid["b"], in_range(-3, 9))
            assert_that(centroid["c"], in_range(-8, 6))
            

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    