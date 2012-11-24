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

from pml.unsupervised import clustering
from pml.data.model import DataSet
from pml.utils.errors import UnlabelledDataSetError

from matchers import in_range, equals_series

class ClusteringTest(unittest.TestCase):

    def create_data(self, n_rows, n_columns):
        """
        Creates data when you don't care what the contents are.
        
        Args:
          n_rows: int
            The number of rows in the data.
          n_columns: int
            The number of columns in the data.
            
        Returns:
          A Python list of lists with arbitrary data in the shape provided.  
          Can then be passed to a DataSet constructor.
        """
        return [[i + j for j in range(n_columns)] for i in range(n_rows)]

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
            
    def test_get_distances_to_centroids(self):
        dataset = DataSet([[1, 5], [2, 1], [6, 5]])
        centroids = [pd.Series([4, 5]), pd.Series([6, 2])]
        
        results = clustering._get_distances_to_centroids(dataset, centroids)
        
        self.assertEqual(results.shape, (3, 2))
        # TODO: really need a DataFrame matcher...
        expected = [[3, 5.83], [4.47, 4.12], [2, 3]]
        for i, row in enumerate(expected):
            for j, element in enumerate(row):
                self.assertAlmostEqual(results.ix[i].tolist()[j], element, 
                                       places=2)
    
    def test_compute_iteration(self):
        dataset = DataSet([[1, 5], [2, 1], [6, 5]])
        centroids = [pd.Series([4, 5]), pd.Series([6, 2])]
        
        new_cent, clusters = clustering._compute_iteration(dataset, centroids)
        
        expected_cent = [{0: 3.5, 1: 5}, {0: 2, 1: 1}]
        
        self.assertEqual(len(new_cent), len(expected_cent))
        for i, cent in enumerate(new_cent):
            assert_that(cent, equals_series(expected_cent[i]))
        
        assert_that(clusters, equals_series({0: 0, 1: 1, 2: 0}))
            
    def test_kmeans_k_3(self):
        dataset = DataSet([[3, 13], [5, 13], [2, 11], [4, 11], [6, 11], 
                           [8, 5], [5, 3], [6, 2], [9, 2], [16, 14], [18, 13], 
                           [16, 11], [19, 10]])
        
        def create_centroids(dataset, k):
            # Want pre-set centroids (not random) so that test outcome is known
            # Therefore ignore input parameters.
            return [pd.Series([4, 9]), pd.Series([10, 6]), pd.Series([17, 9])]
        
        clustered = clustering.kmeans(dataset, k=3, 
                                      create_centroids=create_centroids)
        assert_that(clustered.get_cluster_assignments(), equals_series({0: 0, 
                                            1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 
                                            6: 1, 7: 1, 8: 1, 9: 2, 10: 2, 
                                            11: 2, 12: 2}))
    
    def test_calculate_purity(self):
        # use example from http://nlp.stanford.edu/IR-book/html/htmledition/
        # evaluation-of-clustering-1.html
        data = self.create_data(17, 2)
        unclustered_dataset = DataSet(data,
            labels=["x", "x", "o", "x", "x", "x", "x", "o", "d", "o", "o", 
                    "o", "x", "d", "d", "d", "x"])
        cluster_assignments = pd.Series([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 
                                         3, 3, 3, 3, 3])
        
        clustered_dataset = clustering.ClusteredDataSet(unclustered_dataset, 
                                                        cluster_assignments)
        
        self.assertAlmostEqual(clustered_dataset.calculate_purity(), 0.71, 2)
        
    def test_calculate_purity_unlabelled_dataset(self):
        data = self.create_data(17, 2)
        unclustered_dataset = DataSet(data) # NOTE: no labels
        cluster_assignments = pd.Series([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 
                                         3, 3, 3, 3, 3])
        
        clustered_dataset = clustering.ClusteredDataSet(unclustered_dataset, 
                                                        cluster_assignments)
        
        self.assertRaises(UnlabelledDataSetError, 
                          clustered_dataset.calculate_purity)
    
    def test_calculate_rand_index(self):
        data = self.create_data(17, 2)
        unclustered_dataset = DataSet(data,
            labels=["x", "x", "o", "x", "x", "x", "x", "o", "d", "o", "o", 
                    "o", "x", "d", "d", "d", "x"])
        cluster_assignments = pd.Series([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 
                                         3, 3, 3, 3, 3])
        
        clustered_dataset = clustering.ClusteredDataSet(unclustered_dataset, 
                                                        cluster_assignments)
        
        self.assertAlmostEqual(clustered_dataset.calculate_rand_index(), 0.68, 
                               2)
        
    def test_calculate_rand_index_unlabelled_dataset(self):
        data = self.create_data(17, 2)
        unclustered_dataset = DataSet(data) # NOTE: no labels
        cluster_assignments = pd.Series([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 
                                         3, 3, 3, 3, 3])
        
        clustered_dataset = clustering.ClusteredDataSet(unclustered_dataset, 
                                                        cluster_assignments)
        
        self.assertRaises(UnlabelledDataSetError, 
                          clustered_dataset.calculate_rand_index)
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    