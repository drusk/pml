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
Integration tests which verify the different parts of the system work together 
properly.  These tests should be in the form of a user workflow.

@author: drusk
"""

import unittest

import pandas as pd
from hamcrest import assert_that

from pml.api import *

import base_tests
from matchers.pml_matchers import equals_dataset

class IntegrationTest(base_tests.BaseFileLoadingTest):
    
    def test_knn_classify_iris(self):
        """
        NOTE: if data was split randomly, the accuracy would fluctuate with 
        each run, making it difficult to put a reasonable bound on the 
        assertion.  Therefore, the iris.data set was rearranged (and named 
        iris_rearranged.data) to have a good mix of training data first and 
        then test data second which can be cleanly split.
        
        iris_rearranged.data has 150 samples with 4 features.
        It has headers but no ids.  No missing data.
        """
        data = load(self.relative_to_base("datasets/iris_rearranged.data"), 
                    has_ids=False)
        train, test = data.split(0.8)
        self.assertEqual(train.num_samples(), 120)
        self.assertEqual(test.num_samples(), 30)
        
        knn = Knn(train)
        results = knn.classify_all(test)
        self.assertAlmostEqual(results.compute_accuracy(), 0.97, 2)
    
    def test_naive_bayes_classify_iris(self):
        data = load(self.relative_to_base("datasets/iris_rearranged.data"),
                    has_ids=False)
        train, test = data.split(0.7)
        
        classifier = NaiveBayes(train)
        results = classifier.classify_all(test)
        self.assertAlmostEqual(results.compute_accuracy(), 0.93, 2)
    
    
    @unittest.skip("XXX: separate running unit tests from integration tests")
    def test_pca_semiconductor_variance(self):
        """
        This takes around 10 seconds due to 1500 samples by 590 features.
        Create a separate run script/config for unit tests only that can be 
        run frequently, and then these slower integration tests can just be 
        run before checkins.
        """
        data = load(self.relative_to_base("datasets/semiconductor.data"),
                    has_ids=False, has_labels=False, has_header=False, 
                    delimiter=" ")
        data.fill_missing_with_feature_means()
        
        variances = get_pct_variance_per_principal_component(data)
        expected = [0.593, 0.241, 0.092, 0.023, 0.015, 0.005, 0.003]
        for i in range(7):
            self.assertAlmostEqual(variances[i], expected[i], places=3)

    def test_pca_ingredients(self):
        """
        Verifies against MATLAB example:
        http://www.mathworks.com/help/stats/princomp.html
        
        Note that the MATLAB docs have the sign backwards on the example 
        output for pc (in all the columns except the leftmost).  This can 
        be verified trivially by running the example in MATLAB.
        """
        data = load(self.relative_to_base("datasets/ingredients.data"),
                    has_ids=False, has_labels=False, has_header=False)
        
        reduced = pca(data, 4)
        expected = load(self.relative_to_base("datasets/pca_ingredients.data"),
                        has_ids=False, has_labels=False, has_header=False)
        
        # TODO matcher that accepts another DataSet as the expected value
        expected_as_list = [expected.get_row(index).tolist() 
                            for index in expected.get_sample_ids()]
        assert_that(reduced, equals_dataset(expected_as_list, places=1))
        
        # TODO numpy array matcher
        assert_that(
            DataSet(pd.DataFrame(reduced.get_weights())), 
            equals_dataset([[0.0678, 0.6460, -0.5673, 0.5062],
                            [0.6785, 0.0200, 0.5440, 0.4933],
                            [-0.0290, -0.7553, -0.4036, 0.5156],
                            [-0.7309, 0.1085, 0.4684, 0.4844]], 
                            places=4)
        )
        
        # TODO: generic sequence almost equals matcher
        expected_eigenvalues = [517.7969, 67.4964, 0.2372, 12.4054]
        eigenvalues = reduced.get_eigenvalues()
        for i, expected_eigenvalue in enumerate(expected_eigenvalues):
            self.assertAlmostEqual(eigenvalues[i], expected_eigenvalue, 
                                   places=4)
            
    def create_iris_preset_centroids(self, dataset, k):
        assert k == 3
        
#        index = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        index = dataset.feature_list()
        return [pd.Series([4.5, 3, 1.5, 1], index=index),
                pd.Series([6.8, 3.5, 5.5, 2], index=index),
                pd.Series([5.2, 2.8, 1.5, 0.5], index=index)]
    
    def test_kmeans_vs_matlab(self):
        """
        Checks kmeans clusters against the results achieved by MATLAB's 
        kmeans function.
        
        The only modification made to the MATLAB output is reducing the 
        cluster numbers by 1 (they started at 1 instead of 0).
        
        Note that the initial centroids have a fixed location to eliminate
        the random factor that would make comparison difficult.
        """
        data = load(self.relative_to_base("datasets/iris.data"), 
                    has_ids=False)
        ml_clusters = pd.read_csv(self.relative_to_base(
                                  "datasets/mlab_iris_clusters.data"),
                                  header=None)
        # Cluster assignments were loaded into a DataFrame, slice them out as
        # a Series for convenience.
        ml_clusters = ml_clusters.ix[:, 0]

        clustered = kmeans(data, 3, 
                        create_centroids=self.create_iris_preset_centroids)
        
        pml_clusters = clustered.get_cluster_assignments()
        for index in pml_clusters.index:
            self.assertEqual(pml_clusters[index], ml_clusters[index], 
                             msg="Cluster discrepancy at index %d" % index)

    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    