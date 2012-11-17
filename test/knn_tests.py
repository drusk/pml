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
Unit tests for the KNN algorithm.

@author: drusk
"""

import unittest

from hamcrest import assert_that, contains

from pml.supervised import classifiers
from pml.data.model import DataSet
from pml.utils.errors import UnlabelledDataSetError

class KnnTest(unittest.TestCase):

    def test_two_classes_no_tie(self):
        training_set = DataSet([[8, 4], [8, 5], [8, 6], [9, 4], [9, 6]], 
                               labels=["b", "c", "c", "b", "c"])
        classifier = classifiers.Knn(training_set, k=5)
        result = classifier.classify([9, 5])
        self.assertEqual(result, "c")
    
    def test_three_classes_no_tie(self):
        training_set = DataSet([[1, 0.5], [1, 2], [1, 3], [2, 0.5], [2, 2.5], 
                                [2, 3], [2.5, 1.5], [3, 1]], 
                               labels=["a", "a", "a", "a", "b", "a", "c", "a"])
        classifier = classifiers.Knn(training_set, k=5)
        result = classifier.classify([2, 1.5])
        self.assertEqual(result, "a")
    
    @unittest.skip("Undecided on best tie breaking strategy")
    def test_tie(self):
        training_set = DataSet([[1, 7], [1.5, 6.5], [1.5, 8], [2.5, 6.5], 
                                [2.5, 8]], labels=["c", "b", "a", "b", "a"])
        classifier = classifiers.Knn(training_set, k=5)
        result = classifier.classify([2, 7])
        self.assertEqual(result, "b")
        
    def test_k_greater_than_num_samples(self):
        training_set = DataSet([[1, 1], [1, 3], [3, 2]], 
                               labels=["a", "a", "b"])
        classifier = classifiers.Knn(training_set, k=7)
        result = classifier.classify([2, 2])
        self.assertEqual(result, "a")
        
    def test_classify_all(self):
        training_set = DataSet([[1, 1], [2, 2], [11, 11], [12, 12]], 
                               labels=["a", "a", "b", "b"])
        classifier = classifiers.Knn(training_set, k=3)
        dataset = [[1.5, 1.3], [12.2, 12.9]]
        classes = classifier.classify_all(dataset).get_classifications()
        assert_that(classes, contains("a", "b"))
        
    def test_create_knn_unlabelled_raises_exception(self):
        training_set = DataSet([[1, 2], [3, 4]])
        self.assertRaises(UnlabelledDataSetError, classifiers.Knn, 
                          training_set)
        
    def test_classify_incomplete_sample(self):
        training_set = DataSet([[1, 2, 3], [4, 5, 6]], labels=["a", "a"])
        classifier = classifiers.Knn(training_set)
        self.assertRaises(ValueError, classifier.classify, [1, 2])


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    