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
Unit tests for naive_bayes module.

@author: drusk
"""

import unittest

import pandas as pd
from hamcrest import assert_that

from pml.data import loader
from pml.data.model import DataSet
from pml.supervised.naive_bayes import NaiveBayes
from pml.utils.errors import InconsistentFeaturesError

from test import base_tests
from test.matchers.matchers import equals_series

class NaiveBayesTest(base_tests.BaseFileLoadingTest):

    def load_car_data(self):
        """
        Loads an example training set and related sample.
         
        NOTE: pandas automatically converts columns with just 'yes' and 'no' 
        values into booleans.
        """
        training_set = loader.load(self.relative_to_base("datasets/"
                                                         "car_thefts.data"))
        sample = {
                  "color": "red",
                  "type": "suv",
                  "origin": "domestic"
                  }
        return training_set, sample

    def test_count_examples(self):
        training_set, sample = self.load_car_data()
        classifier = NaiveBayes(training_set)
        count = classifier._count_examples(True, "color", sample["color"])
        self.assertEquals(count, 3)

    def test_calc_prob_feature_given_class(self):
        training_set, sample = self.load_car_data()
        classifier = NaiveBayes(training_set)
        prob = classifier._calc_prob_feature_given_class(True, "color", 
                                                         sample["color"])
        self.assertAlmostEqual(prob, 0.57, places=2)
        
    def test_calc_prob_class(self):
        training_set = DataSet([[1, 2], [3, 4], [5, 6], [7, 8]],
                                    labels=["cat", "dog", "cat", "cat"])
        classifier = NaiveBayes(training_set)
        prob = classifier._calc_prob_class("cat")
        self.assertAlmostEqual(prob, 0.75, places=2)

    def test_classify_single_sample(self):
        training_set, sample = self.load_car_data()
        classifier = NaiveBayes(training_set)
        self.assertFalse(classifier.classify(sample))

    def test_get_classification_probabilities(self):
        training_set, sample = self.load_car_data()
        classifier = NaiveBayes(training_set)
        probabilities = classifier.get_classification_probabilities(sample)
        self.assertAlmostEqual(probabilities[True], 0.035, places=3)
        self.assertAlmostEqual(probabilities[False], 0.070, places=3)
    
    def test_classify_all(self):
        training_set, sample_0 = self.load_car_data()
        sample_1 = {"color": "yellow", "type": "sports", "origin": "domestic"}
        dataset = DataSet(pd.DataFrame([sample_0, sample_1]), 
                          labels=[False, True])
        classifier = NaiveBayes(training_set)
        results = classifier.classify_all(dataset)
        assert_that(results.get_classifications(), equals_series({0: False, 1: False}))
        self.assertEqual(results.compute_accuracy(), 0.5)
    
    def test_classify_inconsistent_features(self):
        training_set, __ = self.load_car_data()
        sample = {"color": "yellow", "type": "sports", "year": 2012}
        classifier = NaiveBayes(training_set)
        self.assertRaises(InconsistentFeaturesError, classifier.classify, sample)
    

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    