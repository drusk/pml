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

from pml.data import loader
from pml.data.model import DataSet
from pml.supervised.naive_bayes import NaiveBayes

import base_tests

class NaiveBayesTest(base_tests.BaseFileLoadingTest):

    def load_car_data(self):
        """
        Loads an example training set and related sample.
         
        NOTE: pandas automatically converts columns with just 'yes' and 'no' 
        values into booleans.
        """
        training_set = loader.load(self.relative("datasets/car_thefts.data"))
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
        prob = classifier._calc_prob_feature_given_class(True, "color", sample["color"])
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

    # TODO test actual probabilities generated
    

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    