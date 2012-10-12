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
import classifiers
from hamcrest import assert_that, contains

class KnnTest(unittest.TestCase):

    def testTwoClassesNoTie(self):
        training_set = [[8, 4, "b"], [8, 5, "c"], [8, 6, "c"], [9, 4, "b"], 
                        [9, 6, "c"]]
        classifier = classifiers.Knn(training_set, k=5)
        result = classifier.classify([9, 5])
        self.assertEqual(result, "c")
    
    def testThreeClassesNoTie(self):
        training_set = [[1, 0.5, "a"], [1, 2, "a"], [1, 3, "a"], [2, 0.5, "a"],
                         [2, 2.5, "b"], [2, 3, "a"], [2.5, 1.5, "c"], 
                         [3, 1, "a"]]
        classifier = classifiers.Knn(training_set, k=5)
        result = classifier.classify([2, 1.5])
        self.assertEqual(result, "a")
    
    @unittest.skip("Undecided on best tie breaking strategy")
    def testTie(self):
        training_set = [[1, 7, "c"], [1.5, 6.5, "b"], [1.5, 8, "a"], 
                        [2.5, 6.5, "b"], [2.5, 8, "a"]]
        classifier = classifiers.Knn(training_set, k=5)
        result = classifier.classify([2, 7])
        self.assertEqual(result, "b")
        
    def testKGreaterThanNumSamples(self):
        training_set = [[1, 1, "a"], [1, 3, "a"], [3, 2, "b"]]
        classifier = classifiers.Knn(training_set, k=7)
        result = classifier.classify([2, 2])
        self.assertEqual(result, "a")
        
    def testClassifyAll(self):
        training_set = [[1, 1, "a"], [2, 2, "a"], [11, 11, "b"], [12, 12, "b"]]
        classifier = classifiers.Knn(training_set, k=3)
        dataset = [[1.5, 1.3], [12.2, 12.9]]
        classes = classifier.classify_all(dataset)
        assert_that(classes, contains("a", "b"))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    