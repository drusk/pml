"""
Unit tests for the KNN algorithm.

@author: drusk
"""

import unittest
import classifiers

class KnnTest(unittest.TestCase):

    def testTwoClassesNoTie(self):
        training_set = [[8, 4, "b"], [8, 5, "c"], [8, 6, "b"], [9, 4, "b"], 
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
    
    def testTie(self):
        training_set = [[1, 7, "c"], [1.5, 6.5, "b"], [1.5, 8, "a"], 
                        [2.5, 6.5, "b"], [2.5, 8, "a"]]
        classifier = classifiers.Knn(training_set, k=5)
        result = classifier.classify([2, 7])
        self.assertEqual(result, "b")
        
    def testKLessThanSamples(self):
        training_set = [[1, 1, "a"], [1, 3, "a"], [3, 2, "b"]]
        classifier = classifiers.Knn(training_set, k=7)
        result = classifier.classify([2, 2])
        self.assertEqual(result, "a")

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    