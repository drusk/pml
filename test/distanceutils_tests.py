"""
Unit tests for the distance_utils module.

@author: drusk
"""

import unittest
import distance_utils

class DistutilsTest(unittest.TestCase):

    def setUp(self):
        self.delta = 0.01

    def testEuclidean2d(self):
        vector1 = [9, 6]
        vector2 = [5, 3]
        distance = distance_utils.euclidean(vector1, vector2)
        self.assertEqual(distance, 5) 

    def testEuclidean2dNegative(self):
        vector1 = [-12, 2]
        vector2 = [6, -5]
        distance = distance_utils.euclidean(vector1, vector2)
        self.assertAlmostEqual(distance, 19.313, delta=self.delta)
        
    def testEuclidean3d(self):
        vector1 = [-4, 9, 6]
        vector2 = [8, -2, -7]
        distance = distance_utils.euclidean(vector1, vector2)
        self.assertAlmostEqual(distance, 20.833, delta=self.delta)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    