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
Unit tests for the distance_utils module.

@author: drusk
"""

import unittest
import distance_utils

class DistutilsTest(unittest.TestCase):

    def setUp(self):
        self.delta = 0.01

    def test_euclidean_2d(self):
        vector1 = [9, 6]
        vector2 = [5, 3]
        distance = distance_utils.euclidean(vector1, vector2)
        self.assertEqual(distance, 5) 

    def test_euclidean_2d_negative(self):
        vector1 = [-12, 2]
        vector2 = [6, -5]
        distance = distance_utils.euclidean(vector1, vector2)
        self.assertAlmostEqual(distance, 19.313, delta=self.delta)
        
    def test_euclidean_3d(self):
        vector1 = [-4, 9, 6]
        vector2 = [8, -2, -7]
        distance = distance_utils.euclidean(vector1, vector2)
        self.assertAlmostEqual(distance, 20.833, delta=self.delta)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    