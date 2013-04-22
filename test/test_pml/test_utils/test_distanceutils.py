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

from pml.utils.distance_utils import euclidean
from pml.utils.distance_utils import cosine_similarity

class DistutilsTest(unittest.TestCase):

    def setUp(self):
        self.delta = 0.01

    def test_euclidean_2d(self):
        vector1 = [9, 6]
        vector2 = [5, 3]
        distance = euclidean(vector1, vector2)
        self.assertEqual(distance, 5) 

    def test_euclidean_2d_negative(self):
        vector1 = [-12, 2]
        vector2 = [6, -5]
        distance = euclidean(vector1, vector2)
        self.assertAlmostEqual(distance, 19.313, delta=self.delta)
        
    def test_euclidean_3d(self):
        vector1 = [-4, 9, 6]
        vector2 = [8, -2, -7]
        distance = euclidean(vector1, vector2)
        self.assertAlmostEqual(distance, 20.833, delta=self.delta)

    def test_cosine_similarity_4d(self):
        """
        http://bioinformatics.oxfordjournals.org/content/suppl/2009/10/24/
        btp613.DC1/bioinf-2008-1835-File004.pdf
        NOTE: the answer they give is slightly off by about 0.013 (easily
        verifiable with hand calculator).
        """
        vector1 = [1, 1, 0, 1]
        vector2 = [2, 0, 1, 1]
        distance = cosine_similarity(vector1, vector2)
        self.assertAlmostEqual(distance, 0.707, delta=self.delta)

    def test_cosine_similarity_right_angles(self):
        vector1 = [0, 1]
        vector2 = [1, 0]
        distance = cosine_similarity(vector1, vector2)
        self.assertAlmostEqual(distance, 0.0, delta=self.delta)

    def test_cosine_similarity_parallel(self):
        vector1 = [1, 1, 1, 1]
        vector2 = [3, 3, 3, 3]
        distance = cosine_similarity(vector1, vector2)
        self.assertAlmostEqual(distance, 1.0, delta=self.delta)

    def test_cosine_similarity_zeros(self):
        vector1 = [1, 1, 1]
        vector2 = [0, 0, 0]
        distance = cosine_similarity(vector1, vector2)
        self.assertAlmostEqual(distance, 0.0, delta=self.delta)

    def test_negative_similarity(self):
        vector1 = [1, 0, -3]
        vector2 = [1, 1, 2]
        distance = cosine_similarity(vector1, vector2)
        self.assertAlmostEqual(distance, -0.645497, delta=self.delta)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    