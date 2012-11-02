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

Allows test workflows to be executed multiple times to average results 
which have random elements.

@author: drusk
"""

import unittest
import base_tests

from loader import load
from classifiers import Knn
from metrics import compute_accuracy

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
        data = load(self.relative("datasets/iris_rearranged.data"), 
                    has_ids=False)
        train, test = data.split(0.8)
        self.assertEqual(train.num_samples(), 120)
        self.assertEqual(test.num_samples(), 30)
        
        knn = Knn(train)
        results = knn.classify_all(test)
        accuracy = results.compute_accuracy()
        self.assertAlmostEqual(accuracy, 0.97, 2)
    
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    