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
Tests for the DataSet class.

@author: drusk
"""

import unittest
from loader import DataSet

class DataSetTest(unittest.TestCase):

    def testReduceRows(self):
        dataset = DataSet.from_list([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        reduced = dataset.reduce_rows(sum)
        expected = [6, 15, 24]
        
        self.assertEqual(len(reduced.values), len(expected))
        
        # TODO collections matchers
        for i in range(len(expected)):
            self.assertEqual(reduced.values[i], expected[i])
        
    def testDropColumn(self):
        original_dataset = DataSet.from_list([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertEqual(original_dataset.num_features(), 3)
        
        filtered_dataset = original_dataset.drop_column(1)
        self.assertEqual(filtered_dataset.num_features(), 2)
        
        self.assertEqual(original_dataset.num_features(), 3)
        
        # TODO should test the exact values as well.  Maybe have an as_list 
        # method and pass results to a collections matcher?
        
    def testGetColumn(self):
        dataset = DataSet.from_list([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        column1 = dataset.get_column(1)
        
        expected = [2, 5, 8]
        self.assertEqual(len(column1.values), len(expected))
        
        # TODO collections matchers
        for i in range(len(expected)):
            self.assertEqual(column1.values[i], expected[i])
    
    def testGetColumnSetValue(self):
        dataset = DataSet.from_list([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        new_val = 1
        dataset.get_column(1)[:] = new_val
        
        # check that original dataset has been altered
        # TODO collections matchers
        column1 = dataset.get_column(1)
        for i in range(len(column1)):
            self.assertEqual(column1[i], new_val)
        
    def testGetRows(self):
        dataset = DataSet.from_list([[1, 2], [3, 4], [5, 6], [7, 8]])
        selection = dataset.get_rows([1, 3])
        
        self.assertEqual(selection.num_samples(), 2)
        # TODO DataSet matcher?  need to verify it is the right 2 rows...
        
    def testSplit(self):
        dataset = DataSet.from_list([[1, 2], [3, 4], [5, 6], [7, 8]])
        first, second = dataset.split(0.5)
        self.assertEqual(first.num_samples(), 2)
        self.assertEqual(second.num_samples(), 2)
        #TODO: check exact elements with collections matcher
        
    def testUnequalSplit(self):
        dataset = DataSet.from_list([[1, 2], [3, 4], [5, 6], [7, 8]])
        first, second = dataset.split(0.3)
        self.assertEqual(first.num_samples(), 1)
        self.assertEqual(second.num_samples(), 3)
        #TODO: check exact elements with collections matcher

    def testSplit0(self):
        dataset = DataSet.from_list([[1, 2], [3, 4], [5, 6], [7, 8]])
        first, second = dataset.split(0)
        self.assertEqual(first.num_samples(), 0)
        self.assertEqual(second.num_samples(), 4)
        #TODO: check exact elements with collections matcher

    def testSplitInvalidPercent(self):
        dataset = DataSet.from_list([[1, 2], [3, 4], [5, 6], [7, 8]])
        self.assertRaises(ValueError, dataset.split, 50)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    