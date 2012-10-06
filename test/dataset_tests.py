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
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    