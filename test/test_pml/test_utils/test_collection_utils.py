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
Unit tests for collection_utils module.

@author: drusk
"""

import unittest

from pml.utils import collection_utils

class CollectionUtilsTest(unittest.TestCase):

    def test_get_key_with_highest_value(self):
        dictionary = {"dog": 5, "cat": 10, "bird": 7}
        key = collection_utils.get_key_with_highest_value(dictionary)
        self.assertEqual(key, "cat")

    def test_get_key_with_highest_value_float(self):
        dictionary = {0: 0.10, 1: 0.0567, 2: 0.72}
        key = collection_utils.get_key_with_highest_value(dictionary)
        self.assertEqual(key, 2)
       
    def test_get_key_with_highest_value_empty(self):
        dictionary = {}
        self.assertIsNone(
                collection_utils.get_key_with_highest_value(dictionary))
        
    def test_are_all_equal_empty(self):
        iterable = []
        self.assertTrue(collection_utils.are_all_equal(iterable))
    
    def test_are_all_equal_one_element(self):
        iterable = ['a']
        self.assertTrue(collection_utils.are_all_equal(iterable))
    
    def test_are_all_equal(self):
        iterable1 = ['a', 'b']
        self.assertFalse(collection_utils.are_all_equal(iterable1))
        
        iterable2 = ['a', 'b', 'b']
        self.assertFalse(collection_utils.are_all_equal(iterable2))
        
        iterable3 = ['b', 'b']
        self.assertTrue(collection_utils.are_all_equal(iterable3))

        iterable4 = ['b', 'b', 'b']
        self.assertTrue(collection_utils.are_all_equal(iterable4))
    
    def test_get_most_common(self):
        collection = ["a", "b", "a", "a", "b"]
        self.assertEqual(collection_utils.get_most_common(collection), "a")
    
    def test_get_most_common_empty(self):
        collection = []
        self.assertIsNone(collection_utils.get_most_common(collection))
    

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()