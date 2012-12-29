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
Unit tests for matcher module used for testing.

@author: drusk
"""

import unittest

import numpy as np
from hamcrest import assert_that

from test.matchers import pml_matchers
from test.matchers.pml_matchers import equals_tree
from pml.supervised.decision_trees.trees import Node, Tree

class MatchersTest(unittest.TestCase):

    def test_equals_places_none(self):
        list1 = [1.1, 2.2, 3.3]
        list2 = [1.1, 2.2, 3.3]
        self.assertTrue(pml_matchers.lists_match(list1, list2))

    def test_equals_nan(self):
        list1 = [1.1, np.NaN, 3.3]
        list2 = [1.1, np.NaN, 3.3]
        self.assertTrue(pml_matchers.lists_match(list1, list2))

    def test_equals_places_2(self):
        list1 = [1.101, 2.202, 3.303]
        list2 = [1.102, 2.199, 3.301]
        self.assertTrue(pml_matchers.lists_match(list1, list2, places=2))

    def create_tree_tennis(self):
        """
        Creates a tree matching the play_tennis.data data's decision tree.
        """
        root_node = Node("Outlook")
        
        humidity_node = Node("Humidity")
        high_humidity_node = Node("No")
        normal_humidity_node = Node("Yes")
        humidity_node.add_child("High", high_humidity_node)
        humidity_node.add_child("Normal", normal_humidity_node)
        root_node.add_child("Sunny", humidity_node)
        
        overcast_node = Node("Yes")
        root_node.add_child("Overcast", overcast_node)
        
        wind_node = Node("Wind")
        strong_wind_node = Node("No")
        weak_wind_node = Node("Yes")
        wind_node.add_child("Strong", strong_wind_node)
        wind_node.add_child("Weak", weak_wind_node)
        root_node.add_child("Rain", wind_node)
        
        return Tree(root_node)

    def test_equals_tree(self):
        assert_that(self.create_tree_tennis(), 
            equals_tree(
                {"Outlook": {
                    "Sunny": {
                        "Humidity": {
                            "High": "No",
                            "Normal": "Yes"
                        }
                    },
                    "Overcast": "Yes",
                    "Rain": {
                        "Wind": {
                            "Strong": "No",
                            "Weak": "Yes"
                        }
                    }
                }}
            )
        )

    def test_equals_tree_wrong_leaf_val(self):
        """
        Test that the matcher picks up on the expected value not being right.
        """
        matcher = pml_matchers.IsTree(
            {"Outlook": {
                    "Sunny": {
                        "Humidity": {
                            "High": "No",
                            "Normal": "Yes"
                        }
                    },
                    "Overcast": "Yes",
                    "Rain": {
                        "Wind": {
                            "Strong": "Yes", # Changed
                            "Weak": "Yes"
                        }
                    }
                }}
        )
        self.assertFalse(matcher.matches(self.create_tree_tennis()))
        
    def test_equals_tree_wrong_branch_val(self):
        """
        Test that the matcher picks up on the expected value not being right.
        """
        matcher = pml_matchers.IsTree(
            {"Outlook": {
                    "Sunny": {
                        "Humidity": {
                            "High": "No",
                            "Normal": "Yes"
                        }
                    },
                    "Overcastt": "Yes", # Changed
                    "Rain": {
                        "Wind": {
                            "Strong": "No",
                            "Weak": "Yes"
                        }
                    }
                }}
        )
        self.assertFalse(matcher.matches(self.create_tree_tennis()))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    