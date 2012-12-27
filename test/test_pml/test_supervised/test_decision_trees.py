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
Unit tests for decision_trees module.

Examples from:
http://www.doc.ic.ac.uk/~sgc/teaching/pre2012/v231/lecture11.html

NOTE: weekends.data has been modified slightly: w5 parents switched to 'yes'
in order to make money unambiguously the best choice for splitting after 
weather on the rainy branch.

@author: drusk
"""

import unittest

import pandas as pd

from pml.supervised.decision_trees import entropy, info_gain
from pml.supervised import id3
from pml.data.model import DataSet
from pml.data.loader import load

from test import base_tests

class DecisionTreesTest(base_tests.BaseFileLoadingTest):

    def create_example_dataset(self):
        """
        Creates the small example data set used in the lecture notes for 
        entropy and information gain calculations (not the weekend data set).
        """
        # 1 feature data set
        sample_ids = ["s1", "s2", "s3", "s4"]
        feature_vals = pd.Series(["v2", "v2", "v3", "v1"], index=sample_ids, 
                                 name="A")
        df = pd.DataFrame(feature_vals)

        labels = pd.Series(["+", "-", "-", "-"], index=sample_ids)
        return DataSet(df, labels=labels)

    def test_entropy(self):
        dataset = self.create_example_dataset()
        self.assertAlmostEqual(entropy(dataset), 0.811, places=3)
        
    def test_entropy_0_proportion(self):
        dataset = self.create_example_dataset().value_filter("A", "v1")
        self.assertEqual(entropy(dataset), 0)

    def test_info_gain(self):
        dataset = self.create_example_dataset()
        self.assertAlmostEqual(info_gain("A", dataset), 0.311, places=3)
    
    def test_id3_choose_feature_to_split(self):
        data = load(self.relative_to_base("/datasets/weekends.data"))
        root = id3.choose_feature_to_split(data)
        self.assertEqual(root, "weather")
    
    def test_id3_build_tree_marine_animals(self):
        dataset = load(self.relative_to_base("/datasets/marine_animal.data"))
        tree = id3.build_tree(dataset)
        
        self.assertDictEqual(
                tree,
                {"no_surfacing": {
                    False: False,
                    True: {
                        "has_flippers": {
                            False: False, 
                            True: True           
                        }
                    }
                 }
                }
        )
        
    def test_id3_build_tree_weekends(self):
        dataset = load(self.relative_to_base("/datasets/weekends.data"))
        tree = id3.build_tree(dataset)
        
        self.assertDictEqual(
                tree, 
                {"weather": {
                    "sunny": {
                        "parents": {
                            True: "cinema",
                            False: "tennis"
                        }
                    },
                    "windy": {
                        "parents": {
                            True: "cinema",
                            False: {
                                "money": {
                                    "rich": "shopping",
                                    "poor": "cinema"
                                }
                            }
                        }
                    },
                    "rainy": {
                        "money": {
                            "poor": "cinema",
                            "rich": "stay in"
                        }
                    }
                }}
        )
    

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    