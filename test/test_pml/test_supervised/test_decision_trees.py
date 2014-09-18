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
and
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/mlbook/ch3.pdf

NOTE: weekends.data has been modified slightly: w5 parents switched to 'yes'
in order to make money unambiguously the best choice for splitting after 
weather on the rainy branch.

@author: drusk
"""

import unittest

import pandas as pd
from hamcrest import assert_that, equal_to

from pml.supervised.decision_trees import id3
from pml.supervised.decision_trees import DecisionTree
from pml.data.loader import load
from pml.data.model import DataSet

from test import base_tests
from test.matchers.pml_matchers import equals_tree
from test.matchers.pandas_matchers import equals_series

class DecisionTreesTest(base_tests.BaseFileLoadingTest):

    def test_id3_choose_feature_to_split(self):
        data = load(self.relative_to_base("/datasets/weekends.data"))
        root = id3.choose_feature_to_split(data)
        self.assertEqual(root, "weather")
    
    def test_id3_build_tree_marine_animals(self):
        dataset = load(self.relative_to_base("/datasets/marine_animal.data"))
        tree = id3.build_tree(dataset)
        
        assert_that(tree,
            equals_tree(
                {"no_surfacing": {
                    "no": "no",
                    "yes": {
                        "has_flippers": {
                            "no": "no",
                            "yes": "yes"
                        }
                    }
                 }
                }
            )
        )
        
    def test_id3_build_tree_weekends(self):
        dataset = load(self.relative_to_base("/datasets/weekends.data"))
        tree = id3.build_tree(dataset)
        
        assert_that(tree,
            equals_tree( 
                {"weather": {
                    "sunny": {
                        "parents": {
                            "yes": "cinema",
                            "no": "tennis"
                        }
                    },
                    "windy": {
                        "parents": {
                            "yes": "cinema",
                            "no": {
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
        )
        
    def test_id3_build_tree_play_tennis(self):
        dataset = load(self.relative_to_base("/datasets/play_tennis.data"),
                       delimiter=" ")
        tree = id3.build_tree(dataset)
        
        assert_that(tree,
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
        
    def test_classify_play_tennis(self):
        training = load(self.relative_to_base("/datasets/play_tennis.data"),
                        delimiter=" ")
        classifier = DecisionTree(training)
        sample = pd.Series(["Rain", "Cool", "High", "Strong"], 
                           index=['Outlook', 'Temperature', 'Humidity', 
                                  'Wind'])
        self.assertEqual(classifier.classify(sample), "No")
        
    def test_classify_weekends(self):
        training = load(self.relative_to_base("/datasets/weekends.data"))
        classifier = DecisionTree(training)
        sample = pd.Series(["windy", "no", "rich"],
                           index=['weather', 'parents', 'money'])
        self.assertEqual(classifier.classify(sample), "shopping")

    def test_classify_all_weekends(self):
        training = load(self.relative_to_base("/datasets/weekends.data"))
        classifier = DecisionTree(training)
        index = ['weather', 'parents', 'money']
        sample_0 = pd.Series(["windy", "no", "rich"], index=index)
        sample_1 = pd.Series(["sunny", "yes", "rich"], index=index)
        results = classifier.classify_all(
                        DataSet(pd.DataFrame([sample_0, sample_1])))
        assert_that(results.get_classifications(), 
                    equals_series({0: "shopping", 1: "cinema"}))

    def test_data_has_value_not_in_training(self):
        training = load(self.relative_to_base("/datasets/play_tennis.data"), 
                        delimiter=" ")
        classifier = DecisionTree(training)
        
        # NOTE: Snowing is not a value of Outlook seen in the training set
        sample = pd.Series({"Outlook": "Snowing", "Temperature": "Cool", 
                            "Humidity": "Normal", "Wind": "Strong"})
        
        assert_that(classifier.classify(sample), equal_to("Yes"))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    