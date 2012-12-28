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
Custom Hamcrest matchers.

@author: drusk
"""

from hamcrest.core.base_matcher import BaseMatcher

from test.matchers.util import equals

class IsDataSet(BaseMatcher):
    """
    Used for asserting the contents of a DataSet object are as expected.
    """
    
    def __init__(self, as_list, places=None):
        """
        Creates a new matcher given an expected input.
        
        Args:
          as_list: 2d list
            A 2d list where each sub-list represents a row in the dataset's 
            underlying DataFrame.
          places: int
            The number of decimal places to check when comparing data values.
            Defaults to None, in which case full equality is checked (good for 
            ints, but not for floats).
        """
        self.as_list = as_list
        self.places = places
    
    def _matches(self, dataset):
        if dataset.num_samples() != len(self.as_list):
            return False

        for i in xrange(dataset.num_samples()):
            # if the dataset has been filtered the indices may not be a 
            # continuous range
            index = dataset.get_sample_ids()[i]
            if not lists_match(self.as_list[i], 
                               dataset.get_row(index).tolist(), 
                               places=self.places):
                return False
        
        return True    
        
    def describe_to(self, description):
        description.append_text("dataset with elements: ")
        description.append_text(self.as_list.__str__())
    

class IsTree(BaseMatcher):
    """
    Used for asserting that a tree data structure has the expected contents.
    """
    
    def __init__(self, expected_dict):
        """
        Creates a new matcher given an expected input.
        
        Args:
          expected_dict: dictionary
            The tree stored as a dictionary of dictionaries.
        """
        self.expected_dict = expected_dict
    
    def _matches(self, actual_tree):
        return self._match_recursively(actual_tree.get_root_node(), 
                                       self.expected_dict)
    
    def _match_recursively(self, actual_node, expected):
        """
        Performs the actual checks that the tree contents are as expected.  
        It is called recursively for each node in the tree.
        """
        if not isinstance(expected, dict) and actual_node.is_leaf():
            # Matched down this branch
            return actual_node.get_value() == expected
        
        node_val = actual_node.get_value()
        # Check current root's value
        if (len(expected) != 1 or 
            node_val != expected.keys()[0]):
            return False

        # Check there are the right number of children
        if (len(expected[node_val]) != 
            len(actual_node.get_branches())):
            return False
        
        # Recursively check each branch
        for branch in actual_node.get_branches():
            child_node = actual_node.get_child(branch)
            try:
                expected_subtree = expected[node_val][branch]
            except KeyError:
                return False

            if not self._match_recursively(child_node, expected_subtree):
                return False
        
        # All branches from the current node matched
        return True
        
    def describe_to(self, description):
        description.append_text("Tree with form:")
        description.append_text(self.expected_dict)
    

def lists_match(list1, list2, places=None):
    """
    Compares two lists and returns True if they are exactly the same, False 
    otherwise.
    
    places: int
            The number of decimal places to check when comparing data values.
            Defaults to None, in which case full equality is checked (good for 
            ints, but not for floats).
    """
    return len(list1) == len(list2) and \
        all([equals(list1[i], list2[i], places=places) 
             for i in xrange(len(list1))])
    
def equals_dataset(as_list, places=None):
    """
    Compares a DataSet object to the given list representation.
    
    Args:
      as_list: Python list of lists
        The expected data.
      places: int
        The number of decimal places to check when comparing data values.
        Defaults to None, in which case full equality is checked (good for 
        ints, but not for floats).
      
    """
    return IsDataSet(as_list, places=places)

def equals_tree(expected_dict):
    """
    Compares a Tree object to the given expected dictionary representation.
    
    Args:
      expected_dict: dictionary
        The tree stored as a dictionary of dictionaries.
            
        For example, a data set with dogs, cats and birds and features 
        "num_legs" and "barks" might have a tree like follows:
        
        {
          "num_legs": {
            4: {
              "barks": {
                True: "dog",
                False: "cat"
              }
            },
            2: "bird"
          }
        }
    """
    return IsTree(expected_dict)
