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

import numpy as np
from hamcrest.core.base_matcher import BaseMatcher

class IsDataSet(BaseMatcher):
    """
    Used for asserting the contents of a DataSet object are as expected.
    """
    
    def __init__(self, as_list):
        """
        Creates a new matcher given an expected input.
        
        Args:
          as_list: 2d list
            A 2d list where each sub-list represents a row in the dataset's 
            underlying DataFrame.
        """
        self.as_list = as_list
    
    def _matches(self, dataset):
        if dataset.num_samples() != len(self.as_list):
            return False

        for i in xrange(dataset.num_samples()):
            # if the dataset has been filtered the indices may not be a 
            # continuous range
            index = dataset._dataframe.index[i]
            if not lists_match_exactly(self.as_list[i], 
                                 dataset.get_row(index).tolist()):
                return False
        
        return True    
        
    def describe_to(self, description):
        description.append_text("dataset with elements: ")
        description.append_text(self.as_list.__str__())
    

class IsSeries(BaseMatcher):
    """
    Matches a pandas Series data structure.
    """
    
    def __init__(self, as_dict):
        """
        Creates a new matcher given the expected input as a dictionary.
        """
        self.as_dict = as_dict
        
    def _matches(self, series):
        if len(self.as_dict) != len(series):
            return False
        
        for key in self.as_dict:
            if series[key] != self.as_dict[key]:
                return False
            
        return True
        
    def describe_to(self, description):
        description.append_text("pandas Series with elements: ")
        description.append_text(self.as_dict.__str__())
        
    
def _equals(val1, val2):
    """
    Special equals method to make NaN's considered equal to each other.
    """
    if np.isnan(val1) and np.isnan(val2):
        return True
    return val1 == val2
    
def lists_match_exactly(list1, list2):
    """
    Compares two lists and returns True if they are exactly the same, False 
    otherwise.
    """
    return len(list1) == len(list2) and \
        all([_equals(list1[i], list2[i]) for i in xrange(len(list1))])
    
def equals_dataset(as_list):
    """
    Compares a DataSet object to the given list representation.
    """
    return IsDataSet(as_list)

def equals_series(as_dict):
    """
    Compares a pandas Series object to the provided dictionary representation.
    """
    return IsSeries(as_dict)
