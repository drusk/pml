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

import util

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
    

class IsSeries(BaseMatcher):
    """
    Matches a pandas Series data structure.
    """
    
    def __init__(self, as_dict, places=None):
        """
        Creates a new matcher given the expected input as a dictionary.
        
        Args:
          as_dict: dictionary
            The expected data in key-value format.
          places: int
            The number of decimal places to check when comparing data values.
            Defaults to None, in which case full equality is checked (good for 
            ints, but not for floats).
        """
        self.as_dict = as_dict
        self.places = places
        
    def _numeric_equals(self, val1, val2):
        return _equals(val1, val2, self.places)
        
    def _non_numeric_equals(self, val1, val2):
        return val1 == val2
        
    def _get_equals_function(self, data):
        if is_numeric_type(data):
            return self._numeric_equals
        else:
            return self._non_numeric_equals
        
    def _matches(self, series):
        equals = self._get_equals_function(series)
        if len(self.as_dict) != len(series):
            return False
        
        for key in self.as_dict:
            if not equals(series[key], self.as_dict[key]):
                return False
            
        return True
        
    def describe_to(self, description):
        description.append_text("pandas Series with elements: ")
        description.append_text(self.as_dict.__str__())
        
        
class InRange(BaseMatcher):
    """
    Matches values within a specified range (inclusive).
    """
    
    def __init__(self, minval, maxval):
        """
        Creates a new matcher given the expected minimum and maximum values.
        """
        self.minval = minval
        self.maxval = maxval
        
    def _matches(self, val):
        return val <= self.maxval and val >= self.minval
        
    def describe_to(self, description):
        description.append_text("value between %s and %s" 
                                %(self.minval, self.maxval))
        
    
def is_numeric_type(data):
    """
    Checks if the data is any of the numeric types.
    
    Args:
      data: data structure with dtype, i.e. pandas.Series, pandas.DataFrame
        The data whose type will be checked.
        
    Returns:
      True if the data type is numeric, false otherwise.
    """
    return "int" in data.dtype.name or "float" in data.dtype.name
    
def _equals(val1, val2, places=None):
    """
    Special equals method to make NaN's considered equal to each other.
    
    places: int
            The number of decimal places to check when comparing data values.
            Defaults to None, in which case full equality is checked (good for 
            ints, but not for floats).
    """
    if np.isnan(val1) and np.isnan(val2):
        return True
    
    if places is None:
        return val1 == val2
    else:
        return util.almost_equal(val1, val2, places)
    
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
        all([_equals(list1[i], list2[i], places=places) 
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

def equals_series(as_dict, places=None):
    """
    Compares a pandas Series object to the provided dictionary representation.
    
    Args:
      as_dict: dictionary
        The expected data in key-value format.
      places: int
        The number of decimal places to check when comparing data values.
        Defaults to None, in which case full equality is checked (good for 
        ints, but not for floats).
    """
    return IsSeries(as_dict, places)

def in_range(minval, maxval):
    """
    Checks if a value is within the range specified by minval and maxval, 
    inclusive.
    """
    return InRange(minval, maxval)
