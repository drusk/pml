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
Custom Hamcrest matchers for pandas data structures.

@author: drusk
"""

from hamcrest.core.base_matcher import BaseMatcher

from test.matchers.util import equals

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
        return equals(val1, val2, self.places)
        
    def _non_numeric_equals(self, val1, val2):
        return val1 == val2
        
    def _get_equals_function(self, data):
        if _is_numeric_type(data):
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
        
   
class IsDataFrame(BaseMatcher):
    """
    Matches a pandas DataFrame data structure.
    """
    
    def __init__(self, as_list, places):
        """
        Creates a new matcher given the expected input as a list of lists.
        
        Args:
          as_list: list(list)
            The expected data as a list of lists.
          places: int
            The number of decimal places to check when comparing data values.
            Defaults to None, in which case full equality is checked (good for 
            ints, but not for floats).
        """
        self.as_list = as_list
        self.places = places
    
    def _matches(self, dataframe):
        expected_num_rows = len(self.as_list)
        if expected_num_rows != dataframe.shape[0]:
            return False
        
        row_lengths = map(len, self.as_list)
        if len(set(row_lengths)) != 1:
            # row lengths are not all the same
            return False
        
        expected_num_columns = row_lengths[0]
        if expected_num_columns != dataframe.shape[1]:
            return False
        
        for i, row in enumerate(self.as_list):
            for j, element in enumerate(row):
                if not equals(dataframe.ix[i].tolist()[j], element, 
                              places=self.places):
                    return False
                
        return True
        
    def describe_to(self, description):
        description.append_text("pandas DataFrame with elements: ")
        description.append_text(self.as_list.__str__())
   
        
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

def equals_dataframe(as_list, places=None):
    """
    Compares a pandas DataFrame object to the provided list representation.
    Since DataFrames are two dimensional, the list should actually be a list 
    of lists.
    """
    return IsDataFrame(as_list, places)

def _is_numeric_type(data):
    """
    Checks if the data is any of the numeric types.
    
    Args:
      data: data structure with dtype, i.e. pandas.Series, pandas.DataFrame
        The data whose type will be checked.
        
    Returns:
      True if the data type is numeric, false otherwise.
    """
    return "int" in data.dtype.name or "float" in data.dtype.name
