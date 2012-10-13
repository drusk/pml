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
            index = dataset.data_frame.index[i]
            if not match_exactly(self.as_list[i], 
                                 dataset.get_row(index).tolist()):
                return False
        
        return True    
        
    def describe_to(self, description):
        description.append_text("dataset with elements: ")
        description.append_text(self.as_list.__str__())
    
def match_exactly(list1, list2):
    """
    Compares two lists and returns True if they are exactly the same, False 
    otherwise.
    """
    return len(list1) == len(list2) and \
        all([list1[i] == list2[i] for i in xrange(len(list1))])
    
def equals_dataset(as_list):
    """
    Compares a DataSet object to the given list representation.
    """
    return IsDataSet(as_list)
