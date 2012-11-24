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
Custom Hamcrest matchers for general purposes not related to pml or pandas 
data structures.

@author: drusk
"""

from hamcrest.core.base_matcher import BaseMatcher

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


def in_range(minval, maxval):
    """
    Checks if a value is within the range specified by minval and maxval, 
    inclusive.
    """
    return InRange(minval, maxval)
