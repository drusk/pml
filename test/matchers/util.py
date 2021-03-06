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
Assorted utilities useful for testing.

@author: drusk
"""

from numbers import Number

import numpy as np

def equals(expected, actual, places=None):
    """
    Special equals method to make NaN's considered equal to each other.
    For convenience, can also handle comparing non-numeric values, in which 
    case places is ignored.
    
    places: int
            The number of decimal places to check when comparing data values.
            Defaults to None, in which case full equality is checked (good for 
            ints, but not for floats).
    """
    if not isinstance(expected, Number):
        # np.nan is counted as a number
        return expected == actual
    
    if np.isnan(expected) and np.isnan(actual):
        return True
    
    if places is None:
        return expected == actual
    else:
        return almost_equal(expected, actual, places)

def almost_equal(val1, val2, places):
    """
    Checks if two values are equal to each other up to a certain number of 
    decimal places.
    
    Args:
      val1: float
      val2: float
      places: int
        The number of decimal places which val1 and val2 must be equal.
        
    Returns:
      True if val1 and val2 are equal up to the number of decimal places 
      specified by places.
    """
    return int(abs(val1 - val2) * 10 ** places) == 0
    