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
Utilities for working with collections.

@author: drusk
"""

import collections
import operator

def get_key_with_highest_value(dictionary):
    """
    Retrieves the key with the highest value from a dictionary.
    
    Args:
      dictionary: dict
        The dictionary whose keys and values are being examined.
        
    Returns:
      key:
        The key with the highest value.  Returns None if the dictionary 
        is empty.
    """
    if len(dictionary) == 0:
        return None
    
    return max(dictionary.iteritems(), key=operator.itemgetter(1))[0]

def are_all_equal(iterable):
    """
    Checks if all elements of a collection are equal.
    
    Args:
      iterable: iterator
        A collection of values such as a list.
        
    Returns:
      equal: boolean
        True if all elements of iterable are equal.  Will also return true if 
        iterable is empty.
    """
    return len(set(iterable)) <= 1

def get_most_common(iterable):
    """
    Finds the item which occurs most often in a collection of values.
    
    Args:
      iterable: iterator
        The items which will be searched to find the most common.
        
    Returns:
      most_common: 
        The most common item in the collection.  Ties are broken arbitrarily.
    """
    counts = collections.defaultdict(int)
    for item in iterable:
        counts[item] += 1
    
    return get_key_with_highest_value(counts)
