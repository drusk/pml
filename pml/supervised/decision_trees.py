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
Decision trees classification algorithm.

@author: drusk
"""

import numpy as np

def entropy(proportions):
    """
    Calculates the entropy of a data set given the proportion of samples in 
    with each classification.
    
    Entropy is the measure of impurity of the data set.  For example, if all 
    the samples have the same classification, the entropy will be 0.
    
    Args:
      proportions: list(float)
        The proportions of the data set which belong to each class.  It does 
        not matter which class each proportion corresponds to, so they should 
        just be passed as an arbitrarily ordered list.  Each proportion 
        should be a float between 0.0 and 1.0.
        
    Returns:
      The entropy of the data with the provided proportions.  Higher values 
      indicate less uniform or more disordered data.
    """
    def entropy_val(proportion):
        """
        Calculates the entropy associated with a single proportion.
        """
        return -1 * proportion * np.log2(proportion) 

    return np.sum(map(entropy_val, proportions))
