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
Algorithms related to information theory.

@author: drusk
"""

import numpy as np

def info_gain(feature, dataset):
    """
    Calculates the information gain of a feature in a data set.
    
    The information gain of a feature is the expected reduction in entropy 
    caused by knowing the value of that feature.
    
    Args:
      feature: string
        The name of a feature in the data set.
      dataset: model.DataSet
        The data set that the feature is a part of.
        
    Returns:
      info_gain: float
          The information gain of the feature.
    """
    feature_value_counts = dataset.get_feature_value_counts(feature)

    value_entropies = 0
    for value, count in feature_value_counts.iteritems():
        weight = float(count) / dataset.num_samples()
        value_entropies += (weight * 
                            entropy(dataset.value_filter(feature, value)))
    
    return entropy(dataset) - value_entropies

def entropy(dataset):
    """
    Calculates the entropy of a data set. 
    
    Entropy is the measure of impurity of the data set.  For example, if all 
    the samples have the same classification, the entropy will be 0.
    
    Args:
      dataset: model.DataSet
        The data set whose entropy is to be calculated.
        
    Returns:
      The entropy of the data.  Higher values indicate less uniform or more 
      disordered data.
    """
    label_counts = dataset.get_label_value_counts()
    
    def calc_proportion(count):
        return float(count) / np.sum(label_counts)
    
    label_proportions = map(calc_proportion, label_counts)
    
    def entropy_val(proportion):
        """
        Calculates the entropy associated with a single proportion.
        """
        if proportion == 0:
            return 0
        
        return -1 * proportion * np.log2(proportion) 

    return np.sum(map(entropy_val, label_proportions))

