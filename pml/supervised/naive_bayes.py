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
Naive Bayes classification algorithm.

@author: drusk
"""

class NaiveBayes(object):
    """
    """
    
    def __init__(self, training_set):
        """
        Constructs a new NaiveBayes classifier.
        
        Args:
          training_set: model.DataSet
            The data used to train the classifier.
        """
        self._training_set = training_set
    
    def _count_examples_with_class_and_feature_val(
            self, clazz, feature, sample):
        """
        Counts the training set examples which have the specified class as 
        well as the same value for the specified feature as the provided 
        sample.
        
        Args:
          clazz:
            The class which training examples must belong to in order to be 
            counted.
          feature:
            The feature for which the training examples must have the same 
            value as the sample.
          sample: dict or dict-like (ex: pandas.Series)
            The sample whose value for the specified feature must be matched 
            in order to count an example.
            
        Returns:
          count: int
            The number of training examples with the specified class and same 
            value as the sample for the specified feature.
        """
        training_classes = self._training_set.get_labels()
        training_feature_vals = self._training_set.get_column(feature)
        
        match_classes = training_classes == clazz
        match_feature_vals = training_feature_vals == sample[feature]
        
        match_both = match_classes & match_feature_vals
        return match_both.value_counts()[True]
    
    