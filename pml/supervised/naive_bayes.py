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

from pml.supervised.classifiers import AbstractClassifier
from pml.utils import collection_utils

class NaiveBayes(AbstractClassifier):
    """
    Naive Bayes classifier.
    
    This algorithm classifies samples using probabilities calculated based 
    on applying Bayes' theorem.
    
    The algorithm is said to be naive because it assumes all features are 
    independent of each other.  While not generally true, the approach is 
    still quite effective and allows the training set to be much smaller.
    """
    
    def __init__(self, training_set):
        """
        Constructs a new NaiveBayes classifier.
        
        Args:
          training_set: model.DataSet
            The data used to train the classifier.
        """
        super(NaiveBayes, self).__init__(training_set)
    
    def classify(self, sample):
        """
        Predicts a sample's classification based on the training set.
        
        Args:
          sample: dict or pandas.Series
            the sample or observation to be classified.
          
        Returns:
          The sample's classification.
          
        Raises:
          ValueError if sample doesn't have the same number of features as 
          the data in the training set.
        """
        class_probabilities = self.get_classification_probabilities(sample)
        return collection_utils.get_key_with_highest_value(class_probabilities)
    
    def get_classification_probabilities(self, sample):
        """
        Determines the probability that a sample belongs to each class that 
        was seen in the training set.
        
        Args:
          sample: dict or pandas.Series
            The sample or observation to be classified.
        
        Returns:
          probabilities: dict
            A dictionary of classifications and their probabilities.
        """
        class_probabilities = {}

        for clazz in set(self.training_set.get_labels()):
            prob_clazz = self._calc_prob_class(clazz)
            
            likelihood = 1
            for feature in self.training_set.feature_list():
                likelihood *= self._calc_prob_feature_given_class(clazz, feature, 
                                                            sample[feature])
            class_probabilities[clazz] = prob_clazz * likelihood
            
        return class_probabilities
    
    def _calc_prob_class(self, clazz):
        """
        Calculate the probability of a training example belonging to the 
        given class.
        
        Args:
          clazz:
            The class which examples must belong to.
            
        Returns:
          probability: float
            The probability as a floating point number between 0.0 and 1.0.
        """
        clazz_count = self.training_set.get_label_value_counts()[clazz]
        return float(clazz_count) / self.training_set.num_samples()
    
    def _calc_prob_feature_given_class(self, clazz, feature, feature_val):
        """
        Calculates the probability of a training example having a given class 
        as well as the given value of the specified feature.

        Args:
          clazz:
            A class from the training set.
          feature:
            The feature whose value must match the provided feature_val.
          feature_val:
            The value of feature which must be matched.
            
        Returns:
          probability: float
            The probability as a floating point number between 0.0 and 1.0. 
        """
        n = self.training_set.get_label_value_counts()[clazz]
        n_c = self._count_examples(clazz, feature, feature_val)
        
        num_feature_vals = len(set(self.training_set.get_column(feature)))
        p = float(1) / num_feature_vals
        m = num_feature_vals
        
        # the use of m and p is called 'm-estimates' and is for the case 
        # where n_c = 0 because otherwise that would make the product of the 
        # probabilities.
        
        return float(n_c + m*p) / (n + m)
        
    def _count_examples(self, clazz, feature, feature_val):
        """
        Counts the training set examples which have the specified class as 
        well as the specified value of the given feature.
        
        Args:
          clazz:
            The class which training examples must belong to in order to be 
            counted.
          feature:
            The feature for which the training examples must have the value 
            feature_val.
          feature_val: 
            The value of feature which must be matched in order to count an 
            example.
            
        Returns:
          count: int
            The number of training examples with the specified class and same 
            value as the sample for the specified feature.
        """
        training_classes = self.training_set.get_labels()
        training_feature_vals = self.training_set.get_column(feature)
        
        match_classes = training_classes == clazz
        match_feature_vals = training_feature_vals == feature_val
        
        match_both = match_classes & match_feature_vals
        return match_both.value_counts()[True]
    
    