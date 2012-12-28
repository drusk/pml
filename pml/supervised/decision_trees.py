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

from pml.supervised.classifiers import AbstractClassifier
from pml.supervised import id3

class DecisionTree(AbstractClassifier):
    """
    Decision tree classifier.
    
    Builds a tree which is like a flow chart.  It allows a decision to be 
    reached by checking the values for various features and following the 
    appropriate branches until a destination is reached.
        
    In addition to being useful as a classifier, the structure of the 
    decision tree can lend insight into the data. 
    """
    
    def __init__(self, training_set):
        """
        Constructs a new decision tree.
        
        Args:
          training_set: model.DataSet
            The training data to use when building the decision tree.
        """
        self.training_set = training_set
        self._tree = id3.build_tree(training_set)
    
    def _classify(self, sample):
        """
        Predicts a sample's classification based on the decision tree that 
        was built from the training data.
        
        Args:
          sample: 
            The sample or observation to be classified.
          
        Returns:
          The sample's classification.
        """
        node = self._tree.get_root_node()
        while not node.is_leaf():
            feature = node.get_value()
            branch = sample[feature]
            node = node.get_child(branch)
        
        return node.get_value()

