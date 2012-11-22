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
K-nearest neighbours classification algorithm.

@author: drusk
"""

import collections

from pml.supervised.classifiers import AbstractClassifier
from pml.utils import distance_utils

class Knn(AbstractClassifier):
    """
    K-Nearest Neighbours classifier.
    
    This algorithm classifies samples based on the 'k' closest training 
    examples in the feature space.  The sample's class is predicted through a 
    majority vote of its neighbours.  
    
    In the case of a tie, the distances to each tied class are summed 
    amongst the neighbours.  The class with the minimum distance to the sample 
    is selected to break the tie.
    
    This is an example of a 'lazy learning' algorithm where all computation is 
    deferred until classification.
    """
    
    def __init__(self, training_set, k=5):
        """
        Constructs a new Knn classifier.
        
        Args:
          training_set: 
            A labelled DataSet object used to train the classifier.
          k: 
            The number of nearest neighbours to consider when voting for a 
            sample's class.  Must be a positive integer, preferably small.  
            Default value is 5.
            
        Raises:
          UnlabelledDataSetError if the training set is not labelled.
        """
        super(Knn, self).__init__(training_set)
        self.k = k
        
    def __str__(self):
        """
        Returns:
          This object's string representation, primarily for debugging 
          purposes.
        """
        return "<KNN Classifier: k=%d, trained on %d samples>" \
            % (self.k, self.training_set.num_samples())

    def classify(self, sample):
        """
        Predicts a sample's classification based on the training set.
        
        Args:
          sample: 
            the sample or observation to be classified.
          
        Returns:
          The sample's classification.
          
        Raises:
          ValueError if sample doesn't have the same number of features as 
          the data in the training set.
        """
        if len(sample) != self.training_set.num_features():
            raise ValueError(("Sample must have the same number of features " 
                              "as the training set."))
            
        # This function is used so that we can reduce each row with respect 
        # to the sample.
        def calc_dist(vector):
            return distance_utils.euclidean(vector, sample)

        distances = self.training_set.reduce_rows(calc_dist)
        
        votes = self._tally_votes(self.training_set.get_labels(), distances)
        most_voted = self._get_most_voted(votes)
        
        # TODO tie breaking?
        assert len(most_voted) >= 1
        return most_voted[0]

    def _tally_votes(self, labels, distances):
        """
        Counts the k nearest neighbours' votes for which classification to 
        give the sample.
        
        Args:
          labels: 
            the training set labels
          distances: 
            the distance from each entry in the training set to the sample.
              
        Returns: 
          a dictionary mapping labels to their number of votes.
        """
        votes = collections.defaultdict(int)
        for i, index in enumerate(distances.order(ascending=True).index):
            if i < self.k:
                votes[labels[index]] += 1
            else:
                break
        return votes
    
    def _get_most_voted(self, votes):
        """
        Determines the labels which received the most votes.
        
        Args:
          votes: 
            a dictionary mapping labels to their number of votes.
          
        Returns:
          a list of the labels which got the most votes.  It is a list because 
          there may be ties.
        """
        max_votes = max(votes.values())
        most_voted = []
        
        for label, num_votes in votes.iteritems():
            if num_votes == max_votes:
                most_voted.append(label)
        
        return most_voted

