"""
Classification algorithms for supervised learning tasks.

@author: drusk
"""

import loader
import distance_utils
import collections

class Knn(object):
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
          training_set: an array-like set of data used to train the classifier.
          k: the number of nearest neighbours to consider when voting for a 
            sample's class.  Must be a positive integer, preferably small.  
            Default value is 5.
        """
        self.training_set = loader.DataSet.from_unknown(training_set)
        self.k = k
        
    def classify(self, sample):
        """
        Predicts a sample's classification based on the training set.
        
        Args:
          sample: the sample or observation to be classified.
          
        Returns:
          The sample's classification.
        """
        num_features = self.training_set.num_features()
        labels = self.training_set.get_column(num_features - 1)
        data = self.training_set.drop_column(num_features - 1)
        
        # This function is used so that we can reduce each row with respect 
        # to the sample.
        def calc_dist(vector):
            return distance_utils.euclidean(vector, sample)
        
        distances = data.reduce_rows(calc_dist)
        
        votes = self._tally_votes(labels, distances)
        most_voted = self._get_most_voted(votes)
        
        # TODO tie breaking?
        assert len(most_voted) >= 1
        return most_voted[0]

    def _tally_votes(self, labels, distances):
        """
        Counts the k nearest neighbours' votes for which classification to 
        give the sample.
        
        Args:
          labels: the training set labels
          distances: the distance from each entry in the training set to the 
            sample.
              
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
          votes: a dictionary mapping labels to their number of votes.
          
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
