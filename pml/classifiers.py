"""
Classification algorithms for supervised learning tasks.

@author: drusk
"""

import loader
import distance_utils
import collections
import collection_utils

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
        
        def calc_dist(vector):
            return distance_utils.euclidean(vector, sample)
        distances = data.apply_row_function(calc_dist)
        
        votes = collections.defaultdict(int)
        for i, index in enumerate(distances.order(ascending=True).index):
            if i < self.k:
                votes[labels[index]] += 1
            else:
                break
        
        return collection_utils.get_key_of_highest_val(votes)
    