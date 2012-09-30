"""
Classification algorithms for supervised learning tasks.

@author: drusk
"""

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
        pass
    
    def classify(self, sample):
        """
        Predicts a sample's classification based on the training set.
        
        Args:
          sample: the sample or observation to be classified.
          
        Returns:
          The sample's classification.
        """
        pass
    