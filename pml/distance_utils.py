"""
Algorithms for calculating distances between vectors in feature (n) space.

@author: drusk
"""

import numpy as np

def euclidean(vector1, vector2):
    """
    Calculates the Euclidean distance between two vectors in n-space.
    
    Args:
      vector1: 
        start point vector
      vector2: 
        end point vector
    
    Returns:
      The distance (magnitude) between vector1 and vector2.
    """
    return np.sqrt(np.power(np.asarray(vector1) - np.asarray(vector2), 2).sum())
