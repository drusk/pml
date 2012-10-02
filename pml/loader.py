"""
Utilities for loading data sets.

@author: drusk
"""

import pandas as pd

class DataSet(object):
    """
    An abstract representation of a data set.
    """
    
    def __init__(self, data_frame):
        """
        Constructs a new data set object.
        
        Args:
          data_frame: a pandas DataFrame object.
        """
        self.data_frame = data_frame
        
    def num_samples(self):
        """
        Returns:
          The number of samples (rows) in the data set.
        """    
        return self.data_frame.shape[0]
    
    def num_features(self):
        """
        Returns:
          The number of features (columns) in the data set.
        """
        return self.data_frame.shape[1]


def load(path, delimiter=","):
    """
    Loads a data set from a delimited text file.
    
    Args:
      path: the path to the file containing the data set.
      delimiter: the symbol used to separate columns in the file.  Default 
        value is ','.
      
    Returns:
      An array-like object.
    """
    return DataSet(pd.read_csv(path, delimiter=delimiter))
