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
        
    @classmethod
    def from_list(cls, data_list):
        """
        Creates a DataSet object from regular Python lists.
        
        Args:
          data_list: a standard Python list or list of lists containing the 
            data.
            
        Returns:
          A new DataSet instance.
        """
        return cls(pd.DataFrame(data_list))
    
    @classmethod
    def from_unknown(cls, data):
        """
        Creates a DataSet object from an object of an unknown data type.
        
        Args:
          data: the raw data set.  It can be stored as a Python list or 
            pandas dataframe.  If it is already a DataSet then it will just be 
            returned.
        
        Returns:
          A DataSet wrapper around the data.  If it is already a DataSet then 
          it will just be returned.
          
        Raises:
          ValueError if the data is not of a supported type.
        """
        if isinstance(data, DataSet):
            return data
        elif isinstance(data, pd.DataFrame):
            return cls(data)
        elif isinstance(data, list):
            return DataSet.from_list(data)
        else:
            raise ValueError("Unsupported representation of data set")

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
    
    def apply_row_function(self, function):
        return self.data_frame.apply(function, axis=1)

    def drop_column(self, index):
        return DataSet(self.data_frame.drop(index, axis=1))

    def get_column(self, index):
        return self.data_frame.ix[:, index]


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
