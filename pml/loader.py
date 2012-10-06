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
    
    def reduce_rows(self, function):
        """
        Performs a row-wise reduction of the data set.
        
        Args:
          function: the function which will be applied to each row in the data 
            set.
        
        Returns:
          a pandas Series object which is the one dimensional result of 
            reduction (one value corresponding to each row).
        """
        return self.data_frame.apply(function, axis=1)

    def drop_column(self, index):
        """
        Creates a copy of the data set with a specified column removed.
        
        Args:
          index: the index (0 based) of the column to drop.
          
        Returns:
          a new DataSet with the specified column removed.  The original 
          DataSet remains unaltered.
        """
        return DataSet(self.data_frame.drop(index, axis=1))

    def get_column(self, index):
        """
        Selects a column from the data set.
        
        Args:
          index: the index (0 based) of the column to select.
          
        Returns:
          the columns at the specified index as a pandas Series object.  This 
          series is a view on the original data set, not a copy.  That means 
          any changes to it will also be applied to the original data set.
        """
        return self.data_frame.ix[:, index]


def load(path, delimiter=","):
    """
    Loads a data set from a delimited text file.
    
    Args:
      path: the path to the file containing the data set.
      delimiter: the symbol used to separate columns in the file.  Default 
        value is ','.  Hint: delimiter for tab-delimited files is '\t'.
      
    Returns:
      An array-like object.
    """
    return DataSet(pd.read_csv(path, delimiter=delimiter))
