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
          data_frame: 
            a pandas DataFrame object.
        """
        self.data_frame = data_frame
        
    def __str__(self):
        """
        Returns:
          This object's string representation, primarily for debugging 
          purposes.
        """
        return self.data_frame.__str__()
    
    def __repr__(self):
        """
        This gets called when the object's name is typed into IPython on its 
        own line, causing a string representation of the object to be 
        displayed.
        
        Returns:
          This object's string representation, primarily for debugging 
          purposes.
        """
        return self.__str__()  
        
    @classmethod
    def from_list(cls, data_list):
        """
        Creates a DataSet object from regular Python lists.
        
        Args:
          data_list: 
            a standard Python list or list of lists containing the data.
            
        Returns:
          A new DataSet instance.
        """
        return cls(pd.DataFrame(data_list))
    
    @classmethod
    def from_unknown(cls, data):
        """
        Creates a DataSet object from an object of an unknown data type.
        
        Args:
          data: 
            the raw data set.  It can be stored as a Python list or pandas 
            dataframe.  If it is already a DataSet then it will just be 
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
          function: 
            the function which will be applied to each row in the data set.
        
        Returns:
          a pandas Series object which is the one dimensional result of 
            reduction (one value corresponding to each row).
        """
        return self.data_frame.apply(function, axis=1)

    def drop_column(self, index):
        """
        Creates a copy of the data set with a specified column removed.
        
        Args:
          index: 
            the index (0 based) of the column to drop.
          
        Returns:
          a new DataSet with the specified column removed.  The original 
          DataSet remains unaltered.
        """
        return DataSet(self.data_frame.drop(index, axis=1))

    def get_column(self, index):
        """
        Selects a column from the data set.
        
        Args:
          index: 
            the index (0 based) of the column to select.
          
        Returns:
          the columns at the specified index as a pandas Series object.  This 
          series is a view on the original data set, not a copy.  That means 
          any changes to it will also be applied to the original data set.
        """
        return self.data_frame.ix[:, index]

    def get_row(self, identifier):
        """
        Selects a single row from the dataset.
        
        Args:
          identifier:
            The id of the row to select.  If the DataSet has special indices 
            set up (ex: through a call to load with has_ids=True) these can 
            be used.  The integer index (0 based) can also be used.
            
        Returns:
          A pandas Series object representing the desired row.  NOTE: this is 
          a view on the original dataset.  Changes made to this Series will 
          also be made to the DataSet.
        """
        return self.data_frame.ix[identifier]

    def get_rows(self, indices):
        """
        Selects specified rows from the dataset.
        
        Args:
          indices: list
            The list of row indices (0 based) which should be selected.
        
        Returns:
          A new DataSet with the specified rows from the original.
        """
        return DataSet(self.data_frame.take(indices))

    def split(self, percent):
        """
        Splits the dataset in two.
        
        Args:
          percent: float
            The percentage of the original dataset samples which should be 
            placed in the first dataset returned.  The remainder are placed 
            in the second dataset.  This percentage must be specified as a 
            value between 0 and 1 inclusive.
        
        Returns:
          dataset1: DataSet object
            A subset of the original dataset with <percent> samples.
          dataset2: DataSet object
            A subset of the original dataset with 1-<percent> samples.
            
        Raises:
          ValueError if percent < 0 or percent > 1.
        """
        if percent < 0 or percent > 1:
            raise ValueError("Percentage value must be >= 0 and <= 1.")
        
        num_set1_samples = int(percent * self.num_samples())
        set1_rows = range(num_set1_samples)
        set2_rows = range(num_set1_samples, self.num_samples())
    
        # XXX refactor factories/constructor
        return DataSet.from_unknown(self.get_rows(set1_rows)), DataSet.from_unknown(self.get_rows(set2_rows))
    
    def fill_missing(self, fill_value):
        """
        Fill in missing data with a constant value.
        
        Args:
          fill_value:
            The value to insert wherever data is missing.
            
        Returns:
          A new DataSet object with missing values filled in.
        """
        # fillna returns a new DataFrame, does not modify the original
        return DataSet(self.data_frame.fillna(fill_value))
    

def load(path, has_ids=True, has_header=True, delimiter=","):
    """
    Loads a data set from a delimited text file.
    
    Args:
      path: 
        the path to the file containing the data set.
      has_ids: boolean
        set to False if the first column in the loaded dataset should not be 
        interpreted as a feature instead of sample identifiers.  Defaults to 
        True, i.e. first column are interpreted as sample identifiers.
      has_header: boolean
        set to False if the data being loaded does not have column headers on 
        the first line.  Defaults to true.
      delimiter: string
        the symbol used to separate columns in the file.  Default value is 
        ','.  Hint: delimiter for tab-delimited files is '\t'.
      
    Returns:
      An array-like object.
    """
    header = 0 if has_header else None
    id_col = 0 if has_ids else None
    return DataSet(pd.read_csv(path, index_col=id_col, header=header, 
                               delimiter=delimiter))
