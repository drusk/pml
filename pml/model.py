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
Models for the data being analysed and manipulated.

@author: drusk
"""

import pandas as pd
import random as rand

class DataSet(object):
    """
    A collection of data that may be analysed and manipulated.
    
    Columns are interpreted as features in the data set, and rows are samples 
    or observations.
    """
    
    def __init__(self, data, labels=None):
        """
        Creates a new DataSet from data of an unknown type.  If data is itself 
        a DataSet object, then its contents are copied and a new DataSet is 
        created from the copies.
    
        Args:
          data: 
            Data of unknown type.  The supported types are:
                1) pandas DataFrame
                2) Python lists or pandas DataFrame
                3) an existing DataSet object 
          labels: pandas Series or Python list
            The classification labels for the samples in data.  If they are 
            not known (i.e. it is an unlabelled data set) the value None 
            should be used.  Default value is None (unlabelled).
        
        Raises:
          ValueError if the data is not of a supported type.  A ValueError is 
          also raised if labels are supplied but there is a mismatch between 
          the number of labels and samples.    
        """
        if isinstance(data, pd.DataFrame):
            self._dataframe = data
        elif isinstance(data, list):
            self._dataframe = pd.DataFrame(data)
        elif isinstance(data, DataSet):
            self._dataframe = data._dataframe.copy()
        else:
            raise ValueError("Unsupported representation of data set")        
        
        if labels is not None and self.num_samples() != len(labels):
            raise ValueError(("Number of labels doesn't match the number ", 
                              "of samples in the data."))

        if isinstance(labels, list):
            self.labels = pd.Series(labels)
        else:
            # will just be None if there are no labels
            self.labels = labels
        
    def __str__(self):
        """
        Returns:
          This object's string representation, primarily for debugging 
          purposes.
        """
        return "%s\nLabelled: %s" % (self._dataframe.__str__(), self.is_labelled())
    
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
        
    def get_data_frame(self):
        """
        Retrieve the DataSet's underlying data as a pandas DataFrame object.
        
        Returns:
          A pandas DataFrame with the DataSet's main data and the labels if 
          they are present attached as the rightmost column.
        """
        if not self.is_labelled():
            return self._dataframe
        
        return pd.concat([self._dataframe, pd.DataFrame(self.labels)], axis=1)
        
    def num_samples(self):
        """
        Returns:
          The number of samples (rows) in the data set.
        """    
        return self._dataframe.shape[0]
    
    def num_features(self):
        """
        Returns:
          The number of features (columns) in the data set.
        """
        return self._dataframe.shape[1]
    
    def is_labelled(self):
        """
        Returns:
          True if the dataset has classification labels for each sample, 
          False otherwise.
        """
        return self.labels is not None
    
    def feature_list(self):
        """
        Returns:
          The list of features in the dataset. 
        """
        return self._dataframe.columns.tolist()
    
    def get_labels(self, indices=None):
        """
        Selects classification labels for the specified samples (rows) in the 
        DataSet.

        Args:
          indices: list
            The list of row indices (0 based) which should be selected.  
            Defaults to None, in which case all labels are selected.
        
        Returns:
          A pandas Series with the classification labels.
        """
        if indices is None:
            return self.labels
        else:
            return self.labels.take(indices)
    
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
        return self._dataframe.apply(function, axis=1)

    def reduce_features(self, function):
        """
        Performs a feature-wise (i.e. column-wise) reduction of the data set.
        
        Args:
          function:
            The function which will be applied to each feature in the data set.
            
        Returns:
          A pandas Series object which is the one dimensional result of the 
          reduction (one value corresponding to each feature).
        """
        return self._dataframe.apply(function, axis=0)

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
        return DataSet(self._dataframe.drop(index, axis=1))

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
        return self._dataframe.ix[:, index]

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
        return self._dataframe.ix[identifier]

    def get_rows(self, indices):
        """
        Selects specified rows from the dataset.
        
        Args:
          indices: list
            The list of row indices (0 based) which should be selected.
        
        Returns:
          A new DataSet with the specified rows from the original.
        """
        labels = self.labels.take(indices) if self.is_labelled() else None
        return DataSet(self._dataframe.take(indices), labels=labels)

    def split(self, percent, random=False):
        """
        Splits the dataset in two.
        
        Args:
          percent: float
            The percentage of the original dataset samples which should be 
            placed in the first dataset returned.  The remainder are placed 
            in the second dataset.  This percentage must be specified as a 
            value between 0 and 1 inclusive.
          random: boolean
            Set to True if the samples selected for each new dataset should 
            be picked randomly.  Defaults to False, meaning the samples are 
            taken in their existing order.
        
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
        
        if not random:
            set1_rows = range(num_set1_samples)
            set2_rows = range(num_set1_samples, self.num_samples())
        else:
            all_rows = range(self.num_samples())
            rand.shuffle(all_rows)
            set1_rows = all_rows[:num_set1_samples]
            set2_rows = all_rows[num_set1_samples:]
    
        return self.get_rows(set1_rows), self.get_rows(set2_rows)
    
    def fill_missing(self, fill_value):
        """
        Fill in missing data with a constant value.  Changes are made in-place.
        
        Args:
          fill_value:
            The value to insert wherever data is missing.
            
        Returns:
          Void.  The changes to the DataSet are made in-place.
        """
        return self._dataframe.fillna(fill_value, inplace=True)
    
    def combine_labels(self, to_combine, new_label):
        """
        Combines classification labels to have some new value.
        
        For example, consider a dataset with labels "cat", "crow" and 
        "pidgeon".  Maybe you are only really worried about whether something 
        is a cat or a bird, so you want to combine the "crow" and "pidgeon" 
        labels into a new one called "bird".
        
        Args:
          to_combine: list
            The list of labels which will be combined to form one new 
            classification label.
          new_label: string
            The new classification label for those which were combined.
        """
        # pd.Series.replace returns a new Series, leaves original unmodified
        self.labels = self.labels.replace(to_combine, value=new_label)


def as_dataset(data):
    """
    Creates a DataSet from the provided data.  If data is already a DataSet, 
    return it directly.  Use this instead of the DataSet constructor if you 
    don't know whether your data is a DataSet already, but you don't want to 
    create a new one if it already is.
    
    Args:
      data: 
        Data of unknown type.  It may be a Python list or pandas DataFrame or 
        DataSet object.
        
    Returns:
      A DataSet object.  If the data was already a DataSet then the input 
      object will be directly returned.
          
    Raises:
      ValueError if the data is not of a supported type.    
    """
    if isinstance(data, DataSet):
        return data
    else:
        return DataSet(data)
    