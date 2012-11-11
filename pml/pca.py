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
Implements principal component analysis (PCA) and related operations.

@author: drusk
"""

import numpy as np
import numpy.linalg as linalg
import pandas as pd

import model

class ReducedDataSet(model.DataSet):
    """
    A DataSet which has had dimensionality reduction performed on it.
    
    Columns are interpreted as features in the data set, and rows are 
    observations.
    
    This dimensionally reduced data set has all of the observations of the 
    original, but its features have been adjusted to be linear combinations 
    of the originals.  
    
    Those features with little variance may have been dropped during the 
    dimensionality reduction process.  Use the percent_variance() method to 
    find out how much of the original variance has been retained in the 
    reduced features.
    """
    
    def __init__(self, data, sample_ids, labels, eigenvalues):
        """
        Creates a new ReducedDataSet.
        
        Args:
          data: numpy.array
            The raw array with the new data.
          sample_ids: list
            The ids for the samples (rows, observations) in the data set.
          labels: pandas.Series
            The labels, if any, provided for the observations.
          eigenvalues: numpy.array (1D)
            The list of eigenvalues produced to determine which components in 
            the new feature space were most important.  This includes all of 
            the eigenvalues, not just the ones for the components selected.
        """
        # build a pandas DataFrame with the original row index
        dataframe = pd.DataFrame(data, index=sample_ids)
        super(ReducedDataSet, self).__init__(dataframe, labels=labels)
        
        self.eigenvalues = eigenvalues

    def percent_variance(self):
        """
        Calculates the percentage of the original DataSet's variance which is  
        still present in this dimensionally reduced DataSet.
        
        Returns:
          A floating point number between 0.0 and 1.0 representing the 
          percentage. 
        """
        return _percent_variance(self.eigenvalues, self.num_features())
    
    
def _percent_variance(eigenvalues, num_components):
    """
    Calculates the percentage of total variance found in the top princpal 
    components.
    
    Args:
      eigenvalues: numpy.array (1D)
        The list of all eigenvalues for a data set.
      num_components: int
        The number of principal components which will be selected.
        
    Returns:
      The percentage of total variance for the top number of principal 
      components selected.  This will be a floating point number between 0.0 
      and 1.0. 
    """
    # make sure eigenvalues are a numpy array (allows fancy indexing)
    eigenvalues = np.array(eigenvalues)
    
    # get indices sorted smallest to largest
    sorted_indices = np.argsort(eigenvalues)
    
    # get largest
    selected_indices = sorted_indices[-num_components:]
    
    return np.sum(eigenvalues[selected_indices]) / np.sum(eigenvalues)

def remove_means(dataset):
    """
    Remove the column mean from each value in the dataset.
    
    For example, if a certain column as values [1, 2, 3], the column mean is 
    2.  When the column means are removed, that column will then have the 
    values [-1, 0, 1].
    
    NOTE: the modifications are made in place in dataset.
    
    Args:
      dataset: model.DataSet
        The dataset to remove the column means from.
    """
    column_means = dataset.reduce_features(np.mean)
    
    for feature in dataset.feature_list():
        def subtract_mean(sample):
            """
            Subtracts the current column/feature's mean value from a sample.
            """
            return sample - column_means[feature]

        dataset.set_column(feature, 
                           dataset.get_column(feature).map(subtract_mean))

def pca(dataset, num_components):
    """
    Performs Principle Component Analysis (PCA) on a dataset.
    
    Args:
      dataset: model.DataSet
        The dataset to be analysed.
      num_components: int
        The number of principal components to select.
    """
    # 1. remove the mean
    dataset = dataset.copy()
    remove_means(dataset)
    
    # 2. compute the covariance matrix
    # rowvar=0 so that rows are interpreted as observations
    cov_mat = np.cov(dataset.get_data_frame(), rowvar=0)
    
    # 3. find the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = linalg.eig(cov_mat)

    # 4. sort the eigenvalues from largest to smallest
    # get a list of indices for the eigenvalues ordered largest to smallest
    indices = np.argsort(eigenvalues).tolist()
    indices.reverse()
    
    # 5. take the top N eigenvectors
    selected_indices = indices[:num_components]

    # 6. transform the data into the new space created by the top N eigenvectors
    transformed_data = np.dot(dataset.get_data_frame(), 
                              eigenvectors[:, selected_indices])
    
    return ReducedDataSet(transformed_data, dataset.get_sample_ids(), 
                          dataset.get_labels(), eigenvalues)
