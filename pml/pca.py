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
    
    def __init__(self, data, row_index, labels, eigen_values):
        """
        Creates a new ReducedDataSet.
        
        Args:
          data: numpy.array
            The raw array with the new data.
          row_index: pandas.Index
            The index of row (observation) names.
          labels: pandas.Series
            The labels, if any, provided for the observations.
          eigen_values: numpy.array (1D)
            The list of eigen values produced to determine which components in 
            the new feature space were most important.
        """
        # build a pandas DataFrame with the original row index
        dataframe = pd.DataFrame(data, index=row_index)
        super(ReducedDataSet, self).__init__(dataframe, labels=labels)
        
        self.eigen_values = eigen_values
    

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
    eigen_values, eigen_vectors = linalg.eig(cov_mat)

    # 4. sort the eigenvalues from largest to smallest
    # get a list of indices for the eigenvalues ordered largest to smallest
    indices = np.argsort(eigen_values).tolist()
    indices.reverse()
    
    # 5. take the top N eigenvectors
    selected_indices = indices[:num_components]

    # 6. transform the data into the new space created by the top N eigenvectors
    transformed_data = np.dot(dataset.get_data_frame(), 
                              eigen_vectors[:, selected_indices])
    
    # TODO DataSet.get_row_index()
    return ReducedDataSet(transformed_data, dataset.get_row_index(), 
                          dataset.get_labels(), eigen_values)
