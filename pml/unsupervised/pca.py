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

from pml.data import model
import plotting

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

def _get_cov_mat_eigen_values_and_vectors(dataset):
    """
    Calculates the eigenvalues and eigenvectors for the covariance matrix of a 
    DataSet. 
    
    Args:
      dataset: model.DataSet
        The data whose covariance matrix will be calculated.
    
    Returns:
      eigenvalues: numpy.array
        A 1D array of the eigenvalues of the covariance matrix.
      eigenvectors: numpy.array
        A 2D array of the eigenvectors of the covariance matrix.
    """
    # rowvar=0 so that rows are interpreted as observations
    cov_mat = np.cov(dataset.get_data_frame(), rowvar=0)
    
    eigenvalues, eigenvectors = linalg.eig(cov_mat)
    
    return eigenvalues, eigenvectors

def _copy_and_remove_means(dataset):
    """
    Copies the DataSet before removing the column means in order to preserve 
    the original data.
    
    Args:
      dataset: model.DataSet
        The DataSet to copy and remove means from.
    
    Returns:
      The new, copied DataSet with column means removed.
    """
    dataset = dataset.copy()
    remove_means(dataset)
    return dataset

def _get_descending_cov_mat_eigenvalues(dataset):
    """
    Get the eigenvalues of the covariance matrix sorted largest to smallest.
    
    Args:
      dataset: model.DataSet
        The data whose covariance matrix will be calculated.
        
    Returns:
      eigenvalues: list
        The list of eigenvalues in descending order of magnitude.
    """
    eigenvalues, _ = _get_cov_mat_eigen_values_and_vectors(dataset)
    eigenvalues = eigenvalues.tolist()
    
    # sort from largest to smallest
    eigenvalues.sort()
    eigenvalues.reverse()
    return eigenvalues

def plot_pct_variance_per_principal_component(dataset, plot_type="bar"):
    """
    Generates a plot to visualize the percentage of variance captured 
    by each principal component in the data set.

    Args:
      dataset: model.DataSet
        The data set whose principal components will be examined.  Should not 
        already be reduced.
      plot_type: string
        The plot type to generate.  Supported plot types are:
          'bar': vertical bar chart
          'barh': horizontal bar chart
          'line': line chart
        Default is 'bar'. 
        
    Returns:
      void, but produces a matplotlib plot. 
      
    Raises:
      UnsupportedPlotTypeError if plot_type is not recognized.
    """
    # Fail early: check plot type here right away even though the plotting 
    # module will check it later.  Don't want a user with a large data set to 
    # wait for all the processing to occur only to find out they made a typo 
    # on the plot type.
    plotting.verify_supported_series_plot_type(plot_type)
    variances = get_pct_variance_per_principal_component(dataset)
    plotting.plot_percent_series(variances, plot_type)

def get_pct_variance_per_principal_component(dataset):
    """
    Determines the percentage of variance captured by each principal component 
    in the data set.
    
    Args:
      dataset: model.DataSet
        The data set whose principal components will be examined.  Should not 
        already be reduced.
        
    Returns:
      variances: pandas.Series
        The percentage of variance (as a float between 0.0 and 1.0) for each 
        principal component.
    """
    eigenvalues = _get_descending_cov_mat_eigenvalues(dataset)
    return pd.Series(eigenvalues) / np.sum(eigenvalues)

def recommend_num_components(dataset, min_pct_variance=0.9):
    """
    Recommends the number of principal components that should be selected in 
    order to keep a minimum specified percentage of the original data's 
    variance while also minimizing dimensionality.
    
    Args:
      dataset: model.DataSet
        The dataset in question.
      min_pct_variance: float
        The minimum percent of variance which should be maintained when 
        selecting the recommended number of principal components.  Should be 
        between 0.0 and 1.0.
        Defaults to 0.9 (i.e. 90%).
        
    Returns:
      The integer number of principal components which should be selected for 
      Principal Component Analysis.
      
    Raises:
      ValueError if min_pct_variance is < 0 or > 1.
    """
    if min_pct_variance < 0 or min_pct_variance > 1:
        raise ValueError("Invalid minimum percent variance "
                         "(must be between 0 and 1): %f" %min_pct_variance)
    
    dataset = _copy_and_remove_means(dataset)
    eigenvalues = _get_descending_cov_mat_eigenvalues(dataset)
    
    cumulative_pct_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    
    num_components = 1
    for pct_variance in cumulative_pct_variance:
        if pct_variance >= min_pct_variance:
            return num_components
        
        num_components += 1
        
    # should never reach this point since if all components are used the 
    # percent variance will be 100%, and the min percent variance specified 
    # can never be greater than 100% 

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
    dataset = _copy_and_remove_means(dataset)
    eigenvalues, eigenvectors = _get_cov_mat_eigen_values_and_vectors(dataset)

    # get a list of indices for the eigenvalues ordered largest to smallest
    indices = np.argsort(eigenvalues).tolist()
    indices.reverse()
    
    # take the top N eigenvectors
    selected_indices = indices[:num_components]

    # transform the data into the new space created by the top N eigenvectors
    transformed_data = np.dot(dataset.get_data_frame(), 
                              eigenvectors[:, selected_indices])
    
    return ReducedDataSet(transformed_data, dataset.get_sample_ids(), 
                          dataset.get_labels(), eigenvalues)
