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

def pca(dataset):
    """
    """
    # 1. remove the mean
    
    # 2. compute the covariance matrix
    
    # 3. find the eigenvalues and eigenvectors of the covariance matrix
    
    # 4. sort the eigenvalues from largest to smallest
    
    # 5. take the top N eigenvectors
    
    # 6. transform the data into the new space created by the top N eigenvectors
