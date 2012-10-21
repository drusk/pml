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
Clustering algorithms for unsupervised learning tasks.

@author: drusk
"""

import random

import pandas as pd

import model

def create_random_centroids(dataset, k):
    """
    Initializes centroids at random positions.
    
    The random value chosen for each feature will always be limited to the 
    range of values found in the dataset.  For example, if a certain feature 
    has a minimum value of 0 in the dataset, and maximum value of 9, the
    
    Args:
      dataset: DataSet
        The DataSet to create the random centroids for.
      k: int
        The number of centroids to create.
        
    Returns:
      A list of centroids.  Each centroid is a pandas Series with the same 
      labels as the dataset's headers.
    """
    min_maxs = zip(dataset.reduce_features(min).values, 
                   dataset.reduce_features(max).values)

    def rand_range(range_tuple):
        """
        Generates a random floating point number in the range specified by 
        the tuple.
        """
        return random.uniform(range_tuple[0], range_tuple[1])

    return [pd.Series(map(rand_range, min_maxs), index=dataset.feature_list()) 
            for _ in range(k)]

def kmeans(dataset, k=2):
    """
    K-means clustering algorithm.
    
    This algorithm partitions a dataset into k clusters in which each 
    observation (sample) belongs to the cluster with the nearest mean.
    
    Args:
      dataset: model.DataSet
        The DataSet to perform the clustering on.
      k: int
        The number of clusters to partition the dataset into.
    """
    # If dataset is not already a model.DataSet object, make it one.
    dataset = model.as_dataset(dataset)
    
    # 1. Initialize k centroids
    centroids = create_random_centroids(dataset, k)
    
    # 2. Calculate distance from each data point to each centroid
    
    # 3. Put datapoints in group with nearest centroid
    
    # 4. Calculate mean position of datapoints in each centroid's cluster
    
    # 5. Repeat 2-4 until clusters are stable
