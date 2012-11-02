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

import itertools
import random

import pandas as pd

import model
from distance_utils import euclidean
from pandas_util import are_dataframes_equal

class UnlabelledDataSetError(Exception):
    """
    A custom exception to be thrown when trying to perform an operation that 
    requires a DataSet to be labelled when it is not.
    """
    
    def __init__(self):
        """
        Constructs a new exception.
        """
        Exception.__init__(self, ("Operation requires the DataSet to be "
                           "labelled, but it is not."))
    

class ClusteredDataSet(model.DataSet):
    """
    A collection of data which has been analysed by a clustering algorithm.  
    It contains both the original DataSet and the results of the clustering.  
    It provides methods for analysing these clustering results.  
    """
    
    def __init__(self, dataset, cluster_assignments):
        """
        Creates a new ClusteredDataSet.
        
        Args:
          dataset: model.DataSet
            A dataset which does not have cluster assignments.
          cluster_assignments: pandas.Series
            A Series with the cluster assignment for each sample in the 
            dataset.
        """
        super(ClusteredDataSet, self).__init__(dataset.get_data_frame(), 
                                               dataset.get_labels())
        self.cluster_assignments = cluster_assignments
    
    def get_cluster_assignments(self):
        """
        Retrieves the cluster assignments produced for this dataset by a 
        clustering algorithm.
        
        Returns:
          A pandas Series.  It contains the index of the original dataset 
          with a numerical value representing the cluster it is a part of.
        """
        return self.cluster_assignments
    
    def calculate_purity(self):
        """
        Calculate the purity, a measurement of quality for the clustering 
        results.
        
        Each cluster is assigned to the class which is most frequent in the 
        cluster.  Using these classes, the percent accuracy is then calculated.
        
        Returns:
          A number between 0 and 1.  Poor clusterings have a purity close to 0 
          while a perfect clustering has a purity of 1.
          
        Raises:
          UnlabelledDataSetError if the dataset is not labelled.
        """
        if not self.is_labelled():
            raise UnlabelledDataSetError()
        
        # get the set of unique cluster ids
        clusters = set(self.cluster_assignments.values)

        # find out what class is most frequent in each cluster
        cluster_classes = {}
        for cluster in clusters:
            # get the indices of rows in this cluster
            indices = self.cluster_assignments[self.cluster_assignments == 
                                               cluster].index

            # filter the labels series down to those in this cluster
            cluster_labels = self.labels[indices]

            # assign the most common label to be the label for this cluster
            cluster_classes[cluster] = cluster_labels.value_counts().idxmax()
        
        def get_label(cluster):
            """
            Get the label for a sample based on its cluster.
            """
            return cluster_classes[cluster]
        
        # get the list of labels as determined by each cluster's most frequent 
        # label
        labels_by_clustering = self.cluster_assignments.map(get_label)

        # See how the clustering labels compare with the actual labels.  
        # Return the percentage of indices in agreement.
        num_agreed = 0
        for ind in labels_by_clustering.index:
            if labels_by_clustering[ind] == self.labels[ind]:
                num_agreed += 1
        
        return float(num_agreed) / labels_by_clustering.size
        
    def calculate_rand_index(self):
        """
        Calculate the Rand index, a measurement of quality for the clustering 
        results.  It is essentially the percent accuracy of the clustering.
        
        The clustering is viewed as a series of decisions.  There are 
        N*(N-1)/2 pairs of samples in the dataset to be considered.  The 
        decision is considered correct if the pairs have the same label and 
        are in the same cluster, or have different labels and are in different 
        clusters.  The number of correct decisions divided by the total number 
        of decisions gives the Rand index, or accuracy.
        
        Returns:
          The accuracy, a number between 0 and 1.  The closer to 1, the better 
          the clustering.
        
        Raises:
          UnlabelledDataSetError if the dataset is not labelled.
        """
        if not self.is_labelled():
            raise UnlabelledDataSetError()
        
        correct = 0
        total = 0
        for index_combo in itertools.combinations(self.get_sample_ids(), 2):
            index1 = index_combo[0]
            index2 = index_combo[1]
                
            same_class = (self.labels[index1] == self.labels[index2])
            same_cluster = (self.cluster_assignments[index1] 
                            == self.cluster_assignments[index2])
            
            if same_class and same_cluster:
                correct += 1
            elif not same_class and not same_cluster:
                correct += 1
                
            total += 1
            
        return float(correct) / total
        

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

    return [pd.Series(map(rand_range, min_maxs), index=dataset.feature_list(), 
                      name = i) for i in range(k)]

def kmeans(dataset, k=2, create_centroids=create_random_centroids):
    """
    K-means clustering algorithm.
    
    This algorithm partitions a dataset into k clusters in which each 
    observation (sample) belongs to the cluster with the nearest mean.
    
    Args:
      dataset: model.DataSet
        The DataSet to perform the clustering on.
      k: int
        The number of clusters to partition the dataset into.
      create_centroids: function
        The function specifying how to create the initial centroids for the 
        clusters.  Defaults to creating them randomly.
        
    Returns:
      A ClusteredDataSet which contains the cluster assignments as well as the 
      original data.  In the cluster assignments, each sample index is 
      assigned a numerical value representing the cluster it is part of.
    """
    # If dataset is not already a model.DataSet object, make it one.
    dataset = model.as_dataset(dataset)
    
    # Initialize k centroids
    centroids = create_centroids(dataset, k)
    
    # Iteratively compute best clusters until they stabilize
    assignments = None
    clusters_changed = True
    while clusters_changed:
        centroids, new_assignments = _compute_iteration(dataset, centroids)
        if are_dataframes_equal(new_assignments, assignments):
            clusters_changed = False
        assignments = new_assignments
    
    return ClusteredDataSet(dataset, assignments)

def _get_distances_to_centroids(dataset, centroids):
    """
    Calculates the calc_distance from each data point to each centroid.
    
    Args:
      dataset: model.DataSet
        The DataSet whose samples are being 
      centroids: list of pandas Series
        The centroids to compare each data point with.
        
    Returns:
      A pandas DataFrame with a row for each sample in dataset and a column 
      for the distance to each centroid.
    """
    distances = {}
    for i, centroid in enumerate(centroids):
        def calc_distance(sample):
            # TODO: parameter to pass in calc_distance function
            return euclidean(sample, centroid)
    
        distances[i] = dataset.reduce_rows(calc_distance)

    # each dictionary entry is interpreted as a column
    return pd.DataFrame(distances)

def _compute_iteration(dataset, centroids):
    """
    Computes an iteration of the k-means algorithm.
    
    Args:
      dataset: model.DataSet
        The dataset being clustered.
      centroids: list of pandas Series
        The current centroids at the start of the iteration.
        
    Returns:
      new_centroids: list of pandas Series
        The updated centroids.
      cluster_assignments: pandas Series
        The current cluster assignments for each sample.
    """
    # 2. Calculate calc_distance from each data point to each centroid
    distances = _get_distances_to_centroids(dataset, centroids)

    # 3. Find each datapoint's nearest centroid
    cluster_assignments = distances.idxmin(axis=1)

    def nearest_centroid(sample_index):
        return cluster_assignments[sample_index]
        
    # 4. Calculate mean position of datapoints in each centroid's cluster
    new_centroids = dataset.get_data_frame().groupby(nearest_centroid).mean()

    # XXX turning each row in dataframe into a series... refactor!    
    list_of_series = [new_centroids.ix[ind] for ind in new_centroids.index]
    
    return list_of_series, cluster_assignments
