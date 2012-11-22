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
Classification algorithms for supervised learning tasks.

@author: drusk
"""

from pml.data import model
from pml.utils.errors import UnlabelledDataSetError

class AbstractClassifier(object):
    """
    This is the base class which classification algorithms should extend.  It 
    provides the common functionality for each classifier. 
    """
    
    def __init__(self, training_set):
        """
        Constructs the classifier.  Subclasses may have additional parameters 
        in their constructors.
        
        Args:
          training_set:
            A labelled DataSet object used to train the classifier.

        Raises:
          UnlabelledDataSetError if the training set is not labelled.
        """
        if not training_set.is_labelled():
            raise UnlabelledDataSetError(custom_message=("Training set must "
                                                         "be labelled."))
        self.training_set = training_set

    def __repr__(self):
        """
        This gets called when the object's name is typed into IPython on its 
        own line, causing a string representation of the object to be 
        displayed.
        
        NOTE: override __str__ in subclass to set this.
        
        Returns:
          This object's string representation, primarily for debugging 
          purposes.
        """
        return self.__str__()

    def classify_all(self, dataset):
        """
        Predicts the classification of each sample in a dataset.
        
        Args:
          dataset: DataSet compatible object (see DataSet constructor)
            the dataset whose samples (observations) will be classified.
            
        Returns:
          A ClassifiedDataSet which contains the classification results for 
          each sample.  It also contains the original data.
        """
        dataset = model.as_dataset(dataset)
        return ClassifiedDataSet(dataset, dataset.reduce_rows(self.classify))
        
    def classify(self, sample):
        """
        Predicts a sample's classification based on the training set.
        
        NOTE: classifiers which subclass this AbstractClassifier must 
        implement this method.
        
        Args:
          sample: 
            the sample or observation to be classified.
          
        Returns:
          The sample's classification.
          
        Raises:
          ValueError if sample doesn't have the same number of features as 
          the data in the training set.
        """
        raise NotImplementedError(("Classifiers must implement the "
                                   "'classify' method."))
        

class ClassifiedDataSet(model.DataSet):
    """
    A collection of data which has been analysed by a classification 
    algorithm.  It contains both the original DataSet and the results of 
    the classification.  It provides methods for analysing these 
    classification results.  
    """
    
    def __init__(self, dataset, classifications):
        """
        Creates a new ClassifiedDataSet.
        
        Args:
          dataset: model.DataSet
            A dataset which has been classified but does not hold the results.
          classifications: pandas.Series
            A Series with the classification results.
        """
        super(ClassifiedDataSet, self).__init__(dataset.get_data_frame(), 
                                                dataset.get_labels())
        self.classifications = classifications
    
    def get_classifications(self):
        """
        Retrieves the classifications computed for this dataset.
        
        Returns:
          A pandas Series containing each sample's classification.
        """
        return self.classifications
    
    def compute_accuracy(self):
        """
        Calculates the percent accuracy of classification results.
        
        Returns:
          The percent accuracy of the classification results, i.e. the number 
          of samples correctly classified divided by the total number of 
          samples.  Should be a floating point number between 0 and 1.
          
        Raises:
          UnlabelledDataSetError if the dataset is not labelled.
        """
        if not self.is_labelled():
            raise UnlabelledDataSetError()
        
        correct = 0
        for ind in self.classifications.index:
            if self.classifications[ind] == self.labels[ind]:
                correct += 1
            
        return float(correct) / len(self.classifications)

