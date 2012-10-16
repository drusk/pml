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
Utilities for calculating metrics related to the performance of algorithms.

@author: drusk
"""

def compute_accuracy(results, dataset):
    """
    Used to calculate the percent accuracy of result labels from a classifier 
    if the classified data was an already labelled DataSet. 
    
    Args:
      results: pandas Series
        The results of classifying each sample in the dataset.
      dataset: DataSet object
        The original DataSet that was classified.  Must be labelled.
        
    Returns:
      The percent accuracy of the classification results, i.e. the number of 
      samples correctly classified divided by the total number of samples.  
      Should be a floating point number between 0 and 1.
      
    Raises:
      ValueError if dataset is not labelled.  Also raises a ValueError if the 
      number of samples in the results and dataset do not match.
    """
    if not dataset.is_labelled():
        raise ValueError(("DataSet must be labelled in order to compute ", 
                          "accuracy"))
    
    # TODO: should the actual indexes of each be compared?
    if len(results) != dataset.num_samples():
        raise ValueError(("Results and DataSet must have same number of ",
                          "samples"))
    
    true_labels = dataset.get_labels()

    correct = 0
    for ind in results.index:
        if results[ind] == true_labels[ind]:
            correct += 1
            
    return float(correct) / len(results)
