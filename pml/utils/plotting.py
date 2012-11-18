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
Provides capabilities for plotting DataSets.

@author: drusk
"""

import matplotlib.pyplot as plt
from pandas.tools.plotting import radviz

from pml.utils import errors

def plot_radviz(dataset):
    """
    Generates a RadViz plot of the provided DataSet.  RadViz is useful for 
    visualizing data with more than two dimensions.
    """
    # radviz takes a pandas DataFrame and the name of the column which 
    # contains class membership info. 
    # therefore need to pass in the dataset's merged data and labels
    radviz(dataset.get_labelled_data_frame(), dataset.get_labels().name)
    plt.show()

def plot_percent_series(series, plot_type):
    """
    Plots a series of values against a y axis formatted for percentages.
    
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
    verify_supported_series_plot_type(plot_type)
    series.plot(kind=plot_type)
    plt.show()
    
def verify_supported_series_plot_type(plot_type):
    """
    Checks if a plot type is among those supported for plotting a series.
    
    Args:
      plot_type: string
        The plot type to generate.  Supported plot types are:
          'bar': vertical bar chart
          'barh': horizontal bar chart
          'line': line chart
    
    Returns:
      void
          
    Raises:
      UnsupportedPlotTypeError if the plot type is not supported.
    """
    supported_plot_types = ["bar", "barh", "line"]
    if plot_type not in supported_plot_types:
        raise errors.UnsupportedPlotTypeError(plot_type, supported_plot_types)

