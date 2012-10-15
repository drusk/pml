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
Utilities for loading data sets.

@author: drusk
"""

import pandas as pd
import model

def load(path, has_ids=True, has_header=True, has_labels=True, delimiter=","):
    """
    Loads a data set from a delimited text file.
    
    Args:
      path: 
        the path to the file containing the data set.
      has_ids: boolean
        set to False if the first column in the loaded dataset should not be 
        interpreted as a feature instead of sample identifiers.  Defaults to 
        True, i.e. first column are interpreted as sample identifiers.
      has_header: boolean
        set to False if the data being loaded does not have column headers on 
        the first line.  Defaults to true.
      has_labels: boolean
        set to False if the data being loaded does not have classification 
        labels for each sample.  Defaults to True.  The labels should be the 
        last column in the dataset being loaded.
      delimiter: string
        the symbol used to separate columns in the file.  Default value is 
        ','.  Hint: delimiter for tab-delimited files is '\t'.
      
    Returns:
      A DataSet object.
    """
    header = 0 if has_header else None
    id_col = 0 if has_ids else None

    dataframe = pd.read_csv(path, index_col=id_col, header=header, 
                            delimiter=delimiter)
    
    if has_labels:
        labels = dataframe.pop(dataframe.columns[-1])

    return model.DataSet(dataframe)
