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
Utility functions for working with pandas datatypes, i.e. DataFrames and 
Series.

@author: drusk
"""

def are_dataframes_equal(dataframe1, dataframe2):
    """
    Compares two pandas DataFrame objects to see if they are equal.
    
    Args:
      dataframe1: pandas.DataFrame
      dataframe2: pandas.DataFrame
      
    Returns:
      True if the DataFrames are identically-labeled and all elements are the 
      same.  False otherwise. 
    """
    try:
        # reduce 2d boolean array down to a single answer
        return (dataframe1 == dataframe2).all().all()
    except Exception:
        # pandas throws:
        # "Exception: Can only compare identically-labeled DataFrame objects"
        return False
    