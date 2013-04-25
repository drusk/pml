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

def find(series, search_for):
    """
    Retrieves the indices of a pandas Series which have a specified value.
    
    Args:
      series: pandas.Series
        The series to search and retrieve indices from.
      search_for:
        Either a single value to search for, or a list of several values
        to search for.  If it is a list, then if an index has any of the
        specified values it will be included.
        
    Returns:
      indices: pandas.Index
        The indices which held one of the specified values.

    Example 1, search for single value:
      series = pd.Series(["hostile", "friendly", "friendly", "neutral"],
                           index=["wolf", "cat", "dog", "mouse"])
      indices = find(series, "friendly")
      assert_that(indices, contains("cat", "dog")

    Example 2, search for multiple values:
      series = pd.Series(["hostile", "friendly", "friendly", "neutral"],
                           index=["wolf", "cat", "dog", "mouse"])
      indices = find(series, ["friendly", "neutral"])
      assert_that(indices, contains("cat", "dog", "mouse")
    """
    # Convert values to search for to a list if they aren't
    values = [search_for] if type(search_for) is not list else search_for

    overall_matches = None
    for value in values:
        matches = (series == value)

        if overall_matches is None:
            overall_matches = matches
        else:
            overall_matches |= matches

    return series.index[overall_matches]

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

def is_series_numeric(series):
    """
    Checks if the series data type is numeric.
    
    Args:
      series: pd.Series
        The series whose data type will be checked.
        
    Returns:
      True if the series is numeric, i.e. values are some form of int or 
      float.
    """
    dtype_name = series.dtype.name
    return dtype_name.startswith("int") or dtype_name.startswith("float")
