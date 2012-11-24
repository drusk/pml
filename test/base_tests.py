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
Base test classes which contain functionality that may be useful for unit 
testing multiple modules.

@author: drusk
"""

import unittest
import os

import pandas as pd

from pml.data.model import DataSet

class BaseFileLoadingTest(unittest.TestCase):
    """
    A test case class which has a method for loading files relative to the 
    subclassing test case.
    """
    
    def relative(self, path):
        """
        Marks a path as being relative, so that it will be converted to 
        absolute.  A
        
        Args:
          path: the relative path.
        
        Returns:
          The absolute path for the relative path.
        """
        if path.startswith("/"):
            path = path[1:]
        path.replace("/", os.sep)
        return os.path.dirname(__file__) + os.sep + path


class BaseDataSetTest(unittest.TestCase):
    """
    A test case class which has methods for creating test DataSets.
    """
    
    def create_dataset(self, labels=None, sample_ids=None):
        """
        Used to create a DataSet for testing purposes with optional labels 
        and sample_ids.
        """
        raw_data = [[1, 2, 3] for _ in xrange(len(labels))]
        df = pd.DataFrame(raw_data, index=sample_ids)
        return DataSet(df, labels=labels)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    