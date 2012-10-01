"""
Base test classes which contain functionality that may be useful for unit 
testing multiple modules.

@author: drusk
"""

import unittest
import os

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

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()