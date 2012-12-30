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
Utilities for the shell.

Learned about redirecting standard output from:
http://stackoverflow.com/questions/2828953/
silent-the-stdout-of-a-function-in-python-without-trashing-sys-stdout-and-restor

@author: drusk
"""

import os.path
import sys
import contextlib

class FakeFile(object):
    """
    Has no-op writes.
    """
    
    def write(self, value):
        pass
    
    
@contextlib.contextmanager
def no_stdout():
    """
    Wrap this around code for which you want to silence standard output 
    (using a with statement).
    """
    saved_stdout = sys.stdout
    sys.stdout = FakeFile()
    yield
    sys.stdout = saved_stdout

def get_samples_basepath():
    """
    Determines the absolute path to the directory containing sample data.
    
    Returns:
      basepath: string
    """
    return os.path.join(os.path.dirname(__file__), "sample_data")

def list_samples():
    """
    Gets a list of the filenames of sample data sets.
    
    Returns:
      sample_names: list
    """
    return os.listdir(get_samples_basepath())
