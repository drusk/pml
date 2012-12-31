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
Project setup.

@author: drusk
"""

from setuptools import setup

setup(
      name="pml",
      version="0.0.1",
      
      author="David Rusk",
      author_email="drusk@uvic.ca",
      
      packages=["pml", "pml.interactive", "pml.data", "pml.supervised", 
                "pml.supervised.decision_trees", "pml.unsupervised", 
                "pml.tools", "pml.utils"],
      
      # Allows sample data to be loaded in shell
      package_data={"pml.interactive": ["sample_data/*"]},
      
      scripts=["scripts/pml"],
      
      url="http://github.com/drusk/pml",
      license="LICENSE",
      
      description="Simple interface to Python machine learning algorithms.",
      long_description=open("README.rst").read(),
      
      install_requires=[
                        "ipython >= 0.11",
                        "pandas >= 0.8.1",
                        "matplotlib",
                        "numpy >= 1.6.1",
                        ]
      )
