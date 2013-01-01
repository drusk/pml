===
pml
===

*Simple interface to Python machine learning algorithms.*

Status: version 0.0.1, i.e. in early development.  Do not consider the API to 
be stable.

Installation
============
An installation script for Ubuntu is provided in the scripts/install 
directory.  This script sets up the required dependencies for you:

*  NumPy: http://docs.scipy.org/doc/numpy/user/install.html
*  matplotlib: http://matplotlib.org/users/installing.html
*  pandas: http://pandas.pydata.org/pandas-docs/stable/install.html
*  IPython: http://ipython.org/ipython-doc/stable/install/install.html

If you already have these dependencies, you can simply install PML using pip:

    sudo pip install git+https://github.com/drusk/pml.git

If you don't have pip but have easy_install, do

    sudo easy_install pip

If you don't have easy_install either, follow these installation instructions:
http://www.pip-installer.org/en/latest/installing.html 

Usage
=====
PML is intended to provide:

* an easy to use programmatic interface to machine learning and data 
  analysis tools
* an interactive environment for exploring data

For the latter, typing "pml" in your favourite shell will start up an 
interactive session using IPython (called the PML shell).

Documentation
=============
Online documentation can be found at:
http://pml.readthedocs.org/en/latest/index.html

Additionally, try out the tutorials by starting the PML shell and typing 
"tutorial".
