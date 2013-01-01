#!/bin/sh

# Installs PML and its dependencies on Ubuntu.

sudo apt-get install -y python-setuptools python-dev git
sudo easy_install pip
sudo pip install numpy
sudo apt-get install -y python-matplotlib
sudo pip install pandas
sudo pip install git+https://github.com/drusk/pml.git
