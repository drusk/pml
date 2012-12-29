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
An interactive tutorial of pml's basic functionality.

@author: drusk
"""

import platform
from subprocess import call

def clear_shell():
    """
    Clears the shell's screen.
    """
    if platform.system() == "Windows":
        # shell=True for reason in this stackoverflow answer
        # http://stackoverflow.com/questions/3022013/windows-cant-find-the-file-on-subprocess-call
        call("cls", shell=True)
    else:
        call("clear")

def begin_tutorial():
    """
    Begin the tutorial.
    """
    clear_shell()
    print "+----------------------------+"
    print " Welcome to the PML tutorial."
    print "+----------------------------+"
    print "\n"
    print "This is an interactive tutorial that will teach you the basics "
    print "of using PML for data analysis and machine learning tasks."
    print "\n"
    
    raw_input("[Press enter to continue]")
    
    tutorial_load()
    
    end_tutorial()

def tutorial_load():
    """
    Tutorial for loading data.
    """
    

def end_tutorial():
    """
    Exit the tutorial.
    """
    print "\n"
    print "-" * 70
    print "That concludes PML's tutorial."
    print "You may also find the module documentation useful.  It can "
    print "be accessed from the PML shell by typing 'docs'."
    print "-" * 70
