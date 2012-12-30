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
from code import InteractiveInterpreter
from subprocess import call

from pml.api import *

# Passing globals() to interpreter gives it access to the pml api imported 
# above.
tutorial_interpreter = InteractiveInterpreter(globals())

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
    
    wait_for_enter_to_continue()
    
    lesson_dataset_intro()
    
    end_tutorial()

def lesson_dataset_intro():
    """
    Introductory lesson on loading and examining data.
    """
    clear_shell()
    print "PML DataSet Introduction"
    print "------------------------"
    print "This lesson introduces the basics of PML's core data structure, the DataSet."
    print "A DataSet consists of a collection of samples which have values for a set of"
    print " features.  An example might be specific course grades for a set of students."
    print "\n"
    print "DataSets can be conveniently created by loading data from a file on disk using" 
    print "the 'load' function."
    print "\n"
    print "Try loading one of the tutorial data sets using the following command:"
    
    let_user_try("data = load(\"iris.csv\")")
    
    print "Correct.  The iris data set has now been loaded into memory."
    print "\n"
    print "By default the data is assumed to be in comma separated value format."
    print "Other formats can be loaded by specifying the \"delimiter\" option."
    print "For example, "
    print "    'data = load(\"iris.tsv\", delimiter=\"\\t\")'"
    print "would load a tab separated file."
    
    print "\n"
    wait_for_enter_to_continue()
    print "\n"
    
    print "Once a data set has been loaded, many operations can be performed on it."
    print "One of the simplest is listing its features.  Try it out:"
    
    let_user_try("data.feature_list()")
    
    print "Good.  You can also check the number of samples in the data set:"
    
    let_user_try("data.num_samples()")
    
    print "In this case we see there are 150 samples."
    print "\n"
    
    print "The iris data set that we loaded is labelled.  This means the actual "
    print "classifications of the samples are known.  You can check if a DataSet "
    print "is labelled like this:"
    
    let_user_try("data.is_labelled()")
    
    print "If you want a quick summary of the DataSet without calling all of these methods "
    print "individually, simply enter the name of the DataSet at the prompt:"
    
    let_user_try("data")
    
    print "To wrap up this lesson, lets try plotting our DataSet to see if there are "
    print "any visual patterns.  Since our DataSets can be more than two dimensional, "
    print "we need a visualization method such as 'Radviz' to fit it on a 2D plot."
    print "Try creating a plot (once it is created, close it to continue):"
    
    let_user_try("data.plot_radviz()")
    
    print "\n"
    print "That completes the introductory lesson on DataSets."
    print "The next lesson will cover using classifiers."
    
    wait_for_enter_to_continue()

def end_tutorial():
    """
    Exit the tutorial.
    """
    clear_shell()
    print "-" * 70
    print "You have reached the end of the PML tutorial, congratulations!"
    print "You may also find the module documentation useful.  It can "
    print "be accessed from the PML shell by typing 'docs'."
    print "-" * 70


########################## Utilities ###########################

def let_user_try(command):
    """
    Lets the user try entering the command.
    
    Args:
      command: string
        The command the user should enter.
    """
    print "\n"
    
    while True:
        print command
        print "\n"
        
        user_command = pml_prompt()
        
        if commands_match(user_command, command):
            tutorial_interpreter.runsource(user_command)
            print "\n"
            return
        
        print "\n"
        print "The command was not entered correctly, try again."
        
def commands_match(user_command, expected_command):
    """
    Checks if the commands are essentially equivalent after whitespace
    differences and quote differences.
    
    Returns:
      match: boolean
        True if the commands appear to be equivalent.
    """
    def normalize(command):
        return command.replace(" ", "").replace("'", "\"")
    
    return normalize(user_command) == normalize(expected_command)

def wait_for_enter_to_continue():
    """
    Prompt the user to press enter before continuing.
    """
    raw_input("[Press enter to continue]")

def pml_prompt():
    """
    Reads user input given the PML prompt.
    """
    return raw_input("pml:> ")

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

def should_quit():
    """
    Asks the user if they want to continue with the next lesson or quit the 
    tutorial.
    
    Returns:
      should_quit: boolean
        True if the user wants to quit the tutorial.
    """
    print "\n"
    print ("Type 'quit' to end the tutorial, or press enter to continue to "
           "the next lesson.")
    selection = pml_prompt()
    return selection.lower() == "quit"
