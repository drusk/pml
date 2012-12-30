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
    
    raw_input("[Press enter to continue]")
    
    lesson_load()
    
    end_tutorial()

def lesson_load():
    """
    Lesson on loading data.
    """
    clear_shell()
    print "PML Data Loading Lesson"
    print "-----------------------"
    print "PML allows you to load data from a file on disk using the 'load' function."
    print "\n"
    print "Try loading one of the tutorial data sets using the following command:"
    
    let_user_try("data = load(\"iris.data\")")
    
    print "Correct.  The iris data set has now been loaded into memory."
    
    print "\n"
    print "That completes the lesson on loading data sets."

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


########################## Utilities ###########################

def let_user_try(command):
    """
    Lets the user try entering the command.
    
    Args:
      command: string
        The command the user should enter.
    """
    while True:
        print command
        print "\n"
        
        user_command = pml_prompt()
        
        if commands_match(user_command, command):
            tutorial_interpreter.runsource(user_command)
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
