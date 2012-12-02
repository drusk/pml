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
Runs a customised IPython shell.

Inspired by: 
https://github.com/ingenuitas/SimpleCV/blob/develop/SimpleCV/Shell/Shell.py

@author: drusk
"""

import sys
import webbrowser

from IPython.config.loader import Config
from IPython.frontend.terminal.embed import InteractiveShellEmbed

from pml.interactive.util import no_stdout

# Import pml library.  These imports will be available in the shell that 
# is created.
from pml.api import *

def magic_docs(self, arg):
    """
    The function called when the 'docs' magic is executed.  IPython requires 
    this function to accept two parameters, even though they are not used in 
    this instance.
    """
    with no_stdout():
        webbrowser.open("http://drusk.github.com/pml/")

def setup_shell():
    banner = "+-----------------------------------------------------------+\n"
    banner += " PML Shell - built on IPython.\n"
    banner += "+-----------------------------------------------------------+\n"
    banner += "Commands: \n"
    banner += "\t'exit', 'quit' or press 'CTRL + D' to exit the shell.\n"
    banner += "\t'docs' will open up the online documentation in a web \n"
    banner += "\tbrowser\n"

    exit_message = "\nExiting PML shell, good bye!"
    
    # XXX: this currently only supports IPython version 0.11 or higher!
    config = Config()
    config.PromptManager.in_template = "pml:\\#> "
    config.PromptManager.out_template = "pml:\\#: "
    
    shell = InteractiveShellEmbed(config=config, banner1=banner, 
                                  exit_msg=exit_message)
    shell.define_magic("docs", magic_docs)
    return shell

def run():
    shell = setup_shell()
    sys.exit(shell())

if __name__ == "__main__":
    run()
    