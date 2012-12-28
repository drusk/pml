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
TODO: Short module level description.

@author: drusk
"""

class Tree(object):
    """
    """
    
    def __init__(self, root_node):
        """
        """
        self._root_node = root_node
        
    def get_root_node(self):
        """
        """
        return self._root_node
    

class Node(object):
    """
    """
    
    def __init__(self, value):
        """
        """
        self._value = value
        self._children = {}
    
    def get_value(self):
        """
        """
        return self._value
    
    def add_child(self, branch, child):
        """
        """
        self._children[branch] = child
    
    def get_child(self, branch):
        """
        """
        return self._children[branch]
    
    def get_branches(self):
        """
        """
        return self._children.keys()
    
    def is_leaf(self):
        """
        """
        return len(self._children) == 0
    
    