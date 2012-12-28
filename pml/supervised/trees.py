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
Data structures for representing trees.

@author: drusk
"""

class Tree(object):
    """
    A tree containing nodes which are connected to each other by directed 
    edges.
    """
    
    def __init__(self, root_node):
        """
        Constructs a new tree.
        
        Args:
          root_node: Node
            The node which will be at the root of the tree.
        """
        self._root_node = root_node
        
        self._all_nodes = root_node.get_all_descendants()
        self._all_nodes.append(root_node)
        
    def get_root_node(self):
        """
        Retrieves the root node of the tree.  This node will have only 
        outgoing edges (children), no incoming edges (parents).
        
        Returns:
          root: Node
            The root node.
        """
        return self._root_node
    
    def get_leaves(self):
        """
        Retrieves all leaf nodes from the tree.
        
        Returns:
          leaves: list(Node)
        """
        return [node for node in self._all_nodes if node.is_leaf()]
    
    def get_num_leaves(self):
        """
        Counts the number of leaves in the tree.
        
        Returns:
          num_leaves: int
        """
        return len(self.get_leaves())
    
    def get_depth(self):
        """
        Calculates the number of nodes on the longest path from root to leaf.
        
        Returns:
          depth: int
        """
        return self._root_node.get_height() + 1
    

class Node(object):
    """
    A node in a tree.  Holds a value and may have branches connecting it to
    other nodes.
    """
    
    def __init__(self, value):
        """
        Constructs a new node.
        
        Args:
          value:
            The data value to be associated with this node.
        """
        self._value = value
        self._children = {}
    
    def get_value(self):
        """
        Retrieves the data value associated with this node.
        
        Returns:
          value:
            The value associated with this node.
        """
        return self._value
    
    def add_child(self, branch, child):
        """
        Creates a branch from this node to another node which will be the 
        child.
        
        Args:
          branch:
            The identifier for the branch connecting this node to the 
            child node.
          child: Node
            Another node which will be a child of the current node.
        
        Returns:
          void
        """
        self._children[branch] = child
    
    def get_child(self, branch):
        """
        Retrieves the child node connected by the specified branch.
        
        Args:
          branch:
            The identifier that was used to associate a child node with
            the current node.
            
        Returns:
          child: Node
            The child node found by following the specified branch.
            
        Raises:
          KeyError if the specified branch does not exist.
        """
        return self._children[branch]
    
    def get_branches(self):
        """
        Retrieves all the branches to children of the current node.
        
        Returns:
          branches: list
            A list of all the branches to child nodes.  Note that this means
            branches TO this node are not included.
        """
        return self._children.keys()
    
    def is_leaf(self):
        """
        Checks if this node is a leaf (has no children).
        
        Returns:
          is_leaf: boolean
            True if this node has no children.
        """
        return len(self._children) == 0
    
    def get_height(self):
        """
        Determines the node's height, i.e. the maximum number of edges 
        between it and a leaf node.
        
        Returns:
          height: int
        """
        max_distance = 0
        
        for branch in self.get_branches():
            distance = self.get_child(branch).get_height() + 1
            if distance > max_distance:
                max_distance = distance
        
        return max_distance
    
    def get_all_descendants(self):
        """
        Retrieves all descendants of the current node, i.e. nodes which can 
        eventually be reached by following outgoing branches from the current 
        node.
        
        Returns:
          descendants: list(Node)
        """
        descendants = []
        
        for branch in self.get_branches():
            child_node = self.get_child(branch)
            descendants.append(child_node)
            descendants.extend(child_node.get_all_descendants())
        
        return descendants
    
    