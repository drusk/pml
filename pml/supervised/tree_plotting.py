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
Plots decision trees.

@author: drusk
"""

import matplotlib.pyplot as plt

from pml.supervised.trees import Tree

class MatplotlibAnnotationTreePlotter(object):
    """
    Plots a decision tree with matplotlib by using annotations.
    
    This is the approach used by Peter Harrington in his book Machine 
    Learning in Action. 
    """
    
    # constants
    decision_node_type = dict(boxstyle="sawtooth", fc="0.8")
    leaf_node_type = dict(boxstyle="round4", fc="0.8")
    arrow_args = dict(arrowstyle="<-")
    
    def __init__(self, tree):
        """
        Constructs a new plotter.
        
        Args:
          tree: Tree
            The decision tree to be plotted.
        """
        self.tree = tree
        
    def _plot_node(self, node, center_point, parent_location, node_type):
        """
        Plots a single node using a matplotlib annotation.
        """
        self.axis.annotate(
                node.get_value(), xy=parent_location, 
                xycoords='axes fraction', xytext=center_point, 
                textcoords='axes fraction', va="center", ha="center", 
                bbox=node_type, arrowprops=self.arrow_args)
    
    def _plot_mid_text(self, center_point, parent_point, text):
        """
        Plots text along the arc between a node and its parent.
        """
        x_mid = (parent_point[0] - center_point[0]) / 2.0 + center_point[0]
        y_mid = (parent_point[1] - center_point[1]) / 2.0 + center_point[1]
        self.axis.text(x_mid, y_mid, text)
    
    def _plot_tree_recursively(self, tree, parent_point, node_text):
        """
        Plots the provided tree.  This function works by recursively plotting
        subtrees.
        """
        current_root = tree.get_root_node()
        center_point = (
            self.x_offset + 
            (1.0 + tree.get_num_leaves()) / (2.0 * self.nodes_across), 
            self.y_offset)
        
        self._plot_mid_text(center_point, parent_point, node_text)
        self._plot_node(current_root, center_point, parent_point, 
                        self.decision_node_type)
        
        self.y_offset -= 1.0 / self.nodes_high
        
        for branch in current_root.get_branches():
            child_node = current_root.get_child(branch)
            
            if child_node.is_leaf():
                self.x_offset += 1.0 / self.nodes_across
                self._plot_node(child_node, (self.x_offset, self.y_offset), 
                                center_point, self.leaf_node_type)
                self._plot_mid_text((self.x_offset, self.y_offset), 
                                    center_point, branch)
                
            else:
                self._plot_tree_recursively(Tree(child_node), 
                                            center_point, branch)
                
        self.y_offset += 1.0 / self.nodes_high
    
    def plot(self):
        """
        Generate a plot of the decision tree.
        
        Returns:
          void 
        """
        fig = plt.figure(1, facecolor="white")
        fig.clf()
        axprops = dict(xticks=[], yticks=[])
        self.axis = plt.subplot(111, frameon=False, **axprops)
        self.nodes_across = float(self.tree.get_num_leaves())
        self.nodes_high = float(self.tree.get_depth())
        self.x_offset = -0.5 / self.nodes_across
        self.y_offset = 1.0
        self._plot_tree_recursively(self.tree, (0.5, 1.0), "")
        plt.show()
        
