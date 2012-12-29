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
Unit tests for trees module.

@author: drusk
"""

import unittest

from hamcrest import assert_that, contains_inanyorder

from pml.supervised.decision_trees.trees import Tree, Node

class TreesTest(unittest.TestCase):

    def create_tree(self):
        """
        Creates play_tennis.data decision tree.
        
        Returns:
          tree: Tree
          leaf_nodes: list(Node)
        """
        root_node = Node("Outlook")
        
        humidity_node = Node("Humidity")
        high_humidity_node = Node("No")
        normal_humidity_node = Node("Yes")
        humidity_node.add_child("High", high_humidity_node)
        humidity_node.add_child("Normal", normal_humidity_node)
        root_node.add_child("Sunny", humidity_node)
        
        overcast_node = Node("Yes")
        root_node.add_child("Overcast", overcast_node)
        
        wind_node = Node("Wind")
        strong_wind_node = Node("No")
        weak_wind_node = Node("Yes")
        wind_node.add_child("Strong", strong_wind_node)
        wind_node.add_child("Weak", weak_wind_node)
        root_node.add_child("Rain", wind_node)
        
        leaves = [high_humidity_node, normal_humidity_node, overcast_node, 
                  strong_wind_node, weak_wind_node]
        
        return Tree(root_node), leaves

    def test_get_branches_no_children(self):
        node = Node("test")
        self.assertListEqual(node.get_branches(), [])

    def test_get_all_descendants(self):
        root_node = Node("Root")
        child1 = Node("Child1")
        child2 = Node("Child2")
        root_node.add_child("child1", child1)
        root_node.add_child("child2", child2)
        
        grandchild1 = Node("GC1")
        grandchild2 = Node("GC2")
        child2.add_child("child1", grandchild1)
        child2.add_child("child2", grandchild2)
        
        assert_that(
            root_node.get_all_descendants(),
            contains_inanyorder(
                child1, child2, grandchild1, grandchild2))

    def test_get_all_descendants_empty(self):
        root_node = Node("Root")
        self.assertListEqual(root_node.get_all_descendants(), [])

    def test_get_leaves(self):
        tree, expected_leaves = self.create_tree()
        actual_leaves = tree.get_leaves()
        
        assert_that(
            actual_leaves, 
            contains_inanyorder(*expected_leaves))

    def test_get_num_leaves(self):
        tree, _ = self.create_tree()
        self.assertEqual(tree.get_num_leaves(), 5)
    
    def test_tree_depth(self):
        tree, _ = self.create_tree()
        self.assertEqual(tree.get_depth(), 3)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()