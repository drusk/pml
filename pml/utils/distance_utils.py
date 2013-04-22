# Copyright (C) 2012, 2013 David Rusk
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
Algorithms for calculating distances between vectors in feature (n) space.

@author: drusk
"""

import numpy as np

def euclidean(vector1, vector2):
    """
    Calculates the Euclidean distance between two vectors in n-space.
    
    Args:
      vector1: 
        start point vector
      vector2: 
        end point vector
    
    Returns:
      The distance (magnitude) between vector1 and vector2.
    """
    return np.sqrt(np.power(np.asarray(vector1) - np.asarray(vector2), 2).sum())

def cosine_similarity(vector1, vector2):
    """
    Calculates the cosine similarity between two vectors.  This is the
    cosine of the angle between them.

    Args:
      vector1: array-like
      vector2: array-like

    Returns:
      The cosine of the angle between the input vectors.
      -1 means they are complete opposites.  0 means they are independent,
      and 1 means they are very similar.
    """
    # Make sure we are working with numpy arrays
    vector1 = np.asarray(vector1)
    vector2 = np.asarray(vector2)

    norm = np.linalg.norm
    product_of_magnitudes = norm(vector1) * norm(vector2)

    if product_of_magnitudes == 0:
        # TODO find a reference supporting this decision
        return 0.0

    return np.dot(vector1, vector2) / product_of_magnitudes

def cosine_distance(vector1, vector2):
    """
    Calculates the cosine distance between two vectors.  It is the complement
    of cosine similarity.  I.e.:

      cosine_distance = 1 - cosine_similarity

    Use this instead of cosine similarity when you want small values to mean
    the vectors are similar as in regular distance measurements such as
    Euclidean distance.

    NOTE: this is not a proper distance metric as it does not have the
    triangle inequality property.

    Args:
      vector1: array-like
      vector2: array-like

    Returns:
      Close to 0 for similar vectors and larger values for dissimilar vectors.
    """
    return 1 - cosine_similarity(vector1, vector2)
