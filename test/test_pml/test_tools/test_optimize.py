# Copyright (C) 2013 David Rusk
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

__author__ = "David Rusk <drusk@uvic.ca>"

import unittest

import pandas as pd
from hamcrest import assert_that

from pml.data import loader
from pml.tools import optimize

from test.base_tests import BaseFileLoadingTest
from test.matchers.pandas_matchers import equals_series


class GradientDescentTest(BaseFileLoadingTest):
    def test_gradient_descent_2_parameters(self):
        """
        Test based on Assignment 1 of the free online
        Stanford Machine Learning online course.

        For population = 35,000, we predict a profit of 4519.767868
        For population = 70,000, we predict a profit of 45342.450129

        Final cost: 4.483388
        """
        dataset = loader.load(self.relative_to_base("datasets/ex1data1.txt"),
                              has_ids=False, has_header=False, has_labels=True,
                              delimiter=",")
        dataset.set_column("bias", pd.Series([1] * dataset.num_samples()))

        learning_rate = 0.01
        iter = 100

        initial_theta = pd.Series({"bias": 0, "X.1": 0})
        theta = optimize.gradient_descent(dataset, initial_theta,
                                          learning_rate, iterations=iter)

        # assert_that(theta.tolist(), contains(-0.576556, 0.859582))
        assert_that(theta, equals_series({"bias": -0.576556,
                                          "X.1": 0.859582},
                                         places=6))

    def test_gradient_descent_3_parameters(self):
        """
        Test based on Assignment 1 of the free online
        Stanford Machine Learning online course.
        """
        dataset = loader.load(self.relative_to_base("datasets/ex1data2.txt"),
                              has_ids=False, has_header=False, has_labels=True,
                              delimiter=",")
        dataset.normalize_features()
        dataset.set_column("bias", pd.Series([1] * dataset.num_samples()))

        learning_rate = 1.0
        iter = 50

        initial_theta = pd.Series({"bias": 0, "X.1": 0, "X.2": 0})
        theta = optimize.gradient_descent(dataset, initial_theta,
                                          learning_rate, iterations=iter)

        assert_that(theta, equals_series({"bias": 340412.659574,
                                          "X.1": 110631.050279,
                                          "X.2": -6649.474271},
                                         places=6))


if __name__ == '__main__':
    unittest.main()
