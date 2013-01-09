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
Unit tests for the model module.

@author: drusk
"""

import unittest

import numpy as np
import pandas as pd
from hamcrest import assert_that, contains

from pml.data.model import DataSet, as_dataset
from pml.utils.errors import InconsistentSampleIdError
from pml.utils.errors import UnlabelledDataSetError

from test import base_tests
from test.matchers.pml_matchers import equals_dataset
from test.matchers.pandas_matchers import equals_series, equals_dataframe

class DataSetTest(base_tests.BaseDataSetTest):

    def test_reduce_rows(self):
        dataset = DataSet([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        reduced = dataset.reduce_rows(sum)
        assert_that(reduced.values, contains(6, 15, 24))
        
    def test_reduce_features(self):
        dataset = DataSet([[4, 9, 8], [2, 1, 7], [5, 6, 1]])
        reduced = dataset.reduce_features(min)
        assert_that(reduced.values, contains(2, 1, 1))
        
    def test_drop_column(self):
        original = DataSet([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertEqual(original.num_features(), 3)
        filtered = original.drop_column(1)
        self.assertEqual(filtered.num_features(), 2)
        assert_that(filtered, equals_dataset([[1, 3], [4, 6], [7, 9]]))
        # make sure original unchanged
        self.assertEqual(original.num_features(), 3)
        assert_that(original, equals_dataset([[1, 2, 3], [4, 5, 6], 
                                              [7, 8, 9]]))
        
    def test_drop_column_keeps_labels(self):
        original = self.create_dataset(labels=["cat", "dog", "bird"],
                                       num_features=3)
        filtered = original.drop_column(1)
        assert_that(filtered.get_labels(), contains("cat", "dog", "bird"))
        
    def test_get_column(self):
        dataset = DataSet([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        column1 = dataset.get_column(1)
        assert_that(column1.values, contains(2, 5, 8))
    
    def test_get_column_set_value(self):
        dataset = DataSet([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        dataset.get_column(1)[:] = 1
        assert_that(dataset.get_column(1), contains(1, 1, 1))
        
    def test_set_column(self):
        dataset = DataSet([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        dataset.set_column(1, [11, 11, 11])
        assert_that(dataset, equals_dataset([[1, 11, 3], [4, 11, 6], 
                                             [7, 11, 9]]))
    
    def test_set_new_column(self):
        dataset = DataSet([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        dataset.set_column(3, [11, 11, 11])
        assert_that(dataset, equals_dataset([[1, 2, 3, 11], [4, 5, 6, 11], 
                                             [7, 8, 9, 11]]))
        
    def test_get_rows(self):
        dataset = DataSet([[1, 2], [3, 4], [5, 6], [7, 8]])
        selection = dataset.get_rows([1, 3])
        
        self.assertEqual(selection.num_samples(), 2)
        assert_that(selection, equals_dataset([[3, 4], [7, 8]]))
        
    def test_get_labelled_rows(self):
        dataset = DataSet([[1, 2], [3, 4], [5, 6], [7, 8]], 
                          labels=["a", "a", "b", "b"])
        selection = dataset.get_rows([1, 3])
        
        self.assertEqual(selection.num_samples(), 2)
        self.assertTrue(selection.is_labelled())
        # TODO incorporate labels equals_series into DataSet matcher?
        assert_that(selection, equals_dataset([[3, 4], [7, 8]]))
        assert_that(selection.get_labels(), equals_series({1: "a", 3: "b"}))
        
    def test_get_label_value_counts(self):
        dataset = DataSet([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]], 
                          labels=["a", "b", "b", "c", "a", "b"])
        expected = {"a": 2, "b": 3, "c": 1}
        value_counts = dataset.get_label_value_counts()
        assert_that(value_counts, equals_series(expected))
        assert_that(value_counts.index, contains("b", "a", "c"))
        
    def test_get_label_value_counts_no_labels(self):
        dataset = DataSet([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
        assert_that(dataset.get_label_value_counts(), equals_series({}))
        
    def test_split(self):
        dataset = DataSet([[1, 2], [3, 4], [5, 6], [7, 8]])
        first, second = dataset.split(0.5)
        self.assertEqual(first.num_samples(), 2)
        assert_that(first, equals_dataset([[1, 2], [3, 4]]))
        self.assertEqual(second.num_samples(), 2)
        assert_that(second, equals_dataset([[5, 6], [7, 8]]))
        
    def test_unequal_split(self):
        dataset = DataSet([[1, 2], [3, 4], [5, 6], [7, 8]])
        first, second = dataset.split(0.3)
        self.assertEqual(first.num_samples(), 1)
        assert_that(first, equals_dataset([[1, 2]]))
        self.assertEqual(second.num_samples(), 3)
        assert_that(second, equals_dataset([[3, 4], [5, 6], [7, 8]]))

    def test_split_0(self):
        dataset = DataSet([[1, 2], [3, 4], [5, 6], [7, 8]])
        first, second = dataset.split(0)
        self.assertEqual(first.num_samples(), 0)
        assert_that(first, equals_dataset([]))
        self.assertEqual(second.num_samples(), 4)
        assert_that(second, equals_dataset([[1, 2], [3, 4], [5, 6], [7, 8]]))

    def test_split_invalid_percent(self):
        dataset = DataSet([[1, 2], [3, 4], [5, 6], [7, 8]])
        self.assertRaises(ValueError, dataset.split, 50)

    def test_split_random(self):
        dataset = DataSet([[1, 2], [3, 4], [5, 6], [7, 8]])
        first, second = dataset.split(0.5, random=True)
        # since the split is random, can't assert that first or second 
        # contain particular rows, just the number of rows
        self.assertEqual(first.num_samples(), 2)
        self.assertEqual(second.num_samples(), 2)
        
    def test_fill_missing(self):
        dataset = DataSet([[1, np.NaN, 3], [np.NaN, 5, np.NaN]])
        dataset.fill_missing(0)
        assert_that(dataset, equals_dataset([[1, 0, 3], [0, 5, 0]]))
        
    def test_fill_missing_with_feature_means(self):
        dataset = DataSet([[2, np.NaN, np.NaN], [np.NaN, 6, 10], 
                           [5, 4, np.NaN]])
        dataset.fill_missing_with_feature_means()
        assert_that(dataset, equals_dataset([[2, 5, 10], [3.5, 6, 10], 
                                             [5, 4, 10]]))
    
    def test_split_labelled(self):
        dataset = DataSet([[1, 2], [3, 4], [5, 6], [7, 8]], 
                          labels=["b", "b", "b", "a"])
        first, second = dataset.split(0.5)
        self.assertTrue(first.is_labelled())
        assert_that(first.get_labels(), equals_series({0: "b", 1: "b"}))
        self.assertTrue(second.is_labelled())
        assert_that(second.get_labels(), equals_series({2: "b", 3: "a"}))
        
    def test_get_row(self):
        dataset = DataSet([[1, 2], [3, 4], [5, 6], [7, 8]])
        row = dataset.get_row(1)
        assert_that(row.values, contains(3, 4))
        # check that changes made to selected row are reflected in original
        row[:] = 1
        assert_that(dataset.get_row(1), contains(1, 1))
    
    def test_get_row_by_id(self):
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 
                          index=["V01", "V02", "V03"])
        dataset = DataSet(df)
        sample = dataset.get_row("V02")
        assert_that(sample, contains(4, 5, 6))
        # make sure position based index is still usable
        sample = dataset.get_row(1)
        assert_that(sample, contains(4, 5, 6))
        
    def test_get_last_row(self):
        dataset = DataSet([[1, 2], [3, 4], [5, 6], [7, 8]])
        assert_that(dataset.get_row(dataset.num_samples() - 1), contains(7, 8))

    def test_contruct_dataset_from_dataset(self):
        original = DataSet([[1, 2], [3, 4], [5, 6]])
        new = DataSet(original)
        self.assertFalse(new is original)
        new._dataframe.ix[1] = 1
        assert_that(new, equals_dataset([[1, 2], [1, 1], [5, 6]]))
        assert_that(original, equals_dataset([[1, 2], [3, 4], [5, 6]]))
        
    def test_as_dataset(self):
        original = DataSet([[1, 2], [3, 4], [5, 6]])
        dataset = as_dataset(original)
        self.assertTrue(dataset is original)
        dataset._dataframe.ix[1] = 1
        assert_that(dataset, equals_dataset([[1, 2], [1, 1], [5, 6]]))
        assert_that(original, equals_dataset([[1, 2], [1, 1], [5, 6]]))

    def test_get_labelled_data_frame(self):
        dataset = DataSet([[1, 2], [3, 4], [5, 6], [7, 8]], 
                          labels=pd.Series(["b", "b", "b", "a"]))
        df = dataset.get_labelled_data_frame()

        # TODO: non-numeric values in DataFrame matcher
        expected = [[1, 2, "b"], [3, 4, "b"], [5, 6, "b"], [7, 8, "a"]]
        for i in range(len(expected)):
            self.assertTrue(df.ix[i].tolist(), expected[i])

    def test_get_labelled_data_frame_no_labels(self):
        dataset = DataSet([[1, 2], [3, 4], [5, 6], [7, 8]])
        df = dataset.get_labelled_data_frame()
        assert_that(df, equals_dataframe([[1, 2], [3, 4], [5, 6], [7, 8]]))

    def test_get_data_frame(self):
        dataset = DataSet([[1, 2], [3, 4], [5, 6], [7, 8]], 
                          labels=pd.Series(["b", "b", "b", "a"]))
        df = dataset.get_data_frame()
        assert_that(df, equals_dataframe([[1, 2], [3, 4], [5, 6], [7, 8]]))

    def test_combine_labels(self):
        dataset = DataSet([[1, 2], [3, 4], [5, 6]], 
                          labels=pd.Series(["cat", "crow", "pidgeon"]))
        dataset.combine_labels(["crow", "pidgeon"], "bird")
        labels = dataset.get_labels()
        assert_that(labels, equals_series({0: "cat", 1: "bird", 2: "bird"}))

    def test_feature_list(self):
        dataset = DataSet(pd.DataFrame([[4, 1, 2], [5, 9, 8], [4, 3, 6]], 
                          columns=["a", "b", "c"]))
        assert_that(dataset.feature_list(), contains("a", "b", "c"))
        
    def test_default_feature_list(self):
        dataset = DataSet(pd.DataFrame([[4, 1, 2], [5, 9, 8], [4, 3, 6]]))
        assert_that(dataset.feature_list(), contains(0, 1, 2))

    def test_has_missing_values(self):
        dataset1 = DataSet([[4.2, np.NaN, 3.1], [2.5, 1.9, np.NaN], 
                            [1.1, 1.2, 1.7]])
        self.assertTrue(dataset1.has_missing_values())
        
        dataset2 = DataSet([[4.2, 3.9, 3.1], [2.5, 1.9, 2.2], [1.1, 1.2, 1.7]])
        self.assertFalse(dataset2.has_missing_values())

    def test_to_string(self):
        df = pd.DataFrame([[4.2, np.NaN, 3.1], [2.5, 1.9, np.NaN], 
                           [1.1, 1.2, 1.7]], 
                          columns=["weight", "height", "length"])
        dataset = DataSet(df, labels=["cat", "bird", "bat"])
        expected = "\n".join(("Features: ['weight', 'height', 'length']",
                              "Samples: 3",
                              "Missing values? yes",
                              "Labelled? yes"))
        self.assertEqual(expected, dataset.__repr__())

    def test_get_sample_ids(self):
        dataset = DataSet([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert_that(dataset.get_sample_ids(), contains(0, 1, 2))

    def test_get_sample_ids_named(self):
        dataset = DataSet(pd.DataFrame([[1, 2], [3, 4], [5, 6]], 
                                       index=["a", "b", "c"]))
        assert_that(dataset.get_sample_ids(), contains("a", "b", "c"))

    def test_copy(self):
        dataset1 = DataSet([[1, 2], [3, 4]], labels=pd.Series(["a", "b"]))
        dataset2 = dataset1.copy()
        dataset2.set_column(1, pd.Series([4, 5]))
        
        assert_that(dataset2, equals_dataset([[1, 4], [3, 5]]))
        assert_that(dataset2.get_labels(), equals_series({0: "a", 1: "b"}))
        assert_that(dataset1, equals_dataset([[1, 2], [3, 4]]))
        assert_that(dataset2.get_labels(), equals_series({0: "a", 1: "b"}))
        
    def test_copy_no_labels(self):
        dataset1 = DataSet([[1, 2], [3, 4]])
        dataset2 = dataset1.copy()
        dataset2.set_column(1, pd.Series([4, 5]))
        
        assert_that(dataset2, equals_dataset([[1, 4], [3, 5]]))
        self.assertFalse(dataset2.is_labelled())
        assert_that(dataset1, equals_dataset([[1, 2], [3, 4]]))
        self.assertFalse(dataset1.is_labelled())
        
    def test_sample_id_mismatch(self):
        labels = pd.Series(["cat", "dog", "bird"], index=[1, 2, 3])
        try:
            self.create_dataset(labels=labels, sample_ids=[0, 1, 2])
            self.fail("Exception should have occurred.")
        except InconsistentSampleIdError:
            pass
        
    def test_labels_as_dict(self):
        labels = {0: "cat", 1: "dog", 2: "bat"}
        dataset = self.create_dataset(labels=labels)
        assert_that(dataset.get_labels(), contains("cat", "dog", "bat"))
        
    def test_filter_by_feature_value(self):
        features = ["name", "hair colour"]
        df = pd.DataFrame([["Bill", "brown"], ["Bob", "black"], 
                           ["Jim", "brown"]], columns=features)
        dataset = DataSet(df)
        filtered = dataset.value_filter("hair colour", "brown")
        self.assertEqual(filtered.feature_list(), features)
        assert_that(filtered.get_sample_ids(), contains(0, 2))
        assert_that(filtered, equals_dataset([["Bill", "brown"], 
                                              ["Jim", "brown"]]))
        
    def test_filter_by_feature_value_with_labels(self):
        features = ["name", "hair colour"]
        df = pd.DataFrame([["Bill", "brown"], ["Bob", "black"], 
                           ["Jim", "brown"]], columns=features)
        dataset = DataSet(df, labels=["SENG", "SENG", "CENG"])
        filtered = dataset.value_filter("hair colour", "brown")
        assert_that(filtered.get_labels(), 
                    equals_series({0: "SENG", 2: "CENG"}))
        
    def test_filter_by_label(self):
        features = ["name", "hair colour"]
        df = pd.DataFrame([["Bill", "brown"], ["Bob", "black"], 
                           ["Jim", "brown"]], columns=features)
        dataset = DataSet(df, labels=["SENG", "SENG", "CENG"])
        filtered = dataset.label_filter("SENG")
        assert_that(filtered, equals_dataset([["Bill", "brown"], 
                                              ["Bob", "black"]]))
        assert_that(filtered.get_labels(), 
                    equals_series({0: "SENG", 1: "SENG"}))
        
    def test_filter_unlabelled_data_by_label(self):
        features = ["name", "hair colour"]
        df = pd.DataFrame([["Bill", "brown"], ["Bob", "black"], 
                           ["Jim", "brown"]], columns=features)
        dataset = DataSet(df)
        self.assertRaises(UnlabelledDataSetError, dataset.label_filter, "SENG")
        
    def test_filter_by_sample_id(self):
        dataset = self.create_dataset(sample_ids=["V01", "V02", "V03", "V04"])
        filtered = dataset.sample_filter(["V02", "V03"])
        assert_that(filtered.get_sample_ids(), contains("V02", "V03"))
        
    def test_get_feature_value_counts(self):
        df = pd.DataFrame([["Jim", 19, 180], ["John", 18, 177], 
                           ["Jack", 19, 185]], 
                          columns=["name", "age", "height"])
        dataset = DataSet(df)
        
        age_value_counts = dataset.get_feature_value_counts("age")
        assert_that(age_value_counts, equals_series({19: 2, 18: 1}))
        
        name_value_counts = dataset.get_feature_value_counts("name")
        assert_that(name_value_counts, 
                    equals_series({"Jim": 1, "John": 1, "Jack": 1}))
    
    def test_get_values(self):
        df = pd.DataFrame([["Jim", 19, 180], ["John", 18, 177], 
                           ["Jack", 19, 185]], 
                          columns=["name", "age", "height"])
        dataset = DataSet(df)
        
        age_values = dataset.get_feature_values("age")
        self.assertEqual(len(age_values), 2)
        self.assertTrue(19 in age_values)
        self.assertTrue(18 in age_values)
        
        height_values = dataset.get_feature_values("height")
        self.assertTrue(180 in height_values)
        self.assertTrue(185 in height_values)
        self.assertTrue(177 in height_values)
        
    def test_bin_feature(self):
        df = pd.DataFrame([[0, 1], [7, 2], [6, 3]], 
                          columns=["MATH100", "PHYS125"])
        dataset = DataSet(df)
        
        dataset.bin("MATH100", [4, 7])
        
        assert_that(dataset, equals_dataset([[0, 1], [2, 2], [1, 3]]))
    
    def test_bin_feature_bin_names(self):
        df = pd.DataFrame([[0, 1], [9, 2], [6, 3]], 
                          columns=["MATH100", "PHYS125"])
        dataset = DataSet(df)
        
        dataset.bin("MATH100", [4, 7], bin_names=["low", "mid", "high"])
        
        assert_that(dataset, equals_dataset([["low", 1], ["high", 2], 
                                             ["mid", 3]]))
    
    def test_bin_feature_1_boundary(self):
        df = pd.DataFrame([[0, 1], [9, 2], [6, 3]], 
                          columns=["MATH100", "PHYS125"])
        dataset = DataSet(df)
        
        dataset.bin("MATH100", [3], bin_names=["low", "high"])
        
        assert_that(dataset, equals_dataset([["low", 1], ["high", 2], 
                                             ["high", 3]]))
        
    def test_bin_feature_floats(self):
        df = pd.DataFrame([[3.5, 1], [9.1, 2], [6.2, 3]], 
                          columns=["MATH100", "PHYS125"])
        dataset = DataSet(df)
        
        dataset.bin("MATH100", [3.9, 7], bin_names=["low", "mid", "high"])
        
        assert_that(dataset, equals_dataset([["low", 1], ["high", 2], 
                                             ["mid", 3]]))
        
    def test_bin_feature_wrong_num_bin_names(self):
        df = pd.DataFrame([[0, 1], [9, 2], [6, 3]], 
                          columns=["MATH100", "PHYS125"])
        dataset = DataSet(df)
        
        self.assertRaises(
            ValueError, 
            dataset.bin, 
            "MATH100", [4, 7], bin_names=["low", "mid"]
        )
    
    def test_bin_feature_not_numeric(self):
        df = pd.DataFrame([["F", 1], ["A+", 2], ["B+", 3]], 
                          columns=["MATH100", "PHYS125"])
        dataset = DataSet(df)
        
        self.assertRaises(
            ValueError, 
            dataset.bin, 
            "MATH100", [4, 7]
        )
        
    def test_bin_feature_no_boundaries(self):
        df = pd.DataFrame([[0, 1], [9, 2], [6, 3]], 
                          columns=["MATH100", "PHYS125"])
        dataset = DataSet(df)
        
        self.assertRaises(
            ValueError, 
            dataset.bin, 
            "MATH100", []
        )
        
    def test_bin_all(self):
        df = pd.DataFrame([[0, 6], [9, 2], [6, 4]], 
                          columns=["MATH100", "PHYS125"])
        dataset = DataSet(df)
        
        dataset.bin("*", [4, 7], bin_names=["low", "mid", "high"])
        
        assert_that(dataset, equals_dataset([["low", "mid"], ["high", "low"], 
                                             ["mid", "mid"]]))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    