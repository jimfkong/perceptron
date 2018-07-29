import unittest

from KFolds import kfolds


class OneHotTests(unittest.TestCase):
    def test_kfolds_should_split_data_evenly(self):
        n_rows = 30
        n_folds = 10

        folds = kfolds(n_rows, n_folds)

        self.assertEqual(len(folds), n_folds)

        for i in range(0, n_folds):
            self.assertEqual((len(folds[i].train_index)), 3)
            self.assertEqual((len(folds[i].test_index)), 27)

    def test_kfolds_should_account_for_uneven_data(self):
        n_rows = 29
        n_folds = 10

        folds = kfolds(n_rows, n_folds)

        self.assertEqual(len(folds), n_folds)

        for i in range(0, n_folds - 1):
            self.assertEqual((len(folds[i].train_index)), 3)
            self.assertEqual((len(folds[i].test_index)), 26)

        self.assertEqual((len(folds[n_folds - 1].train_index)), 2)
        self.assertEqual((len(folds[n_folds - 1].test_index)), 27)

    def test_kfolds_should_throw_exception_if_not_enough_data(self):
        n_rows = 9
        n_folds = 10

        with self.assertRaises(ValueError):
            kfolds(n_rows, n_folds)

    def test_kfolds_should_split_data_evenly_if_n_rows_equals_n_folds(self):
        n_rows_and_folds = 10

        folds = kfolds(n_rows_and_folds, n_rows_and_folds)

        self.assertEqual(len(folds), n_rows_and_folds)

        for i in range(0, n_rows_and_folds):
            self.assertEqual((len(folds[i].train_index)), 1)
            self.assertEqual((len(folds[i].test_index)), 9)

    def test_kfolds_should_not_have_repeated_indexes(self):
        n_rows = 30
        n_folds = 10

        folds = kfolds(n_rows, n_folds)

        for i in range(0, n_folds):
            indexes = folds[i].train_index + folds[i].test_index
            self.assertTrue(len(indexes) == len(set(indexes)))

    def test_kfolds_should_not_generate_test_folds_with_repeated_indexes(self):
        n_rows = 30
        n_folds = 10

        folds = kfolds(n_rows, n_folds)

        train_indexes = set()
        test_indexes = set()

        for i in range(0, n_folds):
            train_indexes |= set(folds[i].train_index)
            test_indexes |= set(folds[i].test_index)

        self.assertEqual(len(train_indexes), n_rows)
        self.assertEqual(len(test_indexes), n_rows)
