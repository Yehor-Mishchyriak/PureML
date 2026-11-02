import unittest

import numpy as np

from pureml.evaluation import confusion_matrix
from pureml.machinery import Tensor, is_grad_enabled
from pureml.training_utils import TensorDataset


class LookupModel:
    def __init__(self, predictions, training=True):
        self.predictions = np.asarray(predictions)
        self.training = training

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __call__(self, inputs):
        assert not self.training
        assert not is_grad_enabled()
        return Tensor(self.predictions[inputs.data.astype(int)], requires_grad=False)


class TestConfusionMatrix(unittest.TestCase):
    def test_counts_follow_explicit_class_order_across_batches(self):
        model = LookupModel([2, 1, 2, 0, 0])
        data = TensorDataset(np.arange(5), np.array([2, 0, 2, 1, 0]))
        result = confusion_matrix(model, data, classes=[2, 0, 1, 3], batch_size=2)
        np.testing.assert_array_equal(result, [
            [2, 0, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0],
        ])
        self.assertEqual(result.dtype, np.dtype(np.int64))
        self.assertTrue(model.training)

    def test_scores_and_one_hot_targets(self):
        model = LookupModel([[0.1, 0.9], [0.7, 0.3], [0.2, 0.8]], training=False)
        data = TensorDataset(np.arange(3), np.eye(2)[[0, 0, 1]])
        np.testing.assert_array_equal(confusion_matrix(model, data, [1, 0]), [[1, 0], [1, 1]])
        self.assertFalse(model.training)

    def test_non_contiguous_class_indices(self):
        model = LookupModel([10, -2, 10])
        data = TensorDataset(np.arange(3), np.array([-2, -2, 10]))
        np.testing.assert_array_equal(confusion_matrix(model, data, [-2, 10]), [[1, 1], [0, 1]])

    def test_empty_dataset_returns_zero_counts(self):
        model = LookupModel([])
        data = TensorDataset(np.array([], dtype=int), np.array([], dtype=int))
        np.testing.assert_array_equal(confusion_matrix(model, data, [2, 0]), np.zeros((2, 2)))
        self.assertTrue(model.training)

    def test_invalid_classes(self):
        data = TensorDataset(np.arange(1), np.array([0]))
        for classes in ([], [0, 0], [0, 1.5]):
            with self.subTest(classes=classes), self.assertRaises(ValueError):
                confusion_matrix(LookupModel([0]), data, classes)

    def test_unknown_labels_restore_training_mode(self):
        for predictions, targets in (([2], [0]), ([0], [2])):
            for training in (True, False):
                with self.subTest(predictions=predictions, targets=targets, training=training):
                    model = LookupModel(predictions, training=training)
                    data = TensorDataset(np.arange(1), np.array(targets))
                    with self.assertRaisesRegex(ValueError, 'classes'):
                        confusion_matrix(model, data, [0, 1])
                    self.assertEqual(model.training, training)

    def test_prediction_failure_restores_training_mode(self):
        class FailingModel(LookupModel):
            def __call__(self, inputs):
                raise RuntimeError('synthetic prediction failure')

        model = FailingModel([])
        data = TensorDataset(np.arange(1), np.array([0]))
        with self.assertRaisesRegex(RuntimeError, 'synthetic'):
            confusion_matrix(model, data, [0])
        self.assertTrue(model.training)

    def test_prediction_and_target_lengths_must_match(self):
        class ShortModel(LookupModel):
            def __call__(self, inputs):
                return Tensor(np.array([0]), requires_grad=False)

        model = ShortModel([])
        data = TensorDataset(np.arange(2), np.array([0, 1]))
        with self.assertRaisesRegex(ValueError, 'shape'):
            confusion_matrix(model, data, [0, 1])
        self.assertTrue(model.training)
