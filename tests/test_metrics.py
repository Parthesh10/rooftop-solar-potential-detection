"""Regression tests for the metric fixes (F-08, F-11, F-12)."""

import math

import numpy as np
import pytest

from loss.loss import aggregate, accuracy, f1, iou, precision, recall, sigmoid


def test_iou_basic():
    pred = np.array([[1, 1], [0, 0]], dtype=bool)
    target = np.array([[1, 0], [0, 0]], dtype=bool)
    # tp=1, fp=1, fn=0 -> 1/2
    assert iou(pred, target) == pytest.approx(0.5)


def test_iou_perfect():
    a = np.array([[1, 0], [0, 1]], dtype=bool)
    assert iou(a, a) == pytest.approx(1.0)


def test_empty_tile_is_undefined_not_perfect():
    """F-08: the old code inverted both arrays and returned 1.0 here."""
    empty = np.zeros((4, 4), dtype=bool)
    assert math.isnan(iou(empty, empty))


def test_empty_target_with_false_positives_is_zero():
    """A tile with no rooftops where the model hallucinated some scores 0, not 1."""
    target = np.zeros((4, 4), dtype=bool)
    pred = np.zeros((4, 4), dtype=bool)
    pred[0, 0] = True
    assert iou(pred, target) == pytest.approx(0.0)


def test_recall_guards_division_by_zero():
    """F-11: recall() had no denominator guard while precision() did."""
    target = np.zeros((4, 4), dtype=bool)
    pred = np.zeros((4, 4), dtype=bool)
    assert math.isnan(recall(pred, target))
    assert math.isnan(precision(pred, target))


def test_aggregate_is_nan_aware_and_counts_undefined():
    mean, n_used, n_undef = aggregate([1.0, 0.0, float("nan"), 0.5])
    assert mean == pytest.approx(0.5)
    assert (n_used, n_undef) == (3, 1)


def test_aggregate_all_undefined():
    mean, n_used, n_undef = aggregate([float("nan")] * 3)
    assert math.isnan(mean)
    assert (n_used, n_undef) == (0, 3)


def test_sigmoid_does_not_overflow():
    """F-12: 1/(1+np.exp(-x)) overflows to inf below about x = -709."""
    with np.errstate(over="raise"):
        out = sigmoid(np.array([-1000.0, -750.0, 0.0, 750.0, 1000.0]))
    assert np.all(np.isfinite(out))
    assert out[0] == pytest.approx(0.0, abs=1e-12)
    assert out[2] == pytest.approx(0.5)
    assert out[-1] == pytest.approx(1.0, abs=1e-12)


def test_accuracy_is_high_on_a_trivial_all_background_prediction():
    """Documents why accuracy is a near-useless headline metric here."""
    target = np.zeros((100, 100), dtype=bool)
    target[:10, :] = True  # 10% positive rate, about what this task has
    pred = np.zeros((100, 100), dtype=bool)
    assert accuracy(pred, target) == pytest.approx(0.90)
    assert math.isnan(iou(pred, target)) is False and iou(pred, target) == 0.0


def test_f1_matches_dice():
    pred = np.array([1, 1, 0, 0], dtype=bool)
    target = np.array([1, 0, 1, 0], dtype=bool)
    # tp=1, fp=1, fn=1 -> 2/(2+1+1) = 0.5
    assert f1(pred, target) == pytest.approx(0.5)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        iou(np.zeros((4, 4), dtype=bool), np.zeros((4, 5), dtype=bool))
