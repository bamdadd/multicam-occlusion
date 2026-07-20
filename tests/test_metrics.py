"""Tests for the MPJPE pose-recovery metric.

Covers the acceptance criteria from issue #7:

  1. Zero error on identical inputs.
  2. A hand-computed small case (Pythagorean joint distances) so a change to the
     norm (L1, or dropping the sqrt) is caught by an exact arithmetic check.
  3. Per-joint visibility masking: the average runs over valid joints only.
  4. Batched-vs-single consistency: the batch value is the mean of the
     per-sample single values (per-sample-then-batch reduction).
"""

from __future__ import annotations

import numpy as np
import pytest

from multicam_occlusion import mpjpe

TOL = 1e-12


def test_zero_error_on_identical_single() -> None:
    """Identical single poses have exactly zero MPJPE."""
    pose = np.array([[0.3, -0.4, 0.8], [1.0, 2.0, 3.0], [-1.0, 0.0, 5.0]])
    assert mpjpe(pose, pose) == pytest.approx(0.0, abs=TOL)


def test_zero_error_on_identical_batch() -> None:
    """Identical batches have exactly zero MPJPE."""
    batch = np.arange(2 * 4 * 3, dtype=np.float64).reshape(2, 4, 3)
    assert mpjpe(batch, batch) == pytest.approx(0.0, abs=TOL)


def test_hand_computed_single() -> None:
    """Two joints with known Pythagorean displacements from the origin.

    gt is the origin for both joints. Predictions are offset by:
      * joint 0: (3, 4, 0)  -> ||.|| = sqrt(9 + 16)      = 5   (the 3-4-5 triple)
      * joint 1: (0, 6, 8)  -> ||.|| = sqrt(36 + 64)     = 10  (the 6-8-10 triple)
    MPJPE = mean(5, 10) = 15 / 2 = 7.5.

    This value is only produced by the L2 norm: an L1 norm gives
    mean(7, 14) = 10.5, and a squared (no-sqrt) distance gives mean(25, 100) =
    62.5 -- so this case pins the norm exactly.
    """
    pred = np.array([[3.0, 4.0, 0.0], [0.0, 6.0, 8.0]])
    gt = np.zeros((2, 3))
    assert mpjpe(pred, gt) == pytest.approx(7.5, abs=TOL)


def test_masked_joints_average_over_valid_only() -> None:
    """A masked-out joint is excluded from the mean.

    Three joints, gt at the origin, with per-joint distances 5, 10, 100. Masking
    the third (the distance-100 joint) must average only the first two:
    mean(5, 10) = 7.5. Averaging over all three would give (5 + 10 + 100) / 3 =
    38.33..., so this pins the mask to *exclude* invalid joints.
    """
    pred = np.array([[3.0, 4.0, 0.0], [0.0, 6.0, 8.0], [100.0, 0.0, 0.0]])
    gt = np.zeros((3, 3))
    mask = np.array([True, True, False])
    assert mpjpe(pred, gt, mask=mask) == pytest.approx(7.5, abs=TOL)


def test_masked_matches_unmasked_when_all_valid() -> None:
    """An all-True mask is a no-op versus omitting the mask."""
    rng = np.random.default_rng(20260717)
    pred = rng.normal(size=(5, 3))
    gt = rng.normal(size=(5, 3))
    mask = np.ones(5, dtype=bool)
    assert mpjpe(pred, gt, mask=mask) == pytest.approx(mpjpe(pred, gt), abs=TOL)


def test_batched_is_mean_of_single() -> None:
    """The batch MPJPE is the equal-weight mean of the per-sample single values."""
    rng = np.random.default_rng(42)
    batch_pred = rng.normal(size=(3, 6, 3))
    batch_gt = rng.normal(size=(3, 6, 3))
    per_sample = [mpjpe(batch_pred[i], batch_gt[i]) for i in range(3)]
    assert mpjpe(batch_pred, batch_gt) == pytest.approx(float(np.mean(per_sample)), abs=TOL)


def test_batched_masked_is_mean_of_single_masked() -> None:
    """Per-sample-then-batch reduction holds under per-sample masks too."""
    rng = np.random.default_rng(7)
    batch_pred = rng.normal(size=(3, 6, 3))
    batch_gt = rng.normal(size=(3, 6, 3))
    mask = rng.random(size=(3, 6)) > 0.3  # each row keeps at least some joints
    per_sample = [mpjpe(batch_pred[i], batch_gt[i], mask=mask[i]) for i in range(3)]
    assert mpjpe(batch_pred, batch_gt, mask=mask) == pytest.approx(
        float(np.mean(per_sample)), abs=TOL
    )


def test_all_masked_single_raises() -> None:
    """A fully-masked single pose has undefined MPJPE and raises."""
    pred = np.ones((3, 3))
    gt = np.zeros((3, 3))
    mask = np.zeros(3, dtype=bool)
    with pytest.raises(ValueError, match="masked out"):
        mpjpe(pred, gt, mask=mask)


def test_all_masked_one_batch_sample_raises() -> None:
    """A fully-masked sample anywhere in a batch raises."""
    pred = np.ones((2, 3, 3))
    gt = np.zeros((2, 3, 3))
    mask = np.array([[True, True, True], [False, False, False]])
    with pytest.raises(ValueError, match="masked out"):
        mpjpe(pred, gt, mask=mask)


def test_shape_mismatch_raises() -> None:
    """pred and gt must share a shape."""
    with pytest.raises(ValueError, match="same shape"):
        mpjpe(np.zeros((3, 3)), np.zeros((4, 3)))


def test_wrong_coordinate_dim_raises() -> None:
    """The trailing coordinate dimension must be 3."""
    with pytest.raises(ValueError, match="J, 3"):
        mpjpe(np.zeros((3, 2)), np.zeros((3, 2)))


def test_wrong_mask_shape_raises() -> None:
    """The mask must carry exactly one flag per joint."""
    pred = np.zeros((3, 3))
    gt = np.zeros((3, 3))
    with pytest.raises(ValueError, match="one flag per joint"):
        mpjpe(pred, gt, mask=np.ones((3, 3), dtype=bool))
