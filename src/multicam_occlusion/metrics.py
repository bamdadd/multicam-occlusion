"""Pose-recovery accuracy metrics.

Depends only on NumPy. The headline metric is **MPJPE** (Mean Per-Joint Position
Error): the mean, over joints, of the Euclidean distance between each predicted
and ground-truth joint position. It is the standard 3D human-pose-estimation
number (Human3.6M et al.) and the natural per-pose analogue of the single-point
recovery error in :mod:`multicam_occlusion.recovery`.

Reference (public method): Ionescu et al., *Human3.6M: Large Scale Datasets and
Predictive Methods for 3D Human Sensing in Natural Environments*, IEEE TPAMI
2014 — Sec. 4, "mean per joint position error".

P-MPJPE (the Procrustes-aligned variant) is intentionally *not* provided here;
MPJPE alone is the metric this module is about.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


def mpjpe(
    pred: FloatArray,
    gt: FloatArray,
    mask: BoolArray | None = None,
) -> float:
    """Mean Per-Joint Position Error between predicted and ground-truth joints.

    Accepts either a single pose ``(J, 3)`` or a batch of poses ``(N, J, 3)``.

    * **Single pose** ``(J, 3)``: the mean over the ``J`` joints of
      ``||pred_j - gt_j||_2`` (the L2 distance per joint).
    * **Batch** ``(N, J, 3)``: the *per-sample-then-batch* mean — MPJPE is
      computed for each of the ``N`` poses independently (mean over that pose's
      valid joints), then those ``N`` values are averaged with equal weight.
      This is the field-standard reduction. It differs from a single pooled mean
      over all ``N * J`` joints only when the per-sample valid-joint counts
      differ (i.e. under per-sample ``mask``); for full masks the two coincide.

    Args:
        pred: predicted joints, ``(J, 3)`` or ``(N, J, 3)``.
        gt: ground-truth joints, the same shape as ``pred``.
        mask: optional per-joint visibility. Shape ``(J,)`` for a single pose or
            ``(N, J)`` for a batch; ``True`` marks a valid joint that is averaged
            in, ``False`` an occluded/invalid joint that is dropped. When
            omitted, every joint is valid.

    Returns:
        The scalar MPJPE, in the same units as the inputs.

    Raises:
        ValueError: if ``pred`` and ``gt`` shapes disagree, the trailing
            coordinate dimension is not 3, the array is neither 2-D nor 3-D, the
            ``mask`` shape does not match the per-joint grid, or any pose has no
            valid joints (a fully-masked pose has an undefined MPJPE).
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)

    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt must have the same shape; got {pred.shape} vs {gt.shape}")
    if pred.ndim not in (2, 3) or pred.shape[-1] != 3:
        raise ValueError(f"expected joints of shape (J, 3) or (N, J, 3); got {pred.shape}")

    # Per-joint Euclidean distance: (J,) for a single pose, (N, J) for a batch.
    dist = np.linalg.norm(pred - gt, axis=-1)

    if mask is None:
        per_sample = dist.mean(axis=-1)
    else:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != dist.shape:
            raise ValueError(
                f"mask must have shape {dist.shape} (one flag per joint); got {mask_arr.shape}"
            )
        valid = mask_arr.sum(axis=-1)
        if np.any(valid == 0):
            raise ValueError("every joint is masked out for at least one pose; MPJPE is undefined")
        per_sample = (dist * mask_arr).sum(axis=-1) / valid

    return float(np.mean(per_sample))
