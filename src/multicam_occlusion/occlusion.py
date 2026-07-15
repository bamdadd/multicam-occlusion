"""Synthetic occlusion masks over multi-view observations.

An occlusion mask is a boolean array where ``True`` means the point is visible
in that view and ``False`` means it is occluded. Masks are deterministic given
a seed so benchmark runs are reproducible.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

BoolArray = npt.NDArray[np.bool_]


def drop_k_mask(
    n_views: int,
    n_points: int,
    k_drop: int,
    seed: int,
) -> BoolArray:
    """Build a visibility mask that occludes exactly ``k_drop`` views per point.

    Args:
        n_views: number of cameras.
        n_points: number of 3D points.
        k_drop: how many views to occlude for each point.
        seed: RNG seed; identical seeds yield identical masks.

    Returns:
        Boolean array of shape ``(n_points, n_views)``; ``True`` == visible.

    Raises:
        ValueError: if ``k_drop`` would leave fewer than two visible views,
            since triangulation then becomes impossible.
    """
    if k_drop < 0:
        raise ValueError("k_drop must be >= 0")
    if n_views - k_drop < 2:
        raise ValueError(
            f"dropping {k_drop} of {n_views} views leaves < 2 visible; "
            "triangulation would be impossible"
        )

    rng = np.random.default_rng(seed)
    mask = np.ones((n_points, n_views), dtype=bool)
    for point_idx in range(n_points):
        occluded = rng.choice(n_views, size=k_drop, replace=False)
        mask[point_idx, occluded] = False
    return mask


def occlude(points2d: npt.NDArray[np.float64], mask: BoolArray) -> npt.NDArray[np.float64]:
    """Apply a visibility mask to per-view 2D observations.

    Occluded observations are replaced with NaN so downstream code cannot
    accidentally consume them without honouring the mask.

    Args:
        points2d: ``(n_points, n_views, 2)`` observations.
        mask: ``(n_points, n_views)`` boolean visibility mask.

    Returns:
        A copy of ``points2d`` with occluded entries set to NaN.
    """
    points2d = np.asarray(points2d, dtype=np.float64)
    if mask.shape != points2d.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} incompatible with points {points2d.shape[:2]}")
    out = points2d.copy()
    out[~mask] = np.nan
    return out
