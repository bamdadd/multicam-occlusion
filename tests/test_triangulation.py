"""Hero smoke test: multi-view recovers occluded points; single view cannot.

Runs with no Blender dependency. A known 3D point is projected into N=6
synthetic ring cameras, occluded in a subset of views, and triangulated from
the surviving observations. We assert:

  1. Multi-view triangulation from the *visible* subset recovers ground truth.
  2. A single view is fundamentally insufficient (it constrains the point only
     to a viewing ray, so two distinct 3D points can share one observation).
"""

from __future__ import annotations

import numpy as np
import pytest

from multicam_occlusion import (
    build_ring_cameras,
    drop_k_mask,
    project_points,
    triangulate_dlt,
)

SEED = 20260715
TOL = 1e-6


def _project_all(proj_mats: np.ndarray, point3d: np.ndarray) -> np.ndarray:
    """Project one 3D point into every camera; returns ``(n_views, 2)``."""
    return np.vstack([project_points(p, point3d)[0] for p in proj_mats])


def test_multiview_recovers_occluded_point() -> None:
    """Occlude 3 of 6 views; triangulate from the rest; recover ground truth."""
    proj_mats = build_ring_cameras(n_cameras=6)
    gt = np.array([0.3, -0.4, 0.8])

    points2d = _project_all(proj_mats, gt)

    # Occlude 3 of 6 views for this single point.
    mask = drop_k_mask(n_views=6, n_points=1, k_drop=3, seed=SEED)[0]
    assert mask.sum() == 3

    recovered = triangulate_dlt(proj_mats, points2d, mask=mask)
    assert np.allclose(recovered, gt, atol=TOL), f"{recovered} != {gt}"


def test_single_view_is_insufficient() -> None:
    """One view cannot recover depth: it constrains the point only to a ray."""
    proj_mats = build_ring_cameras(n_cameras=6)
    gt = np.array([0.3, -0.4, 0.8])
    points2d = _project_all(proj_mats, gt)

    # Only camera 0 is visible -> triangulation must refuse (rank-deficient).
    single = np.zeros(6, dtype=bool)
    single[0] = True
    with pytest.raises(ValueError, match="need >= 2 visible views"):
        triangulate_dlt(proj_mats, points2d, mask=single)


def test_single_view_ray_ambiguity() -> None:
    """Demonstrate the ambiguity: distinct 3D points share one 2D observation.

    Any 3D point along the ray from camera 0 through ``gt`` projects to the same
    pixel in camera 0. This is *why* one view is insufficient — the depth is
    unobservable — yet a second view resolves it.
    """
    proj_mats = build_ring_cameras(n_cameras=6)
    cam0 = proj_mats[0]
    gt = np.array([0.3, -0.4, 0.8])
    pixel_gt = project_points(cam0, gt)[0]

    # Camera 0 centre C: null space of P (P C_h = 0).
    _, _, vt = np.linalg.svd(cam0)
    centre_h = vt[-1]
    centre = centre_h[:3] / centre_h[3]

    # A different point on the same ray: move gt along (gt - C).
    direction = gt - centre
    other = gt + 0.5 * direction
    assert not np.allclose(other, gt)

    pixel_other = project_points(cam0, other)[0]
    assert np.allclose(pixel_other, pixel_gt, atol=1e-6), (
        "two distinct 3D points on the ray must share the same pixel"
    )

    # But adding a second camera disambiguates them.
    two_views = np.zeros(6, dtype=bool)
    two_views[[0, 3]] = True
    points2d = _project_all(proj_mats, gt)
    recovered = triangulate_dlt(proj_mats, points2d, mask=two_views)
    assert np.allclose(recovered, gt, atol=TOL)


def test_drop_k_mask_is_deterministic() -> None:
    """Same seed -> same mask; occlusion is reproducible."""
    a = drop_k_mask(n_views=6, n_points=4, k_drop=2, seed=SEED)
    b = drop_k_mask(n_views=6, n_points=4, k_drop=2, seed=SEED)
    assert np.array_equal(a, b)
    assert np.all(a.sum(axis=1) == 4)  # 6 - 2 visible per point


def test_drop_k_mask_rejects_too_few_visible() -> None:
    """Refuse masks that leave < 2 visible views."""
    with pytest.raises(ValueError, match="< 2 visible"):
        drop_k_mask(n_views=6, n_points=1, k_drop=5, seed=SEED)
