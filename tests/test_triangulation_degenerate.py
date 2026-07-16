"""Characterization: linear DLT degrades as the camera baseline shrinks (#9).

Deterministic pins of the ill-conditioned boundary (fixed seed, monotone and
relative assertions, no absolute thresholds):

  1. Under the DESIGN.md default 0.5px pixel noise, recovery error GROWS as
     two-view angular separation shrinks (45° < 10° < 2°).
  2. Noise-free projections still recover to machine precision there.
  3. Collinear limit (target on the baseline): rank-deficient DLT, silently
     wrong along-baseline coordinate; transverse coordinates recover.
  4. DLT singular-value ratio σmax/σ3 degrades monotonically with separation.
  5. A point at infinity raises the documented ValueError.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from multicam_occlusion import build_ring_cameras, project_points, triangulate_dlt

SEED = 20260715
NOISE_PX = 0.5  # DESIGN.md: default Gaussian pixel-noise level
TOL = 1e-6
SEPARATIONS_DEG = (45.0, 10.0, 2.0)  # wide -> near-degenerate
FloatArray = npt.NDArray[np.float64]


def _project_all(proj_mats: FloatArray, point3d: FloatArray) -> FloatArray:
    """Project one 3D point into every camera; returns ``(n_views, 2)``."""
    return np.vstack([project_points(p, point3d)[0] for p in proj_mats])


def _two_ring_cameras(separation_deg: float) -> FloatArray:
    """Two cameras from a 1-degree-spaced ring, ``separation_deg`` apart."""
    return build_ring_cameras(n_cameras=360)[[0, int(round(separation_deg))]]


def _dlt_system(proj_mats: FloatArray, points2d: FloatArray) -> FloatArray:
    """The DLT constraint matrix, built exactly as DESIGN.md documents it."""
    rows = []
    for i in range(proj_mats.shape[0]):
        p = proj_mats[i]
        rows.append(points2d[i, 0] * p[2] - p[0])
        rows.append(points2d[i, 1] * p[2] - p[1])
    return np.stack(rows)


def _max_abs_error(recovered: FloatArray, gt: FloatArray) -> float:
    """The DESIGN.md metric: max absolute coordinate error."""
    return float(np.max(np.abs(recovered - gt)))


def test_recovery_error_grows_as_baseline_shrinks() -> None:
    """With 0.5px noise the recovery error is strictly monotone in separation."""
    gt = np.array([0.3, -0.4, 0.8])
    errors = []
    for separation in SEPARATIONS_DEG:
        proj_mats = _two_ring_cameras(separation)
        points2d = _project_all(proj_mats, gt)
        noisy = points2d + np.random.default_rng(SEED).normal(0.0, NOISE_PX, points2d.shape)
        errors.append(_max_abs_error(triangulate_dlt(proj_mats, noisy), gt))
    assert errors[0] < errors[1] < errors[2], f"error must grow as baseline shrinks: {errors}"


def test_noise_free_recovery_stays_exact_at_small_baseline() -> None:
    """Exact projections recover to machine precision even at 2 degrees."""
    gt = np.array([0.3, -0.4, 0.8])
    for separation in SEPARATIONS_DEG:
        proj_mats = _two_ring_cameras(separation)
        recovered = triangulate_dlt(proj_mats, _project_all(proj_mats, gt))
        assert np.allclose(recovered, gt, atol=TOL), f"{separation} deg: {recovered} != {gt}"


def test_point_on_baseline_is_silently_unobservable() -> None:
    """Collinear-centres limit: rank-deficient DLT, no error raised, wrong point.

    Opposite cameras of a 6-ring, target on the line joining their centres;
    today's code guards only ``< 2`` views and points at infinity.
    """
    ring = build_ring_cameras(n_cameras=6)
    proj_mats = ring[[0, 3]]  # opposite cameras; baseline is the line y=0, z=1.5
    gt = np.array([0.0, 0.0, 1.5])  # exactly on that baseline
    points2d = _project_all(proj_mats, gt)

    sv = np.linalg.svd(_dlt_system(proj_mats, points2d), compute_uv=False)
    assert sv[2] < 1e-9 * sv[0], "collinear DLT system must be rank-deficient"

    # Must NOT raise today; the off-baseline coordinates are still recovered...
    recovered = triangulate_dlt(proj_mats, points2d)
    assert np.allclose(recovered[1:], gt[1:], atol=TOL)
    # ...but the overall error dwarfs a well-posed solve of the same point.
    err_degenerate = _max_abs_error(recovered, gt)
    err_well_posed = _max_abs_error(
        triangulate_dlt(ring[[0, 1]], _project_all(ring[[0, 1]], gt)), gt
    )
    assert err_degenerate > 1e6 * err_well_posed


def test_dlt_conditioning_degrades_as_baseline_shrinks() -> None:
    """σmax/σ3 of the noise-free DLT system grows strictly as separation shrinks."""
    gt = np.array([0.3, -0.4, 0.8])
    ratios = []
    for separation in SEPARATIONS_DEG:
        proj_mats = _two_ring_cameras(separation)
        sv = np.linalg.svd(_dlt_system(proj_mats, _project_all(proj_mats, gt)), compute_uv=False)
        ratios.append(sv[0] / sv[2])
    assert ratios[0] < ratios[1] < ratios[2], f"conditioning must degrade: {ratios}"


def test_point_at_infinity_raises_value_error() -> None:
    """Observations of a pure direction (point at infinity) trip the guard."""
    proj_mats = build_ring_cameras(n_cameras=6)[[0, 3]]
    direction = np.array([0.1, -0.2, -1.0])
    direction /= np.linalg.norm(direction)
    # A point at infinity (d, 0) projects to x ~ K R d = P[:, :3] d.
    projected = proj_mats[:, :, :3] @ direction
    assert np.all(projected[:, 2] > 0), "direction must lie in front of both cameras"
    points2d = projected[:, :2] / projected[:, 2, None]
    with pytest.raises(ValueError, match="recovered point at infinity"):
        triangulate_dlt(proj_mats, points2d)
