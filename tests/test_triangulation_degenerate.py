"""Characterization: linear DLT degrades as the camera baseline shrinks (#9).

Deterministic pins of the ill-conditioned boundary (fixed seed, monotone and
relative assertions, no absolute thresholds):

  1. Under the DESIGN.md default 0.5px pixel noise, recovery error GROWS as
     two-view angular separation shrinks (45° < 10° < 2°).
  2. Noise-free projections still recover to machine precision there.
  3. Collinear limit (target on the baseline): the DLT system is rank-deficient
     and ``triangulate_dlt`` raises rather than returning the arbitrary vector
     SVD picks out of the 2D null space (#30).
  4. DLT singular-value ratio σmax/σ3 degrades monotonically with separation.
  5. A point at infinity raises the documented ValueError.
  6. The #30 rank guard does not fire on well-conditioned input, and its
     boundary sits where ``DEGENERACY_RATIO_TOL`` was measured to be.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from multicam_occlusion import build_ring_cameras, project_points, triangulate_dlt
from multicam_occlusion.triangulation import DEGENERACY_RATIO_TOL

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


def test_point_on_baseline_raises_value_error() -> None:
    """Collinear-centres limit: rank-deficient DLT must raise, not guess (#30).

    Opposite cameras of a 6-ring, target on the line joining their centres. The
    null space is 2D there, so SVD returns an arbitrary vector from it; before
    #30 that came back as a point whose along-baseline coordinate was silently
    wrong by ~4 units while the transverse ones looked perfect.
    """
    ring = build_ring_cameras(n_cameras=6)
    proj_mats = ring[[0, 3]]  # opposite cameras; baseline is the line y=0, z=1.5
    gt = np.array([0.0, 0.0, 1.5])  # exactly on that baseline
    points2d = _project_all(proj_mats, gt)

    sv = np.linalg.svd(_dlt_system(proj_mats, points2d), compute_uv=False)
    assert sv[2] < 1e-9 * sv[0], "collinear DLT system must be rank-deficient"
    assert sv[-1] / sv[-2] > DEGENERACY_RATIO_TOL, "the rank test must see the 2D null space"

    with pytest.raises(ValueError, match="rank-deficient DLT system"):
        triangulate_dlt(proj_mats, points2d)

    # The same point off the baseline stays well posed and exactly recoverable,
    # so it is the geometry that is refused, not the point.
    well_posed = ring[[0, 1]]
    assert np.allclose(triangulate_dlt(well_posed, _project_all(well_posed, gt)), gt, atol=TOL)


#: Separations at which the guard is required never to fire under NOISE_PX. The
#: 2 degrees of SEPARATIONS_DEG is deliberately absent: at 0.5px noise its ratio
#: distribution (p99 ~0.26) overlaps the on-baseline population (median ~0.21),
#: so no tolerance can both admit it and reject a collinear solve. That is the
#: same configuration test_recovery_error_grows_as_baseline_shrinks pins as the
#: worst of the sweep, and the guard rejects ~11% of its noise draws.
GUARDED_SEPARATIONS_DEG = (45.0, 20.0, 10.0)


def test_rank_guard_does_not_fire_on_well_conditioned_input() -> None:
    """The #30 guard must be invisible to healthy solves, noisy ones included.

    A tolerance set too low silently converts normal use into ValueErrors, so
    sweep view counts and two-view separations, noise-free and at the DESIGN.md
    0.5px level, and require exact recovery whenever the observations are exact.
    """
    gt = np.array([0.3, -0.4, 0.8])
    rng = np.random.default_rng(SEED)
    ring360 = build_ring_cameras(n_cameras=360)

    rigs = [build_ring_cameras(n_cameras=n) for n in (2, 3, 4, 6, 8)]
    rigs += [ring360[[0, int(sep)]] for sep in GUARDED_SEPARATIONS_DEG]

    for proj_mats in rigs:
        points2d = _project_all(proj_mats, gt)
        assert np.allclose(triangulate_dlt(proj_mats, points2d), gt, atol=TOL)
        for _ in range(200):
            noisy = points2d + rng.normal(0.0, NOISE_PX, points2d.shape)
            triangulate_dlt(proj_mats, noisy)  # must not raise


def test_rank_guard_boundary_sits_where_it_was_measured() -> None:
    """A ratio just inside the tolerance still solves, and is genuinely near it.

    Fixed seed 72 on a 2-degree baseline with 0.5px noise lands the ratio at
    ~0.8x ``DEGENERACY_RATIO_TOL`` -- ill-conditioned enough to sit in the
    guard's approach, healthy enough that refusing it would be a regression.
    """
    gt = np.array([0.3, -0.4, 0.8])
    proj_mats = build_ring_cameras(n_cameras=360)[[0, 2]]
    points2d = _project_all(proj_mats, gt)
    noisy = points2d + np.random.default_rng(72).normal(0.0, NOISE_PX, points2d.shape)

    sv = np.linalg.svd(_dlt_system(proj_mats, noisy), compute_uv=False)
    ratio = sv[-1] / sv[-2]
    assert 0.5 * DEGENERACY_RATIO_TOL <= ratio < DEGENERACY_RATIO_TOL, (
        f"boundary case must sit just inside the tolerance, got {ratio}"
    )
    assert np.all(np.isfinite(triangulate_dlt(proj_mats, noisy)))


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
