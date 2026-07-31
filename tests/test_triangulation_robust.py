"""RANSAC triangulation survives outlying observations that break plain DLT (#1).

Deterministic pins (fixed seed, noise-free geometry, no tolerance tuning):

  1. A 20px/coord outlier in one of six views drags ``triangulate_dlt`` off the
     ground truth while ``triangulate_robust`` recovers it exactly. The negative
     control lives inside that same test, so it cannot pass against non-robust
     behaviour.
  2. Same seed, same inputs -> bitwise-identical output (the seeded-RNG
     criterion; the generator is built inside the call, never module-level).
  3. The seed is genuinely consumed: at ``max_iters=1`` the single drawn pair
     decides the answer, so distinct seeds give distinct outcomes. This closes
     the gap test 2 cannot -- a converged RANSAC is seed-independent, so a
     silently ignored seed would still pass a same-seed-same-result test.
  4. On uncontaminated observations robust and plain DLT agree with each other
     and with ground truth (the clean path is not sacrificed).
  5. Visibility-mask semantics are inherited from ``triangulate_dlt``: ``True``
     == visible, occluded rows may be NaN and never reach arithmetic, and < 2
     visible views raises.
  6. Past ~40px/coord the #35 rank guard makes plain DLT *raise* rather than
     return a wrong point, and robust triangulation still recovers there. The
     20px of test 1 is deliberately below that boundary so it pins recovery
     rather than accidentally pinning the raise.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from multicam_occlusion import (
    build_ring_cameras,
    occlude,
    project_points,
    triangulate_dlt,
    triangulate_robust,
)

SEED = 20260715
TOL = 1e-6
N_VIEWS = 6
GROUND_TRUTH = np.array([0.3, -0.4, 0.8])
#: Displacement, per coordinate, of the single outlying observation. Below the
#: ~40px/coord boundary at which the #35 rank guard starts refusing the whole
#: (inconsistent) system, so plain DLT returns a wrong point here rather than
#: raising -- which is the regime the issue describes, an outlier "dragging the
#: whole estimate off". test_gross_outlier_makes_plain_dlt_raise covers the
#: other side of that boundary.
OUTLIER_PX = 20.0
FloatArray = npt.NDArray[np.float64]


def _project_all(proj_mats: FloatArray, point3d: FloatArray) -> FloatArray:
    """Project one 3D point into every camera; returns ``(n_views, 2)``."""
    return np.vstack([project_points(p, point3d)[0] for p in proj_mats])


def _max_abs_error(recovered: FloatArray, gt: FloatArray) -> float:
    """The DESIGN.md metric: max absolute coordinate error."""
    return float(np.max(np.abs(recovered - gt)))


def _contaminated_views(offset_px: float) -> tuple[FloatArray, FloatArray]:
    """Six ring cameras and exact observations, with view 0 displaced by ``offset_px``."""
    proj_mats = build_ring_cameras(n_cameras=N_VIEWS)
    points2d = _project_all(proj_mats, GROUND_TRUTH)
    points2d[0] += offset_px
    return proj_mats, points2d


def test_robust_recovers_ground_truth_when_dlt_does_not() -> None:
    """The acceptance case: one gross outlier, plain DLT is wrong, robust is exact.

    Both halves are asserted here on purpose. The observations are otherwise
    noise-free, so the only thing that can move the estimate is the outlier, and
    plain DLT -- which weights every view equally -- lands ~0.0287 off in the
    DESIGN.md max-abs-coordinate metric while RANSAC reaches machine precision.
    """
    proj_mats, points2d = _contaminated_views(OUTLIER_PX)

    plain_error = _max_abs_error(triangulate_dlt(proj_mats, points2d), GROUND_TRUTH)
    assert plain_error > 1e-2, f"plain DLT must be dragged off the target, got {plain_error}"

    recovered = triangulate_robust(proj_mats, points2d, seed=SEED)
    assert np.allclose(recovered, GROUND_TRUTH, atol=TOL), f"{recovered} != {GROUND_TRUTH}"
    assert _max_abs_error(recovered, GROUND_TRUTH) < plain_error / 1e6


def test_robust_is_deterministic_for_a_fixed_seed() -> None:
    """Same seed, same inputs -> the identical array, bit for bit."""
    proj_mats, points2d = _contaminated_views(OUTLIER_PX)

    first = triangulate_robust(proj_mats, points2d, seed=SEED)
    second = triangulate_robust(proj_mats, points2d, seed=SEED)
    assert np.array_equal(first, second), f"{first} != {second}"


def test_seed_selects_the_sample_sequence() -> None:
    """Distinct seeds draw distinct minimal samples, so the seed is not ignored.

    Only meaningful at ``max_iters=1``: with a single draw the answer is fixed by
    which pair came up, whereas from two iterations on RANSAC has already found
    the clean consensus for every one of these seeds and they all agree. An
    outcome is either the recovered point or the refusal raised when the drawn
    pair contains the outlier and nothing reaches consensus.
    """
    proj_mats, points2d = _contaminated_views(OUTLIER_PX)

    def outcome(seed: int, max_iters: int) -> tuple[float, ...] | None:
        try:
            return tuple(triangulate_robust(proj_mats, points2d, max_iters=max_iters, seed=seed))
        except ValueError:
            return None

    seeds = range(1, 9)
    single_draw = {outcome(seed, max_iters=1) for seed in seeds}
    assert len(single_draw) >= 2, f"the seed must change which pair is drawn, got {single_draw}"

    converged = {outcome(seed, max_iters=2) for seed in seeds}
    assert len(converged) == 1, f"two draws should already converge for every seed, got {converged}"


def test_robust_matches_dlt_on_uncontaminated_observations() -> None:
    """With no outlier the robust path must not cost anything: both are exact."""
    proj_mats = build_ring_cameras(n_cameras=N_VIEWS)
    points2d = _project_all(proj_mats, GROUND_TRUTH)

    plain = triangulate_dlt(proj_mats, points2d)
    robust = triangulate_robust(proj_mats, points2d, seed=SEED)
    assert np.allclose(plain, GROUND_TRUTH, atol=TOL), f"{plain} != {GROUND_TRUTH}"
    assert np.allclose(robust, GROUND_TRUTH, atol=TOL), f"{robust} != {GROUND_TRUTH}"


def test_robust_honours_the_visibility_mask() -> None:
    """Mask semantics are inherited: NaN behind ``False``, and < 2 visible raises."""
    proj_mats, points2d = _contaminated_views(OUTLIER_PX)
    mask = np.ones(N_VIEWS, dtype=bool)
    mask[[1, 4]] = False
    # occlude() writes NaN into the hidden rows, exactly as the benchmark does;
    # nothing masked out may reach the arithmetic.
    observed = occlude(points2d[None], mask[None])[0]
    assert np.isnan(observed[~mask]).all()

    recovered = triangulate_robust(proj_mats, observed, mask=mask, seed=SEED)
    assert np.allclose(recovered, GROUND_TRUTH, atol=TOL), f"{recovered} != {GROUND_TRUTH}"

    single = np.zeros(N_VIEWS, dtype=bool)
    single[0] = True
    with pytest.raises(ValueError, match="need >= 2 visible views"):
        triangulate_robust(proj_mats, points2d, mask=single)


def test_gross_outlier_makes_plain_dlt_raise() -> None:
    """Characterisation: past ~40px/coord the #35 rank guard fires on the outlier.

    A gross outlier keeps ``sigma[-1]`` from collapsing, so an *inconsistent*
    system trips the same guard that rank-deficient ones do and plain DLT refuses
    outright. Robust triangulation is unaffected: the consensus set it refits on
    is consistent.
    """
    proj_mats, points2d = _contaminated_views(50.0)

    with pytest.raises(ValueError, match="rank-deficient DLT system"):
        triangulate_dlt(proj_mats, points2d)

    recovered = triangulate_robust(proj_mats, points2d, seed=SEED)
    assert np.allclose(recovered, GROUND_TRUTH, atol=TOL), f"{recovered} != {GROUND_TRUTH}"
