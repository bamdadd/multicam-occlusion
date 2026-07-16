"""Numpy-only recovery tests over committed analytic-manifest fixtures.

These run inside the core CI gate: they import ONLY ``multicam_occlusion`` (no
multicam-sim, no matplotlib) and read the small sweep manifests committed under
``tests/fixtures/sweep/`` — the exact JSON multicam-sim's producer emits. They
assert the benchmark's thesis on real data:

  1. the rebuilt ``P = K [R | t]`` reproduces every stored pixel (convention
     round-trip);
  2. multi-view recovery is far more accurate than the single-view baseline;
  3. as occlusion rises, the single-view error climbs while multi-view stays low.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from multicam_occlusion.observation import ObservationManifest
from multicam_occlusion.recovery import back_project, recover_trajectory

FIXTURES = Path(__file__).parent / "fixtures" / "sweep"
SEED = 20260716
OCCLUDED_CAM = 1


def _levels() -> list[Path]:
    files = sorted(FIXTURES.glob("level_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    assert files, "no sweep fixtures committed"
    return files


def test_loader_reprojection_round_trips() -> None:
    """Every visible observation reprojects onto its stored uv (convention check)."""
    for path in _levels():
        # from_json runs verify_reprojection() itself; assert it does not raise.
        ObservationManifest.from_json(path, verify=True)


def test_multiview_beats_singleview_every_level() -> None:
    """Multi-view recovery error is far below the single-view baseline throughout."""
    for path in _levels():
        manifest = ObservationManifest.from_json(path)
        rec = recover_trajectory(manifest, single_cam=OCCLUDED_CAM, pixel_noise=0.5, seed=SEED)
        multi, single = rec.multi_mpjpe(), rec.single_mpjpe()
        assert rec.multi_recoverable_rate() == 1.0  # 3 cams, at most one dropped
        assert multi < single, f"{path.name}: multi {multi} !< single {single}"
        # A wide, unambiguous margin — not a coin-flip.
        assert single > 10.0 * multi, f"{path.name}: margin too small ({single / multi:.1f}x)"


def test_singleview_climbs_with_occlusion() -> None:
    """Occlusion blows up the single-view error; multi-view stays low and flat."""
    levels = _levels()
    first = recover_trajectory(
        ObservationManifest.from_json(levels[0]),
        single_cam=OCCLUDED_CAM,
        pixel_noise=0.5,
        seed=SEED,
    )
    last = recover_trajectory(
        ObservationManifest.from_json(levels[-1]),
        single_cam=OCCLUDED_CAM,
        pixel_noise=0.5,
        seed=SEED,
    )
    assert last.occlusion_rate() > first.occlusion_rate()
    assert last.single_mpjpe() > first.single_mpjpe()  # blows up
    # Multi-view barely moves: the extra occlusion is absorbed by other views.
    assert last.multi_mpjpe() < 2.0 * first.multi_mpjpe()


def test_exact_projection_recovers_to_machine_precision() -> None:
    """With no pixel noise, multi-view recovers ground truth to ~machine eps."""
    manifest = ObservationManifest.from_json(_levels()[0])
    rec = recover_trajectory(manifest, single_cam=OCCLUDED_CAM, pixel_noise=0.0)
    assert rec.multi_mpjpe() < 1e-9


def test_back_projection_is_depth_ambiguous() -> None:
    """One pixel + two different depths => two different 3D points on one ray.

    This is *why* a single view fails: without the right depth the back-projected
    point is wrong, and the depth is exactly what one camera cannot observe.
    """
    manifest = ObservationManifest.from_json(_levels()[0])
    cam = manifest.cameras[0]
    c, r, k = cam.centre(), cam.rotation(), cam.intrinsic_matrix()
    uv = np.array([320.0, 240.0])
    near = back_project(c, r, k, uv, depth=2.0)
    far = back_project(c, r, k, uv, depth=5.0)
    assert not np.allclose(near, far)
    # Both lie on the same ray from the camera centre (collinear with C).
    d_near, d_far = near - c, far - c
    cross = np.cross(d_near, d_far)
    assert np.linalg.norm(cross) < 1e-9


def test_recover_trajectory_rejects_unknown_entity() -> None:
    manifest = ObservationManifest.from_json(_levels()[0])
    with pytest.raises(ValueError, match="not in manifest"):
        recover_trajectory(manifest, entity_id="nope")
