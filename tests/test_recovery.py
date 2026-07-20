"""Numpy-only recovery tests over the committed posed-skeleton manifest.

These run inside the core CI gate: they import ONLY ``multicam_occlusion`` (no
multicam-sim, no matplotlib) and replay ``tests/fixtures/pose/manifest.json`` — the
exact analytic manifest the producer emits for the hand-sweep pose scene. They
assert the HONEST finding, not a strawman:

  1. the rebuilt ``P = K [R | t]`` reproduces every stored pixel (convention check);
  2. on RECOVERABLE joints, multi-view error is well below the per-joint-fair
     single-view baseline (multi wins per-joint accuracy, and the baseline is each
     joint's own best camera, not one global worst camera);
  3. the flat multi-view error is partly survivorship: as the occlusion dose rises,
     multi-view coverage drops (it silently stops solving the hard joints);
  4. coverage is not a multi-view advantage here: single-view coverage tracks it, so
     the win is accuracy, not seeing more joints;
  5. seeded pixel noise is a real variance source (no noise => near-exact recovery).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from multicam_occlusion.observation import ObservationManifest
from multicam_occlusion.recovery import (
    PoseRecovery,
    back_project,
    recover_pose,
)

FIXTURE = Path(__file__).parent / "fixtures" / "pose" / "manifest.json"
SEEDS = (0, 1, 2)
PIXEL_NOISE = 1.0


def _manifest() -> ObservationManifest:
    return ObservationManifest.from_json(FIXTURE)


def _frame_phi(manifest: ObservationManifest) -> dict[int, float]:
    """Per-frame occlusion dose phi = mean over (joint, cam) of 1 - visible_fraction."""
    entity = manifest.entities[0]
    phi: dict[int, float] = {}
    for frame in entity.frames:
        drops = [
            1.0 - (obs.visible_fraction if obs.visible_fraction is not None else 1.0)
            for point in frame.points.values()
            for obs in point.per_cam
        ]
        phi[frame.frame] = float(np.mean(drops)) if drops else 0.0
    return phi


def _agg(rec: PoseRecovery, frames: set[int]) -> tuple[float, float, float, float]:
    """(multi_mpjpe, single_mpjpe, multi_coverage, single_coverage) over a frame subset."""
    recs = [r for r in rec.records if r.frame in frames]
    multi = [r.multi_err for r in recs if r.multi_err is not None]
    single = [r.single_err for r in recs]
    multi_cov = sum(r.multi_err is not None for r in recs) / len(recs)
    single_cov = sum(r.single_observed for r in recs) / len(recs)
    return (
        float(np.mean(multi)) if multi else float("nan"),
        float(np.mean(single)) if single else float("nan"),
        multi_cov,
        single_cov,
    )


def test_loader_reprojection_round_trips() -> None:
    """Every visible observation reprojects onto its stored uv (convention check)."""
    ObservationManifest.from_json(FIXTURE, verify=True)


def test_manifest_carries_visibility_labels() -> None:
    """The fixture must carry visible_fraction (the dose this benchmark bins on)."""
    obs = _manifest().entities[0].frames[0].points["nose"].per_cam[0]
    assert obs.visible_fraction is not None


def test_multiview_more_accurate_on_recoverable_joints() -> None:
    """On joints both can produce, multi-view error is well below single-view.

    Per-joint-fair baseline (each joint's best camera + its own depth prior), so
    this is not the global-worst-camera strawman. Multi still wins on accuracy
    because a single pixel cannot observe depth.
    """
    manifest = _manifest()
    all_frames = {f.frame for f in manifest.entities[0].frames}
    for seed in SEEDS:
        rec = recover_pose(manifest, pixel_noise=PIXEL_NOISE, seed=seed)
        multi, single, _, _ = _agg(rec, all_frames)
        assert multi < single, f"seed {seed}: multi {multi} !< single {single}"
        # a clear, not coin-flip, margin (single is depth-blind); NOT the old 10x claim
        assert single > 3.0 * multi, f"seed {seed}: margin {single / multi:.1f}x"


def test_flat_multi_error_is_survivorship_coverage_drops() -> None:
    """As occlusion rises, multi-view coverage falls even while its error stays low.

    This is the honesty gate: the flat MPJPE curve silently drops the hard joints.
    """
    manifest = _manifest()
    phi = _frame_phi(manifest)
    median = float(np.median(list(phi.values())))
    low = {fr for fr, v in phi.items() if v <= median}
    high = {fr for fr, v in phi.items() if v > median}

    rec = recover_pose(manifest, pixel_noise=PIXEL_NOISE, seed=SEEDS[0])
    multi_lo, _, cov_lo, _ = _agg(rec, low)
    multi_hi, _, cov_hi, _ = _agg(rec, high)

    assert cov_hi < cov_lo, f"coverage should fall with occlusion ({cov_hi} !< {cov_lo})"
    # error on the joints it CAN still solve stays low and flat (survivorship).
    assert multi_hi < 2.0 * multi_lo


def test_coverage_tracks_between_estimators() -> None:
    """Multi-view does not recover many more joints than single-view here.

    A physical hand occludes multiple sightlines, so both estimators lose roughly
    the same joint-frames. The multi-view advantage is accuracy, not coverage; this
    guards against re-introducing a 'multi sees far more' strawman.
    """
    manifest = _manifest()
    phi = _frame_phi(manifest)
    median = float(np.median(list(phi.values())))
    high = {fr for fr, v in phi.items() if v > median}
    rec = recover_pose(manifest, pixel_noise=PIXEL_NOISE, seed=SEEDS[0])
    _, _, cov_multi, cov_single = _agg(rec, high)
    assert abs(cov_multi - cov_single) < 0.15


def test_seeded_noise_is_the_variance_source() -> None:
    """No pixel noise => near-exact multi recovery; noise => a real seed spread."""
    manifest = _manifest()
    all_frames = {f.frame for f in manifest.entities[0].frames}

    clean = recover_pose(manifest, pixel_noise=0.0, seed=0)
    multi_clean = [r.multi_err for r in clean.records if r.multi_err is not None]
    assert max(multi_clean) < 1e-9  # DLT recovers GT to ~machine eps without noise

    means = []
    for seed in SEEDS:
        rec = recover_pose(manifest, pixel_noise=PIXEL_NOISE, seed=seed)
        m, _, _, _ = _agg(rec, all_frames)
        means.append(m)
    assert np.std(means) > 0.0  # noise actually moves the estimate across seeds


def test_back_projection_is_depth_ambiguous() -> None:
    """One pixel + two depths => two different 3D points on one ray (why mono fails)."""
    cam = _manifest().cameras[0]
    c, r, k = cam.centre(), cam.rotation(), cam.intrinsic_matrix()
    uv = np.array([320.0, 240.0])
    near = back_project(c, r, k, uv, depth=2.0)
    far = back_project(c, r, k, uv, depth=5.0)
    assert not np.allclose(near, far)
    cross = np.cross(near - c, far - c)
    assert np.linalg.norm(cross) < 1e-9  # collinear with the camera centre


def test_recover_pose_rejects_unknown_entity() -> None:
    with pytest.raises(ValueError, match="not in manifest"):
        recover_pose(_manifest(), entity_id="nope")
