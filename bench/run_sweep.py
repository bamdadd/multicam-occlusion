"""Occlusion dose-response for single-view vs multi-view 3D POSE recovery.

The *producer* side of the benchmark and the only place that drives multicam-sim.
A posed COCO-17 skeleton stands in a 3-camera ring; a hand-proxy sweeps across one
camera's view so different joints occlude at different times. multicam-sim emits the
analytic manifest (projection + hard ``visible`` + image-space ``visible_fraction``
from ``build_manifest(object_radius=...)``); the numpy-only pipeline in
:mod:`multicam_occlusion.recovery` then recovers every joint two ways and we bin the
frames by occlusion dose.

Pre-registered design (fixed BEFORE running; not tuned to the result):

* dose ``phi`` per frame = mean over all (joint, camera) of ``1 - visible_fraction``.
  Secondary readout: fraction of (joint, camera) with ``occluded == True``.
* frames are binned by ``phi`` on the fixed edges ``PHI_EDGES``.
* single-view baseline = per-joint best camera (max mean ``visible_fraction`` for that
  joint) back-projected to that joint's centroid-depth prior. Per-joint-fair, not one
  global camera.
* multi-view = DLT over each joint's visible cameras (>= 2).
* variance source = seeded Gaussian pixel noise on the triangulated ``uv`` (SEEDS);
  visibility/dose are geometric and seed-independent, so coverage has no seed spread.

Every MPJPE is co-reported with its coverage: ``multi_mpjpe`` averages ONLY joints
with >= 2 visible cameras, so a flat MPJPE curve can be survivorship. The coverage
curve is where that truth lives. Run it with ``make sweep``. Outputs:

* ``docs/occlusion_dose_response.json`` — the binned dose-response the plot reads.
* ``tests/fixtures/pose/manifest.json`` — the committed manifest a numpy-only test
  replays (seed-independent, so one file covers every seed).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from multicam_sim import build_manifest
from multicam_sim.dsl import CameraRig, HandSweep
from multicam_sim.pose import PoseFrame, PoseTrajectory, Skeleton
from multicam_sim.scene import Scene
from pydantic import BaseModel

from multicam_occlusion.observation import ObservationManifest
from multicam_occlusion.recovery import PoseRecovery, recover_pose

# ---- fixed benchmark configuration (all pre-registered, all reproducible) --- #
FPS = 30.0
NUM_FRAMES = 41
N_CAMS = 3
RING_RADIUS = 4.0
RING_HEIGHT = 1.5
IMG_W, IMG_H = 640, 480
FOCAL = 800.0
LOOK_AT = (0.0, 0.0, 1.0)
PIXEL_NOISE = 1.0  # px Gaussian keypoint noise (realistic detector)
SEEDS = (0, 1, 2)
OCCLUDED_CAM = 1  # the camera the hand sweeps across
OBJECT_RADIUS = 0.07  # per-joint silhouette proxy for visible_fraction
HAND_RADIUS = 0.60  # large enough to cover most of the skeleton on the occluded cam
HAND_SPAN = 0.55
HAND_TARGET_JOINT = "left_hip"  # mid-body base; the sphere also catches torso joints
# Pre-registered phi bin edges (fixed before the run; empty bins reported as such).
PHI_EDGES = tuple(round(x, 4) for x in np.linspace(0.0, 0.35, 8))

# Standing COCO-17 offsets (dx, dy, dz) from the foot base; +y is the facing dir.
_JOINT_OFFSETS: dict[str, tuple[float, float, float]] = {
    "nose": (0.0, 0.10, 1.60),
    "left_eye": (0.03, 0.10, 1.64),
    "right_eye": (-0.03, 0.10, 1.64),
    "left_ear": (0.08, 0.05, 1.63),
    "right_ear": (-0.08, 0.05, 1.63),
    "left_shoulder": (0.20, 0.0, 1.45),
    "right_shoulder": (-0.20, 0.0, 1.45),
    "left_elbow": (0.26, 0.12, 1.20),
    "right_elbow": (-0.26, 0.12, 1.20),
    "left_wrist": (0.22, 0.28, 1.00),
    "right_wrist": (-0.22, 0.28, 1.00),
    "left_hip": (0.12, 0.0, 0.95),
    "right_hip": (-0.12, 0.0, 0.95),
    "left_knee": (0.12, 0.02, 0.52),
    "right_knee": (-0.12, 0.02, 0.52),
    "left_ankle": (0.10, 0.0, 0.10),
    "right_ankle": (-0.10, 0.0, 0.10),
}


class BinStat(BaseModel):
    """Aggregated dose-response for one phi bin (mean +/- std over seeds)."""

    phi_lo: float
    phi_hi: float
    phi_mid: float
    n_frames: int
    n_joint_frames: int
    occluded_rate: float  # secondary readout: fraction of (joint,cam) with occluded=True
    multi_mpjpe_mean: float | None
    multi_mpjpe_std: float | None
    single_mpjpe_mean: float | None
    single_mpjpe_std: float | None
    multi_recoverable_mean: float
    multi_recoverable_std: float
    single_observed_mean: float
    single_observed_std: float


class SweepConfig(BaseModel):
    seeds: list[int]
    n_cams: int
    num_frames: int
    pixel_noise: float
    occluded_cam: int
    object_radius: float
    hand_radius: float
    hand_span: float
    phi_edges: list[float]


class SweepResult(BaseModel):
    config: SweepConfig
    bins: list[BinStat]


def build_pose_trajectory() -> PoseTrajectory:
    """A standing COCO-17 skeleton that sways and reaches, so joints span depth."""
    frames: list[PoseFrame] = []
    for f in range(NUM_FRAMES):
        phase = math.sin(2.0 * math.pi * f / (NUM_FRAMES - 1))  # -1..1 smooth
        sway = 0.15 * phase  # whole-body sway in x (depth spread across cameras)
        joints: dict[str, list[float]] = {}
        for name, (dx, dy, dz) in _JOINT_OFFSETS.items():
            reach = 0.10 * phase if name.endswith("wrist") else 0.0  # arms reach in +y
            joints[name] = [dx + sway, dy + reach, dz]
        frames.append(PoseFrame(frame=f, joints=joints))
    return PoseTrajectory(id="operator", skeleton=Skeleton.coco17(), frames=frames)


def build_scene() -> Scene:
    """3-camera ring + posed skeleton + a hand sweeping across the occluded camera."""
    cameras = CameraRig.ring(
        N_CAMS,
        radius=RING_RADIUS,
        height=RING_HEIGHT,
        look_at=LOOK_AT,
        width=IMG_W,
        height_px=IMG_H,
        focal=FOCAL,
    )
    entity = build_pose_trajectory().to_entity()
    hand = (
        HandSweep.sphere(HAND_RADIUS, span=HAND_SPAN)
        .blocks(camera=OCCLUDED_CAM)
        .on(entity.id, HAND_TARGET_JOINT)
        .during((0, NUM_FRAMES - 1))
        .realize(cameras, entity.frames)
    )
    return Scene(
        fps=FPS,
        num_frames=NUM_FRAMES,
        cameras=cameras,
        entities=[entity],
        occluders=[hand],
    )


def build_manifest_dict() -> dict[str, object]:
    """The analytic manifest for the posed hand-sweep scene (with visible_fraction)."""
    manifest = build_manifest(build_scene(), object_radius=OBJECT_RADIUS)
    return json.loads(manifest.to_json())


def frame_phi(manifest: ObservationManifest) -> dict[int, float]:
    """Per-frame occlusion dose phi = mean over (joint, cam) of (1 - visible_fraction)."""
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


def frame_occluded_rate(manifest: ObservationManifest) -> dict[int, float]:
    """Per-frame fraction of (joint, cam) observations with occluded == True."""
    entity = manifest.entities[0]
    rate: dict[int, float] = {}
    for frame in entity.frames:
        flags = [
            1.0 if obs.occluded else 0.0 for point in frame.points.values() for obs in point.per_cam
        ]
        rate[frame.frame] = float(np.mean(flags)) if flags else 0.0
    return rate


def _bin_index(phi: float) -> int | None:
    for i in range(len(PHI_EDGES) - 1):
        lo, hi = PHI_EDGES[i], PHI_EDGES[i + 1]
        # include the top edge in the last bin
        if lo <= phi < hi or (i == len(PHI_EDGES) - 2 and phi == hi):
            return i
    return None


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values))


def aggregate(manifest: ObservationManifest, recoveries: dict[int, PoseRecovery]) -> list[BinStat]:
    """Bin frames by phi and aggregate both estimators (mean +/- std over seeds)."""
    phi = frame_phi(manifest)
    occ_rate = frame_occluded_rate(manifest)
    frames_in_bin: dict[int, list[int]] = {i: [] for i in range(len(PHI_EDGES) - 1)}
    for fr, value in phi.items():
        b = _bin_index(value)
        if b is not None:
            frames_in_bin[b].append(fr)

    stats: list[BinStat] = []
    for b, frame_ids in frames_in_bin.items():
        frame_set = set(frame_ids)
        # per-seed bin metrics
        multi_mpjpe_s, single_mpjpe_s, recov_s, obsv_s = [], [], [], []
        n_joint_frames = 0
        for seed in SEEDS:
            recs = [r for r in recoveries[seed].records if r.frame in frame_set]
            if seed == SEEDS[0]:
                n_joint_frames = len(recs)
            if not recs:
                continue
            multi_errs = [r.multi_err for r in recs if r.multi_err is not None]
            if multi_errs:
                multi_mpjpe_s.append(float(np.mean(multi_errs)))
            single_mpjpe_s.append(float(np.mean([r.single_err for r in recs])))
            recov_s.append(sum(r.multi_err is not None for r in recs) / len(recs))
            obsv_s.append(sum(r.single_observed for r in recs) / len(recs))

        mm, ms = _mean_std(multi_mpjpe_s)
        sm, ss = _mean_std(single_mpjpe_s)
        rm, rs = _mean_std(recov_s)
        om, os_ = _mean_std(obsv_s)
        bin_occ = float(np.mean([occ_rate[fr] for fr in frame_ids])) if frame_ids else 0.0
        stats.append(
            BinStat(
                phi_lo=PHI_EDGES[b],
                phi_hi=PHI_EDGES[b + 1],
                phi_mid=0.5 * (PHI_EDGES[b] + PHI_EDGES[b + 1]),
                n_frames=len(frame_ids),
                n_joint_frames=n_joint_frames,
                occluded_rate=bin_occ,
                multi_mpjpe_mean=mm,
                multi_mpjpe_std=ms,
                single_mpjpe_mean=sm,
                single_mpjpe_std=ss,
                multi_recoverable_mean=rm if rm is not None else 0.0,
                multi_recoverable_std=rs if rs is not None else 0.0,
                single_observed_mean=om if om is not None else 0.0,
                single_observed_std=os_ if os_ is not None else 0.0,
            )
        )
    return stats


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    manifest_dict = build_manifest_dict()
    manifest = ObservationManifest.model_validate(manifest_dict)
    manifest.verify_reprojection()

    recoveries = {
        seed: recover_pose(manifest, pixel_noise=PIXEL_NOISE, seed=seed) for seed in SEEDS
    }
    bins = aggregate(manifest, recoveries)

    result = SweepResult(
        config=SweepConfig(
            seeds=list(SEEDS),
            n_cams=N_CAMS,
            num_frames=NUM_FRAMES,
            pixel_noise=PIXEL_NOISE,
            occluded_cam=OCCLUDED_CAM,
            object_radius=OBJECT_RADIUS,
            hand_radius=HAND_RADIUS,
            hand_span=HAND_SPAN,
            phi_edges=list(PHI_EDGES),
        ),
        bins=bins,
    )

    fixtures = root / "tests" / "fixtures" / "pose"
    fixtures.mkdir(parents=True, exist_ok=True)
    (fixtures / "manifest.json").write_text(json.dumps(manifest_dict, indent=2))

    out = root / "docs" / "occlusion_dose_response.json"
    out.write_text(result.model_dump_json(indent=2))

    print(
        f"{'phi_mid':>8} {'n_fr':>5} {'cover_multi':>12} {'cover_single':>13} "
        f"{'multi_mpjpe':>18} {'single_mpjpe':>18}"
    )

    def fmt(mean: float | None, std: float | None) -> str:
        return "n/a" if mean is None else f"{mean:.4f}+-{std:.4f}"

    for s in bins:
        if s.n_frames == 0:
            continue
        mm = fmt(s.multi_mpjpe_mean, s.multi_mpjpe_std)
        sm = fmt(s.single_mpjpe_mean, s.single_mpjpe_std)
        print(
            f"{s.phi_mid:8.3f} {s.n_frames:5d} {s.multi_recoverable_mean:12.3f} "
            f"{s.single_observed_mean:13.3f} {mm:>18} {sm:>18}"
        )
    print(f"\nwrote {out.relative_to(root)}")
    print(f"wrote {(fixtures / 'manifest.json').relative_to(root)}")


if __name__ == "__main__":
    main()
