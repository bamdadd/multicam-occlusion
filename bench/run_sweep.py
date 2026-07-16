"""Generate an occlusion sweep and measure single-view vs multi-view recovery.

This is the *producer* side of the benchmark and the ONLY place that drives
multicam-sim. It builds a series of 3-camera scenes of one moving object at
*increasing* occlusion (a solid grows on camera 1's sightline), emits each as an
analytic observation manifest, then runs the numpy-only recovery pipeline from
:mod:`multicam_occlusion.recovery` on every scene.

It is deliberately OUTSIDE ``src/`` so the package and its CI gate stay
numpy-only: ``multicam_sim`` is an optional ``bench`` dependency, never installed
by ``uv sync --dev``. Run it with ``make sweep`` (or ``uv run --group bench
python bench/run_sweep.py``). Outputs:

* ``docs/occlusion_dose_response.json`` — the sweep curve (occlusion level,
  single/multi error) the plot script reads.
* ``tests/fixtures/sweep/level_*.json`` — a few committed manifests so a fast
  numpy-only test asserts multi-view beats single-view without this dependency.

Fixed seeds + fixed geometry make it fully reproducible.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from multicam_sim.dsl import CameraRig, Occlusion, SceneBuilder
from multicam_sim.dsl import Path as SimPath
from multicam_sim.manifest import build_manifest

from multicam_occlusion.observation import ObservationManifest
from multicam_occlusion.recovery import recover_trajectory

# ---- fixed benchmark configuration (all reproducible) --------------------- #
SEED = 20260716
FPS = 30.0
NUM_FRAMES = 21
N_CAMS = 3
RING_RADIUS = 4.0
RING_HEIGHT = 1.5
IMG_W, IMG_H = 640, 480
FOCAL = 800.0
LOOK_AT = (0.0, 0.0, 0.8)
# Object crosses the scene through a range of depths so a single fixed-depth
# prior is provably wrong for most of the trajectory.
PATH_START = (-1.3, -0.4, 0.5)
PATH_END = (1.3, 0.4, 1.1)
# Occluder radius on camera 1's sightline, swept from none to a large solid.
DOSE_RADII = [0.0, 0.08, 0.14, 0.20, 0.28, 0.38, 0.50]
PIXEL_NOISE = 0.5  # px Gaussian keypoint noise (realistic detector)
OCCLUDED_CAM = 1  # the camera the solid targets; single-view baselines on it


@dataclass(frozen=True)
class SweepPoint:
    dose_radius: float
    occlusion_rate: float
    multi_mpjpe: float
    single_mpjpe: float
    multi_recoverable_rate: float
    single_observed_rate: float
    min_visible: int


def build_sweep_manifest(dose_radius: float) -> dict[str, object]:
    """Build one scene's analytic manifest at the given occluder radius."""
    cameras = CameraRig.ring(
        N_CAMS,
        radius=RING_RADIUS,
        height=RING_HEIGHT,
        look_at=LOOK_AT,
        width=IMG_W,
        height_px=IMG_H,
        focal=FOCAL,
    )
    builder = (
        SceneBuilder(fps=FPS, num_frames=NUM_FRAMES)
        .cameras(cameras)
        .entity("obj", SimPath.linear(PATH_START, PATH_END))
    )
    if dose_radius > 0.0:
        builder = builder.occlude(
            Occlusion.sphere(dose_radius).blocks(camera=OCCLUDED_CAM).during((0, NUM_FRAMES - 1))
        )
    return build_manifest(builder.build())


def measure(manifest_dict: dict[str, object], dose_radius: float) -> SweepPoint:
    """Load a manifest dict and measure both estimators over the trajectory."""
    manifest = ObservationManifest.model_validate(manifest_dict)
    manifest.verify_reprojection()
    rec = recover_trajectory(
        manifest,
        single_cam=OCCLUDED_CAM,
        pixel_noise=PIXEL_NOISE,
        seed=SEED,
    )
    return SweepPoint(
        dose_radius=dose_radius,
        occlusion_rate=rec.occlusion_rate(),
        multi_mpjpe=rec.multi_mpjpe(),
        single_mpjpe=rec.single_mpjpe(),
        multi_recoverable_rate=rec.multi_recoverable_rate(),
        single_observed_rate=rec.single_observed_rate(),
        min_visible=min(fr.n_visible for fr in rec.frames),
    )


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    fixtures = root / "tests" / "fixtures" / "sweep"
    fixtures.mkdir(parents=True, exist_ok=True)

    points: list[SweepPoint] = []
    for i, radius in enumerate(DOSE_RADII):
        manifest_dict = build_sweep_manifest(radius)
        point = measure(manifest_dict, radius)
        points.append(point)
        # Commit a subset of manifests as numpy-only test fixtures.
        (fixtures / f"level_{i}.json").write_text(json.dumps(manifest_dict, indent=2))

    print(
        f"{'dose_r':>7} {'occ_rate':>9} {'min_vis':>8} "
        f"{'single_mpjpe':>13} {'multi_mpjpe':>12} {'ratio':>8}"
    )
    for p in points:
        ratio = p.single_mpjpe / p.multi_mpjpe if p.multi_mpjpe else float("inf")
        print(
            f"{p.dose_radius:7.2f} {p.occlusion_rate:9.3f} {p.min_visible:8d} "
            f"{p.single_mpjpe:13.4f} {p.multi_mpjpe:12.5f} {ratio:8.1f}x"
        )

    out = root / "docs" / "occlusion_dose_response.json"
    out.write_text(
        json.dumps(
            {
                "config": {
                    "seed": SEED,
                    "n_cams": N_CAMS,
                    "num_frames": NUM_FRAMES,
                    "pixel_noise": PIXEL_NOISE,
                    "occluded_cam": OCCLUDED_CAM,
                    "single_cam": OCCLUDED_CAM,
                },
                "points": [asdict(p) for p in points],
            },
            indent=2,
        )
    )
    print(f"\nwrote {out.relative_to(root)}")
    print(f"wrote {len(DOSE_RADII)} fixtures under {fixtures.relative_to(root)}/")


if __name__ == "__main__":
    main()
