"""Generate the fusion order-verification fixture from real multicam-sim output.

This is the *producer* side of the fusion mode's order-verification path and,
like ``bench/run_sweep.py``, the only place that drives multicam-sim. It rebuilds
the asymmetric-visibility assembly-station scene shipped as multicam-sim's
``examples/assembly_station.py`` — an operator (COCO-17) framed by an overview
camera and three parts framed by a worktop camera, with complementary per-camera
``visible`` labels — and emits the two committed sidecars fusion consumes:

* ``tests/fixtures/sim_assembly_station/manifest.json`` — the full scene manifest
  (multicam-sim's ``build_manifest``: projection + in_view/visible/occ_frac);
* ``tests/fixtures/sim_assembly_station/order.json``    — the order (a bill of
  materials: one each of ``part_a`` / ``part_b`` / ``part_c``).

The scene is DETERMINISTIC — it has no RNG and takes no seed, so the committed
fixture is exact and regenerating it is a no-op diff. It lives OUTSIDE ``src/``
so the package and its CI gate stay numpy-only: ``multicam_sim`` is an optional
``bench`` dependency, never installed by ``uv sync --dev``.

Regenerate with::

    make fusion-scene
    # i.e. uv run --group bench python bench/gen_fusion_scene.py

The geometry mirrors multicam-sim ``examples/assembly_station.py`` (its shipped
preset); the manifest itself is computed entirely by multicam-sim's producer
code, so the fixture is genuine generated output, not hand-authored.
"""

from __future__ import annotations

import math
from pathlib import Path

from multicam_sim import write_manifest
from multicam_sim.dsl.rig import CameraRig, StationView
from multicam_sim.entities import Entity, EntityFrame
from multicam_sim.order import BillOfMaterials, Order, write_order_json
from multicam_sim.pose import PoseFrame, PoseTrajectory, Skeleton
from multicam_sim.scene import Scene

FPS = 30.0
NUM_FRAMES = 11

_OPERATOR_BASE = (0.0, 0.0)
_CONTAINER = (2.9, 0.0, 0.92)
_ITEM_STAGING = {
    "part_a": (2.75, -0.30, 0.90),
    "part_b": (2.90, -0.30, 0.90),
    "part_c": (3.05, -0.30, 0.90),
}
_PLACED_AT = {"part_a": 2, "part_b": 5, "part_c": 8}

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


def operator_pose() -> PoseTrajectory:
    bx, by = _OPERATOR_BASE
    frames: list[PoseFrame] = []
    for f in range(NUM_FRAMES):
        phase = math.sin(2.0 * math.pi * f / (NUM_FRAMES - 1))
        joints: dict[str, list[float]] = {}
        for name, (dx, dy, dz) in _JOINT_OFFSETS.items():
            reach = 0.06 * phase if name.endswith("wrist") else 0.0
            joints[name] = [bx + dx, by + dy + reach, dz]
        frames.append(PoseFrame(frame=f, joints=joints))
    return PoseTrajectory(id="operator", skeleton=Skeleton.coco17(), frames=frames)


def item_entity(item_id: str) -> Entity:
    staging = _ITEM_STAGING[item_id]
    placed_at = _PLACED_AT[item_id]
    frames = [
        EntityFrame(
            frame=f,
            points={"center": list(_CONTAINER if f >= placed_at else staging)},
        )
        for f in range(NUM_FRAMES)
    ]
    return Entity(id=item_id, frames=frames)


def build_scene() -> Scene:
    cameras = CameraRig.stations(
        [
            StationView(position=(0.0, 4.2, 2.4), look_at=(0.0, 0.0, 1.2), fov_deg=40.0),
            StationView(position=(2.9, -1.1, 1.7), look_at=(2.9, 0.0, 0.9), fov_deg=44.0),
        ],
        width=1280,
        height_px=720,
    )
    entities = [operator_pose().to_entity(), *(item_entity(i) for i in _ITEM_STAGING)]
    return Scene(fps=FPS, num_frames=NUM_FRAMES, cameras=cameras, entities=entities)


def build_order() -> Order:
    bom = BillOfMaterials.from_counts({item: 1 for item in _ITEM_STAGING})
    return Order(order_id="ORD-1", bom=bom)


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "sim_assembly_station"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(build_scene(), out_dir / "manifest.json")
    write_order_json(build_order(), out_dir / "order.json")
    print(f"wrote manifest.json + order.json to {out_dir}")


if __name__ == "__main__":
    main()
