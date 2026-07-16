"""Generate the MTMC handoff fixture from REAL multicam-sim output.

Regenerate with::

    make mtmc-scene          # or: uv run --group bench python bench/gen_mtmc_scene.py

Writes ``tests/fixtures/mtmc_stations.json`` — a manifest produced by
``multicam_sim.manifest.build_manifest`` on a genuinely non-overlapping two-station
rig (``multicam_sim.dsl.CameraRig.stations``, disjoint fields of view). Because the
committed fixture is the recorded output, the numpy-only test suite consumes real
sim geometry without importing ``multicam-sim`` in CI (it is an optional ``bench``
dependency).

Seed: ``SEED = 0`` (nominal). The scene uses no RNG — projection is deterministic and
the committed JSON is byte-reproducible — so the seed is recorded for provenance only.

Scene (deterministic; no RNG — pure projection, byte-reproducible):

* **station 1** (camera 0) and **station 2** (camera 1) stand back along -y watching
  a shared track line, but aim at ``x = -6`` and ``x = +6`` with a 55° FOV, so their
  view cones are disjoint: an entity near one station is far off the other's axis and
  projects out of frame. The stretch around ``x = 0`` is a genuine blind gap seen by
  neither camera.
* **two entities** sweep left→right along the track at the SAME constant velocity, with
  ``item_b`` released ``STAGGER_FRAMES`` frames after ``item_a``. Equal velocity ⇒ equal
  true transit time across the gap; the stagger makes the correct pairing (a→a, b→b) the
  only one whose exit/entry gap sits at the transit-time peak, so the spatio-temporal
  handoff must disambiguate on time alone (the default appearance backend is neutral).

The stable ``entity.id`` is the cross-camera ground-truth identity a tracker must hold
across the blind gap; it enters only the metrics layer, never the matcher.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from multicam_sim.dsl import CameraRig, StationView
from multicam_sim.dsl import Path as SimPath
from multicam_sim.entities import Entity
from multicam_sim.manifest import build_manifest
from multicam_sim.scene import Scene
from multicam_sim.topology import CameraTopology, Station, TransitEdge

SEED = 0  # nominal: the scene is deterministic (no RNG); recorded for provenance.

FPS = 10.0
NUM_FRAMES = 12
WIDTH = 640
HEIGHT_PX = 480
FOV_DEG = 60.0

# Cameras stand back along -y and aim at their own station centre on the track line.
CAM_Y = -5.0
CAM_Z = 1.5
TRACK_Z = 1.0
STATION_A_X = -6.0
STATION_B_X = 6.0

# Left→right sweep shared by both entities (constant velocity ⇒ equal gap transit).
# Tuned so item_a is visible in station 1 for frames 0-2 and re-appears in station 2
# from frame 7: a 5-frame (0.5 s) blind-gap crossing, the scene's designed transit.
START_X = -6.0
VELOCITY_X = 1.3  # world units per frame
STAGGER_FRAMES = 2  # item_b released this many frames after item_a
TRANSIT_S = 5.0 / FPS  # designed gap-crossing time (exit station 1 -> enter station 2)

STATION_A = "station-1"
STATION_B = "station-2"


def _cameras() -> Any:
    """Two separated stations, one camera each, with disjoint fields of view."""
    views = [
        StationView(position=(STATION_A_X, CAM_Y, CAM_Z), look_at=(STATION_A_X, 0.0, TRACK_Z)),
        StationView(position=(STATION_B_X, CAM_Y, CAM_Z), look_at=(STATION_B_X, 0.0, TRACK_Z)),
    ]
    return CameraRig.stations(views, width=WIDTH, height_px=HEIGHT_PX, fov_deg=FOV_DEG)


def _sweep_entity(entity_id: str, delay_frames: int) -> Entity:
    """One entity sweeping +x at constant velocity, released ``delay_frames`` late.

    A single straight ``Path`` timed so frame ``f`` sits at ``START_X + V*(f-delay)``:
    both entities share ``V`` (equal true transit), only the release differs.
    """
    total_s = (NUM_FRAMES - 1) / FPS
    x_at = lambda f: START_X + VELOCITY_X * (f - delay_frames)  # noqa: E731
    a = (x_at(0), 0.0, TRACK_Z)
    b = (x_at(NUM_FRAMES - 1), 0.0, TRACK_Z)
    frames = SimPath.linear(a, b).over(total_s).compile_frames(FPS, NUM_FRAMES, name="center")
    return Entity(id=entity_id, frames=frames)


def build_scene() -> Scene:
    cameras = _cameras()
    entities = [
        _sweep_entity("item_a", delay_frames=0),
        _sweep_entity("item_b", delay_frames=STAGGER_FRAMES),
    ]
    # Station adjacency + transit time (the blind-gap crossing) as real sim topology.
    topology = CameraTopology(
        stations=[
            Station(id=STATION_A, camera_ids=[0]),
            Station(id=STATION_B, camera_ids=[1]),
        ],
        edges=[
            TransitEdge(src=STATION_A, dst=STATION_B, transit_time_s=TRANSIT_S),
            TransitEdge(src=STATION_B, dst=STATION_A, transit_time_s=TRANSIT_S),
        ],
    )
    return Scene(
        fps=FPS,
        num_frames=NUM_FRAMES,
        cameras=cameras,
        entities=entities,
        occluders=[],
        topology=topology,
    )


def _visible_frames(manifest: dict[str, Any]) -> dict[str, dict[int, list[int]]]:
    """entity -> camera -> sorted visible frame indices (diagnostic only)."""
    report: dict[str, dict[int, list[int]]] = {}
    for entity in manifest["entities"]:
        per_cam: dict[int, list[int]] = {}
        for frame_entry in entity["frames"]:
            f = int(frame_entry["frame"])
            for obs in frame_entry["points"]["center"]["per_cam"]:
                if obs["visible"]:
                    per_cam.setdefault(int(obs["cam"]), []).append(f)
        report[str(entity["id"])] = {c: sorted(v) for c, v in sorted(per_cam.items())}
    return report


def main() -> None:
    manifest = build_manifest(build_scene())
    out = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "mtmc_stations.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out}")
    for entity, per_cam in _visible_frames(manifest).items():
        print(f"  {entity}: {per_cam}")


if __name__ == "__main__":
    main()
