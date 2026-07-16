"""Generate the packing-station fixture manifest (multicam-sim scene schema).

Regenerate with::

    python tests/fixtures/build_packing_station.py

The scene is a deliberately small, asymmetric two-camera rig:

* **camera 0** — north, wide: sees the ``worker`` entity (never the items).
* **camera 1** — east, at the worktop: sees the ``item_*`` entities (never the
  worker).

Every point carries a ``per_cam`` entry for *both* cameras (visible in one,
``visible=false`` in the other) — exactly as ``multicam-sim``'s ``build_manifest``
emits — so the asymmetric-visibility router is genuinely exercised.

The worker's hand dips to the worktop three times (a "place"); two of those
dips cause an item to move, the third is a **distractor** (a reach with no
resulting change). A third item moves late, *outside* the causal lag window — a
change with no cause. Ground truth is therefore two real interactions, and a
faithful estimator must refuse both distractors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FPS = 10.0
NUM_FRAMES = 21

# Frames at which the worker's hand dips to the worktop (a "place" action).
PLACE_FRAMES = (5, 9, 13)  # 9 is the distractor: a reach with no item change.


def _camera(cam_id: int, tx: float) -> dict[str, Any]:
    """A minimally-shaped camera record (fusion ignores calibration values)."""
    return {
        "id": cam_id,
        "K": [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
        "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "t": [tx, 0.0, 5.0],
        "width": 640,
        "height": 480,
        "convention": "opencv_rdf",
    }


def _obs(cam: int, xyz: tuple[float, float, float], visible: bool) -> dict[str, Any]:
    """A per-camera observation; uv is a plausible placeholder (unused by fusion)."""
    u = round(xyz[0] * 100.0 + 320.0, 3)
    v = round(xyz[2] * 100.0 + 240.0, 3)
    return {"cam": cam, "uv": [u, v], "visible": visible, "occ_frac": 0.0}


def _point(xyz: tuple[float, float, float], human_view: bool) -> dict[str, Any]:
    """A point visible in exactly one camera (cam 0 for the worker, cam 1 items)."""
    return {
        "xyz_gt": [xyz[0], xyz[1], xyz[2]],
        "per_cam": [
            _obs(0, xyz, visible=human_view),
            _obs(1, xyz, visible=not human_view),
        ],
    }


def _worker_hand_z() -> list[float]:
    """Hand height per frame: flat at 1.2, dipping to 0.7 at each place frame."""
    z = [1.2] * NUM_FRAMES
    for d in PLACE_FRAMES:
        z[d - 1] = 1.0
        z[d] = 0.7  # strict local minimum, below the 0.95 midline
        z[d + 1] = 1.0
    return z


def _worker_entity() -> dict[str, Any]:
    z = _worker_hand_z()
    frames = [
        {"frame": f, "points": {"hand": _point((0.0, 0.0, z[f]), human_view=True)}}
        for f in range(NUM_FRAMES)
    ]
    return {"id": "worker", "frames": frames}


def _item_entity(
    item_id: str, staging: tuple[float, float, float], move_frame: int, dy: float
) -> dict[str, Any]:
    """An item stationary at ``staging`` until ``move_frame``, then shifted by dy."""
    moved = (staging[0], staging[1] + dy, staging[2])
    frames = [
        {
            "frame": f,
            "points": {"center": _point(staging if f < move_frame else moved, human_view=False)},
        }
        for f in range(NUM_FRAMES)
    ]
    return {"id": item_id, "frames": frames}


def build() -> dict[str, Any]:
    return {
        "fps": FPS,
        "num_frames": NUM_FRAMES,
        "cameras": [_camera(0, -1.0), _camera(1, 1.0)],
        "entities": [
            _worker_entity(),
            # Placed just after the worker's dips at frames 5 and 13 (lag 0.1s).
            _item_entity("item_0", (0.5, 0.0, 0.8), move_frame=6, dy=0.3),
            _item_entity("item_1", (-0.5, 0.0, 0.8), move_frame=14, dy=0.3),
            # Moves at frame 19 (t=1.9): no action within the causal window.
            _item_entity("item_2", (0.0, -0.5, 0.8), move_frame=19, dy=-0.3),
        ],
    }


def main() -> None:
    out = Path(__file__).with_name("packing_station.json")
    out.write_text(json.dumps(build(), indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
