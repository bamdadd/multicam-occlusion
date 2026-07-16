"""Extract per-camera tracklets from a validated multicam-sim manifest.

Consumes the typed :class:`~multicam_occlusion.mtmc.reader.SimManifest` parsed from
the schema ``multicam-sim`` emits (``build_manifest``): cameras with ``width`` /
``height``, and entities whose per-frame points carry a ``per_cam`` list of
observations. For a chosen point name it walks each entity × camera and splits the
``visible`` observations into **contiguous runs** — each run is one tracklet (a target
leaving and re-entering a camera yields two tracklets there).

Two deliberate outputs:

* the tracklets, each with a **local** id (``"c{cam}:e{entity}:r{run}"``);
* a separate ``tracklet id → GT identity`` map (the entity id), used **only** by
  the metrics layer.

The extractor here is intentionally minimal (a real single-view tracker and a
learned zone model are out of scope — see the tracklet-extractor issue). Entry /
exit zones come from which image border the first / last observation sits
nearest; timestamps are ``frame / fps`` because this manifest has no per-
observation timestamp.
"""

from __future__ import annotations

from .reader import CameraRecord, SimManifest
from .tracklet import Observation, Tracklet, Zone


def classify_zone(camera_id: int, uv: tuple[float, float], width: float, height: float) -> Zone:
    """Name the image border nearest to ``uv`` as ``"cam{id}:{side}"``.

    Sides are ``left`` / ``right`` / ``top`` / ``bottom``; the nearest border by
    pixel distance wins, ties resolved in that fixed order. This is a stand-in
    for a learned entry/exit-zone model; the topology only needs stable labels.
    """
    u, v = uv
    distances = [("left", u), ("right", width - u), ("top", v), ("bottom", height - v)]
    side = min(distances, key=lambda d: d[1])[0]
    return f"cam{camera_id}:{side}"


def _camera_dims(cameras: list[CameraRecord]) -> dict[int, tuple[float, float]]:
    return {cam.id: (float(cam.width), float(cam.height)) for cam in cameras}


def extract_tracklets(
    manifest: SimManifest,
    point_name: str = "center",
) -> tuple[list[Tracklet], dict[str, str]]:
    """Return ``(tracklets, gt_identity_of)`` for one point across all entities.

    ``tracklets`` have local, camera-scoped ids; ``gt_identity_of`` maps each
    tracklet id to its source entity id (the ground-truth identity across the
    blind gap). Cameras are visited in manifest order for determinism.
    """
    fps = manifest.fps
    dims = _camera_dims(manifest.cameras)
    camera_ids = [cam.id for cam in manifest.cameras]

    tracklets: list[Tracklet] = []
    gt_identity_of: dict[str, str] = {}

    for entity in manifest.entities:
        entity_id = entity.id
        # frame -> {cam -> uv} for visible observations of the chosen point.
        per_frame: dict[int, dict[int, tuple[float, float]]] = {}
        for frame_entry in entity.frames:
            frame = frame_entry.frame
            point = frame_entry.points.get(point_name)
            if point is None:
                continue
            for obs in point.per_cam:
                if not obs.visible:
                    continue
                per_frame.setdefault(frame, {})[obs.cam] = (obs.uv[0], obs.uv[1])

        for cam in camera_ids:
            width, height = dims[cam]
            # Collect this camera's visible (frame, uv) for the entity, in order,
            # then split into contiguous-frame runs (one tracklet per run).
            seq = sorted((f, cols[cam]) for f, cols in per_frame.items() if cam in cols)
            runs: list[list[tuple[int, tuple[float, float]]]] = []
            for frame, uv in seq:
                if runs and frame == runs[-1][-1][0] + 1:
                    runs[-1].append((frame, uv))
                else:
                    runs.append([(frame, uv)])

            for run_index, run in enumerate(runs):
                tid = f"c{cam}:e{entity_id}:r{run_index}"
                tracklets.append(
                    Tracklet(
                        id=tid,
                        camera_id=cam,
                        observations=[
                            Observation(frame=f, timestamp=f / fps, uv=uv) for f, uv in run
                        ],
                        entry_zone=classify_zone(cam, run[0][1], width, height),
                        exit_zone=classify_zone(cam, run[-1][1], width, height),
                        entry_time=run[0][0] / fps,
                        exit_time=run[-1][0] / fps,
                    )
                )
                gt_identity_of[tid] = entity_id

    return tracklets, gt_identity_of
