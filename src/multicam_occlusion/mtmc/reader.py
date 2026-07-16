"""Typed reader for the multicam-sim manifest — the MTMC pipeline's front door.

``multicam-sim``'s ``build_manifest`` emits a JSON contract: calibrated cameras,
entities whose per-frame named points carry a ``per_cam`` observation list, and an
optional MTMC ``topology`` block (stations + directed transit edges). This module
parses that JSON into a validated :class:`SimManifest` so the rest of the pipeline
reads *typed attributes*, not raw dict lookups, and a malformed manifest fails at the
boundary with a pydantic error rather than deep inside extraction.

It also derives the matcher's :class:`~multicam_occlusion.mtmc.topology.CameraTopology`
from the manifest's own station adjacency + transit times
(:func:`matcher_topology_from_manifest`), so the handoff runs on real sim topology —
not a hand-built graph. Only the *entry/exit zones* come from the extracted tracklets
(which image border a run used); those are geometry, never ground-truth identity, so
the derivation stays non-circular.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from .topology import CameraTopology, TransitDistribution, TransitionEdge
from .tracklet import Tracklet

#: Design constant: the transit-time spread (seconds). The mean comes from the
#: manifest's ``transit_time_s`` (a property of the scene); the std is the matcher's
#: tolerance, deliberately NOT fitted to the observed tracklet gaps (that would be
#: circular). Mirrors the value the hand-authored fixture used.
DEFAULT_TRANSIT_STD = 0.15


class PerCamObservation(BaseModel):
    """One camera's view of a point at one frame, as emitted by multicam-sim."""

    cam: int
    uv: tuple[float, float]
    visible: bool
    in_view: bool | None = None
    occ_frac: float | None = None


class PointRecord(BaseModel):
    """A named point: its ground-truth world coordinate and per-camera views."""

    xyz_gt: list[float]
    per_cam: list[PerCamObservation]


class FrameRecord(BaseModel):
    """One entity frame: a mapping of point name -> :class:`PointRecord`."""

    frame: int
    points: dict[str, PointRecord]


class EntityRecord(BaseModel):
    """A tracked entity: a stable id and its per-frame points. ``id`` is the
    cross-camera ground-truth identity — read only by the metrics layer."""

    id: str
    frames: list[FrameRecord]
    edges: list[tuple[str, str]] | None = None


class CameraRecord(BaseModel):
    """A calibrated camera record (only ``id``/``width``/``height`` drive MTMC)."""

    id: int
    K: list[list[float]]
    R: list[list[float]]
    t: list[float]
    width: int
    height: int
    convention: str | None = None


class StationRecord(BaseModel):
    """A named station holding one or more cameras (multicam-sim topology)."""

    id: str
    camera_ids: list[int]


class TransitEdgeRecord(BaseModel):
    """A directed station adjacency with the transit time across the blind gap."""

    src: str
    dst: str
    transit_time_s: float


class TopologyRecord(BaseModel):
    """The manifest's optional MTMC block: stations + directed transit edges."""

    stations: list[StationRecord]
    edges: list[TransitEdgeRecord] = []


class SimManifest(BaseModel):
    """A validated multicam-sim manifest — the typed input to the MTMC pipeline."""

    cameras: list[CameraRecord]
    fps: float
    num_frames: int
    entities: list[EntityRecord]
    topology: TopologyRecord | None = None


def read_manifest(path: str | Path) -> SimManifest:
    """Load and validate a multicam-sim manifest JSON file into a typed model."""
    return SimManifest.model_validate(json.loads(Path(path).read_text()))


def matcher_topology_from_manifest(
    manifest: SimManifest,
    tracklets: Sequence[Tracklet],
    *,
    transit_std: float = DEFAULT_TRANSIT_STD,
) -> CameraTopology:
    """Build the matcher :class:`CameraTopology` from the manifest + tracklet zones.

    Adjacency and transit time come from the manifest's own ``topology`` block: each
    directed station edge ``A -> B`` becomes matcher edges between A's camera and B's
    camera, with the station transit as the Gaussian mean (std is a design constant).
    The exit/entry *zones* on those edges are the image borders the extracted
    tracklets actually used on each camera — a geometric label, never a GT identity,
    so this is not circular.

    This builder is for the non-overlapping regime: it requires exactly one camera per
    station (the MTMC handoff has no in-station overlap to fuse).
    """
    if manifest.topology is None:
        raise ValueError("manifest has no topology block; cannot build a handoff graph")

    station_camera: dict[str, int] = {}
    for station in manifest.topology.stations:
        if len(station.camera_ids) != 1:
            raise ValueError(
                f"station {station.id!r} has {len(station.camera_ids)} cameras; the "
                "non-overlapping handoff builder needs exactly one camera per station"
            )
        station_camera[station.id] = station.camera_ids[0]

    exit_zones: dict[int, set[str]] = defaultdict(set)
    entry_zones: dict[int, set[str]] = defaultdict(set)
    for tracklet in tracklets:
        exit_zones[tracklet.camera_id].add(tracklet.exit_zone)
        entry_zones[tracklet.camera_id].add(tracklet.entry_zone)

    edges: list[TransitionEdge] = []
    for edge in manifest.topology.edges:
        src_camera = station_camera[edge.src]
        dst_camera = station_camera[edge.dst]
        transit = TransitDistribution(mean=edge.transit_time_s, std=transit_std)
        for exit_zone in sorted(exit_zones.get(src_camera, set())):
            for entry_zone in sorted(entry_zones.get(dst_camera, set())):
                edges.append(
                    TransitionEdge(
                        src_camera=src_camera,
                        dst_camera=dst_camera,
                        exit_zone=exit_zone,
                        entry_zone=entry_zone,
                        transit=transit,
                    )
                )

    return CameraTopology(cameras=[cam.id for cam in manifest.cameras], edges=edges)
