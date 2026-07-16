"""Typed reader for the multicam-sim *scene* manifest (the entities schema).

This is a **different** on-disk schema from :class:`multicam_occlusion.sources.
Manifest` (which describes frame/video *streams*). ``multicam-sim``'s
``build_manifest`` emits a scene record shaped like::

    {"cameras": [...], "fps": 30.0, "num_frames": 11, "entities": [...]}

where every entity carries, per frame, per named point, a per-camera observation
with a hard ``visible`` boolean and a continuous ``occ_frac``. That ``visible``
label — computed geometrically upstream — is the contract this module reads: it
is what decides, in an *asymmetric* multi-view rig, which camera contributes
which entity.

The reader is dependency-light: it mirrors the JSON as pydantic models (no
``multicam-sim`` import — the producer emits plain-dict JSON, nothing
importable) and exposes a ``frame -> seconds`` helper so downstream fusion works
in real time, not frame indices. Unknown fields are ignored so the reader keeps
working as the producer's schema grows.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

# Camera ids in the multicam-sim manifest are integers (``Camera.id: int``).
CamId = int


class CamObservation(BaseModel):
    """One camera's view of one world point at one frame.

    ``visible`` is the hard boolean the upstream simulator computes
    geometrically (in front of the camera, inside the image, sightline
    unblocked). It is the asymmetric-visibility signal fusion routes on.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    cam: CamId
    uv: tuple[float, float]
    visible: bool
    occ_frac: float = 0.0


class PointObservation(BaseModel):
    """A named point's ground-truth world coordinate plus its per-camera views."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    xyz_gt: tuple[float, float, float]
    per_cam: list[CamObservation]

    def camera(self, cam: CamId) -> CamObservation | None:
        """Return this point's observation from camera ``cam`` (or ``None``)."""
        for obs in self.per_cam:
            if obs.cam == cam:
                return obs
        return None

    def visible_in(self, cam: CamId) -> bool:
        """Whether the point is visible in camera ``cam`` (missing -> ``False``)."""
        obs = self.camera(cam)
        return obs is not None and obs.visible


class ManifestFrame(BaseModel):
    """One frame of an entity: a mapping of point name -> observation."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    frame: int
    points: dict[str, PointObservation]


class ManifestEntity(BaseModel):
    """A tracked thing: named points across frames, optional skeleton ``edges``."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    edges: list[tuple[str, str]] | None = None
    frames: list[ManifestFrame]

    def point_names(self) -> list[str]:
        """Every point name that appears in any frame, in first-seen order."""
        seen: list[str] = []
        for f in self.frames:
            for name in f.points:
                if name not in seen:
                    seen.append(name)
        return seen

    def visible_frame_count(self, cam: CamId) -> int:
        """How many frames have *any* point of this entity visible in ``cam``."""
        return sum(1 for f in self.frames if any(p.visible_in(cam) for p in f.points.values()))


class SceneManifest(BaseModel):
    """The multicam-sim scene manifest: cameras, timing, and entities.

    Use :meth:`time_of` to convert a frame index to seconds (``frame / fps``);
    fusion correlates events in *time*, so every event downstream carries a
    real timestamp, not a frame number.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    fps: float
    num_frames: int
    cameras: list[dict[str, object]]
    entities: list[ManifestEntity]

    def time_of(self, frame: int) -> float:
        """Convert a frame index to seconds using the scene ``fps``."""
        if self.fps <= 0.0:
            raise ValueError(f"fps must be positive; got {self.fps}")
        return frame / self.fps

    def entity(self, entity_id: str) -> ManifestEntity:
        """Look up an entity by id (raises ``KeyError`` if absent)."""
        for e in self.entities:
            if e.id == entity_id:
                return e
        raise KeyError(entity_id)

    @classmethod
    def from_json(cls, path: str | Path) -> SceneManifest:
        """Load and validate a scene manifest from a JSON file."""
        return cls.model_validate(json.loads(Path(path).read_text()))
