"""Per-camera single-view tracklets — the atoms of cross-camera stitching.

A :class:`Tracklet` is what a *single-camera* multi-object tracker already
produces: one target's run of observations inside one camera's field of view,
tagged with the image **zone** where it entered and exited and the timestamps of
that entry/exit. The MTMC (multi-target multi-camera) layer never re-does
single-view tracking; it treats the tracklet as given and only decides which
per-camera tracklets are the *same physical object* seen at different cameras
across a blind gap.

Crucially a tracklet carries a **local** id (unique within its camera). The GT
identity that links tracklets across cameras is deliberately *not* stored here —
recovering it is the whole task, and leaking it into the matcher would make the
benchmark circular. Ground truth lives only in the evaluation path
(:mod:`multicam_occlusion.mtmc.metrics`).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

#: A named image region a target crosses through, scoped to its camera, e.g.
#: ``"cam0:right"``. Zones are opaque labels: the topology maps an exit zone on
#: one camera to an entry zone on another, so a matcher never inspects geometry.
Zone = str


class Observation(BaseModel):
    """One camera's detection of a target at one frame: a pixel and its time."""

    model_config = ConfigDict(frozen=True)

    frame: int
    timestamp: float
    uv: tuple[float, float]


class Tracklet(BaseModel):
    """A single target's contiguous run of observations within one camera.

    ``id`` is local to ``camera_id`` (the single-view tracker assigns it); it is
    never assumed to align across cameras. ``entry_zone`` / ``exit_zone`` name
    where the target crossed into and out of view, and ``entry_time`` /
    ``exit_time`` are the timestamps of the first and last observation — the two
    signals the spatio-temporal handoff reasons over.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    camera_id: int
    observations: list[Observation]
    entry_zone: Zone
    exit_zone: Zone
    entry_time: float
    exit_time: float

    @model_validator(mode="after")
    def _check(self) -> Tracklet:
        if not self.observations:
            raise ValueError(f"tracklet {self.id!r}: needs at least one observation")
        frames = [o.frame for o in self.observations]
        if frames != sorted(frames):
            raise ValueError(f"tracklet {self.id!r}: observations must be frame-ordered")
        if self.exit_time < self.entry_time:
            raise ValueError(f"tracklet {self.id!r}: exit_time precedes entry_time")
        return self

    @property
    def duration(self) -> float:
        """Wall-clock span from entry to exit, in seconds."""
        return self.exit_time - self.entry_time
