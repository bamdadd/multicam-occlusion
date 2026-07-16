"""Camera topology — the directed graph of *possible* cross-camera transitions.

Nodes are cameras; a directed edge says "a target leaving camera ``src`` through
``exit_zone`` may reappear at camera ``dst`` through ``entry_zone`` after roughly
this transit time". The topology encodes the physical layout (which station
feeds which, how long the blind gap between them takes) once, as data; the
handoff matcher reads it and never hard-codes geometry.

The transit time is a small distribution (Gaussian by mean/std), not a point
value, so the matcher can (a) hard-gate on a plausibility window and (b) score
inside it. Keeping this a typed, serialisable model means a richer transit model
(log-normal, learned histogram) is an open-closed swap, not a rewrite.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, model_validator

from .tracklet import Zone


class TransitDistribution(BaseModel):
    """Expected transit time across a blind gap, as a Gaussian (seconds).

    ``plausibility`` is the Gaussian evaluated so its peak is ``1.0`` at the
    mean — a unitless score, not a density — and ``window`` is the hard
    ``mean ± k*std`` gate outside which a transit is rejected outright.
    """

    model_config = ConfigDict(frozen=True)

    mean: float
    std: float

    @model_validator(mode="after")
    def _check(self) -> TransitDistribution:
        if self.mean < 0.0:
            raise ValueError("transit mean must be >= 0")
        if self.std <= 0.0:
            raise ValueError("transit std must be > 0")
        return self

    def window(self, k_sigma: float = 3.0) -> tuple[float, float]:
        """The ``[mean - k*std, mean + k*std]`` acceptance interval."""
        half = k_sigma * self.std
        return (self.mean - half, self.mean + half)

    def plausibility(self, dt: float) -> float:
        """Peak-normalised Gaussian score in ``(0, 1]`` for an observed ``dt``."""
        z = (dt - self.mean) / self.std
        return math.exp(-0.5 * z * z)


class TransitionEdge(BaseModel):
    """A directed handoff channel: ``src`` exit-zone → ``dst`` entry-zone."""

    model_config = ConfigDict(frozen=True)

    src_camera: int
    dst_camera: int
    exit_zone: Zone
    entry_zone: Zone
    transit: TransitDistribution

    @model_validator(mode="after")
    def _check(self) -> TransitionEdge:
        if self.src_camera == self.dst_camera:
            raise ValueError("a transition edge must connect two distinct cameras")
        return self


class CameraTopology(BaseModel):
    """The full transition graph: camera nodes + directed handoff edges."""

    model_config = ConfigDict(frozen=True)

    cameras: list[int]
    edges: list[TransitionEdge]

    @model_validator(mode="after")
    def _check(self) -> CameraTopology:
        known = set(self.cameras)
        if len(known) != len(self.cameras):
            raise ValueError("duplicate camera id in topology")
        for edge in self.edges:
            missing = {edge.src_camera, edge.dst_camera} - known
            if missing:
                raise ValueError(f"edge references unknown camera id(s): {sorted(missing)}")
        return self

    def edge_for(
        self, src_camera: int, exit_zone: Zone, dst_camera: int, entry_zone: Zone
    ) -> TransitionEdge | None:
        """The edge matching an exit→entry crossing, or ``None`` if none exists.

        Deterministic: returns the first edge in declaration order whose four
        fields all match. A topology with duplicate (src, exit, dst, entry)
        edges is a modelling error, not something this resolves by scoring.
        """
        for edge in self.edges:
            if (
                edge.src_camera == src_camera
                and edge.dst_camera == dst_camera
                and edge.exit_zone == exit_zone
                and edge.entry_zone == entry_zone
            ):
                return edge
        return None
