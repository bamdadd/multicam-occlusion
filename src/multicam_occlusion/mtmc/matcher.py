"""Handoff / ReID matching — spatio-temporal plausibility first, appearance last.

The contribution under test is a **spatio-temporal** handoff: an exiting
tracklet matches an entering one only if the camera topology admits the
crossing (exit-zone → edge → entry-zone) *and* the observed gap time falls in
the edge's transit window. Appearance / ReID is a **pluggable, secondary**
backend behind the :class:`AppearanceBackend` Protocol; the default contributes
nothing (returns ``1.0``), so a match on a non-overlapping rig is decided by
geometry and time alone — exactly the regime where view overlap (and hence
triangulation) is impossible.

Both the matcher and the appearance backend are Protocols, so a learned ReID
embedding or a different scoring rule drops in without touching assignment or
metrics (open-closed).
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from .topology import CameraTopology
from .tracklet import Tracklet

FloatArray = npt.NDArray[np.float64]


@runtime_checkable
class AppearanceBackend(Protocol):
    """Appearance similarity between two tracklets, in ``[0, 1]``."""

    def similarity(self, a: Tracklet, b: Tracklet) -> float: ...


class NoAppearance:
    """The default: appearance is neutral, so scoring is purely spatio-temporal."""

    def similarity(self, a: Tracklet, b: Tracklet) -> float:
        return 1.0


class StubEmbedding:
    """A deterministic, GPU-free stand-in for a real ReID embedding.

    Hashes each tracklet's id to a fixed unit vector and returns rescaled cosine
    similarity in ``[0, 1]``. It carries no real appearance signal — it exists to
    prove the seam: a genuine embedding backend swaps in here with no change to
    the matcher, assignment, or metrics.
    """

    def __init__(self, dim: int = 16) -> None:
        if dim < 2:
            raise ValueError("embedding dim must be >= 2")
        self._dim = dim

    def _embed(self, tracklet: Tracklet) -> FloatArray:
        digest = hashlib.sha256(tracklet.id.encode("utf-8")).digest()
        # Tile the 32-byte digest up to `dim` values, centre, and normalise.
        raw = np.frombuffer((digest * (self._dim // 32 + 1))[: self._dim], dtype=np.uint8)
        vec = raw.astype(np.float64) - 127.5
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0.0 else vec

    def similarity(self, a: Tracklet, b: Tracklet) -> float:
        cosine = float(np.dot(self._embed(a), self._embed(b)))
        return (cosine + 1.0) / 2.0


@runtime_checkable
class HandoffMatcher(Protocol):
    """Scores an exiting→entering tracklet handoff, or vetoes it with ``None``."""

    def score(
        self, exiting: Tracklet, entering: Tracklet, topology: CameraTopology
    ) -> float | None: ...


class SpatioTemporalMatcher:
    """Default handoff matcher: topology + transit window gate, then appearance.

    ``score`` returns ``None`` (a hard veto) unless *all* of the following hold:

    1. the two tracklets are on different cameras;
    2. the topology has an edge from ``exiting``'s camera/exit-zone to
       ``entering``'s camera/entry-zone;
    3. the gap ``dt = entering.entry_time - exiting.exit_time`` is positive
       (effect follows cause) and inside the edge's transit window.

    When admitted, the score is ``temporal ** (1 - w) * appearance ** w`` with
    ``w = appearance_weight`` — a geometric blend so the default
    (``appearance = 1.0``) collapses to the pure temporal plausibility.
    """

    def __init__(
        self,
        appearance: AppearanceBackend | None = None,
        *,
        k_sigma: float = 3.0,
        appearance_weight: float = 0.0,
    ) -> None:
        if not 0.0 <= appearance_weight <= 1.0:
            raise ValueError("appearance_weight must be in [0, 1]")
        self._appearance: AppearanceBackend = NoAppearance() if appearance is None else appearance
        self._k_sigma = k_sigma
        self._w = appearance_weight

    def score(
        self, exiting: Tracklet, entering: Tracklet, topology: CameraTopology
    ) -> float | None:
        if exiting.camera_id == entering.camera_id:
            return None
        edge = topology.edge_for(
            exiting.camera_id, exiting.exit_zone, entering.camera_id, entering.entry_zone
        )
        if edge is None:
            return None
        dt = entering.entry_time - exiting.exit_time
        if dt <= 0.0:
            return None
        lo, hi = edge.transit.window(self._k_sigma)
        if not (lo <= dt <= hi):
            return None
        temporal = edge.transit.plausibility(dt)
        if self._w == 0.0:
            return temporal
        appearance = self._appearance.similarity(exiting, entering)
        return float(temporal ** (1.0 - self._w) * appearance**self._w)
