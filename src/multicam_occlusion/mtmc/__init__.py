"""Multi-target multi-camera (MTMC) handoff across non-overlapping cameras.

This is the *camera-relationship* path, distinct from the overlap/triangulation
core. When cameras do **not** share a field of view (station cameras along a
route — an entity leaves station 1's camera, crosses an unobserved blind gap,
and reappears at station 2's camera), 3D triangulation is impossible — there is
no common ray to intersect. The problem instead is **identity handoff**: keep
one global id on an entity that exits one camera, crosses the gap, and reappears
at the next.

Pipeline (all typed, deterministic, GPU-free, open-closed):

1. :mod:`~multicam_occlusion.mtmc.extract` — per-camera :class:`Tracklet` s from
   a multicam-sim manifest (single-view tracking assumed solved).
2. :class:`~multicam_occlusion.mtmc.topology.CameraTopology` — the directed
   graph of possible transitions with transit-time distributions.
3. :class:`~multicam_occlusion.mtmc.matcher.SpatioTemporalMatcher` — score a
   handoff by topology + transit window first; appearance/ReID is a pluggable
   :class:`~multicam_occlusion.mtmc.matcher.AppearanceBackend`.
4. :func:`~multicam_occlusion.mtmc.assignment.assign_global_ids` — stitch
   matched tracklets into consistent global identities.
5. :func:`~multicam_occlusion.mtmc.metrics.compute_identity_metrics` — IDF1 and
   ID-switches against GT identities (identity consistency, not 3D error).
"""

from multicam_occlusion.mtmc.assignment import (
    assign_global_ids,
    per_tracklet_global_ids,
)
from multicam_occlusion.mtmc.extract import (
    classify_zone,
    extract_tracklets,
    load_manifest,
)
from multicam_occlusion.mtmc.matcher import (
    AppearanceBackend,
    HandoffMatcher,
    NoAppearance,
    SpatioTemporalMatcher,
    StubEmbedding,
)
from multicam_occlusion.mtmc.metrics import (
    IdentityMetrics,
    compute_identity_metrics,
    idf1_from_counts,
)
from multicam_occlusion.mtmc.topology import (
    CameraTopology,
    TransitDistribution,
    TransitionEdge,
)
from multicam_occlusion.mtmc.tracklet import Observation, Tracklet, Zone

__all__ = [
    "AppearanceBackend",
    "CameraTopology",
    "HandoffMatcher",
    "IdentityMetrics",
    "NoAppearance",
    "Observation",
    "SpatioTemporalMatcher",
    "StubEmbedding",
    "Tracklet",
    "TransitDistribution",
    "TransitionEdge",
    "Zone",
    "assign_global_ids",
    "classify_zone",
    "compute_identity_metrics",
    "extract_tracklets",
    "idf1_from_counts",
    "load_manifest",
    "per_tracklet_global_ids",
]
