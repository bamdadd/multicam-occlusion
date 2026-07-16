"""Complementary / asymmetric multi-view fusion.

A camera-relationship mode for rigs where two cameras see *different aspects of
the same event* rather than the same point from different angles. The motivating
case is a packing station: a north camera sees the **worker** (the action); an
east camera near the worktop sees the **items** (the assembly state). You cannot
triangulate a human action against an object — so instead of geometric recovery
you **fuse** the two partial views in time, correlating each action with the
assembly change it caused (*who did what to which item, when*).

Pipeline (each stage pluggable via a Protocol, defaults deterministic):

    SceneManifest  ──partition_by_visibility──▶  actors / items   (asymmetric)
        │                                              │
        │ ReachActionDetector          DisplacementStateDetector
        ▼                                              ▼
    HumanViewObservation                       WorktopViewObservation
        └───────────────  FusionEstimator  ────────────┘
                                │
                        JointSceneState  ──association_metric──▶ precision/recall
"""

from multicam_occlusion.fusion.detectors import (
    ActionDetector,
    CameraRoles,
    DisplacementStateDetector,
    ReachActionDetector,
    ScenePartition,
    WorktopStateDetector,
    partition_by_visibility,
)
from multicam_occlusion.fusion.estimator import (
    FusionEstimator,
    TemporalProximityEstimator,
    fuse_scene,
)
from multicam_occlusion.fusion.metrics import AssociationMetrics, association_metric
from multicam_occlusion.fusion.observations import (
    ActionEvent,
    AssemblyChangeEvent,
    FusedInteraction,
    GroundTruthInteraction,
    HumanViewObservation,
    JointSceneState,
    WorktopViewObservation,
)
from multicam_occlusion.fusion.scene_manifest import (
    CamObservation,
    ManifestEntity,
    ManifestFrame,
    PointObservation,
    SceneManifest,
)

__all__ = [
    "ActionDetector",
    "ActionEvent",
    "AssemblyChangeEvent",
    "AssociationMetrics",
    "CamObservation",
    "CameraRoles",
    "DisplacementStateDetector",
    "FusedInteraction",
    "FusionEstimator",
    "GroundTruthInteraction",
    "HumanViewObservation",
    "JointSceneState",
    "ManifestEntity",
    "ManifestFrame",
    "PointObservation",
    "ReachActionDetector",
    "SceneManifest",
    "ScenePartition",
    "TemporalProximityEstimator",
    "WorktopStateDetector",
    "WorktopViewObservation",
    "association_metric",
    "fuse_scene",
    "partition_by_visibility",
]
