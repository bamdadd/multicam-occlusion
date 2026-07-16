"""Complementary / asymmetric multi-view fusion.

A camera-relationship mode for rigs where two cameras see *different aspects of
the same event* rather than the same point from different angles. The motivating
case is an assembly / order-fulfilment station: an overview camera sees the
**operator** (the activity — a reach, a place); a second camera at the station
sees the **items** (the container filling, a part's assembly state changing). You
cannot triangulate an operator action against a part — so instead of geometric
recovery you **fuse** the two partial views in time, correlating each action with
the assembly change it caused (*who did what to which item, when*), and read the
result against the order (**order verification**: fulfilled / missing / wrong /
extra per item).

Pipeline (each stage pluggable via a Protocol, defaults deterministic):

    SceneManifest  ──partition_by_visibility──▶  actors / items   (asymmetric)
        │                                              │
        │ ReachActionDetector            DisplacementStateDetector
        ▼                                              ▼
    OperatorViewObservation                     ItemViewObservation
        └───────────────  FusionEstimator  ────────────┘
                                │
                        JointSceneState  ──association_metric──▶ precision/recall
                                │
                          verify_order  ──▶  fulfilled / missing / wrong / extra
"""

from multicam_occlusion.fusion.detectors import (
    ActionDetector,
    CameraRoles,
    DisplacementStateDetector,
    ItemStateDetector,
    ReachActionDetector,
    ScenePartition,
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
    ItemViewObservation,
    JointSceneState,
    OperatorViewObservation,
)
from multicam_occlusion.fusion.scene_manifest import (
    CamObservation,
    ManifestEntity,
    ManifestFrame,
    PointObservation,
    SceneManifest,
)
from multicam_occlusion.fusion.verification import (
    ItemVerification,
    OrderStatus,
    OrderVerification,
    verify_order,
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
    "ItemStateDetector",
    "ItemVerification",
    "ItemViewObservation",
    "JointSceneState",
    "ManifestEntity",
    "ManifestFrame",
    "OperatorViewObservation",
    "OrderStatus",
    "OrderVerification",
    "PointObservation",
    "ReachActionDetector",
    "SceneManifest",
    "ScenePartition",
    "TemporalProximityEstimator",
    "association_metric",
    "fuse_scene",
    "partition_by_visibility",
    "verify_order",
]
