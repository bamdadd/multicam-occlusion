"""Multi-camera synthetic occlusion benchmark.

Core thesis: with multiple calibrated cameras a 3D point can be recovered even
when it is occluded in some views; a single view is fundamentally insufficient
(the observation constrains the point only to a ray).
"""

from multicam_occlusion.metrics import mpjpe
from multicam_occlusion.observation import (
    ObsCamera,
    ObsEntity,
    ObservationManifest,
    ObsFrame,
    PerCam,
    PointObs,
)
from multicam_occlusion.occlusion import drop_k_mask, occlude
from multicam_occlusion.recovery import (
    FrameRecovery,
    JointFrame,
    PoseRecovery,
    TrajectoryRecovery,
    back_project,
    best_camera_per_joint,
    recover_pose,
    recover_trajectory,
)
from multicam_occlusion.sources import (
    CameraCalibration,
    CameraStream,
    FileFrameSource,
    Frame,
    FrameBundle,
    FrameRef,
    FrameSource,
    Manifest,
)
from multicam_occlusion.triangulation import (
    build_ring_cameras,
    project_points,
    triangulate_dlt,
    triangulate_robust,
)

__all__ = [
    "CameraCalibration",
    "CameraStream",
    "FileFrameSource",
    "Frame",
    "FrameBundle",
    "FrameRecovery",
    "FrameRef",
    "FrameSource",
    "JointFrame",
    "Manifest",
    "ObsCamera",
    "ObsEntity",
    "ObsFrame",
    "ObservationManifest",
    "PerCam",
    "PointObs",
    "PoseRecovery",
    "TrajectoryRecovery",
    "back_project",
    "best_camera_per_joint",
    "build_ring_cameras",
    "drop_k_mask",
    "mpjpe",
    "occlude",
    "project_points",
    "recover_pose",
    "recover_trajectory",
    "triangulate_dlt",
    "triangulate_robust",
]

__version__ = "0.1.0"
