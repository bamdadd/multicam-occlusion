"""Multi-camera synthetic occlusion benchmark.

Core thesis: with multiple calibrated cameras a 3D point can be recovered even
when it is occluded in some views; a single view is fundamentally insufficient
(the observation constrains the point only to a ray).
"""

from multicam_occlusion.occlusion import drop_k_mask, occlude
from multicam_occlusion.triangulation import (
    build_ring_cameras,
    project_points,
    triangulate_dlt,
)

__all__ = [
    "build_ring_cameras",
    "drop_k_mask",
    "occlude",
    "project_points",
    "triangulate_dlt",
]

__version__ = "0.1.0"
