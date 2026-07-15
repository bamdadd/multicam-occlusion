"""Linear DLT triangulation and synthetic pinhole-camera helpers.

All routines are Blender-free and depend only on NumPy. Cameras follow the
standard pinhole model P = K [R | t], mapping a world point X (homogeneous) to
image coordinates via x ~ P X.

References (public methods only):
  * Hartley & Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed.,
    Chapter 12 (triangulation) and the Direct Linear Transformation.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def look_at_rotation(eye: FloatArray, target: FloatArray, up: FloatArray) -> FloatArray:
    """Return a world-to-camera rotation R for a camera at ``eye`` looking at ``target``.

    Uses the OpenCV/computer-vision convention: camera +z points along the
    viewing direction (from eye toward target), +x right, +y down.
    """
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    # Right = forward x up_world (then re-orthogonalise a down vector).
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    # Rows of R are the camera axes expressed in world coordinates.
    return np.stack([right, down, forward], axis=0)


def build_ring_cameras(
    n_cameras: int,
    radius: float = 4.0,
    height: float = 1.5,
    focal: float = 800.0,
    image_size: tuple[int, int] = (640, 480),
    look_at: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> FloatArray:
    """Build ``n_cameras`` synthetic pinhole cameras on a ring around the origin.

    Cameras are placed on a circle of the given ``radius`` at a fixed ``height``,
    each looking at ``look_at``. Returns an array of shape ``(n_cameras, 3, 4)``
    of projection matrices ``P = K [R | t]``.
    """
    if n_cameras < 1:
        raise ValueError("n_cameras must be >= 1")

    width, height_px = image_size
    cx, cy = width / 2.0, height_px / 2.0
    k_intrinsic = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    target = np.asarray(look_at, dtype=np.float64)
    up_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    proj_mats = np.empty((n_cameras, 3, 4), dtype=np.float64)
    for i in range(n_cameras):
        angle = 2.0 * np.pi * i / n_cameras
        eye = np.array([radius * np.cos(angle), radius * np.sin(angle), height], dtype=np.float64)
        rotation = look_at_rotation(eye, target, up_world)
        translation = -rotation @ eye  # t = -R C
        extrinsic = np.hstack([rotation, translation.reshape(3, 1)])
        proj_mats[i] = k_intrinsic @ extrinsic
    return proj_mats


def project_points(proj_mat: FloatArray, points3d: FloatArray) -> FloatArray:
    """Project world points through a single 3x4 projection matrix.

    ``points3d`` has shape ``(N, 3)``; returns pixel coordinates of shape
    ``(N, 2)``. Points at or behind the image plane (w <= 0) raise ``ValueError``.
    """
    points3d = np.atleast_2d(np.asarray(points3d, dtype=np.float64))
    homogeneous = np.hstack([points3d, np.ones((points3d.shape[0], 1))])
    projected = homogeneous @ proj_mat.T  # (N, 3)
    w = projected[:, 2]
    if np.any(w <= 0):
        raise ValueError("point projects behind the camera (w <= 0)")
    return projected[:, :2] / w[:, None]


def triangulate_dlt(
    proj_mats: FloatArray,
    points2d: FloatArray,
    mask: npt.NDArray[np.bool_] | None = None,
) -> FloatArray:
    """Linear DLT triangulation of one 3D point from N views.

    Args:
        proj_mats: ``(N, 3, 4)`` projection matrices.
        points2d: ``(N, 2)`` pixel observations, one per view.
        mask: optional ``(N,)`` boolean visibility mask; ``False`` marks a view
            in which the point is occluded and is dropped from the solve.

    Returns:
        The recovered 3D point, shape ``(3,)``.

    Raises:
        ValueError: if fewer than two views are visible (a single view
            constrains the point only to a ray and cannot recover depth).
    """
    proj_mats = np.asarray(proj_mats, dtype=np.float64)
    points2d = np.asarray(points2d, dtype=np.float64)
    n_views = proj_mats.shape[0]

    visibility = np.ones(n_views, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)

    visible = np.flatnonzero(visibility)
    if visible.size < 2:
        raise ValueError(
            f"need >= 2 visible views to triangulate; got {visible.size}. "
            "A single view constrains the point only to a ray."
        )

    rows = []
    for i in visible:
        p = proj_mats[i]
        u, v = points2d[i]
        # Each view contributes two independent rows of the DLT constraint
        # x cross (P X) = 0  ->  u*P3 - P1 = 0 and v*P3 - P2 = 0.
        rows.append(u * p[2] - p[0])
        rows.append(v * p[2] - p[1])
    a_matrix = np.stack(rows, axis=0)

    # Homogeneous least squares: solution is the right-singular vector of the
    # smallest singular value.
    _, _, vt = np.linalg.svd(a_matrix)
    solution = vt[-1]
    if abs(solution[3]) < 1e-12:
        raise ValueError("degenerate configuration: recovered point at infinity")
    point3d: FloatArray = solution[:3] / solution[3]
    return point3d
