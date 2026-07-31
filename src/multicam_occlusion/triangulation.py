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

#: Rank test for the DLT system: ``sigma[-1] / sigma[-2]`` above this is treated
#: as rank-deficient. A well-posed solve leaves ``sigma[-2]`` at signal scale
#: while ``sigma[-1]`` collapses, so the ratio is ~1e-14 on exact observations
#: and stays below ~4e-2 under the DESIGN.md 0.5px pixel noise; a 2D null space
#: collapses both together and drives the ratio to O(1). 0.1 is the log-midpoint
#: of those measured populations -- see ``tests/test_triangulation_degenerate.py``.
DEGENERACY_RATIO_TOL = 0.1

#: Reprojection residual, in pixels, at or below which a view joins the RANSAC
#: consensus set. Measured over the DESIGN.md camera-count sweep (ring rigs,
#: n = 3..8) at the DESIGN.md 0.5px pixel noise: the residual a clean two-view
#: hypothesis leaves at the other clean views has median 0.88px and p99.9
#: 4.71px over 151200 samples, while a view displaced by 20px per coordinate
#: has a measured minimum residual of 23.8px over 22400 samples.
#: 5.0 is the smallest round number above that clean
#: p99.9 -- also 10x the DESIGN.md noise level -- and sits 4.8x below the
#: cheapest gross outlier. Biased low on purpose: too low only drops a clean
#: view from the refit, too high admits the outlier and defeats the function.
#: See ``tests/test_triangulation_robust.py``.
RANSAC_INLIER_THRESH_PX = 5.0

#: Number of minimal two-view hypotheses drawn per call. Measured with one
#: 20px-per-coordinate outlier and 0.5px noise, 200 seeds per camera count:
#: 10 iterations already recover ground truth 200/200 times at every n in
#: 3..8 (5 iterations still fails 25/200 at n = 3). 100 keeps a 10x margin on
#: that and stays cheap -- the widest rig DESIGN.md sweeps has only C(8, 2) = 28
#: distinct two-view subsets, and each hypothesis costs one small SVD.
RANSAC_MAX_ITERS = 100


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
            constrains the point only to a ray and cannot recover depth); if
            the DLT system is rank-deficient, which happens when the target
            lies on the baseline of the contributing views; or if the
            recovered point is at infinity.
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
    _, sigma, vt = np.linalg.svd(a_matrix)
    solution = vt[-1]

    # Rank test before anything is read off the solution vector. A well-posed
    # system has a 1D null space, so only sigma[-1] collapses. When the target
    # lies on the baseline joining the contributing camera centres the null
    # space is 2D: sigma[-2] collapses with it and vt[-1] is an arbitrary vector
    # from that space -- a point that is silently wrong along the baseline
    # rather than one detectably at infinity. Checked first because rank
    # deficiency invalidates the whole vector, its last coordinate included.
    if sigma[-2] <= 0.0 or float(sigma[-1] / sigma[-2]) > DEGENERACY_RATIO_TOL:
        raise ValueError(
            "degenerate configuration: rank-deficient DLT system "
            f"(sigma[-1]/sigma[-2] > {DEGENERACY_RATIO_TOL}); the target may lie on "
            "the baseline joining the contributing camera centres"
        )
    if abs(solution[3]) < 1e-12:
        raise ValueError("degenerate configuration: recovered point at infinity")
    point3d: FloatArray = solution[:3] / solution[3]
    return point3d


def _reprojection_residuals(
    proj_mats: FloatArray,
    points2d: FloatArray,
    visible: npt.NDArray[np.intp],
    point3d: FloatArray,
) -> FloatArray:
    """Euclidean pixel residual of ``point3d`` at each of the ``visible`` views.

    A hypothesis that falls behind one of the cameras cannot be scored there --
    ``project_points`` raises for ``w <= 0`` -- so that view is charged an
    infinite residual, which reads as "outlier" rather than aborting the solve.
    """
    residuals = np.empty(visible.size, dtype=np.float64)
    for k, view in enumerate(visible):
        try:
            predicted = project_points(proj_mats[view], point3d)[0]
        except ValueError:
            residuals[k] = np.inf
            continue
        residuals[k] = float(np.linalg.norm(predicted - points2d[view]))
    return residuals


def triangulate_robust(
    proj_mats: FloatArray,
    points2d: FloatArray,
    mask: npt.NDArray[np.bool_] | None = None,
    *,
    inlier_thresh_px: float = RANSAC_INLIER_THRESH_PX,
    max_iters: int = RANSAC_MAX_ITERS,
    seed: int = 0,
) -> FloatArray:
    """RANSAC triangulation of one 3D point, tolerant of outlying observations.

    A single grossly wrong observation -- a detector that latched onto the wrong
    person, a mis-associated keypoint -- drags the least-squares DLT solution off
    the true point, because every view enters the solve with equal weight. This
    draws ``max_iters`` minimal two-view subsets from the visible views, solves
    each with :func:`triangulate_dlt`, scores it by how many views reproject
    within ``inlier_thresh_px``, and refits :func:`triangulate_dlt` on the
    winning consensus set. Hypotheses tied on inlier count are separated by the
    summed residual over their inliers (lower wins), which is what distinguishes
    a clean pair from an outlying one at three views, where both score two.

    Args:
        proj_mats: ``(N, 3, 4)`` projection matrices.
        points2d: ``(N, 2)`` pixel observations, one per view.
        mask: optional ``(N,)`` boolean visibility mask; ``False`` marks a view
            in which the point is occluded and is dropped entirely.
        inlier_thresh_px: reprojection residual admitting a view to the
            consensus set; defaults to ``RANSAC_INLIER_THRESH_PX`` (10x the
            DESIGN.md 0.5px noise level).
        max_iters: number of minimal subsets drawn.
        seed: RNG seed. The generator is built inside this call, so identical
            arguments give a bitwise-identical result, the same convention as
            :func:`multicam_occlusion.occlusion.drop_k_mask`.

    Returns:
        The recovered 3D point, shape ``(3,)``.

    Raises:
        ValueError: if fewer than two views are visible; if no hypothesis ever
            gathered two inliers, i.e. no two observations agree at this
            threshold; or, from the refit, if the consensus set is itself
            geometrically degenerate.
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

    rng = np.random.default_rng(seed)
    best_inliers: npt.NDArray[np.intp] | None = None
    best_count = 0
    best_cost = np.inf

    for _ in range(max_iters):
        sample_mask = np.zeros(n_views, dtype=bool)
        sample_mask[visible[rng.choice(visible.size, size=2, replace=False)]] = True
        try:
            hypothesis = triangulate_dlt(proj_mats, points2d, mask=sample_mask)
        except ValueError:
            # A minimal sample can be degenerate on its own (near-collinear pair,
            # badly skew rays); that hypothesis is discarded, not the whole call.
            continue

        residuals = _reprojection_residuals(proj_mats, points2d, visible, hypothesis)
        is_inlier = residuals <= inlier_thresh_px
        count = int(is_inlier.sum())
        cost = float(residuals[is_inlier].sum())
        # Strict comparisons on both, so ties keep the first-seen hypothesis and
        # the outcome is fixed by the draw sequence alone.
        if count > best_count or (count == best_count and cost < best_cost):
            best_count, best_cost, best_inliers = count, cost, visible[is_inlier]

    if best_inliers is None or best_count < 2:
        raise ValueError(
            f"no consensus: no two of {visible.size} visible views agreed within "
            f"{inlier_thresh_px}px over {max_iters} iterations. Every observation "
            "may be an outlier, or the threshold may be too tight."
        )

    inlier_mask = np.zeros(n_views, dtype=bool)
    inlier_mask[best_inliers] = True
    return triangulate_dlt(proj_mats, points2d, mask=inlier_mask)
