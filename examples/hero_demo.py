"""Hero demo: single view fails, multi-view recovers an occluded 3D point.

Blender-free. Projects one known 3D point into six synthetic ring cameras,
occludes it in three of them, and triangulates from the surviving views.

Run with::

    uv run python examples/hero_demo.py
"""

from __future__ import annotations

import numpy as np

from multicam_occlusion import (
    build_ring_cameras,
    drop_k_mask,
    project_points,
    triangulate_dlt,
)

SEED = 20260715


def project_all(proj_mats: np.ndarray, point3d: np.ndarray) -> np.ndarray:
    """Project one 3D point into every camera; returns ``(n_views, 2)``."""
    return np.vstack([project_points(p, point3d)[0] for p in proj_mats])


def main() -> None:
    proj_mats = build_ring_cameras(n_cameras=6)
    gt = np.array([0.3, -0.4, 0.8])
    points2d = project_all(proj_mats, gt)

    # Occlude 3 of 6 views for this point (deterministic given the seed).
    mask = drop_k_mask(n_views=6, n_points=1, k_drop=3, seed=SEED)[0]
    visible = np.flatnonzero(mask).tolist()

    print(f"ground truth point : {gt}")
    print(f"visible views      : {visible} ({mask.sum()} of 6)")

    # A single view cannot recover depth: it constrains the point to a ray.
    single = np.zeros(6, dtype=bool)
    single[0] = True
    try:
        triangulate_dlt(proj_mats, points2d, mask=single)
    except ValueError as exc:
        print(f"single view        : REFUSED -- {str(exc).split(';')[0]}")

    # The visible subset recovers ground truth to machine precision.
    recovered = triangulate_dlt(proj_mats, points2d, mask=mask)
    max_err = float(np.max(np.abs(recovered - gt)))
    print(f"multi-view recovery: {recovered}")
    print(f"max abs error      : {max_err:.2e}")


if __name__ == "__main__":
    main()
