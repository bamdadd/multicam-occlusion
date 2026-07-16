"""Recover a moving point's 3D trajectory from an observation manifest.

Two estimators, run over the same occluded observations, are the whole benchmark:

* **multi-view** — for each frame, mask the per-camera observations on the hard
  ``visible`` boolean and triangulate (linear DLT) from every camera that still
  sees the point. Needs >= 2 visible views; when occlusion drops a frame below
  that it is reported as *unrecoverable*, never faked.
* **single-view** — a monocular baseline that is *provably* unable to recover
  depth. One camera's pixel constrains the point only to a ray, so we back-
  project it to a fixed depth prior (the scene's mean camera-depth). When that
  one camera is occluded there is no observation at all and the best a monocular
  system can do is fall back to the prior mean (the scene centroid). Both are
  honest monocular behaviours; neither invents depth it cannot see.

Optional deterministic Gaussian pixel noise (fixed seed) models a realistic
keypoint detector, so the multi-view curve *responds* to the occlusion dose
(3-view frames tighten the solve; 2-view frames loosen it) instead of sitting at
machine epsilon.

All routines are NumPy-only — no multicam-sim, no renderer — so they run inside
the numpy-only CI gate against a committed fixture manifest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from multicam_occlusion.observation import ObsEntity, ObservationManifest, PointObs
from multicam_occlusion.triangulation import triangulate_dlt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class FrameRecovery:
    """One frame's recovery outcome for both estimators.

    ``multi``/``single`` are the recovered points (``None`` when the estimator
    could not produce one — e.g. multi-view with < 2 visible views). ``n_visible``
    is how many cameras saw the point that frame.
    """

    frame: int
    gt: FloatArray
    n_visible: int
    multi: FloatArray | None
    single: FloatArray
    single_observed: bool  # did the single camera actually see the point?

    def multi_error(self) -> float | None:
        if self.multi is None:
            return None
        return float(np.linalg.norm(self.multi - self.gt))

    def single_error(self) -> float:
        return float(np.linalg.norm(self.single - self.gt))


@dataclass(frozen=True)
class TrajectoryRecovery:
    """The whole trajectory's recovery, plus aggregate error metrics."""

    frames: list[FrameRecovery]
    single_cam: int
    n_cams: int

    def occlusion_rate(self) -> float:
        """Fraction of (frame x camera) observations that were occluded.

        Measured from the manifest, so it reflects the *achieved* occlusion the
        placed geometry produced, not a requested window.
        """
        total = len(self.frames) * self.n_cams
        occluded = sum(self.n_cams - fr.n_visible for fr in self.frames)
        return occluded / total if total else 0.0

    def multi_mpjpe(self) -> float:
        """Mean per-frame Euclidean error over *recoverable* multi-view frames."""
        errs = [e for fr in self.frames if (e := fr.multi_error()) is not None]
        return float(np.mean(errs)) if errs else float("inf")

    def single_mpjpe(self) -> float:
        """Mean per-frame Euclidean error of the monocular baseline (all frames)."""
        return float(np.mean([fr.single_error() for fr in self.frames]))

    def multi_recoverable_rate(self) -> float:
        """Fraction of frames the multi-view estimator could solve (>= 2 views)."""
        n = len(self.frames)
        return sum(fr.multi is not None for fr in self.frames) / n if n else 0.0

    def single_observed_rate(self) -> float:
        """Fraction of frames the single camera actually saw the point."""
        n = len(self.frames)
        return sum(fr.single_observed for fr in self.frames) / n if n else 0.0


def _entity(manifest: ObservationManifest, entity_id: str | None) -> ObsEntity:
    if entity_id is None:
        return manifest.entities[0]
    for e in manifest.entities:
        if e.id == entity_id:
            return e
    raise ValueError(f"entity {entity_id!r} not in manifest")


def back_project(
    cam_center: FloatArray,
    rotation: FloatArray,
    intrinsics: FloatArray,
    uv: FloatArray,
    depth: float,
) -> FloatArray:
    """Back-project a pixel to the 3D point at a chosen camera-space ``depth``.

    The ray through pixel ``uv`` is ``X = C + lambda * R^T K^-1 [u, v, 1]``. Since
    the third component of ``K^-1 [u, v, 1]`` is 1 for a standard pinhole ``K``,
    the camera-space z of ``X`` equals ``lambda`` — so ``lambda = depth`` places
    the point at exactly that depth. This is the monocular ambiguity made
    explicit: without a depth prior a single pixel fixes only the ray.
    """
    ray = rotation.T @ (np.linalg.inv(intrinsics) @ np.append(uv, 1.0))
    result: FloatArray = cam_center + depth * ray
    return result


def _scene_centroid(entity: ObsEntity, point_name: str) -> FloatArray:
    pts = [f.points[point_name].gt() for f in entity.frames if point_name in f.points]
    return np.mean(np.stack(pts, axis=0), axis=0)


def _aligned_observations(point: PointObs, n_cams: int) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
    """Return ``(uv (N,2), visible (N,))`` aligned to camera index order."""
    uv = np.zeros((n_cams, 2), dtype=np.float64)
    visible = np.zeros(n_cams, dtype=bool)
    for obs in point.per_cam:
        uv[obs.cam] = obs.uv_array()
        visible[obs.cam] = obs.visible
    return uv, visible


def recover_trajectory(
    manifest: ObservationManifest,
    *,
    entity_id: str | None = None,
    point_name: str = "center",
    single_cam: int = 0,
    pixel_noise: float = 0.0,
    seed: int = 0,
) -> TrajectoryRecovery:
    """Recover the trajectory with both estimators over every frame.

    Args:
        single_cam: which camera the monocular baseline is restricted to.
        pixel_noise: Gaussian std (px) added to *visible* observations before
            triangulation; ``0`` for exact projections. Deterministic given
            ``seed``.
    """
    entity = _entity(manifest, entity_id)
    proj_mats = manifest.projection_matrices()
    n_cams = len(manifest.cameras)
    cam = manifest.cameras[single_cam]
    rng = np.random.default_rng(seed)

    centroid = _scene_centroid(entity, point_name)
    # Depth prior: the scene centroid's camera-space z as seen by the single cam.
    depth_prior = float((cam.rotation() @ (centroid - cam.centre()))[2])
    c_center, c_rot, c_k = cam.centre(), cam.rotation(), cam.intrinsic_matrix()

    frames: list[FrameRecovery] = []
    for frame in entity.frames:
        if point_name not in frame.points:
            continue
        point = frame.points[point_name]
        gt = point.gt()
        uv, visible = _aligned_observations(point, n_cams)

        noisy = uv.copy()
        if pixel_noise > 0.0:
            noisy[visible] += rng.normal(0.0, pixel_noise, size=(int(visible.sum()), 2))

        # Multi-view: triangulate from the visible subset (>= 2 views).
        multi: FloatArray | None = None
        if int(visible.sum()) >= 2:
            multi = triangulate_dlt(proj_mats, noisy, mask=visible)

        # Single-view: back-project the one camera to the depth prior; if that
        # camera is occluded, fall back to the zero-information prior (centroid).
        if visible[single_cam]:
            single = back_project(c_center, c_rot, c_k, noisy[single_cam], depth_prior)
            observed = True
        else:
            single = centroid.copy()
            observed = False

        frames.append(
            FrameRecovery(
                frame=frame.frame,
                gt=gt,
                n_visible=int(visible.sum()),
                multi=multi,
                single=single,
                single_observed=observed,
            )
        )

    return TrajectoryRecovery(frames=frames, single_cam=single_cam, n_cams=n_cams)
