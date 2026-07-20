"""Typed loader for the *analytic observation manifest* (multicam-sim's output).

This is a **distinct ingest** from :mod:`multicam_occlusion.sources`. That seam
(:class:`~multicam_occlusion.sources.FrameSource`) carries per-camera *pixels* for
an image/video pipeline. This module reads the other manifest multicam-sim emits:
a pure-analytic record where, for every named point of every entity at every
frame, it stores the ground-truth world coordinate and, per camera, the pixel
projection plus a hard ``visible`` boolean and a continuous ``occ_frac`` knob.

The schema is multicam-sim's contract of record (see its ``DESIGN.md``); the
camera convention is the same ``P = K [R | t]`` OpenCV pinhole this package's
:mod:`~multicam_occlusion.triangulation` already uses, so a manifest produced
there is consumed convention-for-convention here. All floats arrive at full
double precision, so rebuilding ``P`` recovers ground truth to ~machine epsilon —
:meth:`ObsCamera.reprojection_error` verifies exactly that.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

FloatArray = npt.NDArray[np.float64]

#: The camera convention multicam-sim tags every camera with.
CONVENTION = "opencv_rdf"


class ObsCamera(BaseModel):
    """One camera's calibration in the observation manifest.

    ``K`` (3x3 intrinsics), ``R`` (3x3 world->camera rotation) and ``t``
    (length-3 translation) are stored as JSON-native nested lists; the accessors
    return NumPy. ``projection_matrix`` rebuilds ``P = K [R | t]`` — the exact
    3x4 that :func:`multicam_occlusion.triangulation.triangulate_dlt` consumes.
    """

    model_config = ConfigDict(frozen=True)

    id: int
    K: list[list[float]] = Field(description="3x3 intrinsic matrix")
    R: list[list[float]] = Field(description="3x3 world-to-camera rotation")
    t: list[float] = Field(description="length-3 world-to-camera translation")
    width: int
    height: int
    convention: str = CONVENTION

    @model_validator(mode="after")
    def _check(self) -> ObsCamera:
        if len(self.K) != 3 or any(len(row) != 3 for row in self.K):
            raise ValueError(f"camera {self.id}: K must be 3x3")
        if len(self.R) != 3 or any(len(row) != 3 for row in self.R):
            raise ValueError(f"camera {self.id}: R must be 3x3")
        if len(self.t) != 3:
            raise ValueError(f"camera {self.id}: t must have length 3")
        if self.convention != CONVENTION:
            raise ValueError(
                f"camera {self.id}: convention {self.convention!r} != {CONVENTION!r}; "
                "this loader consumes the OpenCV RDF pinhole contract only"
            )
        return self

    def intrinsic_matrix(self) -> FloatArray:
        return np.asarray(self.K, dtype=np.float64)

    def rotation(self) -> FloatArray:
        return np.asarray(self.R, dtype=np.float64)

    def translation(self) -> FloatArray:
        return np.asarray(self.t, dtype=np.float64)

    def centre(self) -> FloatArray:
        """Camera centre ``C = -R^T @ t`` in world coordinates."""
        result: FloatArray = -self.rotation().T @ self.translation()
        return result

    def projection_matrix(self) -> FloatArray:
        """``P = K [R | t]``, a ``(3, 4)`` matrix (core-compatible)."""
        extrinsic = np.hstack([self.rotation(), self.translation().reshape(3, 1)])
        result: FloatArray = self.intrinsic_matrix() @ extrinsic
        return result

    def reprojection_error(self, xyz: FloatArray, uv: FloatArray) -> float:
        """Pixel distance between projecting ``xyz`` and the stored ``uv``.

        A near-zero value proves we rebuilt the producer's ``P`` correctly. Used
        by the loader's optional self-check; catches any K/R/t transcription bug
        for free.
        """
        p = self.projection_matrix()
        homog = np.append(np.asarray(xyz, dtype=np.float64), 1.0)
        projected = p @ homog
        if projected[2] <= 0.0:
            return float("inf")
        proj_uv = projected[:2] / projected[2]
        return float(np.linalg.norm(proj_uv - np.asarray(uv, dtype=np.float64)))


class PerCam(BaseModel):
    """One camera's observation of one point: pixel + occlusion signals.

    ``visible_fraction`` / ``occluded`` are multicam-sim's opt-in image-space
    occlusion labels (the fraction of the object's silhouette this camera still
    sees, and whether any nearer occluder eats into it). They are ``None`` when
    the producer did not emit them, so an older manifest still loads. When
    present, ``visible_fraction`` is the continuous occlusion dose this benchmark
    bins on.
    """

    model_config = ConfigDict(frozen=True)

    cam: int
    uv: list[float] = Field(description="pixel [u, v]")
    visible: bool = Field(description="hard DLT mask: in-front AND in-image AND unblocked")
    occ_frac: float = Field(default=0.0, description="continuous occlusion knob; never masks")
    visible_fraction: float | None = Field(
        default=None, description="image-space silhouette fraction visible in [0,1]; None if absent"
    )
    occluded: bool | None = Field(
        default=None, description="any nearer occluder covers the silhouette (visible_fraction<1)"
    )

    @model_validator(mode="after")
    def _check(self) -> PerCam:
        if len(self.uv) != 2:
            raise ValueError(f"camera {self.cam}: uv must be [u, v]")
        if self.visible_fraction is not None and not 0.0 <= self.visible_fraction <= 1.0:
            raise ValueError(f"camera {self.cam}: visible_fraction must be in [0, 1]")
        return self

    def uv_array(self) -> FloatArray:
        return np.asarray(self.uv, dtype=np.float64)


class PointObs(BaseModel):
    """A named point at one frame: ground-truth 3D + per-camera observations."""

    model_config = ConfigDict(frozen=True)

    xyz_gt: list[float] = Field(description="ground-truth world [x, y, z]")
    per_cam: list[PerCam]

    @model_validator(mode="after")
    def _check(self) -> PointObs:
        if len(self.xyz_gt) != 3:
            raise ValueError("xyz_gt must be [x, y, z]")
        return self

    def gt(self) -> FloatArray:
        return np.asarray(self.xyz_gt, dtype=np.float64)


class ObsFrame(BaseModel):
    """One frame of an entity: frame index + its named points."""

    model_config = ConfigDict(frozen=True)

    frame: int
    points: dict[str, PointObs]


class ObsEntity(BaseModel):
    """A tracked entity: an ordered list of frames, optional skeleton ``edges``."""

    model_config = ConfigDict(frozen=True)

    id: str
    frames: list[ObsFrame]
    edges: list[list[str]] | None = None


class ObservationManifest(BaseModel):
    """The full analytic manifest multicam-sim emits.

    Load with :meth:`from_json`; then read cameras via :attr:`cameras` and iterate
    entities/frames/points. :meth:`projection_matrices` returns the ``(N, 3, 4)``
    stack aligned with camera declaration order, ready for
    :func:`multicam_occlusion.triangulation.triangulate_dlt`.
    """

    model_config = ConfigDict(frozen=True)

    cameras: list[ObsCamera]
    fps: float
    num_frames: int
    entities: list[ObsEntity]

    @model_validator(mode="after")
    def _check(self) -> ObservationManifest:
        ids = [c.id for c in self.cameras]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate camera id in manifest")
        return self

    @classmethod
    def from_json(cls, path: str | Path, *, verify: bool = True) -> ObservationManifest:
        """Load and validate a manifest; optionally self-check reprojection.

        With ``verify=True`` (default) every stored ``uv`` is checked against a
        fresh projection of its ``xyz_gt`` to within ``1e-6`` px, proving the
        rebuilt ``P = K [R | t]`` matches the producer convention-for-convention.
        """
        manifest = cls.model_validate(json.loads(Path(path).read_text()))
        if verify:
            manifest.verify_reprojection()
        return manifest

    def projection_matrices(self) -> FloatArray:
        """``(N, 3, 4)`` projection matrices in camera declaration order."""
        return np.stack([c.projection_matrix() for c in self.cameras], axis=0)

    def verify_reprojection(self, tol: float = 1e-6) -> None:
        """Assert every visible observation reprojects onto its stored ``uv``.

        Only ``visible`` observations are checked: an occluded point may sit
        behind the camera or outside the frame, where the stored ``uv`` is still
        the raw projection but the ``w <= 0`` guard would trip.
        """
        by_id = {c.id: c for c in self.cameras}
        for entity in self.entities:
            for frame in entity.frames:
                for name, point in frame.points.items():
                    gt = point.gt()
                    for obs in point.per_cam:
                        if not obs.visible:
                            continue
                        err = by_id[obs.cam].reprojection_error(gt, obs.uv_array())
                        if err > tol:
                            raise ValueError(
                                f"reprojection mismatch {err:.2e}px > {tol:.0e} for "
                                f"entity {entity.id!r} point {name!r} frame {frame.frame} "
                                f"cam {obs.cam}: rebuilt P does not match producer"
                            )
