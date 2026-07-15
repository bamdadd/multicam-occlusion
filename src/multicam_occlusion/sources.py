"""Frame/video source seam: a typed ingestion interface for multi-camera input.

The benchmark core (triangulation, occlusion, metrics) never reads pixels off a
disk or a camera directly. It consumes the :class:`FrameSource` Protocol — a
narrow interface that (a) exposes per-camera calibration and (b) iterates
synchronized frame bundles across cameras. This keeps the core *open-closed*:

* v1 backend (:class:`FileFrameSource`) reads a JSON manifest plus a frame
  sequence (or an ``mp4`` path as metadata) from disk. It is the format a
  synthetic producer (``multicam-sim``) writes.
* A live-capture backend (GStreamer / RTSP / DeepStream) can be added later as
  another :class:`FrameSource` implementation *without touching the core*.

Calibration follows the same pinhole convention as
:mod:`multicam_occlusion.triangulation`: ``P = K [R | t]`` with world-to-camera
rotation ``R`` and translation ``t``. Matrices travel through the manifest as
plain nested lists (JSON-friendly) and are exposed as NumPy arrays via methods,
so the schema stays serializable and dependency-light.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

FloatArray = npt.NDArray[np.float64]


class CameraCalibration(BaseModel):
    """Intrinsic + extrinsic calibration for one camera.

    ``K`` is the ``3x3`` intrinsic matrix, ``R`` the ``3x3`` world-to-camera
    rotation, and ``t`` the length-3 translation. Stored as nested lists so a
    manifest round-trips through JSON; use the accessor methods for NumPy.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    K: list[list[float]] = Field(description="3x3 intrinsic matrix")
    R: list[list[float]] = Field(description="3x3 world-to-camera rotation")
    t: list[float] = Field(description="length-3 translation")

    @model_validator(mode="after")
    def _check_shapes(self) -> CameraCalibration:
        if len(self.K) != 3 or any(len(row) != 3 for row in self.K):
            raise ValueError(f"camera {self.id!r}: K must be 3x3")
        if len(self.R) != 3 or any(len(row) != 3 for row in self.R):
            raise ValueError(f"camera {self.id!r}: R must be 3x3")
        if len(self.t) != 3:
            raise ValueError(f"camera {self.id!r}: t must have length 3")
        return self

    def intrinsic_matrix(self) -> FloatArray:
        """Return ``K`` as a ``(3, 3)`` array."""
        result: FloatArray = np.asarray(self.K, dtype=np.float64)
        return result

    def rotation(self) -> FloatArray:
        """Return ``R`` as a ``(3, 3)`` array."""
        result: FloatArray = np.asarray(self.R, dtype=np.float64)
        return result

    def translation(self) -> FloatArray:
        """Return ``t`` as a ``(3,)`` array."""
        result: FloatArray = np.asarray(self.t, dtype=np.float64)
        return result

    def projection_matrix(self) -> FloatArray:
        """Return ``P = K [R | t]``, a ``(3, 4)`` matrix matching the core.

        Compatible with :func:`multicam_occlusion.triangulation.project_points`
        and :func:`~multicam_occlusion.triangulation.triangulate_dlt`.
        """
        extrinsic = np.hstack([self.rotation(), self.translation().reshape(3, 1)])
        result: FloatArray = self.intrinsic_matrix() @ extrinsic
        return result


class FrameRef(BaseModel):
    """A single frame on disk, tagged with its capture timestamp (seconds)."""

    model_config = ConfigDict(frozen=True)

    timestamp: float
    path: str = Field(description="frame path, relative to the manifest directory")


class CameraStream(BaseModel):
    """Where one camera's frames come from: a frame sequence *or* an mp4 path.

    Exactly one of ``frames`` (a frame sequence, each an :class:`FrameRef`) or
    ``video`` (an ``mp4`` path plus its per-frame ``timestamps``) must be set.
    The v1 file backend loads frame-sequence arrays with NumPy; for a video it
    exposes the path as metadata only (no decoder dependency) — a live backend
    is what actually decodes.
    """

    model_config = ConfigDict(frozen=True)

    camera_id: str
    frames: list[FrameRef] | None = None
    video: str | None = Field(default=None, description="mp4 path, relative to the manifest")
    timestamps: list[float] | None = Field(
        default=None, description="per-frame timestamps for a video stream"
    )

    @model_validator(mode="after")
    def _check_exactly_one_backing(self) -> CameraStream:
        has_frames = self.frames is not None
        has_video = self.video is not None
        if has_frames == has_video:
            raise ValueError(
                f"stream for camera {self.camera_id!r}: set exactly one of 'frames' or 'video'"
            )
        if has_video and not self.timestamps:
            raise ValueError(
                f"stream for camera {self.camera_id!r}: a video stream needs 'timestamps'"
            )
        return self

    def frame_timestamps(self) -> list[float]:
        """Timestamps of every frame in this stream, in declared order."""
        if self.frames is not None:
            return [f.timestamp for f in self.frames]
        assert self.timestamps is not None  # guaranteed by the validator
        return list(self.timestamps)


class Manifest(BaseModel):
    """The on-disk schema a synthetic producer (``multicam-sim``) emits.

    Bundles per-camera calibration with per-camera stream sources and the frame
    timestamps used to synchronize cameras. This is the *same* schema the file
    backend reads, so producer and consumer share one contract.
    """

    model_config = ConfigDict(frozen=True)

    version: int = 1
    cameras: list[CameraCalibration]
    streams: list[CameraStream]

    @model_validator(mode="after")
    def _check_stream_camera_ids(self) -> Manifest:
        camera_ids = {c.id for c in self.cameras}
        if len(camera_ids) != len(self.cameras):
            raise ValueError("duplicate camera id in manifest")
        stream_ids = [s.camera_id for s in self.streams]
        if len(stream_ids) != len(set(stream_ids)):
            raise ValueError("more than one stream for a single camera")
        unknown = set(stream_ids) - camera_ids
        if unknown:
            raise ValueError(f"stream references unknown camera id(s): {sorted(unknown)}")
        return self

    @classmethod
    def from_json(cls, path: str | Path) -> Manifest:
        """Load and validate a manifest from a JSON file."""
        text = Path(path).read_text()
        return cls.model_validate(json.loads(text))


class Frame(BaseModel):
    """One camera's observation at one timestamp.

    ``image`` holds the loaded pixel array for a frame-sequence source, or
    ``None`` for a video-backed source (whose ``path`` is exposed as metadata
    only). ``path`` is absolute, resolved against the manifest directory.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    camera_id: str
    timestamp: float
    path: Path
    image: FloatArray | None = None


class FrameBundle(BaseModel):
    """All cameras' frames at one synchronized timestamp."""

    model_config = ConfigDict(frozen=True)

    timestamp: float
    frames: list[Frame]

    def camera_ids(self) -> list[str]:
        """Ids of the cameras present in this bundle, in order."""
        return [f.camera_id for f in self.frames]


@runtime_checkable
class FrameSource(Protocol):
    """The ingestion seam the benchmark core consumes.

    Any backend — the v1 file reader, or a future GStreamer/RTSP live capture —
    implements these two methods. The core never imports a concrete backend, so
    a new one plugs in without a core change (open-closed).
    """

    def cameras(self) -> Sequence[CameraCalibration]:
        """Return every camera's calibration."""
        ...

    def frames(self) -> Iterator[FrameBundle]:
        """Iterate synchronized frame bundles in ascending timestamp order."""
        ...


class FileFrameSource:
    """v1 :class:`FrameSource`: reads a manifest + frame sequences from disk.

    Frame-sequence frames are loaded with :func:`numpy.load` (``.npy``), so CI
    stays numpy-only. Video streams contribute :class:`Frame` entries whose
    ``image`` is ``None`` and whose ``path`` points at the ``mp4`` — decoding is
    a live backend's job, not this one's.

    Iteration is deterministic: bundles are yielded in ascending timestamp
    order, and within a bundle cameras follow manifest declaration order.
    """

    def __init__(self, manifest: Manifest, root: str | Path) -> None:
        self._manifest = manifest
        self._root = Path(root)
        self._streams = {s.camera_id: s for s in manifest.streams}

    @classmethod
    def from_manifest_file(cls, path: str | Path) -> FileFrameSource:
        """Build a source from a manifest JSON file; paths resolve beside it."""
        manifest_path = Path(path)
        manifest = Manifest.from_json(manifest_path)
        return cls(manifest, root=manifest_path.parent)

    def cameras(self) -> Sequence[CameraCalibration]:
        return self._manifest.cameras

    def _load_frame(self, camera_id: str, timestamp: float, rel_path: str) -> Frame:
        path = self._root / rel_path
        image: FloatArray | None = None
        if path.suffix == ".npy":
            image = np.asarray(np.load(path), dtype=np.float64)
        return Frame(camera_id=camera_id, timestamp=timestamp, path=path, image=image)

    def frames(self) -> Iterator[FrameBundle]:
        # Collect (timestamp, camera_id) -> Frame across all streams, then emit
        # one bundle per distinct timestamp in ascending order. Camera order
        # within a bundle follows manifest declaration order for determinism.
        per_timestamp: dict[float, dict[str, Frame]] = {}
        for camera in self._manifest.cameras:
            stream = self._streams.get(camera.id)
            if stream is None:
                continue
            if stream.frames is not None:
                for ref in stream.frames:
                    frame = self._load_frame(camera.id, ref.timestamp, ref.path)
                    per_timestamp.setdefault(ref.timestamp, {})[camera.id] = frame
            else:
                assert stream.video is not None and stream.timestamps is not None
                for ts in stream.timestamps:
                    frame = Frame(camera_id=camera.id, timestamp=ts, path=self._root / stream.video)
                    per_timestamp.setdefault(ts, {})[camera.id] = frame

        order = [c.id for c in self._manifest.cameras]
        for ts in sorted(per_timestamp):
            by_camera = per_timestamp[ts]
            frames = [by_camera[cid] for cid in order if cid in by_camera]
            yield FrameBundle(timestamp=ts, frames=frames)
