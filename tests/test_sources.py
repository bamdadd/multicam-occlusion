"""FrameSource seam: the v1 file backend loads calibration and iterates frames.

Everything here is numpy-only (frame sequences are ``.npy`` files) so it runs in
CI without a video decoder. We assert that :class:`FileFrameSource`:

  1. round-trips a manifest through JSON and exposes per-camera calibration,
  2. iterates synchronized frame bundles deterministically (ascending
     timestamp, manifest camera order), loading frame-sequence arrays,
  3. exposes a video stream as path metadata only (no decode), and
  4. satisfies the :class:`FrameSource` Protocol at runtime.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from multicam_occlusion import (
    CameraCalibration,
    FileFrameSource,
    FrameSource,
    Manifest,
    build_ring_cameras,
)


def _simple_camera(cam_id: str, offset: float) -> dict[str, object]:
    """A JSON-ready camera record with a distinct, well-formed calibration."""
    return {
        "id": cam_id,
        "K": [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
        "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "t": [offset, 0.0, 5.0],
    }


def _write_manifest(root: Path) -> tuple[Path, dict[str, np.ndarray]]:
    """Write a 3-camera manifest under ``root``; return its path + frame arrays.

    Cameras ``cam0`` and ``cam1`` are backed by two-frame ``.npy`` sequences at
    timestamps 0.0 and 0.1; ``cam2`` is backed by an (undecoded) mp4 at the same
    timestamps. Frame arrays are deterministic so the test can assert content.
    """
    frames_dir = root / "frames"
    frames_dir.mkdir()

    expected: dict[str, np.ndarray] = {}
    streams: list[dict[str, object]] = []
    timestamps = [0.0, 0.1]

    for cam_idx, cam_id in enumerate(("cam0", "cam1")):
        refs: list[dict[str, object]] = []
        for ts_idx, ts in enumerate(timestamps):
            arr = np.full((4, 4), float(cam_idx * 10 + ts_idx), dtype=np.float64)
            rel = f"frames/{cam_id}_{ts_idx}.npy"
            np.save(root / rel, arr)
            expected[f"{cam_id}@{ts}"] = arr
            refs.append({"timestamp": ts, "path": rel})
        streams.append({"camera_id": cam_id, "frames": refs})

    # A video-backed camera: path metadata only, no decoder needed.
    streams.append({"camera_id": "cam2", "video": "frames/cam2.mp4", "timestamps": timestamps})

    manifest = {
        "version": 1,
        "cameras": [
            _simple_camera("cam0", -1.0),
            _simple_camera("cam1", 1.0),
            _simple_camera("cam2", 0.0),
        ],
        "streams": streams,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, expected


def test_file_source_loads_cameras_and_projection(tmp_path: Path) -> None:
    """Calibration round-trips; projection matrix matches K [R | t]."""
    manifest_path, _ = _write_manifest(tmp_path)
    source = FileFrameSource.from_manifest_file(manifest_path)

    cams = source.cameras()
    assert [c.id for c in cams] == ["cam0", "cam1", "cam2"]

    cam0 = cams[0]
    p = cam0.projection_matrix()
    assert p.shape == (3, 4)
    expected = cam0.intrinsic_matrix() @ np.hstack(
        [cam0.rotation(), cam0.translation().reshape(3, 1)]
    )
    assert np.allclose(p, expected)


def test_file_source_iterates_deterministically(tmp_path: Path) -> None:
    """Bundles come in ascending-timestamp, manifest-camera order, repeatably."""
    manifest_path, expected = _write_manifest(tmp_path)
    source = FileFrameSource.from_manifest_file(manifest_path)

    bundles = list(source.frames())
    assert [b.timestamp for b in bundles] == [0.0, 0.1]
    for bundle in bundles:
        assert bundle.camera_ids() == ["cam0", "cam1", "cam2"]

    # Frame-sequence cameras carry loaded arrays; the video camera carries none.
    first = bundles[0]
    assert np.array_equal(first.frames[0].image, expected["cam0@0.0"])
    assert np.array_equal(first.frames[1].image, expected["cam1@0.0"])
    assert first.frames[2].image is None
    assert first.frames[2].path.name == "cam2.mp4"

    # Determinism: a second pass yields identical timestamps and arrays.
    again = list(source.frames())
    assert [b.timestamp for b in again] == [b.timestamp for b in bundles]
    assert np.array_equal(again[1].frames[0].image, bundles[1].frames[0].image)


def test_file_source_satisfies_protocol(tmp_path: Path) -> None:
    """The concrete backend is a structural :class:`FrameSource`."""
    manifest_path, _ = _write_manifest(tmp_path)
    source = FileFrameSource.from_manifest_file(manifest_path)
    assert isinstance(source, FrameSource)


def test_manifest_rejects_unknown_stream_camera(tmp_path: Path) -> None:
    """A stream referencing a camera not in the manifest is rejected."""
    with pytest.raises(ValueError, match="unknown camera id"):
        Manifest.model_validate(
            {
                "cameras": [_simple_camera("cam0", 0.0)],
                "streams": [
                    {"camera_id": "ghost", "frames": [{"timestamp": 0.0, "path": "a.npy"}]}
                ],
            }
        )


def test_stream_requires_exactly_one_backing() -> None:
    """A stream with neither (or both) frames and video is rejected."""
    from multicam_occlusion import CameraStream

    with pytest.raises(ValueError, match="exactly one of 'frames' or 'video'"):
        CameraStream.model_validate({"camera_id": "cam0"})


def test_calibration_rejects_bad_shape() -> None:
    """A non-3x3 intrinsic matrix is rejected at validation time."""
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    with pytest.raises(ValueError, match="K must be 3x3"):
        CameraCalibration.model_validate(
            {"id": "cam0", "K": [[1.0, 0.0], [0.0, 1.0]], "R": identity, "t": [0.0, 0.0, 0.0]}
        )


def test_projection_matrix_is_triangulation_compatible(tmp_path: Path) -> None:
    """A manifest built from ring cameras yields triangulation-ready matrices.

    Decompose the core's ring projection matrices back into K/R/t, store them in
    a manifest, and confirm the reconstructed projection matrix matches — the
    seam feeds the same P the triangulation core expects.
    """
    proj_mats = build_ring_cameras(n_cameras=3)
    # build_ring_cameras uses a shared K with focal 800 and principal point at
    # the (640x480) image centre; recover [R|t] as K^-1 P.
    k = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    cameras = []
    for i, p in enumerate(proj_mats):
        rt = np.linalg.inv(k) @ p
        cameras.append(
            {
                "id": f"cam{i}",
                "K": k.tolist(),
                "R": rt[:, :3].tolist(),
                "t": rt[:, 3].tolist(),
            }
        )
    manifest = Manifest.model_validate({"cameras": cameras, "streams": []})
    for p, cam in zip(proj_mats, manifest.cameras, strict=True):
        assert np.allclose(cam.projection_matrix(), p)
