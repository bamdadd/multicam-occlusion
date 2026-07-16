"""MTMC handoff proof: keep one identity across a blind gap on a non-overlap rig.

The fixture is a **hand-authored** manifest in the multicam-sim schema (its ring
rig always shares a look-at target, so it can't yet emit a non-overlapping rig —
tracked upstream). Two station cameras — station 1 (``cam0``) and station 2
(``cam1``) — with **no field-of-view overlap**: each entity is ``visible`` in one camera at
a time and invisible during the gap. Two entities cross the gap **staggered** in
time, so spatio-temporal plausibility alone (default no-appearance backend) must
disambiguate — a single-entity fixture would be satisfied by assigning
everything one id and prove nothing.

Headline: with the handoff, IDF1 = 1.0 and 0 ID-switches; the no-handoff
baseline scores strictly worse (IDF1 = 0.5, 2 switches). That contrast is the
proof.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from multicam_occlusion.mtmc import (
    CameraTopology,
    SpatioTemporalMatcher,
    StubEmbedding,
    TransitDistribution,
    TransitionEdge,
    assign_global_ids,
    compute_identity_metrics,
    extract_tracklets,
    idf1_from_counts,
    load_manifest,
    per_tracklet_global_ids,
)

FPS = 10.0
WIDTH = 640.0
HEIGHT = 480.0


def _cam(cam_id: int) -> dict[str, Any]:
    return {
        "id": cam_id,
        "K": [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
        "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "t": [0.0, 0.0, 5.0],
        "width": int(WIDTH),
        "height": int(HEIGHT),
        "convention": "opencv_rdf",
    }


def _obs(cam: int, u: float, v: float) -> dict[str, Any]:
    return {"cam": cam, "uv": [u, v], "visible": True, "occ_frac": 0.0}


def _hidden(cam: int) -> dict[str, Any]:
    return {"cam": cam, "uv": [0.0, 0.0], "visible": False, "occ_frac": 1.0}


def _frame(
    frame: int, per_cam: list[dict[str, Any]], xyz: tuple[float, float, float]
) -> dict[str, Any]:
    return {
        "frame": frame,
        "points": {"center": {"xyz_gt": list(xyz), "per_cam": per_cam}},
    }


def _entity_frames(
    visible_windows: dict[int, list[tuple[int, float]]],
    all_frames: range,
) -> list[dict[str, Any]]:
    """Frames for one entity. ``visible_windows`` maps camera -> [(frame, u)].

    Row ``v`` is fixed per entity elsewhere; here every visible obs sits at row
    240 (mid-height) so the nearest border is always left/right, never top/bottom.
    """
    seen: dict[int, dict[int, float]] = {}
    for cam, points in visible_windows.items():
        for frame, u in points:
            seen.setdefault(frame, {})[cam] = u
    frames: list[dict[str, Any]] = []
    for f in all_frames:
        per_cam: list[dict[str, Any]] = []
        for cam in (0, 1):
            if f in seen and cam in seen[f]:
                per_cam.append(_obs(cam, seen[f][cam], 240.0))
            else:
                per_cam.append(_hidden(cam))
        frames.append(_frame(f, per_cam, (float(f), 0.0, 0.0)))
    return frames


def build_fixture_manifest() -> dict[str, Any]:
    """A non-overlapping two-camera, two-entity blind-gap rig.

    ``item_a`` exits cam0 (frames 0-2, moving to the right border) and re-enters
    cam1 (frames 7-9, from the left border); ``item_b`` is offset later — exits
    cam0 at frames 2-4 and re-enters cam1 at frames 9-11. Both gaps are ~0.5 s of
    transit, but the staggered exit/entry order is what makes the correct pairing
    (a→a, b→b) strictly more plausible than the cross pairing.
    """
    all_frames = range(0, 12)
    entity_a = _entity_frames(
        {0: [(0, 100.0), (1, 300.0), (2, 560.0)], 1: [(7, 60.0), (8, 300.0), (9, 500.0)]},
        all_frames,
    )
    entity_b = _entity_frames(
        {0: [(2, 120.0), (3, 320.0), (4, 555.0)], 1: [(9, 65.0), (10, 300.0), (11, 505.0)]},
        all_frames,
    )
    return {
        "cameras": [_cam(0), _cam(1)],
        "fps": FPS,
        "num_frames": 12,
        "entities": [
            {"id": "item_a", "frames": entity_a},
            {"id": "item_b", "frames": entity_b},
        ],
    }


def station_topology() -> CameraTopology:
    """station 1 (cam0) right-exit → station 2 (cam1) left-entry, ~0.5 s transit."""
    return CameraTopology(
        cameras=[0, 1],
        edges=[
            TransitionEdge(
                src_camera=0,
                dst_camera=1,
                exit_zone="cam0:right",
                entry_zone="cam1:left",
                transit=TransitDistribution(mean=0.5, std=0.15),
            )
        ],
    )


def test_extractor_yields_non_overlapping_tracklets() -> None:
    """Each entity yields exactly one tracklet per camera; zones are right→left."""
    manifest = build_fixture_manifest()
    tracklets, gt = extract_tracklets(manifest)

    assert len(tracklets) == 4  # 2 entities x 2 cameras, no re-entry within a camera
    by_id = {t.id: t for t in tracklets}

    a0 = by_id["c0:eitem_a:r0"]
    a1 = by_id["c1:eitem_a:r0"]
    assert a0.camera_id == 0 and a1.camera_id == 1
    assert a0.exit_zone == "cam0:right"  # left station 1 on the right
    assert a1.entry_zone == "cam1:left"  # entered station 2 on the left
    # Non-overlap: cam0 run ends before the cam1 run begins (a real blind gap).
    assert a0.exit_time < a1.entry_time
    assert gt["c0:eitem_a:r0"] == "item_a" and gt["c1:eitem_a:r0"] == "item_a"


def test_handoff_keeps_identity_across_blind_gap() -> None:
    """The headline: handoff → IDF1 1.0 / 0 switches; baseline is strictly worse."""
    manifest = build_fixture_manifest()
    tracklets, gt = extract_tracklets(manifest)
    topology = station_topology()

    predicted = assign_global_ids(tracklets, topology, SpatioTemporalMatcher())
    metrics = compute_identity_metrics(tracklets, predicted, gt)

    # Each entity's two per-camera tracklets are stitched into one global id.
    assert predicted["c0:eitem_a:r0"] == predicted["c1:eitem_a:r0"]
    assert predicted["c0:eitem_b:r0"] == predicted["c1:eitem_b:r0"]
    # ...and the two entities stay distinct (no wrong cross-camera match).
    assert predicted["c0:eitem_a:r0"] != predicted["c0:eitem_b:r0"]

    assert metrics.idf1 == 1.0
    assert metrics.id_switches == 0

    # No-handoff baseline: every tracklet its own id -> each entity fractures.
    baseline = per_tracklet_global_ids(tracklets)
    base_metrics = compute_identity_metrics(tracklets, baseline, gt)
    assert base_metrics.idf1 == 0.5
    assert base_metrics.id_switches == 2
    assert metrics.idf1 > base_metrics.idf1


def test_wrong_cross_match_is_less_plausible_than_correct() -> None:
    """The correct pairing outscores the cross pairing — matcher isn't fooled."""
    manifest = build_fixture_manifest()
    tracklets, _ = extract_tracklets(manifest)
    by_id = {t.id: t for t in tracklets}
    topology = station_topology()
    matcher = SpatioTemporalMatcher()

    a0, a1 = by_id["c0:eitem_a:r0"], by_id["c1:eitem_a:r0"]
    b0, b1 = by_id["c0:eitem_b:r0"], by_id["c1:eitem_b:r0"]

    correct = matcher.score(a0, a1, topology)
    cross = matcher.score(a0, b1, topology)
    assert correct is not None and cross is not None
    assert correct > cross
    assert matcher.score(b0, b1, topology) == correct  # symmetric stagger

    # Reverse direction has no topology edge -> hard veto.
    assert matcher.score(a1, a0, topology) is None


def test_appearance_backend_is_pluggable() -> None:
    """A ReID backend swaps in without changing topology/assignment/metrics."""
    manifest = build_fixture_manifest()
    tracklets, gt = extract_tracklets(manifest)
    topology = station_topology()

    matcher = SpatioTemporalMatcher(StubEmbedding(), appearance_weight=0.5)
    predicted = assign_global_ids(tracklets, topology, matcher)
    metrics = compute_identity_metrics(tracklets, predicted, gt)
    # The spatio-temporal gate still carries the fixture; appearance only reweights.
    assert metrics.idf1 == 1.0
    assert metrics.id_switches == 0


def test_idf1_optimal_matching_beats_greedy() -> None:
    """IDF1 is defined by *optimal* identity matching; greedy would under-count.

    Overlap matrix (rows GT, cols pred):

        ==== pX  pY
        gt1   3   2
        gt2   2   0

    Greedy takes the largest cell first (gt1↔pX = 3) then is stuck with
    gt2↔pY = 0, for IDTP = 3. The optimal assignment (gt1↔pY = 2, gt2↔pX = 2)
    gives IDTP = 4. Our Hungarian must return the optimal 4.
    """
    counts: list[tuple[str | None, str | None, int]] = [
        ("gt1", "pX", 3),
        ("gt1", "pY", 2),
        ("gt2", "pX", 2),
        ("gt2", "pY", 0),
    ]
    idtp, idfp, idfn = idf1_from_counts(counts)
    assert idtp == 4  # optimal, not greedy's 3
    total = 3 + 2 + 2 + 0
    assert idfp == total - 4 and idfn == total - 4
    # IDF1 = 2*IDTP / (2*IDTP + IDFP + IDFN) = 8 / 14
    assert abs((2 * idtp) / (2 * idtp + idfp + idfn) - 8 / 14) < 1e-12


def test_idf1_handles_false_positive_and_negative_detections() -> None:
    """None gt marks false positives; None pred marks false negatives."""
    counts: list[tuple[str | None, str | None, int]] = [
        ("g", "p", 5),
        (None, "p", 2),  # 2 predicted obs with no GT -> false positives
        ("g", None, 3),  # 3 GT obs never predicted -> false negatives
    ]
    idtp, idfp, idfn = idf1_from_counts(counts)
    assert (idtp, idfp, idfn) == (5, 2, 3)


def test_fixture_round_trips_through_json(tmp_path: Path) -> None:
    """The hand-authored manifest is valid JSON and loads via the extractor."""
    manifest = build_fixture_manifest()
    path = tmp_path / "stations.json"
    path.write_text(json.dumps(manifest, indent=2))

    loaded = load_manifest(path)
    tracklets, gt = extract_tracklets(loaded)
    assert len(tracklets) == 4
    assert set(gt.values()) == {"item_a", "item_b"}
