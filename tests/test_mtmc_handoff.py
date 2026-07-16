"""MTMC handoff proof: keep one identity across a blind gap on a non-overlap rig.

The fixture is **real multicam-sim output** — ``tests/fixtures/mtmc_stations.json``,
produced by ``bench/gen_mtmc_scene.py`` from a genuinely non-overlapping two-station
rig (``multicam_sim.dsl.CameraRig.stations``, disjoint fields of view) and committed so
CI stays numpy-only (multicam-sim is an optional ``bench`` dep, never imported here).
See ``make mtmc-scene`` to regenerate.

Two station cameras — station 1 (``cam0``) and station 2 (``cam1``) — with **no
field-of-view overlap**: each entity is ``visible`` in one camera at a time and
invisible during the blind gap. Two entities cross the gap **staggered** in time, so
spatio-temporal plausibility alone (default no-appearance backend) must disambiguate —
a single-entity fixture would be satisfied by assigning everything one id and prove
nothing.

Headline (measured on the generated scene): with the handoff, IDF1 = 1.0 and 0
ID-switches; the no-handoff baseline scores strictly worse (IDF1 = 0.625, 2 switches).
That contrast is the proof. (The baseline is 0.625, not the hand fixture's old 0.5,
because the real sweep gives the two entities different-length visible runs — reported
honestly rather than tuned to a round number.)
"""

from __future__ import annotations

from pathlib import Path

from multicam_occlusion.mtmc import (
    SpatioTemporalMatcher,
    StubEmbedding,
    assign_global_ids,
    compute_identity_metrics,
    extract_tracklets,
    idf1_from_counts,
    matcher_topology_from_manifest,
    per_tracklet_global_ids,
    read_manifest,
)

FIXTURE = Path(__file__).parent / "fixtures" / "mtmc_stations.json"

# Local, camera-scoped tracklet ids the extractor assigns (one visible run each).
A0, A1 = "c0:eitem_a:r0", "c1:eitem_a:r0"
B0, B1 = "c0:eitem_b:r0", "c1:eitem_b:r0"


def test_fixture_is_real_sim_output() -> None:
    """The committed fixture carries fields only multicam-sim's build_manifest emits."""
    manifest = read_manifest(FIXTURE)
    # A topology block, per-camera in_view, and a convention tag are sim-produced —
    # the old hand-authored fixture had none of them.
    assert manifest.topology is not None
    assert {s.id for s in manifest.topology.stations} == {"station-1", "station-2"}
    first_obs = manifest.entities[0].frames[0].points["center"].per_cam[0]
    assert first_obs.in_view is not None
    assert manifest.cameras[0].convention == "opencv_rdf"


def test_extractor_yields_non_overlapping_tracklets() -> None:
    """Each entity yields exactly one tracklet per camera; zones are right→left."""
    manifest = read_manifest(FIXTURE)
    tracklets, gt = extract_tracklets(manifest)

    assert len(tracklets) == 4  # 2 entities x 2 cameras, no re-entry within a camera
    by_id = {t.id: t for t in tracklets}

    a0, a1 = by_id[A0], by_id[A1]
    assert a0.camera_id == 0 and a1.camera_id == 1
    assert a0.exit_zone == "cam0:right"  # left station 1 on the right
    assert a1.entry_zone == "cam1:left"  # entered station 2 on the left
    # Non-overlap: cam0 run ends before the cam1 run begins (a real blind gap).
    assert a0.exit_time < a1.entry_time
    assert gt[A0] == "item_a" and gt[A1] == "item_a"

    # Both entities exit station 1 on the right and enter station 2 on the left, so a
    # single topology edge carries both handoffs (checked below).
    assert by_id[B0].exit_zone == "cam0:right"
    assert by_id[B1].entry_zone == "cam1:left"


def test_topology_derives_from_manifest() -> None:
    """The matcher graph is built from the manifest's own stations + transit times.

    Guards the silent-veto trap: if the derived edge's zones didn't match the
    extractor's, every score would be ``None`` and IDF1 would collapse to the
    baseline while *looking* like the handoff simply didn't help.
    """
    manifest = read_manifest(FIXTURE)
    tracklets, _ = extract_tracklets(manifest)
    topology = matcher_topology_from_manifest(manifest, tracklets)

    assert topology.cameras == [0, 1]
    # The forward handoff edge exists with exactly the zones the extractor produced.
    forward = topology.edge_for(0, "cam0:right", 1, "cam1:left")
    assert forward is not None
    assert forward.transit.mean == 0.5  # the manifest's transit_time_s
    # No spurious *forward* edge from a zone no cam0 tracklet actually exited through.
    assert topology.edge_for(0, "cam0:left", 1, "cam1:left") is None


def test_handoff_keeps_identity_across_blind_gap() -> None:
    """The headline: handoff → IDF1 1.0 / 0 switches; baseline is strictly worse."""
    manifest = read_manifest(FIXTURE)
    tracklets, gt = extract_tracklets(manifest)
    topology = matcher_topology_from_manifest(manifest, tracklets)

    predicted = assign_global_ids(tracklets, topology, SpatioTemporalMatcher())
    metrics = compute_identity_metrics(tracklets, predicted, gt)

    # Each entity's two per-camera tracklets are stitched into one global id.
    assert predicted[A0] == predicted[A1]
    assert predicted[B0] == predicted[B1]
    # ...and the two entities stay distinct (no wrong cross-camera match).
    assert predicted[A0] != predicted[B0]

    assert metrics.idf1 == 1.0
    assert metrics.id_switches == 0

    # No-handoff baseline: every tracklet its own id -> each entity fractures.
    baseline = per_tracklet_global_ids(tracklets)
    base_metrics = compute_identity_metrics(tracklets, baseline, gt)
    assert base_metrics.idf1 == 0.625
    assert base_metrics.id_switches == 2
    assert metrics.idf1 > base_metrics.idf1


def test_wrong_cross_match_is_less_plausible_than_correct() -> None:
    """The correct pairing outscores the cross pairing — matcher isn't fooled."""
    manifest = read_manifest(FIXTURE)
    tracklets, _ = extract_tracklets(manifest)
    by_id = {t.id: t for t in tracklets}
    topology = matcher_topology_from_manifest(manifest, tracklets)
    matcher = SpatioTemporalMatcher()

    a0, a1 = by_id[A0], by_id[A1]
    b0, b1 = by_id[B0], by_id[B1]

    correct = matcher.score(a0, a1, topology)
    cross = matcher.score(a0, b1, topology)
    assert correct is not None and cross is not None
    assert correct > cross
    assert matcher.score(b0, b1, topology) == correct  # symmetric stagger

    # Reverse direction is vetoed: the gap time would be negative (effect before cause).
    assert matcher.score(a1, a0, topology) is None


def test_appearance_backend_is_pluggable() -> None:
    """A ReID backend swaps in without changing topology/assignment/metrics."""
    manifest = read_manifest(FIXTURE)
    tracklets, gt = extract_tracklets(manifest)
    topology = matcher_topology_from_manifest(manifest, tracklets)

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
