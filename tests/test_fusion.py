"""Complementary-fusion mode: action <-> assembly-change association on a fixture.

The packing-station fixture (``tests/fixtures/packing_station.json``, a
multicam-sim scene manifest) has an asymmetric two-camera rig: camera 0 sees the
worker, camera 1 sees the items. The worker places twice and reaches a third
time (a distractor); a third item moves outside the causal window (a change with
no cause). Ground truth is two interactions. These tests assert the pipeline
recovers exactly those two, refuses both distractors, and that the association
metric measures the outcome — including that a too-wide lag window scores a
false positive.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from multicam_occlusion.fusion import (
    ActionDetector,
    CameraRoles,
    FusionEstimator,
    GroundTruthInteraction,
    SceneManifest,
    TemporalProximityEstimator,
    WorktopStateDetector,
    association_metric,
    fuse_scene,
    partition_by_visibility,
)

FIXTURE = Path(__file__).parent / "fixtures" / "packing_station.json"
ROLES = CameraRoles(human_camera=0, worktop_camera=1)

# Known by construction of the fixture (see build_packing_station.py).
GROUND_TRUTH = [
    GroundTruthInteraction(actor_id="worker", item_id="item_0", action_time=0.5, change_time=0.6),
    GroundTruthInteraction(actor_id="worker", item_id="item_1", action_time=1.3, change_time=1.4),
]


@pytest.fixture
def manifest() -> SceneManifest:
    return SceneManifest.from_json(FIXTURE)


def test_asymmetric_visibility_routes_worker_and_items(manifest: SceneManifest) -> None:
    """The worker routes to the human camera, items to the worktop camera."""
    partition = partition_by_visibility(manifest, ROLES)
    assert partition.actors == ["worker"]
    assert partition.items == ["item_0", "item_1", "item_2"]


def test_default_detectors_are_pluggable_protocols() -> None:
    """The default detectors satisfy their extension Protocols at runtime."""
    from multicam_occlusion.fusion import DisplacementStateDetector, ReachActionDetector

    assert isinstance(ReachActionDetector(), ActionDetector)
    assert isinstance(DisplacementStateDetector(), WorktopStateDetector)
    assert isinstance(TemporalProximityEstimator(), FusionEstimator)


def test_fusion_recovers_two_interactions_and_refuses_distractors(
    manifest: SceneManifest,
) -> None:
    """Two real (action, change) pairs recovered; both distractors refused."""
    state = fuse_scene(manifest, ROLES)

    # Exactly the two real interactions, correctly linked worker->item.
    assert len(state.interactions) == 2
    linked = {
        (i.item_id, round(i.action_time, 3), round(i.change_time, 3)) for i in state.interactions
    }
    assert linked == {("item_0", 0.5, 0.6), ("item_1", 1.3, 1.4)}
    assert all(i.actor_id == "worker" and i.action_label == "place" for i in state.interactions)
    assert all(0.0 < i.lag <= 0.5 and 0.0 < i.confidence <= 1.0 for i in state.interactions)

    # The distractor reach (t=0.9) is refused, not forced into a pair.
    assert [round(a.time, 3) for a in state.unassociated_actions] == [0.9]
    # The out-of-window item move (t=1.9) is refused too.
    assert [(c.item_id, round(c.time, 3)) for c in state.unassociated_changes] == [("item_2", 1.9)]


def test_association_metric_is_perfect_on_the_fixture(manifest: SceneManifest) -> None:
    """Precision/recall are 1.0 — by correctly refusing the traps, not by luck."""
    state = fuse_scene(manifest, ROLES)
    metrics = association_metric(state.interactions, GROUND_TRUTH)

    assert metrics.true_positives == 2
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    # Action/change times land on ground truth; the fused lag is the real 0.1s.
    assert metrics.mean_action_time_error == pytest.approx(0.0, abs=1e-9)
    assert metrics.mean_change_time_error == pytest.approx(0.0, abs=1e-9)
    assert metrics.mean_lag == pytest.approx(0.1, abs=1e-9)


def test_too_wide_lag_window_scores_a_false_positive(manifest: SceneManifest) -> None:
    """A 1.0s window wrongly links the distractor action to item_2 -> precision drops.

    This is the metric doing its job: the extra pairing is a false positive, so
    precision falls to 2/3 while recall stays 1.0. It is what makes the perfect
    score on the default window meaningful.
    """
    state = fuse_scene(manifest, ROLES, estimator=TemporalProximityEstimator(max_lag=1.0))
    metrics = association_metric(state.interactions, GROUND_TRUTH)

    assert len(state.interactions) == 3  # item_2 spuriously associated
    assert metrics.true_positives == 2
    assert metrics.false_positives == 1
    assert metrics.precision == pytest.approx(2.0 / 3.0)
    assert metrics.recall == 1.0
