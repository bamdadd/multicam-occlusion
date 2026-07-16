"""Distractor stress test of the causal estimator (controlled in-memory scene).

Both fusion metrics now run end-to-end on **real** multicam-sim output — see
``tests/test_fusion_sim.py`` (order verification *and* causal action↔change
association on the generated ``actions[]``). That generated scene is clean: three
actions, three changes, all matched. This module is retained to exercise what the
clean scene deliberately does not contain — the estimator's *refusal* behaviour —
on a CONTROLLED, hand-authored in-memory scene
(``tests/controlled_assembly_scene.py``), not on sim output. It tests estimator
logic, not sim integration.

The controlled scene is an asymmetric two-camera rig: camera 0 sees the operator,
camera 1 sees the parts. The operator places twice and reaches a third time (a
distractor); a third part moves outside the causal window (a change with no
cause). Ground truth is two interactions — the order is ``part_a`` + ``part_b``.
These tests assert the pipeline recovers exactly those two, refuses both
distractors, that the association metric measures the outcome (including a false
positive under a too-wide lag window), and that order verification reads the
fused scene back against the order.

Here the operator actions are *detected* from the trajectory (``ReachActionDetector``
via ``fuse_scene``); the real-data path instead consumes the producer's
placement-synced ``actions[]`` directly.
"""

from __future__ import annotations

import pytest

from controlled_assembly_scene import build as build_controlled_scene
from multicam_occlusion.fusion import (
    ActionDetector,
    CameraRoles,
    FusionEstimator,
    GroundTruthInteraction,
    ItemStateDetector,
    OrderStatus,
    SceneManifest,
    TemporalProximityEstimator,
    association_metric,
    fuse_scene,
    partition_by_visibility,
    verify_order,
)

ROLES = CameraRoles(operator_camera=0, item_camera=1)
ORDER = ["part_a", "part_b"]

# Known by construction of the controlled scene (see controlled_assembly_scene.py).
GROUND_TRUTH = [
    GroundTruthInteraction(actor_id="operator", item_id="part_a", action_time=0.5, change_time=0.6),
    GroundTruthInteraction(actor_id="operator", item_id="part_b", action_time=1.3, change_time=1.4),
]


@pytest.fixture
def manifest() -> SceneManifest:
    return SceneManifest.model_validate(build_controlled_scene())


def test_asymmetric_visibility_routes_operator_and_parts(manifest: SceneManifest) -> None:
    """The operator routes to the operator camera, parts to the item camera."""
    partition = partition_by_visibility(manifest, ROLES)
    assert partition.actors == ["operator"]
    assert partition.items == ["part_a", "part_b", "part_c"]


def test_default_detectors_are_pluggable_protocols() -> None:
    """The default detectors satisfy their extension Protocols at runtime."""
    from multicam_occlusion.fusion import DisplacementStateDetector, ReachActionDetector

    assert isinstance(ReachActionDetector(), ActionDetector)
    assert isinstance(DisplacementStateDetector(), ItemStateDetector)
    assert isinstance(TemporalProximityEstimator(), FusionEstimator)


def test_fusion_recovers_two_interactions_and_refuses_distractors(
    manifest: SceneManifest,
) -> None:
    """Two real (action, change) pairs recovered; both distractors refused."""
    state = fuse_scene(manifest, ROLES)

    # Exactly the two real interactions, correctly linked operator->part.
    assert len(state.interactions) == 2
    linked = {
        (i.item_id, round(i.action_time, 3), round(i.change_time, 3)) for i in state.interactions
    }
    assert linked == {("part_a", 0.5, 0.6), ("part_b", 1.3, 1.4)}
    assert all(i.actor_id == "operator" and i.action_label == "place" for i in state.interactions)
    assert all(0.0 < i.lag <= 0.5 and 0.0 < i.confidence <= 1.0 for i in state.interactions)

    # The distractor reach (t=0.9) is refused, not forced into a pair.
    assert [round(a.time, 3) for a in state.unassociated_actions] == [0.9]
    # The out-of-window part move (t=1.9) is refused too.
    assert [(c.item_id, round(c.time, 3)) for c in state.unassociated_changes] == [("part_c", 1.9)]


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
    """A 1.0s window wrongly links the distractor action to part_c -> precision drops.

    This is the metric doing its job: the extra pairing is a false positive, so
    precision falls to 2/3 while recall stays 1.0. It is what makes the perfect
    score on the default window meaningful.
    """
    state = fuse_scene(manifest, ROLES, estimator=TemporalProximityEstimator(max_lag=1.0))
    metrics = association_metric(state.interactions, GROUND_TRUTH)

    assert len(state.interactions) == 3  # part_c spuriously associated
    assert metrics.true_positives == 2
    assert metrics.false_positives == 1
    assert metrics.precision == pytest.approx(2.0 / 3.0)
    assert metrics.recall == 1.0


def test_order_verification_marks_the_order_fulfilled(manifest: SceneManifest) -> None:
    """Verifying the fused scene against the order: both lines fulfilled."""
    state = fuse_scene(manifest, ROLES)
    verified = verify_order(state, ORDER)

    assert verified.ok()
    assert [(line.item_id, line.status) for line in verified.lines] == [
        ("part_a", OrderStatus.FULFILLED),
        ("part_b", OrderStatus.FULFILLED),
    ]


def test_order_verification_flags_extra_when_lag_window_is_too_wide(
    manifest: SceneManifest,
) -> None:
    """The spurious part_c interaction surfaces as an EXTRA line, not silently."""
    state = fuse_scene(manifest, ROLES, estimator=TemporalProximityEstimator(max_lag=1.0))
    verified = verify_order(state, ORDER)

    assert not verified.ok()
    extras = verified.by_status(OrderStatus.EXTRA)
    assert [e.item_id for e in extras] == ["part_c"]
