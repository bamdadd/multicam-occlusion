"""Both fusion metrics on REAL multicam-sim output (the generated fixture).

``tests/fixtures/sim_assembly_station/{manifest.json,order.json}`` is genuine
multicam-sim producer output — regenerate with ``make fusion-scene`` (drives
multicam-sim; deterministic, no seed). The scene is the shipped
``examples/assembly_station.py`` preset: an overview camera frames the operator,
a worktop camera frames three parts, complementary per-entity ``visible`` labels,
and — since multicam-sim commit 3199ee4 — ``order.json`` carries placement-synced
operator ``actions[]``.

Both halves of the mode now run end-to-end on this generated data:

* **order verification** — route by asymmetric visibility, reconstruct assembled
  contents from the item camera, reconcile against the order's BOM;
* **causal action↔change association** — consume the operator ``actions[]`` from
  ``order.json``, pair each against the item's assembly change detected from the
  manifest, and score precision/recall/timing against the producer's declared
  ground truth.

The controlled in-memory scene in ``tests/test_fusion.py`` is retained only as a
distractor stress test of the estimator logic (refusing a reach that assembles
nothing, a change outside the lag window) — behaviours this clean generated scene
does not contain.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from multicam_occlusion.fusion import (
    CameraRoles,
    DisplacementStateDetector,
    OrderStatus,
    SceneManifest,
    SceneOrder,
    association_metric,
    fuse_order_actions,
    ground_truth_from_order,
    operator_actions_from_order,
    partition_by_visibility,
    reconstruct_assembled,
    verify_assembly,
    verify_order_from_manifest,
)


def _order(**counts: int) -> SceneOrder:
    """A SceneOrder from ``name=count`` kwargs (for direct verify_assembly tests)."""
    return SceneOrder(order_id="ORD-T", expected=dict(counts))


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "sim_assembly_station"
MANIFEST = FIXTURE_DIR / "manifest.json"
ORDER = FIXTURE_DIR / "order.json"
ROLES = CameraRoles(operator_camera=0, item_camera=1)


@pytest.fixture
def manifest() -> SceneManifest:
    return SceneManifest.from_json(MANIFEST)


@pytest.fixture
def order() -> SceneOrder:
    return SceneOrder.from_json(ORDER)


def test_generated_manifest_has_complementary_visibility(manifest: SceneManifest) -> None:
    """The generated scene really is asymmetric: operator in cam 0, parts in cam 1."""
    partition = partition_by_visibility(manifest, ROLES)
    assert partition.actors == ["operator"]
    assert partition.items == ["part_a", "part_b", "part_c"]


def test_order_reader_reads_generated_bom(order: SceneOrder) -> None:
    """The order.json reader recovers the generated bill of materials."""
    assert order.order_id == "ORD-1"
    assert order.expected_counts() == {"part_a": 1, "part_b": 1, "part_c": 1}


def test_assembled_contents_reconstructed_from_item_camera(manifest: SceneManifest) -> None:
    """Each part's placement into the container is recovered from the item camera."""
    assembled = reconstruct_assembled(manifest, ROLES)
    assert assembled == {"part_a": 1, "part_b": 1, "part_c": 1}


def test_order_verifies_fulfilled_on_generated_scene(
    manifest: SceneManifest, order: SceneOrder
) -> None:
    """End-to-end on generated data: all three ordered parts assembled -> fulfilled."""
    verified = verify_order_from_manifest(manifest, order, ROLES)

    assert verified.ok()
    assert [(line.item_id, line.status) for line in verified.lines] == [
        ("part_a", OrderStatus.FULFILLED),
        ("part_b", OrderStatus.FULFILLED),
        ("part_c", OrderStatus.FULFILLED),
    ]
    assert all(line.expected == 1 and line.assembled == 1 for line in verified.lines)


def test_missing_line_when_a_part_is_never_placed(
    manifest: SceneManifest, order: SceneOrder
) -> None:
    """A too-strict detector sees no placements -> every ordered line is missing.

    Verifies the reconciliation reports shortfalls honestly rather than silently
    passing; the ``min_step`` here is larger than any real inter-frame move.
    """
    blind = DisplacementStateDetector(min_step=100.0)
    verified = verify_order_from_manifest(manifest, order, ROLES, state_detector=blind)

    assert not verified.ok()
    assert [line.status for line in verified.lines] == [OrderStatus.MISSING] * 3
    assert all(line.assembled == 0 for line in verified.lines)


def test_verify_assembly_flags_foreign_item_as_wrong() -> None:
    """An assembled item the order never listed is WRONG (quantity-aware path)."""
    verified = verify_assembly({"part_a": 1, "part_x": 1}, _order(part_a=1))

    lines = {line.item_id: line for line in verified.lines}
    assert lines["part_a"].status is OrderStatus.FULFILLED
    assert lines["part_x"].status is OrderStatus.WRONG
    assert lines["part_x"].expected == 0 and lines["part_x"].assembled == 1
    assert not verified.ok()


def test_verify_assembly_flags_surplus_of_expected_item_as_extra() -> None:
    """More of an expected item than ordered is EXTRA, not WRONG."""
    verified = verify_assembly({"part_a": 2}, _order(part_a=1))

    (line,) = verified.lines
    assert line.status is OrderStatus.EXTRA
    assert line.expected == 1 and line.assembled == 2
    assert verified.by_status(OrderStatus.EXTRA) == [line]


# --- causal action<->change association on the generated actions[] ---------- #


def test_order_reader_reads_synced_operator_actions(order: SceneOrder) -> None:
    """The reader recovers the placement-synced operator actions from order.json."""
    assert [(a.item_id, a.frame, a.action) for a in order.actions] == [
        ("part_a", 2, "place"),
        ("part_b", 5, "place"),
        ("part_c", 8, "place"),
    ]
    assert all(a.entity_id == "operator" and a.hand_joint == "right_wrist" for a in order.actions)


def test_operator_actions_carry_hand_position_and_time(
    manifest: SceneManifest, order: SceneOrder
) -> None:
    """Each converted ActionEvent is timestamped in seconds and keeps the hand xyz."""
    actions = operator_actions_from_order(order, manifest)
    # fps is 30 in the generated scene: t = frame / 30.
    assert [round(a.time, 4) for a in actions] == [round(f / 30.0, 4) for f in (2, 5, 8)]
    assert all(a.actor_id == "operator" and a.label == "place" for a in actions)
    assert all(a.location is not None for a in actions)


def test_causal_fusion_associates_every_action_on_generated_data(
    manifest: SceneManifest, order: SceneOrder
) -> None:
    """Each operator action is linked to the part it placed; nothing is refused."""
    state = fuse_order_actions(manifest, order, ROLES)

    assert len(state.interactions) == 3
    linked = {(i.actor_id, i.item_id) for i in state.interactions}
    assert linked == {("operator", "part_a"), ("operator", "part_b"), ("operator", "part_c")}
    assert all(i.action_label == "place" for i in state.interactions)
    assert all(i.lag == pytest.approx(0.0) for i in state.interactions)
    assert state.unassociated_actions == []
    assert state.unassociated_changes == []


def test_association_metric_is_perfect_on_generated_scene(
    manifest: SceneManifest, order: SceneOrder
) -> None:
    """Precision/recall/f1 = 1.0 with zero timing error, scored on real data."""
    state = fuse_order_actions(manifest, order, ROLES)
    metrics = association_metric(state.interactions, ground_truth_from_order(order, manifest))

    assert (metrics.true_positives, metrics.false_positives, metrics.false_negatives) == (3, 0, 0)
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.mean_action_time_error == pytest.approx(0.0, abs=1e-9)
    assert metrics.mean_change_time_error == pytest.approx(0.0, abs=1e-9)
    assert metrics.mean_lag == pytest.approx(0.0, abs=1e-9)
