"""Order verification on REAL multicam-sim output (the generated fixture).

``tests/fixtures/sim_assembly_station/{manifest.json,order.json}`` is genuine
multicam-sim producer output — regenerate with ``make fusion-scene`` (drives
multicam-sim; deterministic, no seed). The scene is the shipped
``examples/assembly_station.py`` preset: an overview camera frames the operator,
a worktop camera frames three parts, and the two cameras' per-entity ``visible``
labels are complementary.

These tests drive the order-verification path end-to-end on that generated data:
route by asymmetric visibility, reconstruct assembled contents from the item
camera, and reconcile against the generated order's BOM. The *causal* action↔
change association metric is exercised separately, on a controlled in-memory
scene, in ``tests/test_fusion.py`` — the sim preset does not yet emit operator
action events timestamped to placements (tracked upstream).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from multicam_occlusion.fusion import (
    CameraRoles,
    DisplacementStateDetector,
    OrderBom,
    OrderLine,
    OrderStatus,
    SceneManifest,
    SceneOrder,
    partition_by_visibility,
    reconstruct_assembled,
    verify_assembly,
    verify_order_from_manifest,
)


def _order(**counts: int) -> SceneOrder:
    """A SceneOrder from ``name=count`` kwargs (for direct verify_assembly tests)."""
    lines = [OrderLine(name=n, count=c) for n, c in counts.items()]
    return SceneOrder(order_id="ORD-T", bom=OrderBom(items=lines))


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
