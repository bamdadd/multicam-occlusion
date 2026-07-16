"""Order verification: read a fused scene against the order it should fulfil.

These tests drive :func:`verify_order` with hand-built joint scene states so each
status — fulfilled, missing, wrong (substitution), extra — is exercised on its
own, independent of the detectors and estimator.
"""

from __future__ import annotations

from multicam_occlusion.fusion import (
    FusedInteraction,
    JointSceneState,
    OrderStatus,
    verify_order,
)


def _interaction(item_id: str) -> FusedInteraction:
    """A minimal fused interaction assembling ``item_id`` (times are placeholders)."""
    return FusedInteraction(
        actor_id="operator",
        action_label="place",
        item_id=item_id,
        change="placed",
        action_time=0.0,
        change_time=0.1,
        lag=0.1,
        confidence=0.8,
    )


def _state(*item_ids: str) -> JointSceneState:
    return JointSceneState(
        interactions=[_interaction(i) for i in item_ids],
        unassociated_actions=[],
        unassociated_changes=[],
    )


def test_fulfilled_and_missing() -> None:
    """An assembled line is fulfilled; an unmet line with no stand-in is missing."""
    verified = verify_order(_state("part_a"), ["part_a", "part_b"])
    assert not verified.ok()
    assert [(line.item_id, line.status) for line in verified.lines] == [
        ("part_a", OrderStatus.FULFILLED),
        ("part_b", OrderStatus.MISSING),
    ]


def test_wrong_is_a_substitution() -> None:
    """An unmet line with an unexpected item assembled in its place is WRONG."""
    verified = verify_order(_state("part_a", "part_x"), ["part_a", "part_b"])
    statuses = {line.item_id: line for line in verified.lines}

    assert statuses["part_a"].status is OrderStatus.FULFILLED
    wrong = statuses["part_b"]
    assert wrong.status is OrderStatus.WRONG
    assert wrong.observed_item == "part_x"
    # The substitution consumed the unexpected item; it is not also an EXTRA.
    assert verified.by_status(OrderStatus.EXTRA) == []


def test_extra_when_nothing_is_missing() -> None:
    """An unexpected item with no missing line to explain it is EXTRA."""
    verified = verify_order(_state("part_a", "part_x"), ["part_a"])
    assert [(line.item_id, line.status) for line in verified.lines] == [
        ("part_a", OrderStatus.FULFILLED),
        ("part_x", OrderStatus.EXTRA),
    ]


def test_empty_order_is_trivially_ok() -> None:
    """No order lines: nothing to fulfil, nothing assembled -> ok."""
    verified = verify_order(_state(), [])
    assert verified.ok()
    assert verified.lines == []
