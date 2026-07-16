"""Order verification: read a fused scene against the order it should fulfil.

This is the operational payoff of the fusion mode. The estimator already answers
*who did what to which item, when*; order verification reads that joint scene
state against the **order** (a bill of materials — the item ids expected to be
assembled) and assigns each line a status:

* **fulfilled** — an expected item was assembled (a fused interaction links an
  operator action to that item's assembly change);
* **missing** — an expected item was never assembled (no interaction, and nothing
  unexpected took its place);
* **wrong** — an expected item was not assembled, but an *unexpected* item was —
  a mis-pick / substitution stands in for the missing line;
* **extra** — an item was assembled that the order never called for, with no
  missing line for it to explain.

Nothing here changes the estimator, the association, or the metric — it is a thin
interpretation over :class:`JointSceneState`. The unit is the item, and the
reading is deterministic given the fused scene.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from .observations import FusedInteraction, JointSceneState


class OrderStatus(StrEnum):
    """Per-item verification outcome against the order."""

    FULFILLED = "fulfilled"
    MISSING = "missing"
    WRONG = "wrong"
    EXTRA = "extra"


class ItemVerification(BaseModel):
    """One line of the verified order.

    ``item_id`` is the order line (for ``EXTRA``, the unexpected item itself).
    ``observed_item`` names what was actually assembled when it differs from the
    expected line (set for ``WRONG``). ``interaction`` is the fused pair that
    justifies the status, when there is one.
    """

    model_config = ConfigDict(frozen=True)

    item_id: str
    status: OrderStatus
    observed_item: str | None = None
    interaction: FusedInteraction | None = None


class OrderVerification(BaseModel):
    """The verified order: one :class:`ItemVerification` per line, plus extras."""

    model_config = ConfigDict(frozen=True)

    lines: list[ItemVerification]

    def ok(self) -> bool:
        """True iff every line is :attr:`OrderStatus.FULFILLED`."""
        return all(line.status is OrderStatus.FULFILLED for line in self.lines)

    def by_status(self, status: OrderStatus) -> list[ItemVerification]:
        """The lines with a given status, in order."""
        return [line for line in self.lines if line.status is status]


def verify_order(state: JointSceneState, order: list[str]) -> OrderVerification:
    """Verify a fused scene against ``order`` (the expected item ids / BOM).

    Fulfilled lines match an assembled item directly. An expected item with no
    assembly, paired with an unexpected assembled item, is a **wrong** (mis-pick)
    line; without such a stand-in it is **missing**. Any assembled item still
    unclaimed after that is an **extra** line. Deterministic in the assembled
    items' first-seen order.
    """
    assembled: dict[str, FusedInteraction] = {}
    for interaction in state.interactions:
        assembled.setdefault(interaction.item_id, interaction)

    order_set = set(order)
    extras_queue = [iid for iid in assembled if iid not in order_set]

    lines: list[ItemVerification] = []
    for expected in order:
        if expected in assembled:
            lines.append(
                ItemVerification(
                    item_id=expected,
                    status=OrderStatus.FULFILLED,
                    interaction=assembled[expected],
                )
            )
        elif extras_queue:
            wrong_item = extras_queue.pop(0)
            lines.append(
                ItemVerification(
                    item_id=expected,
                    status=OrderStatus.WRONG,
                    observed_item=wrong_item,
                    interaction=assembled[wrong_item],
                )
            )
        else:
            lines.append(ItemVerification(item_id=expected, status=OrderStatus.MISSING))

    for leftover in extras_queue:
        lines.append(
            ItemVerification(
                item_id=leftover,
                status=OrderStatus.EXTRA,
                interaction=assembled[leftover],
            )
        )

    return OrderVerification(lines=lines)
