"""Order verification driven by the *item camera* of a real scene manifest.

This is the half of the fusion mode that runs end-to-end on **generated
multicam-sim output** today. Where :func:`verification.verify_order` reads a
*fused* joint scene state (operator actions linked in time to item changes), this
path reconstructs the **assembled contents** directly from the item-camera view
of a scene manifest — no operator action stream required — and reconciles them
against the order's bill of materials.

The reconstruction is genuinely camera-derived, in two steps that mirror the rest
of the mode:

1. :func:`partition_by_visibility` routes each entity by its per-camera
   ``visible`` labels; the items are exactly the entities seen in the item camera
   but not the operator camera.
2. an :class:`ItemStateDetector` (default :class:`DisplacementStateDetector`)
   reads each item entity's trajectory and counts its **placements** — a part
   moving into the container is one unit assembled.

The result is compared, quantity-aware, against the order (read from the
generated ``order.json``) into a per-item status: ``fulfilled`` / ``missing`` /
``wrong`` / ``extra``.

The *causal* action↔change association metric is a separate concern: it needs an
operator whose reaches are timestamped to placements, which the current
multicam-sim assembly-station preset does not emit — so that metric stays on a
controlled in-memory scene (see ``docs/fusion-design.md`` and the linked
multicam-sim issue). Routing + item detection + order reconciliation, though,
run on real generated data here.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

from .detectors import (
    CameraRoles,
    DisplacementStateDetector,
    ItemStateDetector,
    partition_by_visibility,
)
from .scene_manifest import SceneManifest
from .verification import OrderStatus


class OrderLine(BaseModel):
    """One expected item and how many the order needs (a BOM line)."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    name: str
    count: int

    @field_validator("count")
    @classmethod
    def _positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("OrderLine count must be >= 1")
        return value


class OrderBom(BaseModel):
    """The order's bill of materials: the item lines it expects."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    items: list[OrderLine]


class SceneOrder(BaseModel):
    """A generated ``order.json``: an order id plus its bill of materials.

    Mirrors the multicam-sim ``order.json`` schema (``{order_id, bom: {items:
    [{name, count}]}}``) as read-only pydantic; unknown fields are ignored so the
    reader keeps working as the producer's schema grows.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    order_id: str
    bom: OrderBom

    def expected_counts(self) -> dict[str, int]:
        """Expected quantity per item id (aggregated over BOM lines)."""
        counts: Counter[str] = Counter()
        for line in self.bom.items:
            counts[line.name] += line.count
        return dict(counts)

    @classmethod
    def from_json(cls, path: str | Path) -> SceneOrder:
        """Load and validate a generated order from a JSON file."""
        return cls.model_validate(json.loads(Path(path).read_text()))


class AssemblyLine(BaseModel):
    """One reconciled order line: expected vs assembled counts and the verdict.

    For a ``wrong`` line ``item_id`` is the foreign item that was assembled but
    never ordered (``expected`` is then 0).
    """

    model_config = ConfigDict(frozen=True)

    item_id: str
    status: OrderStatus
    expected: int
    assembled: int


class AssemblyVerification(BaseModel):
    """The reconciled order: one :class:`AssemblyLine` per item, expected or foreign."""

    model_config = ConfigDict(frozen=True)

    lines: list[AssemblyLine]

    def ok(self) -> bool:
        """True iff every line is :attr:`OrderStatus.FULFILLED`."""
        return all(line.status is OrderStatus.FULFILLED for line in self.lines)

    def by_status(self, status: OrderStatus) -> list[AssemblyLine]:
        """The lines with a given status, in order."""
        return [line for line in self.lines if line.status is status]


def reconstruct_assembled(
    manifest: SceneManifest,
    roles: CameraRoles,
    *,
    state_detector: ItemStateDetector | None = None,
) -> dict[str, int]:
    """Reconstruct assembled contents from the item camera of ``manifest``.

    Routes entities by asymmetric visibility, then counts each item entity's
    placement events (a move into the container = one unit assembled) with the
    (pluggable) item-state detector. Items with no placement are absent from the
    result. Deterministic given the manifest and detector.
    """
    detector = state_detector or DisplacementStateDetector()
    partition = partition_by_visibility(manifest, roles)
    assembled: dict[str, int] = {}
    for item_id in partition.items:
        placements = len(detector.detect(manifest, item_id))
        if placements:
            assembled[item_id] = placements
    return assembled


def verify_assembly(assembled: dict[str, int], order: SceneOrder) -> AssemblyVerification:
    """Reconcile assembled counts against the order's BOM, quantity-aware.

    Per item id, comparing assembled count ``got`` to expected count ``want``:

    * ``got == want`` -> **fulfilled**;
    * ``got  < want`` -> **missing** (short by ``want - got``);
    * ``got  > want`` -> **extra** (surplus of an expected item);
    * an assembled item the order never listed -> **wrong** (a foreign item).

    Expected lines come first (in BOM order), foreign items after (sorted), so the
    verdict is deterministic.
    """
    expected = order.expected_counts()
    lines: list[AssemblyLine] = []
    for name, want in expected.items():
        got = assembled.get(name, 0)
        if got == want:
            status = OrderStatus.FULFILLED
        elif got < want:
            status = OrderStatus.MISSING
        else:
            status = OrderStatus.EXTRA
        lines.append(AssemblyLine(item_id=name, status=status, expected=want, assembled=got))
    for name in sorted(assembled):
        if name not in expected:
            lines.append(
                AssemblyLine(
                    item_id=name, status=OrderStatus.WRONG, expected=0, assembled=assembled[name]
                )
            )
    return AssemblyVerification(lines=lines)


def verify_order_from_manifest(
    manifest: SceneManifest,
    order: SceneOrder,
    roles: CameraRoles,
    *,
    state_detector: ItemStateDetector | None = None,
) -> AssemblyVerification:
    """End-to-end order verification over a real scene manifest.

    Reconstructs assembled contents from the item camera (:func:`reconstruct_assembled`)
    and reconciles them against the order (:func:`verify_assembly`). This is the
    fusion mode's order-verification path running on generated multicam-sim output.
    """
    assembled = reconstruct_assembled(manifest, roles, state_detector=state_detector)
    return verify_assembly(assembled, order)
