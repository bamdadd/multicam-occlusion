"""Read a generated ``order.json`` and run both fusion metrics on real sim output.

This is where the fusion mode meets **generated multicam-sim output** end-to-end,
for BOTH of its metrics:

* **Order verification** — reconstruct the assembled contents from the item
  camera of the scene manifest and reconcile them, quantity-aware, against the
  order's expected counts: ``fulfilled`` / ``missing`` / ``wrong`` / ``extra``.
* **Causal action↔change association** — consume the operator ``actions[]`` the
  producer now emits in ``order.json`` (one placement-synced ``place`` event per
  item, carrying the operator's hand-joint world position), pair each against the
  item's assembly change detected from the manifest, and score the association.

The ``order.json`` schema is multicam-sim's ``OrderResult`` sidecar
(``{status, expected, placed, missing, extra, wrong, order_id, actions}``); the
byte-golden manifest never carries the actions. This reader mirrors it as
read-only pydantic (no loose dicts), ignoring unknown fields so it keeps working
as the producer's schema grows. :func:`SceneOrder.expected_counts` gives the BOM
counts; :attr:`SceneOrder.actions` gives the synced operator actions.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .detectors import (
    CameraRoles,
    DisplacementStateDetector,
    ItemStateDetector,
    partition_by_visibility,
)
from .estimator import FusionEstimator, TemporalProximityEstimator
from .observations import (
    ActionEvent,
    AssemblyChangeEvent,
    GroundTruthInteraction,
    ItemViewObservation,
    JointSceneState,
    OperatorViewObservation,
)
from .scene_manifest import SceneManifest
from .verification import OrderStatus

Vec3 = tuple[float, float, float]


class SceneActionEvent(BaseModel):
    """One operator action from a generated ``order.json`` ``actions[]`` entry.

    Mirrors multicam-sim's ``ActionEvent``: a ``place`` action synced to an item's
    placement ``frame``, carrying the operator's ``hand_joint`` world position at
    that frame. This is the placement-synced operator signal the causal-fusion
    metric correlates against each item's assembly change.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    frame: int
    action: str = "place"
    item_id: str
    entity_id: str
    hand_joint: str = "right_wrist"
    hand_position: Vec3


class SceneOrder(BaseModel):
    """A generated ``order.json`` (multicam-sim ``OrderResult`` sidecar).

    Read-only pydantic over ``{expected, order_id, actions, ...}``; unknown fields
    (``status`` / ``placed`` / ``missing`` / ``extra`` / ``wrong``) are ignored.
    ``expected`` is the bill of materials as ``{item: count}``; ``actions`` are the
    placement-synced operator events for causal fusion.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    expected: dict[str, int]
    order_id: str | None = None
    actions: list[SceneActionEvent] = []

    def expected_counts(self) -> dict[str, int]:
        """Expected quantity per item id (the order's BOM)."""
        return dict(self.expected)

    @classmethod
    def from_json(cls, path: str | Path) -> SceneOrder:
        """Load and validate a generated order sidecar from a JSON file."""
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


# --- causal action<->change association on real generated data ------------- #


def operator_actions_from_order(order: SceneOrder, manifest: SceneManifest) -> list[ActionEvent]:
    """Convert the generated ``actions[]`` into fusion :class:`ActionEvent`s.

    Each producer :class:`SceneActionEvent` becomes an operator ``ActionEvent``
    timestamped in seconds (``frame / fps`` via the manifest) and carrying the
    operator's hand-joint world position as its ``location``. This is the operator
    action stream — sourced from ``order.json``, not detected from geometry — that
    the estimator correlates with the item camera's assembly changes.
    """
    return [
        ActionEvent(
            actor_id=a.entity_id,
            label=a.action,
            frame=a.frame,
            time=manifest.time_of(a.frame),
            location=a.hand_position,
        )
        for a in order.actions
    ]


def ground_truth_from_order(
    order: SceneOrder, manifest: SceneManifest
) -> list[GroundTruthInteraction]:
    """The known (action, change) pairs, by construction of the generated scene.

    Each emitted action is synced to its item's placement frame, so the ground
    truth pairs that action's item and time with the assembly change at the same
    frame. This is the producer's declared truth the association metric scores
    the estimator against.
    """
    return [
        GroundTruthInteraction(
            actor_id=a.entity_id,
            item_id=a.item_id,
            action_time=manifest.time_of(a.frame),
            change_time=manifest.time_of(a.frame),
        )
        for a in order.actions
    ]


def fuse_order_actions(
    manifest: SceneManifest,
    order: SceneOrder,
    roles: CameraRoles,
    *,
    state_detector: ItemStateDetector | None = None,
    estimator: FusionEstimator | None = None,
) -> JointSceneState:
    """Causal fusion on real generated data: order ``actions[]`` × manifest changes.

    Routes entities by asymmetric visibility, takes the operator action stream
    from the generated ``order.json`` (not from a geometry detector), detects each
    item's assembly change from the manifest, and correlates the two with the
    (pluggable) estimator. Unlike :func:`estimator.fuse_scene` — which *derives*
    actions from the operator trajectory — this consumes the producer's
    placement-synced action ground truth directly.
    """
    detector = state_detector or DisplacementStateDetector()
    est = estimator or TemporalProximityEstimator()
    partition = partition_by_visibility(manifest, roles)

    actions = operator_actions_from_order(order, manifest)
    actor_id = actions[0].actor_id if actions else (partition.actors[0] if partition.actors else "")
    operator = OperatorViewObservation(
        camera_id=roles.operator_camera, actor_id=actor_id, actions=actions
    )

    changes: list[AssemblyChangeEvent] = []
    for item_id in partition.items:
        changes.extend(detector.detect(manifest, item_id))
    item_view = ItemViewObservation(camera_id=roles.item_camera, changes=changes)

    return est.fuse(operator, item_view)
