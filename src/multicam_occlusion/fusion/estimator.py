"""The fusion estimator: correlate operator actions with assembly changes.

This is the heart of the complementary-fusion mode. Cross-view *triangulation*
is impossible here by construction — the operator action and the item change are
different 3D things seen by different cameras — so recovery is not geometric but
*temporal*: an action at time ``t`` in the operator view should be followed,
within a short causal window, by an assembly change at ``~t`` in the item view.
That correspondence is the joint scene state.

:class:`FusionEstimator` is the pluggable Protocol (mirroring the ``FrameSource``
seam); :class:`TemporalProximityEstimator` is the deterministic default. Its
association rule is three plain constraints — no learning, no RNG:

1. **Causality** — the action precedes the change (``action_time <= change_time``
   within a small epsilon).
2. **Max lag** — the change follows within ``max_lag`` seconds.
3. **One-to-one** — each action pairs with at most one change and vice versa;
   ties break to the *smallest lag* (nearest action to the change).

An optional spatial gate (``max_distance``) additionally requires the action and
change world locations to be close; it is off by default because the thesis is
timing-correlation across complementary views, and the two locations are, in
general, different points (a hand vs an item).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .detectors import (
    ActionDetector,
    CameraRoles,
    DisplacementStateDetector,
    ItemStateDetector,
    ReachActionDetector,
    partition_by_visibility,
)
from .observations import (
    ActionEvent,
    AssemblyChangeEvent,
    FusedInteraction,
    ItemViewObservation,
    JointSceneState,
    OperatorViewObservation,
)
from .scene_manifest import SceneManifest


@runtime_checkable
class FusionEstimator(Protocol):
    """Consumes the two partial views, produces one joint scene state."""

    def fuse(
        self,
        operator: OperatorViewObservation,
        item: ItemViewObservation,
    ) -> JointSceneState:
        """Correlate actions with assembly changes into a joint scene state."""
        ...


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return float(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)) ** 0.5)


class TemporalProximityEstimator:
    """Default :class:`FusionEstimator`: nearest causal action within a lag window.

    Greedy and deterministic. Changes are processed in time order; each change
    takes the still-unused action that precedes it by the smallest non-negative
    lag ``<= max_lag`` (and, if ``max_distance`` is set, is within that distance).
    Unmatched actions and changes are surfaced, not discarded.
    """

    def __init__(
        self,
        max_lag: float = 0.5,
        causal_eps: float = 1e-9,
        max_distance: float | None = None,
    ) -> None:
        if max_lag <= 0.0:
            raise ValueError("max_lag must be positive")
        self.max_lag = max_lag
        self.causal_eps = causal_eps
        self.max_distance = max_distance

    def _plausible(self, action: ActionEvent, change: AssemblyChangeEvent) -> float | None:
        """Return the lag if the pair is admissible, else ``None``."""
        lag = change.time - action.time
        if lag < -self.causal_eps or lag > self.max_lag:
            return None
        if self.max_distance is not None:
            if action.location is None or change.location is None:
                return None
            if _distance(action.location, change.location) > self.max_distance:
                return None
        return max(lag, 0.0)

    def fuse(
        self,
        operator: OperatorViewObservation,
        item: ItemViewObservation,
    ) -> JointSceneState:
        actions = sorted(operator.actions, key=lambda a: a.time)
        changes = sorted(item.changes, key=lambda c: c.time)
        used_action: set[int] = set()
        interactions: list[FusedInteraction] = []
        matched_changes: set[int] = set()

        for ci, change in enumerate(changes):
            best_ai: int | None = None
            best_lag: float | None = None
            for ai, action in enumerate(actions):
                if ai in used_action:
                    continue
                lag = self._plausible(action, change)
                if lag is None:
                    continue
                if best_lag is None or lag < best_lag:
                    best_lag, best_ai = lag, ai
            if best_ai is None or best_lag is None:
                continue
            action = actions[best_ai]
            used_action.add(best_ai)
            matched_changes.add(ci)
            interactions.append(
                FusedInteraction(
                    actor_id=action.actor_id,
                    action_label=action.label,
                    item_id=change.item_id,
                    change=change.change,
                    action_time=action.time,
                    change_time=change.time,
                    lag=best_lag,
                    confidence=1.0 - best_lag / self.max_lag,
                )
            )

        return JointSceneState(
            interactions=interactions,
            unassociated_actions=[a for i, a in enumerate(actions) if i not in used_action],
            unassociated_changes=[c for i, c in enumerate(changes) if i not in matched_changes],
        )


def fuse_scene(
    manifest: SceneManifest,
    roles: CameraRoles,
    *,
    action_detector: ActionDetector | None = None,
    state_detector: ItemStateDetector | None = None,
    estimator: FusionEstimator | None = None,
) -> JointSceneState:
    """End-to-end fusion over a scene manifest.

    Routes entities by asymmetric visibility, extracts partial observations with
    the (pluggable) detectors, then fuses them with the (pluggable) estimator.
    Every stage defaults to its deterministic implementation, so the whole
    pipeline runs from a manifest and a :class:`CameraRoles` alone.

    Assumes a single operator (the assembly-station case). If the router finds
    more than one actor entity, their actions are merged into one operator view.
    """
    action_detector = action_detector or ReachActionDetector()
    state_detector = state_detector or DisplacementStateDetector()
    estimator = estimator or TemporalProximityEstimator()

    partition = partition_by_visibility(manifest, roles)

    actions: list[ActionEvent] = []
    actor_id = partition.actors[0] if partition.actors else "operator"
    for aid in partition.actors:
        actions.extend(action_detector.detect(manifest, aid))
    operator = OperatorViewObservation(
        camera_id=roles.operator_camera, actor_id=actor_id, actions=actions
    )

    changes: list[AssemblyChangeEvent] = []
    for iid in partition.items:
        changes.extend(state_detector.detect(manifest, iid))
    item_view = ItemViewObservation(camera_id=roles.item_camera, changes=changes)

    return estimator.fuse(operator, item_view)
