"""Partial per-camera observations and the fused joint scene state.

The complementary-fusion mode rests on a distinction the geometry cannot erase:
in an asymmetric rig two cameras see *different aspects of the same event*. The
north camera sees the **worker** (an action unfolding); the east camera, close
to the worktop, sees the **items** (an assembly state changing). Neither view
alone answers *who did what to which item* — you cannot triangulate a human
action against an object, because they are not the same 3D point. You **fuse**.

These models are the typed vocabulary of that fusion:

* an :class:`ActionEvent` — a labelled thing the worker did, at a time, from the
  human-view camera; bundled per actor in a :class:`HumanViewObservation`.
* an :class:`AssemblyChangeEvent` — an item's state changing, at a time, from the
  worktop-view camera; bundled in a :class:`WorktopViewObservation`.
* a :class:`FusedInteraction` — the correlation the estimator produces: *this
  action caused that change*, with the timing lag between them.
* a :class:`JointSceneState` — the full result: confirmed interactions plus the
  actions and changes left deliberately unassociated (the plausibility-reject
  path, which is what keeps the association metric honest).

:class:`GroundTruthInteraction` is the same (action, change) pairing known by
construction of a fixture; the metric scores predicted interactions against it.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

Vec3 = tuple[float, float, float]


class ActionEvent(BaseModel):
    """A worker action seen in the human-view camera, timestamped in seconds.

    ``location`` is the actor keypoint's world coordinate at the action frame
    (from ``xyz_gt``); it is optional context for an estimator that wants a
    spatial plausibility gate, never required for temporal fusion.
    """

    model_config = ConfigDict(frozen=True)

    actor_id: str
    label: str
    frame: int
    time: float = Field(description="seconds = frame / fps")
    location: Vec3 | None = None


class HumanViewObservation(BaseModel):
    """One human-view camera's partial read of the scene: an actor's actions."""

    model_config = ConfigDict(frozen=True)

    camera_id: int
    actor_id: str
    actions: list[ActionEvent]


class AssemblyChangeEvent(BaseModel):
    """An item's assembly state changing, seen in the worktop-view camera."""

    model_config = ConfigDict(frozen=True)

    item_id: str
    change: str = Field(description="e.g. 'placed', 'moved', 'assembled'")
    frame: int
    time: float = Field(description="seconds = frame / fps")
    location: Vec3 | None = None


class WorktopViewObservation(BaseModel):
    """One worktop-view camera's partial read: the item state-changes it saw."""

    model_config = ConfigDict(frozen=True)

    camera_id: int
    changes: list[AssemblyChangeEvent]


class FusedInteraction(BaseModel):
    """A correlated (action -> change) pair: who did what to which item, when.

    ``lag`` is ``change_time - action_time`` (the delay between the worker's
    action in one view and the item's state change in the other); ``confidence``
    is in ``(0, 1]`` and decays with lag.
    """

    model_config = ConfigDict(frozen=True)

    actor_id: str
    action_label: str
    item_id: str
    change: str
    action_time: float
    change_time: float
    lag: float
    confidence: float


class JointSceneState(BaseModel):
    """The fused scene: confirmed interactions plus what stayed unassociated.

    Keeping the leftovers explicit is deliberate — an action with no resulting
    change (or a change outside the causal window) must be *refused*, not forced
    into a spurious pair. That refusal is exactly what a precision score should
    reward.
    """

    model_config = ConfigDict(frozen=True)

    interactions: list[FusedInteraction]
    unassociated_actions: list[ActionEvent]
    unassociated_changes: list[AssemblyChangeEvent]


class GroundTruthInteraction(BaseModel):
    """A known (action, change) pairing, by construction of a fixture scene."""

    model_config = ConfigDict(frozen=True)

    actor_id: str
    item_id: str
    action_time: float
    change_time: float
