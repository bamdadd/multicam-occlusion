"""Extract partial observations from a scene manifest — the pluggable front end.

Fusion needs two streams of events derived from the raw 3D trajectories, because
the manifest schema carries *geometry*, not semantics: there is no "action
label" or "assembly state" field, so those must be read off the motion. Two
extractor Protocols do that, each with a deterministic default:

* :class:`ActionDetector` turns the operator-view entity's trajectory into
  :class:`ActionEvent`s. The default :class:`ReachActionDetector` reads a
  reach-and-return: a local minimum of a chosen keypoint's height (the operator's
  hand dipping to the station) below the trajectory's own midline is a "place".
* :class:`ItemStateDetector` turns each item's trajectory into
  :class:`AssemblyChangeEvent`s. The default :class:`DisplacementStateDetector`
  fires when an item's tracked point jumps more than a threshold between frames.

Both are Protocols (``@runtime_checkable``, mirroring the ``FrameSource`` seam),
so a learned appearance/action backend drops in without touching the estimator.

:func:`partition_by_visibility` is the asymmetric-visibility router: given which
camera is the operator view and which is the item view, it reads each entity's
per-camera ``visible`` labels to decide which camera contributes which entity —
an entity seen in the operator camera but not the item one is an actor; the
reverse is an item. That asymmetry *is* the mode.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from .observations import ActionEvent, AssemblyChangeEvent
from .scene_manifest import CamId, ManifestEntity, SceneManifest


class CameraRoles(BaseModel):
    """Which camera plays which role in the asymmetric rig.

    ``operator_camera`` sees the operator (actions); ``item_camera`` sees the
    items (assembly state). They must differ — complementary fusion is
    meaningless if one camera sees everything.
    """

    model_config = ConfigDict(frozen=True)

    operator_camera: CamId
    item_camera: CamId

    def _check(self) -> None:
        if self.operator_camera == self.item_camera:
            raise ValueError("operator_camera and item_camera must differ")


class ScenePartition(BaseModel):
    """The router's verdict: which entities are actors vs items (by visibility)."""

    model_config = ConfigDict(frozen=True)

    actors: list[str]
    items: list[str]


def partition_by_visibility(manifest: SceneManifest, roles: CameraRoles) -> ScenePartition:
    """Split entities into actors and items by their per-camera visibility.

    An entity visible (in at least one frame) in the operator camera but never in
    the item camera is an **actor**; visible in the item camera but never in the
    operator camera, an **item**. An entity visible in both (or neither) is not
    complementary and is left out of both lists — this router is precisely the
    asymmetric case.
    """
    roles._check()
    actors: list[str] = []
    items: list[str] = []
    for entity in manifest.entities:
        in_operator = entity.visible_frame_count(roles.operator_camera) > 0
        in_item = entity.visible_frame_count(roles.item_camera) > 0
        if in_operator and not in_item:
            actors.append(entity.id)
        elif in_item and not in_operator:
            items.append(entity.id)
    return ScenePartition(actors=actors, items=items)


def _track_point(entity: ManifestEntity, point: str | None) -> str:
    """Resolve which named point to track: ``point`` if given, else the first."""
    names = entity.point_names()
    if not names:
        raise ValueError(f"entity {entity.id!r} has no points to track")
    if point is None:
        return names[0]
    if point not in names:
        raise ValueError(f"entity {entity.id!r} has no point {point!r}; has {names}")
    return point


@runtime_checkable
class ActionDetector(Protocol):
    """Turns an operator-view entity into timed :class:`ActionEvent`s."""

    def detect(self, manifest: SceneManifest, actor_id: str) -> list[ActionEvent]:
        """Emit the actions of ``actor_id`` (an entity id), in time order."""
        ...


@runtime_checkable
class ItemStateDetector(Protocol):
    """Turns an item entity into timed :class:`AssemblyChangeEvent`s."""

    def detect(self, manifest: SceneManifest, item_id: str) -> list[AssemblyChangeEvent]:
        """Emit the assembly-state changes of ``item_id``, in time order."""
        ...


class ReachActionDetector:
    """Default :class:`ActionDetector`: a reach-down is a "place" action.

    Tracks one keypoint (``point``, else the entity's first point) and finds
    local minima of its height along ``axis`` (default z) that fall below the
    trajectory's own midline ``(min + max) / 2``. Each such dip — the operator
    reaching down to the station and lifting away — is one ``label`` action.
    Deterministic: no RNG, no thresholds tied to a specific fixture's scale.
    """

    def __init__(self, point: str | None = None, axis: int = 2, label: str = "place") -> None:
        self.point = point
        self.axis = axis
        self.label = label

    def detect(self, manifest: SceneManifest, actor_id: str) -> list[ActionEvent]:
        entity = manifest.entity(actor_id)
        name = _track_point(entity, self.point)
        frames = [f for f in entity.frames if name in f.points]
        heights = [f.points[name].xyz_gt[self.axis] for f in frames]
        if not heights:
            return []
        midline = (min(heights) + max(heights)) / 2.0

        events: list[ActionEvent] = []
        for i in range(1, len(frames) - 1):
            h = heights[i]
            # Strict local minimum below the midline: a distinct reach-and-return.
            if h < heights[i - 1] and h <= heights[i + 1] and h < midline:
                frame = frames[i]
                events.append(
                    ActionEvent(
                        actor_id=actor_id,
                        label=self.label,
                        frame=frame.frame,
                        time=manifest.time_of(frame.frame),
                        location=frame.points[name].xyz_gt,
                    )
                )
        return events


class DisplacementStateDetector:
    """Default :class:`ItemStateDetector`: a jump in position is a change.

    Tracks one item point (``point``, else first) and fires a change whenever the
    Euclidean displacement between consecutive frames exceeds ``min_step``. Models
    an item being placed or repositioned at the station; stationary items produce
    nothing. Deterministic and scale-explicit via ``min_step``.
    """

    def __init__(
        self, point: str | None = None, min_step: float = 0.05, change: str = "placed"
    ) -> None:
        self.point = point
        self.min_step = min_step
        self.change = change

    def detect(self, manifest: SceneManifest, item_id: str) -> list[AssemblyChangeEvent]:
        entity = manifest.entity(item_id)
        name = _track_point(entity, self.point)
        frames = [f for f in entity.frames if name in f.points]
        events: list[AssemblyChangeEvent] = []
        for i in range(1, len(frames)):
            prev = frames[i - 1].points[name].xyz_gt
            cur = frames[i].points[name].xyz_gt
            step = sum((a - b) ** 2 for a, b in zip(prev, cur, strict=True)) ** 0.5
            if step > self.min_step:
                frame = frames[i]
                events.append(
                    AssemblyChangeEvent(
                        item_id=item_id,
                        change=self.change,
                        frame=frame.frame,
                        time=manifest.time_of(frame.frame),
                        location=cur,
                    )
                )
        return events
