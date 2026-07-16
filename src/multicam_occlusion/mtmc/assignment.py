"""Global cross-camera ID assignment — stitch matched tracklets into identities.

Given per-camera tracklets, a topology, and a handoff matcher, this decides
which tracklets are the same physical object and hands each a stable **global
id**. The policy is greedy over handoff plausibility with a strict 1-to-1
constraint (a target exits a camera once and re-enters the next once), which is
deterministic and cheap; a Hungarian assignment over the same score matrix is a
drop-in future upgrade (see the design note). Determinism comes from a total
tie-break order on ``(-score, exiting.id, entering.id)`` and from labelling
components by their earliest, lexicographically-smallest member.

The greedy accept rule mirrors bipartite matching on the handoff graph: an
exiting tracklet donates to at most one successor and an entering tracklet
accepts from at most one predecessor. Accepted handoffs are unioned; the
connected components of the union are the global identities.
"""

from __future__ import annotations

from collections.abc import Sequence

from .matcher import HandoffMatcher, SpatioTemporalMatcher
from .topology import CameraTopology
from .tracklet import Tracklet


class _UnionFind:
    """Minimal union-find over tracklet ids with lexicographic root preference."""

    def __init__(self, ids: Sequence[str]) -> None:
        self._parent = {i: i for i in ids}

    def find(self, x: str) -> str:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:  # path compression
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Keep the lexicographically smaller id as root for stable labelling.
        low, high = (ra, rb) if ra < rb else (rb, ra)
        self._parent[high] = low


def assign_global_ids(
    tracklets: Sequence[Tracklet],
    topology: CameraTopology,
    matcher: HandoffMatcher | None = None,
) -> dict[str, str]:
    """Map each tracklet id to a global identity id (``"G0"``, ``"G1"`` …).

    Scores every ordered cross-camera pair through ``matcher`` (default
    :class:`SpatioTemporalMatcher`), greedily accepts the highest-scoring
    handoffs under the 1-to-1 constraint, and unions accepted pairs. Global ids
    are numbered by each identity's earliest entry time (ties broken by smallest
    member id) so the labelling is stable across runs.
    """
    match = SpatioTemporalMatcher() if matcher is None else matcher
    ids = [t.id for t in tracklets]
    by_id = {t.id: t for t in tracklets}

    # Score every admissible ordered cross-camera handoff.
    candidates: list[tuple[float, str, str]] = []
    for exiting in tracklets:
        for entering in tracklets:
            if exiting.id == entering.id:
                continue
            score = match.score(exiting, entering, topology)
            if score is not None:
                candidates.append((score, exiting.id, entering.id))

    # Greedy 1-to-1: highest score first, deterministic tie-break.
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
    used_out: set[str] = set()
    used_in: set[str] = set()
    uf = _UnionFind(ids)
    for _score, src, dst in candidates:
        if src in used_out or dst in used_in:
            continue
        used_out.add(src)
        used_in.add(dst)
        uf.union(src, dst)

    # Label components by (earliest entry_time, smallest member id).
    components: dict[str, list[str]] = {}
    for tid in ids:
        components.setdefault(uf.find(tid), []).append(tid)

    def sort_key(root: str) -> tuple[float, str]:
        members = components[root]
        return (min(by_id[m].entry_time for m in members), min(members))

    global_of: dict[str, str] = {}
    for index, root in enumerate(sorted(components, key=sort_key)):
        for member in components[root]:
            global_of[member] = f"G{index}"
    return global_of


def per_tracklet_global_ids(tracklets: Sequence[Tracklet]) -> dict[str, str]:
    """The no-handoff baseline: every tracklet is its own global identity.

    This is what a system with *no* cross-camera reasoning produces — an object
    crossing a blind gap fractures into one identity per camera. It is the
    control the handoff result is measured against.
    """
    return {t.id: f"G{i}" for i, t in enumerate(tracklets)}
