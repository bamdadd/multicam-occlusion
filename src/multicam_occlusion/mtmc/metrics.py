"""Cross-camera identity metrics — IDF1 and ID-switches (not 3D error).

The MTMC question is *identity consistency across cameras*, so the metric is not
reprojection error but how well predicted global ids preserve ground-truth
identities. Two standard measures:

* **IDF1** (Ristani et al., 2016) — the identity F1. It is *defined* by the
  optimal one-to-one matching between predicted and GT identities (the pairing
  that maximises correctly-identified observations, IDTP). We solve that
  matching exactly with a compact Hungarian assignment; a greedy match would
  under-count IDTP whenever an identity's observations split across predictions,
  so greedy is *not* IDF1.
* **ID-switches** — how many times a GT identity's predicted id changes as its
  tracklets are followed in time. Zero means every object kept one id across the
  blind gap.

HOTA is noted as roadmap in the design note; it needs the detection-association
decomposition this fixture does not exercise.

Ground truth enters *only here*. Nothing in tracklet / topology / matcher /
assignment reads a GT identity, so the score is not circular.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from .tracklet import Tracklet

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class IdentityMetrics(BaseModel):
    """IDF1 and its parts, plus the ID-switch count."""

    idtp: int
    idfp: int
    idfn: int
    idp: float
    idr: float
    idf1: float
    id_switches: int


def _hungarian_max(weight: IntArray) -> list[tuple[int, int]]:
    """Return row→col pairs maximising total weight on a square cost matrix.

    Classic ``O(n^3)`` Kuhn–Munkres on ``cost = maxw - weight`` (min-cost form).
    ``weight`` must be square; pad with zero rows/cols before calling. Only pairs
    are returned — the caller decides which correspond to real identities.
    """
    n = weight.shape[0]
    if n == 0:
        return []
    maxw = int(weight.max())
    cost = maxw - weight  # minimise cost == maximise weight
    inf = float("inf")
    # 1-indexed potentials method (e-maxx); row/col 0 are sentinels.
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)  # p[j] = row assigned to column j
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = float(cost[i0 - 1, j - 1]) - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0 != 0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    return [(p[j] - 1, j - 1) for j in range(1, n + 1) if p[j] > 0]


def idf1_from_counts(
    counts: Sequence[tuple[str | None, str | None, int]],
) -> tuple[int, int, int]:
    """Optimal IDTP / IDFP / IDFN from ``(gt_id, pred_id, count)`` overlaps.

    ``gt_id is None`` marks false-positive observations (predicted but no GT);
    ``pred_id is None`` marks false negatives (GT but unpredicted). The optimal
    one-to-one GT↔pred matching maximising overlap gives IDTP; the rest fall out
    as IDFP / IDFN. This is the pure identity-matching core of IDF1, decoupled
    from tracklets so it can be unit-tested on hand-built overlap matrices.
    """
    gt_ids = sorted({g for g, _p, _c in counts if g is not None})
    pred_ids = sorted({p for _g, p, _c in counts if p is not None})
    gt_index = {g: i for i, g in enumerate(gt_ids)}
    pred_index = {p: i for i, p in enumerate(pred_ids)}

    total_gt = sum(c for g, _p, c in counts if g is not None)
    total_pred = sum(c for _g, p, c in counts if p is not None)

    n = max(len(gt_ids), len(pred_ids))
    idtp = 0
    if n > 0:
        overlap: IntArray = np.zeros((n, n), dtype=np.int64)
        for g, p, c in counts:
            if g is not None and p is not None:
                overlap[gt_index[g], pred_index[p]] += c
        for r, col in _hungarian_max(overlap):
            if r < len(gt_ids) and col < len(pred_ids):
                idtp += int(overlap[r, col])

    idfp = total_pred - idtp
    idfn = total_gt - idtp
    return idtp, idfp, idfn


def _count_id_switches(
    tracklets: Sequence[Tracklet],
    predicted: Mapping[str, str],
    ground_truth: Mapping[str, str],
) -> int:
    """ID-switches: per GT identity, changes in predicted id along time order."""
    by_gt: dict[str, list[Tracklet]] = {}
    for t in tracklets:
        by_gt.setdefault(ground_truth[t.id], []).append(t)
    switches = 0
    for members in by_gt.values():
        ordered = sorted(members, key=lambda t: (t.entry_time, t.camera_id, t.id))
        previous: str | None = None
        for t in ordered:
            current = predicted[t.id]
            if previous is not None and current != previous:
                switches += 1
            previous = current
    return switches


def compute_identity_metrics(
    tracklets: Sequence[Tracklet],
    predicted: Mapping[str, str],
    ground_truth: Mapping[str, str],
) -> IdentityMetrics:
    """IDF1 + ID-switches for a global-id assignment against GT identities.

    Each observation of a tracklet is weighted by count, so a long tracklet
    contributes more than a short one — the standard IDF1 weighting. ``predicted``
    and ``ground_truth`` map tracklet id → identity id.
    """
    counts: list[tuple[str | None, str | None, int]] = [
        (ground_truth[t.id], predicted[t.id], len(t.observations)) for t in tracklets
    ]
    idtp, idfp, idfn = idf1_from_counts(counts)
    idp = idtp / (idtp + idfp) if (idtp + idfp) > 0 else 0.0
    idr = idtp / (idtp + idfn) if (idtp + idfn) > 0 else 0.0
    denom = 2 * idtp + idfp + idfn
    idf1 = (2 * idtp / denom) if denom > 0 else 0.0
    return IdentityMetrics(
        idtp=idtp,
        idfp=idfp,
        idfn=idfn,
        idp=idp,
        idr=idr,
        idf1=idf1,
        id_switches=_count_id_switches(tracklets, predicted, ground_truth),
    )
