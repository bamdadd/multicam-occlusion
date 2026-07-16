# MTMC handoff across non-overlapping cameras — design note

This note describes the **camera-relationship** path in `multicam_occlusion`: keeping
one identity on an object as it moves between cameras that share **no field of view**.
It is a separate mode from the overlap / triangulation core and does not touch it.

## Why a second mode

The triangulation core answers a geometry question: given several calibrated cameras
that **do** see the same point, recover its 3D position even when some views are
occluded. That whole machinery presupposes overlap — two rays that intersect.

Non-overlapping station cameras break that assumption. Two station cameras along a
route watch **disjoint** areas. An entity leaves station 1's view, crosses a **blind
gap** the rig never observes, and reappears at station 2 a few seconds later. There is
no shared ray, so triangulation is undefined. The question is no longer *where in 3D*
but *which observations are the same entity* — **multi-target multi-camera (MTMC)
identity handoff**. This module is that path.

```
   cam0 (station 1)          blind gap                 cam1 (station 2)
   [===entity===] -- exits right -- ??? -- enters left -- [===entity===]
   frames 0..2        (unobserved, ~0.5 s)                 frames 7..9
        \___________________ one global identity ___________________/
```

## Pipeline

Five typed, deterministic, GPU-free stages, each an open-closed seam:

| Stage | Module | Responsibility |
| --- | --- | --- |
| 1. Tracklets | `mtmc.extract` / `mtmc.tracklet` | Per-camera single-view runs of one target: observations, entry/exit **zone**, entry/exit time. |
| 2. Topology | `mtmc.topology` | Directed graph of *possible* transitions: exit-zone → entry-zone with a transit-time distribution. |
| 3. Handoff matcher | `mtmc.matcher` | Score an exiting→entering pair by spatio-temporal plausibility first; appearance/ReID is a pluggable backend. |
| 4. Global IDs | `mtmc.assignment` | Stitch matched tracklets into consistent global identities. |
| 5. Metrics | `mtmc.metrics` | IDF1 and ID-switches against GT identities (identity consistency, **not** 3D error). |

### 1. Tracklet (`mtmc.tracklet`)

A `Tracklet` is what a single-view tracker already produces: one target's contiguous
observation run inside **one** camera, tagged with the image **zone** it entered and
exited (an opaque label like `cam0:right`) and the entry/exit timestamps. Its `id` is
**local** to its camera. The GT identity that links tracklets across cameras is
deliberately **not** on the model — recovering it is the task.

### 2. CameraTopology (`mtmc.topology`)

Nodes are cameras; a directed `TransitionEdge` says "leaving `src` through `exit_zone`
may reappear at `dst` through `entry_zone` after roughly this transit time." Transit is
a `TransitDistribution` (Gaussian mean/std), not a point value, so the matcher can both
hard-gate on a window (`mean ± k·std`) and score inside it (peak-normalised Gaussian).
A richer transit model (log-normal, learned histogram) is an open-closed swap.

### 3. Handoff matcher (`mtmc.matcher`)

`SpatioTemporalMatcher.score(exiting, entering, topology)` returns `None` (a hard veto)
unless **all** hold: different cameras; a topology edge matches the exit→entry zones;
and the gap `dt = entering.entry_time − exiting.exit_time` is positive and inside the
edge's transit window. When admitted the score is `temporal^(1−w) · appearance^w`.

**Spatio-temporal first, appearance second.** Appearance/ReID is a pluggable
`AppearanceBackend` Protocol. The default `NoAppearance` returns `1.0`, so on a
non-overlapping rig a match is decided by geometry and time alone — precisely the regime
where overlap (and triangulation, and pixel-level cross-view appearance constraints) is
unavailable. `StubEmbedding` is a deterministic, GPU-free stand-in that proves the seam;
a real embedding drops in with no change to assignment or metrics.

### 4. Global ID assignment (`mtmc.assignment`)

`assign_global_ids` scores every ordered cross-camera pair, then greedily accepts the
highest-scoring handoffs under a strict 1-to-1 constraint (an object exits a camera once
and re-enters the next once) and unions accepted pairs; connected components are the
global identities. Determinism comes from a total tie-break order
(`-score, exiting.id, entering.id`) and from labelling components by earliest,
lexicographically-smallest member. Greedy is sufficient and cheap here; a **Hungarian**
assignment over the same score matrix is a drop-in upgrade (roadmap).

### 5. Metrics (`mtmc.metrics`)

The MTMC question is identity consistency, so the metric is not reprojection error:

* **IDF1** (Ristani et al., 2016) — the identity F1, *defined* by the **optimal**
  one-to-one matching between predicted and GT identities that maximises correctly
  identified observations (IDTP). We solve that matching exactly with a compact
  Hungarian assignment (pure NumPy, no new dependency). A greedy match under-counts IDTP
  whenever an identity's observations split across predictions, so greedy is **not**
  IDF1 — a dedicated unit test builds an overlap matrix where greedy scores 3 and the
  optimal (and our) answer is 4.
* **ID-switches** — per GT identity, the number of times its predicted id changes along
  time order. Zero means every object kept one id across the blind gap.
* **HOTA** — roadmap. HOTA needs the detection–association decomposition this
  fixture-level evaluation does not yet exercise.

## Non-circularity

Ground truth enters **only** the metrics stage. Nothing in tracklet / topology / matcher
/ assignment reads a GT identity. Single-camera tracking is *assumed solved* (that is why
the extractor may group a camera's observations by their per-camera entity label); the
contribution under test is purely the **cross-camera stitch**, decided on topology and
time. So the score is not "using the answer to get the answer."

## The fixture and the proof

The manifest core (`multicam-sim`) is the intended producer, but its ring/line rigs all
share a look-at target, so every camera sees the scene centre — it cannot yet emit a
**non-overlapping** rig. So the test uses a **hand-authored** manifest in the exact
multicam-sim schema, with `visible` set directly to encode non-overlap: each entity is
visible in exactly one camera at a time and invisible during the gap. (Upstream issue
[bamdadd/multicam-sim#13](https://github.com/bamdadd/multicam-sim/issues/13) requests a
non-overlap rig preset; when it lands, the fixture becomes real sim output with no
consumer change.)

The fixture is a two-station rig (station 1 `cam0`, station 2 `cam1`) with **two** entities
crossing the gap **staggered** in time. Two entities are essential: with a single entity,
assigning *everything* one id trivially scores IDF1 = 1.0 and the matcher is never
exercised. Because the default appearance backend is neutral, the correct pairing
(a→a, b→b) must be more plausible than the cross pairing (a→b, b→a) on **time alone** —
and it is (correct `dt ≈ 0.5 s` at the transit peak; cross `dt ≈ 0.3 / 0.7 s` off-peak).

**Result on the fixture:**

| Assignment | IDF1 | ID-switches |
| --- | --- | --- |
| Spatio-temporal handoff | **1.0** | **0** |
| No-handoff baseline (each tracklet its own id) | 0.5 | 2 |

The contrast is the proof: without cross-camera reasoning each object fractures into one
identity per camera (IDF1 0.5, 2 switches); the handoff recovers a single identity across
the blind gap (IDF1 1.0, 0 switches).

## Roadmap

* Hungarian global assignment over the plausibility matrix (assignment step).
* HOTA metric alongside IDF1 / ID-switches.
* A learned ReID `AppearanceBackend` (the Protocol already exists; `StubEmbedding` proves
  the swap).
* A robust single-view tracklet extractor (detector + tracker + learned zone model),
  replacing the minimal border-based extractor here.
* Upstream: a non-overlapping rig preset in `multicam-sim` so the fixture becomes real
  producer output.
