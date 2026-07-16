# Complementary / asymmetric multi-view fusion

This note specifies a camera-relationship mode in which two cameras observe
**different aspects of the same event** rather than the same 3D point from
different angles. Where the triangulation core recovers *one* point seen in
*several* overlapping views, this mode fuses *disjoint* partial views into a
joint account of an interaction: **who did what to which item, and when.**

> **Clean-room note.** Built from first principles on top of this repository's
> existing manifest contract and the public multicam-sim scene schema. No
> proprietary code, data, or design is used or copied.

## The motivating case: a packing station

A worker assembles an order at a bench. Two cameras watch, from complementary
positions:

- **Camera A** (north, wide) sees the **worker** — the human, the reach, the
  action — but the worktop surface is foreshortened and the hands occlude the
  goods.
- **Camera B** (east, low, close to the worktop) sees the **items** — a part
  appearing, a box filling, an assembly changing state — but barely the human.

Neither view alone answers the operational question. Camera A knows *an action
happened at t*; camera B knows *the assembly changed at ~t*. Only together do
they say *that action caused that change*.

### Why this is not triangulation (mode contrast)

The overlap-triangulation mode answers "where is this point in 3D?" by
intersecting sightlines from cameras that **see the same point**. That is
impossible here **by construction**: a human action and an item are not the same
3D point, and each is visible in only one camera. There is no ray to intersect.

So recovery is not geometric, it is **temporal**. The correspondence signal is
*co-timing across complementary views*: an action at time `t` in camera A is
followed, within a short causal window, by an assembly change at `~t` in camera
B. Fusion is correlation in time, not intersection in space.

## Asymmetric visibility is the input contract

The mode reads the same multicam-sim scene manifest the rest of the benchmark
consumes: entities, each with named 3D points per frame, each point carrying a
per-camera observation with a hard `visible` boolean (computed geometrically
upstream). This mode leans on that boolean being **asymmetric**:

| entity   | visible in A (human) | visible in B (worktop) | role  |
| -------- | -------------------- | ---------------------- | ----- |
| `worker` | ✅                   | ❌                     | actor |
| `item_*` | ❌                   | ✅                     | item  |

`partition_by_visibility` reads exactly these labels to route each entity: seen
in the human camera but never the worktop camera → **actor**; the reverse →
**item**. An entity seen in both (or neither) is not complementary and is left
out — this router is precisely the asymmetric case. The routing is *data*, not
configuration: change the rig and the manifest's `visible` labels re-route it.

## Pipeline

```
 SceneManifest ──partition_by_visibility──▶  actors / items   (by visibility)
     │                                              │
     │ ActionDetector                WorktopStateDetector
     │  (ReachActionDetector)         (DisplacementStateDetector)
     ▼                                              ▼
 HumanViewObservation                       WorktopViewObservation
   actions: [ActionEvent]                     changes: [AssemblyChangeEvent]
     └───────────────  FusionEstimator  ────────────┘
                  (TemporalProximityEstimator)
                            │
                     JointSceneState
              interactions + unassociated actions/changes
                            │
                   association_metric ──▶ precision / recall / timing error
```

Every stage is a `Protocol` with a deterministic default (mirroring the repo's
`FrameSource` seam), so an appearance/pose/action model backend drops in without
touching the estimator or the metric.

### Partial-observation models

- `ActionEvent` — an `(actor_id, label, frame, time, location?)` the worker did,
  from camera A. Bundled per actor in a `HumanViewObservation`.
- `AssemblyChangeEvent` — an `(item_id, change, frame, time, location?)` an item
  underwent, from camera B. Bundled in a `WorktopViewObservation`.

Because the manifest schema carries **geometry, not semantics** (there is no
"action label" or "assembly state" field), these events are *derived* from the
trajectories. The deterministic defaults:

- **`ReachActionDetector`** — a "place" is a strict local minimum of a tracked
  keypoint's height (the hand dipping to the worktop) below the trajectory's own
  midline. No RNG; no threshold tied to a scene's absolute scale.
- **`DisplacementStateDetector`** — a change fires when an item's tracked point
  jumps more than `min_step` between consecutive frames.

A learned backend (2D pose + an action classifier for A; an object/state
detector for B) is the open/closed extension point: implement the Protocol,
emit the same event types, and the fusion and metric are unchanged.

### The estimator

`TemporalProximityEstimator` associates each assembly change with the human
action that best explains it, under three plain constraints:

1. **Causality** — the action precedes the change (within a tiny epsilon).
2. **Max lag** — the change follows within `max_lag` seconds (default `0.5`).
3. **One-to-one** — each action pairs with at most one change and vice versa;
   ties break to the smallest lag (nearest action).

An optional spatial gate (`max_distance` over world locations) is available but
off by default: the two locations are, in general, different points (a hand vs
an item), so timing is the primary signal. Unmatched actions and changes are
**surfaced, not discarded** — refusing to pair them is a first-class outcome.

## Metric: action ↔ assembly-change association

The unit of evaluation is the **(action, item-change) pair**, generalizing
"did we correctly link the box's contents to the open order?" A predicted
`FusedInteraction` is a **true positive** when a ground-truth interaction exists
with the same `(actor_id, item_id)` and an action time within `time_tol`. From
the greedy one-to-one matching:

- **precision** = TP / predicted — of the links asserted, how many are real;
- **recall** = TP / ground-truth — of the real links, how many recovered;
- **timing error** = mean absolute error of action time and change time versus
  ground truth over matched pairs, plus the mean fused lag.

### Why the score is honest (the distractor design)

A clean two-action / two-change scene scores 1.0/1.0 trivially — that proves the
plumbing runs, not that the metric measures anything. The packing-station
fixture therefore plants two traps:

- a **distractor action** (a third reach that causes no item change), and
- a **distractor change** (a fourth item that moves *outside* the lag window,
  with no action to explain it).

A faithful estimator must **refuse both**. Precision stays 1.0 *because it
correctly rejected the traps*, not because there was nothing to get wrong — the
distractors sit in `unassociated_actions` / `unassociated_changes`. And when the
lag window is widened to 1.0 s, the estimator wrongly links the distractor
action to the late item and precision falls to 2/3 — the metric registering a
real mistake. Both behaviours are asserted in `tests/test_fusion.py`.

## Fixture and reproducibility

`tests/fixtures/packing_station.json` is a small scene in the **multicam-sim
scene schema** (`{fps, num_frames, cameras, entities}`), regenerable with
`python tests/fixtures/build_packing_station.py`. It is a two-camera asymmetric
rig (camera 0 = worker, camera 1 = items). Every point carries a `per_cam` entry
for **both** cameras — `visible` in one, `false` in the other — exactly as
`build_manifest` emits, so the visibility router is genuinely exercised rather
than reading a trivially-partitioned input.

The fixture is a stand-in until multicam-sim can script asymmetric visibility
directly; requested upstream as an **asymmetric-visibility packing-station
preset** (bamdadd/multicam-sim). The whole pipeline is deterministic — no RNG,
no wall-clock — so the association metric on the fixture is exact and pinned in
CI (`ruff` + `mypy --strict` + `pytest`).

## Roadmap

- Multi-actor scenes (route several human entities; disambiguate which actor by
  proximity or by a learned assignment).
- A learned detector backend pair (2D-pose action recognition for the human
  view; object-state detection for the worktop view) behind the existing
  Protocols.
- A soft/probabilistic estimator (associate with calibrated confidence rather
  than a hard greedy one-to-one) and a PR-curve sweep over the lag window.
