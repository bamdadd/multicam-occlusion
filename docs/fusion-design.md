# Complementary / asymmetric multi-view fusion

This note specifies a camera-relationship mode in which two cameras observe
**different aspects of the same event** rather than the same 3D point from
different angles. Where the triangulation core recovers *one* point seen in
*several* overlapping views, this mode fuses *disjoint* partial views into a
joint account of an interaction: **who did what to which item, and when** — and
reads that back against an order (**order verification**).

> **Clean-room note.** Built from first principles on top of this repository's
> existing manifest contract and the public multicam-sim scene schema. No
> proprietary code, data, or design is used or copied.

## The motivating case: an assembly / order-fulfilment station

An operator assembles an order — a bill of materials (BOM) — into a container at
a station. Two cameras watch, from complementary positions:

- **Overview camera** (camera A) sees the **operator** — the activity, the reach,
  the action — but the station surface is foreshortened and the hands occlude the
  parts.
- **Item camera** (camera B, low, at the station) sees the **items** — a part
  appearing, the container filling, an assembly changing state — but barely the
  operator.

Neither view alone answers the operational question. Camera A knows *an action
happened at t*; camera B knows *the assembly changed at ~t*. Only together do
they say *that action assembled that part* — and only then can the assembled
contents be reconciled against the order.

### Why this is not triangulation (mode contrast)

The overlap-triangulation mode answers "where is this point in 3D?" by
intersecting sightlines from cameras that **see the same point**. That is
impossible here **by construction**: an operator action and a part are not the
same 3D point, and each is visible in only one camera. There is no ray to
intersect.

So recovery is not geometric, it is **temporal**. The correspondence signal is
*co-timing across complementary views*: an action at time `t` in camera A is
followed, within a short causal window, by an assembly change at `~t` in camera
B. Fusion is correlation in time, not intersection in space.

## Asymmetric visibility is the input contract

The mode reads the same multicam-sim scene manifest the rest of the benchmark
consumes: entities, each with named 3D points per frame, each point carrying a
per-camera observation with a hard `visible` boolean (computed geometrically
upstream). This mode leans on that boolean being **asymmetric**:

| entity     | visible in A (operator) | visible in B (items) | role  |
| ---------- | ----------------------- | -------------------- | ----- |
| `operator` | ✅                      | ❌                   | actor |
| `part_*`   | ❌                      | ✅                   | item  |

`partition_by_visibility` reads exactly these labels to route each entity: seen
in the operator camera but never the item camera → **actor**; the reverse →
**item**. An entity seen in both (or neither) is not complementary and is left
out — this router is precisely the asymmetric case. The routing is *data*, not
configuration: change the rig and the manifest's `visible` labels re-route it.

## Pipeline

```
 SceneManifest ──partition_by_visibility──▶  actors / items   (by visibility)
     │                                              │
     │ ActionDetector                ItemStateDetector
     │  (ReachActionDetector)         (DisplacementStateDetector)
     ▼                                              ▼
 OperatorViewObservation                     ItemViewObservation
   actions: [ActionEvent]                     changes: [AssemblyChangeEvent]
     └───────────────  FusionEstimator  ────────────┘
                  (TemporalProximityEstimator)
                            │
                     JointSceneState
              interactions + unassociated actions/changes
                    │                         │
       association_metric            verify_order
     precision / recall / timing    fulfilled / missing / wrong / extra
```

Every stage is a `Protocol` with a deterministic default (mirroring the repo's
`FrameSource` seam), so an appearance/pose/action model backend drops in without
touching the estimator or the metric.

### Partial-observation models

- `ActionEvent` — an `(actor_id, label, frame, time, location?)` the operator
  did, from camera A. Bundled per actor in an `OperatorViewObservation`.
- `AssemblyChangeEvent` — an `(item_id, change, frame, time, location?)` a part
  underwent, from camera B. Bundled in an `ItemViewObservation`.

Because the manifest schema carries **geometry, not semantics** (there is no
"action label" or "assembly state" field), these events are *derived* from the
trajectories. The deterministic defaults:

- **`ReachActionDetector`** — a "place" is a strict local minimum of a tracked
  keypoint's height (the hand dipping to the station) below the trajectory's own
  midline. No RNG; no threshold tied to a scene's absolute scale.
- **`DisplacementStateDetector`** — a change fires when a part's tracked point
  jumps more than `min_step` between consecutive frames.

A learned backend (2D pose + an action classifier for A; an object/state
detector for B) is the open/closed extension point: implement the Protocol,
emit the same event types, and the fusion and metric are unchanged.

### The estimator

`TemporalProximityEstimator` associates each assembly change with the operator
action that best explains it, under three plain constraints:

1. **Causality** — the action precedes the change (within a tiny epsilon).
2. **Max lag** — the change follows within `max_lag` seconds (default `0.5`).
3. **One-to-one** — each action pairs with at most one change and vice versa;
   ties break to the smallest lag (nearest action).

An optional spatial gate (`max_distance` over world locations) is available but
off by default: the two locations are, in general, different points (a hand vs a
part), so timing is the primary signal. Unmatched actions and changes are
**surfaced, not discarded** — refusing to pair them is a first-class outcome.

## Metric: action ↔ assembly-change association

The unit of evaluation is the **(action, item-change) pair**, generalizing
"did we correctly reconcile the assembled contents against the order?" A
predicted `FusedInteraction` is a **true positive** when a ground-truth
interaction exists with the same `(actor_id, item_id)` and an action time within
`time_tol`. From the greedy one-to-one matching:

- **precision** = TP / predicted — of the links asserted, how many are real;
- **recall** = TP / ground-truth — of the real links, how many recovered;
- **timing error** = mean absolute error of action time and change time versus
  ground truth over matched pairs, plus the mean fused lag.

### Why the score is honest (the distractor design)

A clean two-action / two-change scene scores 1.0/1.0 trivially — that proves the
plumbing runs, not that the metric measures anything. The assembly-station
fixture therefore plants two traps:

- a **distractor action** (a third reach that assembles no part), and
- a **distractor change** (a fourth part that moves *outside* the lag window,
  with no action to explain it).

A faithful estimator must **refuse both**. Precision stays 1.0 *because it
correctly rejected the traps*, not because there was nothing to get wrong — the
distractors sit in `unassociated_actions` / `unassociated_changes`. And when the
lag window is widened to 1.0 s, the estimator wrongly links the distractor
action to the late part and precision falls to 2/3 — the metric registering a
real mistake. Both behaviours are asserted in `tests/test_fusion.py`.

## Order verification (the operational output)

`verify_order` reads the joint scene state against the order (the expected item
ids / BOM) and assigns each line a status — nothing here changes the estimator,
the association, or the metric; it is a thin interpretation over
`JointSceneState`:

- **fulfilled** — an expected item was assembled (a fused interaction links it);
- **missing** — an expected item was never assembled, with no stand-in;
- **wrong** — an expected item was not assembled, but an *unexpected* item was
  assembled in its place (a mis-pick / substitution);
- **extra** — an item was assembled that the order never called for.

On the fixture the order `[part_a, part_b]` verifies **ok** (both fulfilled); the
spurious `part_c` under a too-wide window surfaces as an `extra` line rather than
passing silently. The four statuses are unit-tested directly in
`tests/test_verification.py`.

## Fixture and reproducibility

`tests/fixtures/assembly_station.json` is a small scene in the **multicam-sim
scene schema** (`{fps, num_frames, cameras, entities}`), regenerable with
`python tests/fixtures/build_assembly_station.py`. It is a two-camera asymmetric
rig (camera 0 = operator, camera 1 = parts). Every point carries a `per_cam`
entry for **both** cameras — `visible` in one, `false` in the other — exactly as
`build_manifest` emits, so the visibility router is genuinely exercised rather
than reading a trivially-partitioned input.

The fixture stands in until multicam-sim can script asymmetric visibility
directly; requested upstream as an **asymmetric-visibility assembly-station
preset** (bamdadd/multicam-sim). The whole pipeline is deterministic — no RNG,
no wall-clock — so the association metric on the fixture is exact and pinned in
CI (`ruff` + `mypy --strict` + `pytest`).

## Roadmap

- Multi-operator scenes (route several actor entities; disambiguate which actor
  by proximity or by a learned assignment).
- A learned detector backend pair (2D-pose action recognition for the operator
  view; object-state detection for the item view) behind the existing Protocols.
- A soft/probabilistic estimator (associate with calibrated confidence rather
  than a hard greedy one-to-one) and a PR-curve sweep over the lag window.
- Quantity-aware orders (a BOM line requiring N of a part), extending order
  verification beyond a presence check.
