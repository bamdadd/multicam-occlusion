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
plumbing runs, not that the metric measures anything. The controlled
assembly-station scene therefore plants two traps:

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

Order verification comes in two forms, one per data source:

1. **On real generated data (`verify_order_from_manifest`).** This is the path
   that runs on multicam-sim output. It does not need an operator action stream:
   it routes entities by asymmetric visibility, reconstructs the assembled
   contents directly from the *item camera* (an `ItemStateDetector` counting each
   part's placement into the container), and reconciles those counts against the
   order's BOM, quantity-aware. On the generated assembly-station scene, all three
   ordered parts are placed, so the order verifies **fulfilled**.

2. **On the fused joint scene state (`verify_order`).** This reads a
   `JointSceneState` (operator actions linked to item changes) against the order —
   a thin interpretation used on the controlled causal scene, where an interaction
   *is* the evidence a line was assembled.

Both assign four statuses, but the two paths differ on `wrong` vs `extra`, and
the difference is intentional:

- The **generated-data path** (`verify_order_from_manifest` / `verify_assembly`)
  is **quantity-aware**, matching multicam-sim's own `order.py` vocabulary:
  *fulfilled* = assembled count equals the ordered count; *missing* = short;
  *extra* = **surplus of an expected item** (more assembled than ordered); *wrong*
  = a **foreign item** assembled that the order never listed. Verified on real
  generated data in `tests/test_fusion_sim.py`.
- The **fused path** (`verify_order`) is **presence/substitution-based**:
  *fulfilled* = an interaction assembled the item; *missing* = no interaction and
  no stand-in; *wrong* = an expected line unmet **with an unexpected item standing
  in for it** (a mis-pick); *extra* = an unexpected item assembled with no missing
  line to explain it. So on the controlled causal scene the order `[part_a,
  part_b]` verifies **ok**, and the spurious `part_c` under a too-wide window
  surfaces as an `extra` line (there is no unmet line for it to substitute).
  Unit-tested directly in `tests/test_verification.py`.

## What runs on real generated data, and what is still controlled

The two halves of the mode consume different data, and the split is deliberate
and disclosed:

- **Order verification runs on REAL generated multicam-sim output.**
  `tests/fixtures/sim_assembly_station/{manifest.json,order.json}` is genuine
  producer output from multicam-sim's shipped `examples/assembly_station.py`
  preset (an overview camera framing the operator, a worktop camera framing the
  parts, with complementary per-camera `visible` labels). Regenerate it with
  `make fusion-scene` (drives multicam-sim via the optional `bench` group;
  **deterministic, no seed**). `verify_order_from_manifest` routes by asymmetric
  visibility, reconstructs the assembled contents from the *item camera*, and
  reconciles them against the generated order's BOM. On this scene the order
  `part_a + part_b + part_c` verifies **fulfilled** (each part placed once).
  See `tests/test_fusion_sim.py`.

- **The causal action↔change association metric runs on a CONTROLLED in-memory
  scene** (`tests/controlled_assembly_scene.py`, clearly labelled as
  hand-authored, not sim-generated). The shipped sim preset frames an operator
  but does **not** emit reaches timestamped to placements — its operator makes a
  single continuous wrist motion, not a dip synced to each part — so there is no
  per-placement action stream to correlate. Until multicam-sim emits those
  action events (requested upstream: **bamdadd/multicam-sim#34**, placement-synced
  operator action events), the distractor/lag-window metric is exercised on the
  controlled scene, which tests the estimator logic rather than the sim
  integration.

Both halves are deterministic — no RNG, no wall-clock — so the numbers are exact
and pinned in CI (`ruff` + `mypy --strict` + `pytest`), and CI never installs
multicam-sim (the committed fixture keeps the gate numpy-only).

## Roadmap

- Multi-operator scenes (route several actor entities; disambiguate which actor
  by proximity or by a learned assignment).
- A learned detector backend pair (2D-pose action recognition for the operator
  view; object-state detection for the item view) behind the existing Protocols.
- A soft/probabilistic estimator (associate with calibrated confidence rather
  than a hard greedy one-to-one) and a PR-curve sweep over the lag window.
- Quantity-aware orders (a BOM line requiring N of a part), extending order
  verification beyond a presence check.
