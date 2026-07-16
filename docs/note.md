# Three Camera-Relationship Modes for Multi-View Occlusion, Handoff, and Fusion: A Deterministic Synthetic Benchmark

**Bamdad Dashtban**

---

## Abstract

When several cameras observe one event, the geometric relationship between the
views — not the number of cameras — determines what can be recovered and how.
We identify three distinct regimes, divided by **view overlap**, and build one
deterministic, pure-NumPy benchmark that treats each as a separate typed mode
over a shared scene-manifest contract. (1) *Overlapping* views of the same point
support **triangulation**: 3D position is recovered from whichever views still
see the point under occlusion. On a 3-camera occlusion dose-response sweep,
single-view recovery under a fixed depth prior sits at 0.134 world-unit error
before any occlusion and climbs to 0.243 as its one view is blocked, while
multi-view triangulation stays at ≈0.003 throughout — 48× to 79× better, with
the gap widening as occlusion increases. (2) *Non-overlapping* views separated by
a blind gap require **identity handoff** (multi-target multi-camera, MTMC): a
spatio-temporal matcher keeps one global identity across the gap, scoring
IDF1 = 1.0 with 0 ID-switches versus 0.625 / 2 for a no-handoff baseline. (3)
*Complementary / asymmetric* views, where each camera sees a different facet of
the event, require **temporal fusion**: correlating an operator-action view with
an item/assembly-state view yields action↔change associations at
precision = recall = F1 = 1.00 (TP = 3, FP = 0, FN = 0; mean fused lag 0.0 s on
the producer's placement-synced operator actions), which drive an
order-verification status (all three ordered parts fulfilled). All three modes
now run on real `multicam-sim` producer output, are deterministic (fixed seeds,
no GPU, no renderer), and gated in CI. The perfect identity and
association scores are a property of a controllable synthetic benchmark with
exact ground truth and deliberately naive baselines, not a claim of real-world
state of the art; we name the missing sources of difficulty (detector noise,
calibration drift, appearance ambiguity) as the next knobs.

---

## 1. Introduction and contribution

Multi-camera perception is usually framed as one problem — "use more cameras to
see better." But several cameras can relate to a single event in fundamentally
different geometric regimes, and the right algorithm, the right metric, and even
the right *question* change with the regime. A calibrated view maps a world
point to a pixel via `x ~ P X`; inverting one view recovers only the viewing
ray, not depth. What a second or third camera buys you depends entirely on how
its field of view relates to the first.

**The contribution of this benchmark is the three-mode framing itself**, with the
dividing line made explicit and each mode implemented, measured, and reproduced:

- **Overlapping views of the same point → Triangulate.** Two rays intersect;
  the 3D point is recoverable even when occlusion removes some views. The
  question is *where in 3D*, and the metric is recovery error versus occlusion
  dose.
- **Non-overlapping views with a blind gap → Handoff / MTMC.** No shared ray
  exists, so triangulation is undefined. An entity exits one station, crosses an
  unobserved gap, and reappears at another. The question is no longer *where* but
  *which observations are the same entity*, and the metric is identity
  consistency (IDF1 / ID-switches).
- **Complementary / asymmetric views → Fuse and verify.** Each camera sees a
  *different aspect* of the event (an operator's action in one view, an item's
  state change in the other). There is no common point to intersect; recovery is
  temporal, not geometric — co-timing across the two views. The question is *who
  did what to which item, and when*, and the output is an
  action↔change association plus an order-verification status.

The organizing principle is **view overlap**: overlap present on a shared point →
triangulate; no overlap → hand off identity across the gap; overlap present but
each camera observing a *distinct facet* → fuse rather than triangulate. All
three modes consume the same analytic scene-manifest schema — per-entity,
per-camera `visible` labels plus ground-truth geometry — so a scene *declares*
its mode by its per-camera visibility structure rather than by configuration. An
entity visible in several cameras on a common point is a triangulation target; an
entity visible in exactly one camera at a time with a temporal gap is a handoff
target; an entity visible in one camera but never another (its complement seen
only in that other) is a fusion actor or item. The framing is domain-neutral
(assembly / order fulfilment: an operator assembles an order into a container at
a station), so the modes read across warehouse, manufacturing, and logistics
settings.

The producer of these manifests is the companion synthetic scene generator
`multicam-sim`, and **all three modes now consume real producer output**: the
triangulation mode a 3-camera occlusion sweep, the handoff mode a genuinely
non-overlapping two-station rig (`multicam_sim.dsl.CameraRig.stations`, disjoint
fields of view), and the fusion mode an asymmetric-visibility assembly-station
scene whose `order.json` carries placement-synced operator `actions[]`
(multicam-sim commit `3199ee4`). Each generated scene is committed as a
numpy-only fixture so CI never imports the sim (§4). One contract, three modes;
the producer integration is complete for all three.

---

## 2. Methods

Each mode is an open-closed pipeline of typed, deterministic, GPU-free stages
over the shared manifest. New backends (detectors, ReID embeddings, transit
models) are *added* behind Protocols; the cores are not *modified*.

### 2.1 Triangulate — overlapping views of one point

Cameras follow the pinhole model `P = K[R | t]` (OpenCV convention).
`build_ring_cameras` places `n` cameras on a circle, each looking at the origin
with a shared intrinsic `K`, giving a controllable ring with known ground-truth
calibration. An occlusion mask is a boolean array of shape `(n_points, n_views)`
where `True` means visible; `drop_k_mask` occludes exactly `k` views per point
under a seeded RNG and refuses any configuration leaving fewer than 2 visible
views (triangulation would be undefined). `occlude` replaces occluded 2D
observations with `NaN` so downstream code cannot silently consume them.

Recovery uses **linear DLT triangulation** (Hartley & Zisserman, Ch. 12). Each
visible view `i` contributes two rows of the constraint `x × (P X) = 0`:

```
u_i · P_i[2] − P_i[0] = 0
v_i · P_i[2] − P_i[1] = 0
```

Stacking rows from all visible views gives a homogeneous system `A X = 0`; the
solution is the right-singular vector of `A` for its smallest singular value
(SVD). `triangulate_dlt` takes the visibility mask directly, solves from the
visible subset only, and raises on fewer than 2 visible views or a recovered
point at infinity. Because exact geometry recovers to machine precision
regardless of camera count, the informative experiments inject Gaussian pixel
noise (default 0.5 px): more cameras over-determine the linear system and absorb
noise, which is what makes "how much does another camera buy you" non-trivial.
The single-view baseline is a monocular depth-prior estimator: with depth
unobservable from one ray, it fixes depth to a prior and reports the resulting
world error.

### 2.2 Handoff — non-overlapping views, one identity across a blind gap

Two station cameras watch disjoint areas. An entity leaves station 1's view,
crosses a blind gap the rig never observes (≈0.5 s), and reappears at station 2.
There is no shared ray, so the task is identity, not geometry. Five typed stages:

| Stage | Module | Responsibility |
| --- | --- | --- |
| 1. Tracklets | `mtmc.extract` / `mtmc.tracklet` | Per-camera single-view run of one target: observations, entry/exit **zone**, entry/exit time. The `id` is *local* to its camera; the linking GT identity is deliberately absent. |
| 2. Topology | `mtmc.topology` | Directed `CameraTopology` of *possible* transitions: exit-zone → entry-zone with a Gaussian transit-time distribution (mean ± k·std gate). |
| 3. Matcher | `mtmc.matcher` | `SpatioTemporalMatcher` scores an exiting→entering pair by spatio-temporal plausibility; appearance/ReID is a pluggable `AppearanceBackend` Protocol (default `NoAppearance` returns 1.0, so geometry and time alone decide). |
| 4. Global IDs | `mtmc.assignment` | `assign_global_ids` greedily accepts the highest-scoring handoffs under a strict 1-to-1 constraint and unions accepted pairs; connected components are the global identities, with a total tie-break order for determinism. |
| 5. Metrics | `mtmc.metrics` | IDF1 and ID-switches against GT identities. |

The matcher returns a hard veto (`None`) unless all hold: different cameras; a
topology edge matches the exit→entry zones; and the gap
`dt = entry_time − exit_time` is positive and inside the edge's transit window.
When admitted, the score is `temporal^(1−w) · appearance^w`. **IDF1** (Ristani et
al., 2016) is *defined* by the optimal one-to-one matching between predicted and
GT identities that maximizes correctly-identified observations (IDTP); we solve
that matching exactly with a compact Hungarian assignment (pure NumPy). A greedy
match under-counts IDTP whenever an identity's observations split across
predictions, so greedy is not IDF1 — a dedicated unit test builds an overlap
matrix where greedy scores IDTP = 3 and the optimal (and our) answer is 4. Ground
truth enters only the metrics stage; nothing in tracklet/topology/matcher/
assignment reads a GT identity, so the score is not "using the answer to get the
answer."

### 2.3 Fuse — complementary / asymmetric views, then verify the order

An operator assembles an order (a bill of materials) into a container at a
station. An **overview camera** sees the operator's action but foreshortens the
surface and occludes parts behind the hands; an **item camera** sees the parts
and the container filling but barely the operator. Neither view alone answers the
operational question; only together do they say *that action assembled that
part*, and only then can the assembled contents be reconciled against the order.

`partition_by_visibility` routes each entity by its asymmetric `visible` labels:
seen in the operator camera but never the item camera → **actor**; the reverse →
**item**; seen in both or neither → left out (not complementary). On the real
generated scene the operator actions are read directly from the producer's
placement-synced `order.json` `actions[]` (`operator_actions_from_order`), so the
action side needs no detection heuristic; the assembly-state changes are still
*derived* from the item camera's trajectory. When a scene ships no `actions[]`,
the same events are recovered from trajectories by deterministic default
detectors, each a Protocol:

- **`ReachActionDetector`** emits an `ActionEvent` at a strict local minimum of a
  tracked keypoint's height below the trajectory's own midline (the hand dipping
  to the station). No RNG, no absolute-scale threshold.
- **`DisplacementStateDetector`** emits an `AssemblyChangeEvent` when a part's
  tracked point jumps more than `min_step` between consecutive frames.

`TemporalProximityEstimator` (a `FusionEstimator`) associates each assembly
change with the action that best explains it under three constraints:
**causality** (action precedes change within an epsilon), **max lag** (change
follows within `max_lag`, default 0.5 s), and **one-to-one** (each action pairs
with at most one change, ties breaking to the smallest lag). An optional spatial
gate is off by default, since the action and the part are, in general, different
points — timing is the primary signal. Unmatched actions and changes are
*surfaced, not discarded*: refusing to pair them is a first-class outcome.
`verify_order` then reads the resulting `JointSceneState` against the order and
labels each line **fulfilled**, **missing**, **wrong** (an unexpected item
substituted for an expected one), or **extra** (an item the order never called
for) — a thin interpretation that changes neither the estimator nor the metric.

---

## 3. Results

All figures below are exact and pinned in CI, and **all three modes run on real
`multicam-sim` producer output** — the triangulation sweep, a non-overlapping
two-station rig (handoff), and an asymmetric-visibility assembly-station scene
(fusion) — each committed as a numpy-only fixture (§4).

**Triangulate — occlusion dose-response (3 cameras, seed `20260716`, 0.5 px
noise, 21 frames; one camera progressively occluded).** As the occluder grows on
camera 1's sightline, single-view recovery under a fixed depth prior degrades
steadily while multi-view triangulation from the surviving views barely moves.
See `docs/occlusion_dose_response.png`.

| Occlusion rate | Single-view error | Multi-view error | Multi-view advantage |
| --------------: | ----------------: | ---------------: | -------------------: |
| 0.0% | 0.1338 | 0.00276 | 48× |
| 4.8% | 0.1446 | 0.00278 | 52× |
| 7.9% | 0.1664 | 0.00306 | 54× |
| 14.3% | 0.2426 | 0.00307 | 79× |

The multi-view estimate stays recoverable at 100% throughout (it always retains
≥2 visible views); the single-view observed rate falls from 100% to 57% as its
one view is blocked. The advantage grows from 48× to 79× because the single-view
error climbs with the dose while the multi-view error is essentially flat.

**Handoff — MTMC identity across the blind gap** (real `multicam-sim`
non-overlapping two-station rig via `bench/gen_mtmc_scene.py`, two entities
crossing staggered in time, default no-appearance backend):

| Assignment | IDF1 | ID-switches |
| --- | ---: | ---: |
| Spatio-temporal handoff | **1.0** | **0** |
| No-handoff baseline (each tracklet its own id) | 0.625 | 2 |

Without cross-camera reasoning each object fractures into one identity per camera
(IDF1 0.625, 2 switches); the handoff recovers a single identity across the gap
(IDF1 1.0, 0 switches). The baseline is 0.625, not a round 0.5, because the real
generated sweep gives the two entities different-length visible runs — reported
as measured rather than tuned. Two entities are essential — with one, assigning
everything a single id trivially scores IDF1 = 1.0 — so the correct pairing
(a→a, b→b) must beat the cross pairing (a→b, b→a) on time alone, and it does. A
`StubEmbedding` ReID backend swapped in at appearance-weight 0.5 leaves the
result unchanged (IDF1 1.0, 0 switches), demonstrating the Protocol seam.

**Fuse — action / assembly-change association and order verification** (real
`multicam-sim` assembly-station scene via `bench/gen_fusion_scene.py`: an operator
places `part_a`, `part_b`, `part_c`, with the placement-synced `actions[]` and the
order BOM read from the producer's `order.json`; ground truth is three
interactions):

| Metric | Value |
| --- | ---: |
| True positives | 3 |
| False positives | 0 |
| False negatives | 0 |
| Precision | 1.00 |
| Recall | 1.00 |
| F1 | 1.00 |
| Mean action-time error | 0.0 s |
| Mean change-time error | 0.0 s |
| Mean fused lag | 0.0 s |

Each operator action links to the part it placed; nothing is refused. The mean
fused lag is 0.0 s because the producer's `actions[]` are placement-synced — the
action and the assembly change it causes are stamped at the same instant — so the
metric measures association, not a timing artefact. Reading the fused scene back
against the order, all three lines verify **fulfilled**. The reconciliation
reports shortfalls honestly on the same real data: a too-strict item detector
(seeing no placements) marks every line **missing**; an assembled item the order
never listed is **wrong**; a surplus of an expected item is **extra**.

The estimator's *refusal* behaviour — pairing nothing where a reach assembles
nothing, or a change falls outside the lag window — is what this clean generated
scene deliberately does not contain, so it is exercised separately as a controlled
estimator stress test (`tests/test_fusion.py`): there a too-wide 1.0 s lag window
wrongly links a distractor action, precision falls to 2/3 (recall stays 1.0), and
the spurious pairing surfaces as an **extra** order line rather than passing
silently — the metric registering a real mistake, which is what makes the perfect
score on the real scene meaningful.

---

## 4. Reproducibility

The core depends only on NumPy — no Blender, no Kubric, no GPU, no renderer, no
wall-clock. All randomness flows through explicit seeds; there is no global RNG
state. Two seeds appear in the benchmark: the dose-response sweep and its pixel
noise use `20260716` (recorded in `docs/occlusion_dose_response.json`), and the
triangulation hero smoke test pins seed `20260715` with a `1e-6` tolerance. The
handoff and fusion pipelines are seed-free (no RNG at all), so their fixtures are
exact.

One `make` target regenerates each mode's fixture from real `multicam-sim`
output; all are optional (they need the `bench` group) and CI runs only the
committed JSON:

```bash
make demo         # drive multicam-sim, run the numpy-only sweep, redraw the figure
make sweep        # triangulate: regen analytic manifests + curve JSON + test fixtures
make mtmc-scene   # handoff: regen the non-overlap two-station fixture (mtmc_stations.json)
make fusion-scene # fusion: regen the assembly-station manifest + order.json fixtures
make plot         # redraw docs/occlusion_dose_response.png from the committed JSON
make test         # the fast numpy-only gate: pytest over committed fixtures
make check        # full local gate: ruff check + ruff format --check + mypy src + pytest
```

The triangulation sweep and its pixel noise use seed `20260716`; the handoff and
fusion generators are seed-free (deterministic, no RNG), so their committed
fixtures are exact.

`make test` / `make check` need no `multicam-sim` and no matplotlib — CI runs
exactly this numpy-only gate over the committed fixtures under
`tests/fixtures/`. Regenerating the sweep or the figures (`make demo` / `make
plot`) uses the optional `bench` (multicam-sim) and `docs` (matplotlib)
dependency groups, keeping the core install lean. The hero block is
`uv run python examples/hero_demo.py`; the recovery-vs-views figure is
`uv run --group docs python docs/plot_recovery_vs_views.py`.

CI (`ruff` lint + format, `mypy --strict` on `src`, `pytest`) gates every change.

---

## 5. Limitations

These are prominent by design; the perfect identity and association scores must
be read against them.

- **Synthetic ground truth.** Every result comes from real `multicam-sim`
  producer output — exact analytic geometry and producer-declared events, not
  rendered or captured imagery. There is no perception detector: 2D observations
  are exact projections (triangulation) and the fusion operator actions are the
  producer's placement-synced `actions[]`, not detections. Perfect scores are a
  property of a controllable benchmark with exact ground truth, **not** a claim
  of real-world state of the art.
- **Naive baselines.** The single-view baseline is a fixed monocular depth prior;
  the MTMC no-handoff baseline gives each tracklet its own id; the fusion
  estimator is a greedy temporal-proximity matcher. These are honest lower
  bounds, not competitive methods — the contrast they provide is the proof, but
  they are not the strongest possible comparison.
- **Missing sources of difficulty — the next knobs.** The benchmark has no
  detector noise, no calibration drift, and no appearance ambiguity yet. These
  are exactly what would move the perfect scores off 1.0, and each is a tracked
  extension point:
  - *Detector / rendering noise*: photorealistic Kubric/Blender scenes with
    controllable occluders and detected (not projected) observations
    (multicam-occlusion #2), a robust/RANSAC triangulation variant for
    contaminated observations (#1), and fusion under noisy / incomplete
    observations — missed and false detections, timing jitter, dropped frames
    (#24).
  - *Harder regimes off the saturated scores*: harder MTMC scenes (longer blind
    gaps, simultaneous crossers, ID ambiguity) to unsaturate the IDF1 = 1.0 (#22),
    and an end-to-end 2D-pose → triangulated 3D-pose path with MPJPE under
    occlusion (#23).
  - *Calibration / geometric edge cases*: near-degenerate baseline tests (#9)
    and a richer transit model with graph validation for handoff (#11).
  - *Appearance ambiguity*: a learned ReID `AppearanceBackend` for handoff (#12)
    and a learned pose+action / object-state detector backend for fusion (#16),
    both behind Protocols that already exist.
  - *Metric and scale*: HOTA alongside IDF1 and an optimal Hungarian global
    assignment for handoff (#13); a PR-curve sweep over the fusion lag window and
    multi-operator scoring (#17); and a runnable assembly-station example driving
    fusion from the real asymmetric scene (#18).

---

## 6. Related work

**Multi-view 3D and triangulation.** The pinhole model, the DLT, and linear
triangulation are standard (Hartley & Zisserman, *Multiple View Geometry in
Computer Vision*, 2nd ed., 2004). Established multi-view 3D human-pose benchmarks
provide the real-world analogue of the triangulation mode: **Human3.6M**
(Ionescu et al., 2014), the **CMU Panoptic Studio** (Joo et al., 2015/2017), and
**TotalCapture** (Trumble et al., 2017) all recover 3D structure from calibrated
overlapping views and quantify occlusion robustness on captured imagery — the
regime our synthetic sweep abstracts.

**Multi-target multi-camera tracking.** The handoff mode sits in the MTMC / re-ID
literature. **IDF1** as an identity-based tracking metric is due to Ristani et
al. (2016); **HOTA** (Luiten et al., 2021) later decomposed detection and
association accuracy. The **NVIDIA AI City Challenge** and the **CityFlow**
benchmark (Tang et al., 2019) are the standard public MTMC / vehicle re-ID
testbeds, with real non-overlapping camera networks and appearance ambiguity our
geometry-and-time-only baseline deliberately omits.

**Complementary / contextual multi-view fusion.** The fusion mode draws on
multi-view activity and human-object-interaction understanding, where
distinct viewpoints contribute complementary evidence rather than redundant views
of one point. Human-object interaction detection (e.g. Gkioxari et al., 2018) and
multi-view action recognition frame the "who did what to which item" question our
temporal estimator addresses in miniature; the order-verification output connects
to assembly / procedure-step verification in industrial vision.

---

## References

1. R. Hartley and A. Zisserman. *Multiple View Geometry in Computer Vision*, 2nd
   ed. Cambridge University Press, 2004.
2. E. Ristani, F. Solera, R. Zou, R. Cucchiara, and C. Tomasi. Performance
   Measures and a Data Set for Multi-Target, Multi-Camera Tracking. *ECCV
   Workshops*, 2016. (IDF1)
3. J. Luiten, A. Ošep, P. Dendorfer, P. Torr, A. Geiger, L. Leal-Taixé, and
   B. Leibe. HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking.
   *International Journal of Computer Vision*, 2021.
4. C. Ionescu, D. Papava, V. Olaru, and C. Sminchisescu. Human3.6M: Large Scale
   Datasets and Predictive Methods for 3D Human Sensing in Natural Environments.
   *IEEE TPAMI*, 2014.
5. H. Joo et al. Panoptic Studio: A Massively Multiview System for Social Motion
   Capture. *ICCV*, 2015 (and *IEEE TPAMI*, 2017).
6. M. Trumble, A. Gilbert, C. Malleson, A. Hilton, and J. Collomosse. Total
   Capture: 3D Human Pose Estimation Fusing Video and Inertial Sensors. *BMVC*,
   2017.
7. Z. Tang et al. CityFlow: A City-Scale Benchmark for Multi-Target Multi-Camera
   Vehicle Tracking and Re-Identification. *CVPR*, 2019. (NVIDIA AI City
   Challenge)
8. G. Gkioxari, R. Girshick, P. Dollár, and K. He. Detecting and Recognizing
   Human-Object Interactions. *CVPR*, 2018.
9. K. Greff et al. Kubric: A Scalable Dataset Generator. *CVPR*, 2022. (planned
   synthetic scene generation)
