# Design

This document specifies the **multicam-occlusion benchmark**: what it measures,
how synthetic occlusion scenes are generated, the single-view-vs-multi-view
metric, the triangulation method, and the evaluation protocol that keeps runs
reproducible.

> **Clean-room note.** Everything here is built from public multi-view-geometry
> material — Hartley & Zisserman's *Multiple View Geometry in Computer Vision*
> (2nd ed.) for the pinhole model and the DLT, and the public
> [Kubric](https://github.com/google-research/kubric) / Blender documentation
> for planned scene generation. No proprietary code or data is used or copied.

## Research question

> *How much does a second or third camera buy you when an object is occluded in
> one view?*

A single calibrated view maps a world point to a pixel via `x ~ P X`; inverting
it recovers only the **viewing ray**, not depth. Occlusion in that view removes
the observation entirely. The benchmark quantifies how additional calibrated
views (a) make recovery possible at all, and (b) make it *robust* — so that
losing one view to occlusion costs progressively less as the camera count grows.

## Three camera-relationship modes

Multiple cameras relate to one event in three fundamentally different geometric
regimes. This project treats each as a separate, typed, open-closed mode over the
same analytic manifest; they are complementary, not competing.

| Mode | Camera relationship | What it recovers | Metric | Where |
| --- | --- | --- | --- | --- |
| **Triangulate** (runs today, the hero) | **Overlapping** views of the same point | 3D position from the views that still see it despite occlusion | recovery error vs occlusion dose | `triangulation.py`, `recovery.py` |
| **Handoff / MTMC** | **Non-overlapping** views, a blind gap between stations | one consistent identity as an entity crosses the gap | IDF1 / ID-switches | `mtmc/`, [`docs/mtmc-design.md`](docs/mtmc-design.md) |
| **Fusion** | **Complementary / asymmetric** views (each sees a *different aspect* of the event) | a joint scene state — which operator action produced which assembly change | action↔assembly-change association + timing; order-verification status | `fusion/`, [`docs/fusion-design.md`](docs/fusion-design.md) |

The dividing line is **view overlap**: overlap → triangulate; no overlap → hand
off identity across the gap; overlap present but each camera observes a *distinct
facet* (operator activity vs item/assembly state) → fuse rather than triangulate.
All three consume `multicam-sim`'s analytic manifest (per-entity, per-camera
`visible` labels), so a scene declares which mode applies by its visibility
structure. Framing is domain-neutral (assembly / order fulfilment: an operator
assembles an order into a container at an assembly station), so the modes read
across warehouse, manufacturing, and logistics settings.

## Scene model

### Cameras (runs today)

Cameras follow the standard pinhole model `P = K [R | t]`, OpenCV convention
(+z along the view direction, +x right, +y down). `build_ring_cameras` places
`n` cameras on a circle of fixed radius and height, each `look_at` the origin,
with a shared intrinsic matrix `K` (focal length + principal point at the image
centre). This gives a controllable **camera ring** with known ground-truth
calibration — the geometric backbone of every experiment.

### Occlusion masks (runs today)

An occlusion mask is a boolean array of shape `(n_points, n_views)` where `True`
means visible. `drop_k_mask(n_views, n_points, k_drop, seed)` occludes exactly
`k_drop` views per point, chosen by a seeded RNG so identical seeds yield
identical masks. It refuses any configuration that leaves `< 2` visible views,
since triangulation is then impossible. `occlude` applies a mask to per-view 2D
observations, replacing occluded entries with `NaN` so downstream code cannot
silently consume an occluded observation.

### Synthetic Kubric/Blender scenes (roadmap)

The projection-based core uses *exact* geometry. The planned extension renders
photorealistic scenes with [Kubric](https://github.com/google-research/kubric)
(a Blender + PyBullet pipeline) to replace exact projections with detected
image observations:

- **Camera ring** around a scene, mirroring `build_ring_cameras`.
- **Controllable occluders** — foreground objects placed to hide the target in a
  chosen subset of views, producing a range of occlusion levels.
- **Per-view visibility masks** derived from Kubric's ground-truth segmentation
  / depth passes, so occlusion labels are exact rather than heuristic.
- Ground-truth 3D target positions from the scene graph, for error measurement.

This stays *planned* until the render pipeline lands; the metric and protocol
below are defined so they apply unchanged to rendered observations.

## Frame/Video source seam

Where do the per-view observations *come from*? The core (triangulation,
occlusion, metrics) must not care. It consumes a single narrow Protocol —
`FrameSource` (`src/multicam_occlusion/sources.py`) — and never imports a
concrete backend. This is the **open-closed seam**: new ingestion backends are
*added*, the core is not *modified*.

```
                 ┌─────────────────────────────┐
   producers ──▶ │  manifest (pydantic schema)  │ ◀── multicam-sim writes this
                 └─────────────┬───────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  FrameSource (Proto) │  ← the only thing the core sees
                    └──────────┬──────────┘
        ┌──────────────────────┼───────────────────────┐
        ▼                      ▼                        ▼
  FileFrameSource      (GStreamer/RTSP —          (DeepStream — future
  (v1, on disk)         future plugin)             hardware-decode plugin)
```

**The interface.** `FrameSource` exposes exactly two things:

- `cameras()` → per-camera calibration (`CameraCalibration`: `id`, intrinsics
  `K`, extrinsics `R | t`, plus a `projection_matrix()` that returns the same
  `P = K[R|t]` the triangulation core already consumes).
- `frames()` → an iterator of `FrameBundle`s, one per synchronized timestamp,
  yielded in ascending timestamp order (cameras within a bundle in manifest
  order) so a run is deterministic.

**The manifest** is a typed, JSON-serializable pydantic schema
(`Manifest` → `CameraCalibration[]` + `CameraStream[]`). Each `CameraStream` is
backed by *exactly one* of a **frame sequence** (`FrameRef[]`, each a path +
timestamp) or a **video** (`mp4` path + per-frame `timestamps`). Matrices travel
as nested lists (no NumPy in the wire format) and surface as arrays via methods,
so the schema stays dependency-light and round-trips cleanly.

**v1 backend — `FileFrameSource`.** Reads a manifest + frame sequences from
disk. Frame-sequence frames load with `numpy.load` (`.npy`), keeping CI
numpy-only; a video stream is exposed as **path metadata only** — no decoder
dependency, because the synthetic benchmark needs no live capture. `image` is
`None` for video-backed frames until a decoding backend fills it.

**Symmetry with multicam-sim.** The manifest the file backend *reads* is the
exact schema a synthetic producer (`multicam-sim`: `mp4` / frame-seq + manifest)
*writes*. One contract, two ends — the producer and the benchmark never drift.

**Live-capture backend (roadmap).** Real multi-camera capture — hardware
decode, RTSP ingest, cross-camera sync — is the domain of GStreamer / RTSP /
NVIDIA DeepStream, the industry-standard stack. Such a backend is added as
another `FrameSource` implementation (its own optional dependency group) and
plugs in **without touching the core**: same `cameras()` / `frames()` contract,
live pixels instead of `.npy` files. This is deliberately *not* built here — the
synthetic benchmark has no live-capture need — but the seam is where it lands.
Tracked as a `help wanted` issue.

## Triangulation method (runs today)

Recovery uses **linear DLT triangulation** (Hartley & Zisserman, Ch. 12). Each
visible view `i` contributes two rows of the constraint `x × (P X) = 0`:

```
u_i * P_i[2] - P_i[0] = 0
v_i * P_i[2] - P_i[1] = 0
```

Stacking the rows from all visible views gives a homogeneous system `A X = 0`;
the solution is the right-singular vector of `A` for its smallest singular
value (SVD). `triangulate_dlt` takes the visibility mask directly and solves
from the visible subset only, raising on `< 2` visible views (a ray, not a
point), on a rank-deficient system (`sigma[-1] / sigma[-2]` above
`DEGENERACY_RATIO_TOL`, which is what a target on the baseline of the
contributing views produces), or on a recovered point at infinity.

**Roadmap:** a robust variant (RANSAC over view subsets, or an M-estimator) for
observations contaminated by outliers, and an optional non-linear refinement
(minimising reprojection error) seeded from the DLT estimate.

## Metric

For a target with ground-truth position `X*`, recovered `X̂` from the visible
views, the primary error is **max absolute coordinate error**
`max_j |X̂_j - X*_j|` (tight, unit-interpretable). We report:

- **Single-view outcome** — categorical: recovery is *refused* (unobservable).
- **Multi-view recovery error** — the metric above, over the visible subset.
- **Occlusion cost** — the increase in recovery error from dropping one visible
  view to occlusion (`n` views vs `n-1`), as a function of camera count.

Because exact geometry recovers to machine precision regardless of camera
count, the informative experiments inject **Gaussian pixel noise** (default
`0.5px`): more cameras over-determine the linear system and absorb noise, which
is what makes "how much does another camera buy you" a non-trivial question.

## Evaluation protocol

Every run is pinned for reproducibility:

- **Seeds.** Occlusion masks and pixel noise are drawn from
  `numpy.random.default_rng(seed)`. Results are averaged over a fixed seed range
  (the figure uses 200 seeds); the seed set is part of the reported config.
- **Camera-count sweep.** Vary `n` over a fixed range (figure: 3–8). The minimum
  is 3 so that, after occluding one view, `n-1 ≥ 2` remain.
- **Occlusion levels.** Vary `k_drop` (views occluded per point) from 0 up to
  `n-2`. Each `(n, k_drop)` cell is a benchmark condition.
- **Deterministic assertions.** The hero smoke test pins a fixed seed
  (`20260715`) and tolerance (`1e-6`), and asserts both that single-view solves
  are refused and that the visible subset recovers ground truth. This runs in CI.
- **Figure regeneration.** `docs/plot_recovery_vs_views.py` reproduces the
  recovery-error-vs-views plot from scratch; matplotlib is an optional
  `docs`-group dependency, so CI stays lean (numpy-only core).

## Reproducibility & determinism

- Core depends only on NumPy; no Blender/Kubric needed to run the geometry.
- All randomness flows through explicit seeds — no global RNG state, no wall-clock.
- CI (`ruff` lint + format, `mypy --strict` on `src`, `pytest`) gates every change.
- The planned Kubric pipeline will pin Blender/Kubric versions and scene seeds so
  rendered scenes are regenerable bit-for-bit.

## References

- R. Hartley & A. Zisserman, *Multiple View Geometry in Computer Vision*, 2nd
  ed., Cambridge University Press, 2004 — pinhole model, DLT, Chapter 12
  triangulation.
- K. Greff et al., *Kubric: A scalable dataset generator*, and the public
  [Kubric repository](https://github.com/google-research/kubric) — planned
  synthetic scene generation.
