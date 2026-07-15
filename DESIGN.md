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
point) or a recovered point at infinity (degenerate configuration).

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
