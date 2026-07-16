# multicam-occlusion

[![CI](https://github.com/bamdadd/multicam-occlusion/actions/workflows/ci.yml/badge.svg)](https://github.com/bamdadd/multicam-occlusion/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

📄 **Writeup:** [docs/note.md](docs/note.md) — the three camera-relationship modes
(triangulate / handoff / fusion), methods, and full results in one place.

> **As an object is occluded in more views, single-view 3D recovery blows up
> while multi-view recovery barely moves.**

![occlusion dose-response: single view blows up, multi view stays low](docs/occlusion_dose_response.png)

A single camera can never recover depth — its pixel fixes only a ray — so with a
fixed depth prior it sits at **~0.13 world-unit error even before any occlusion**
and climbs to **~0.24** as its one view is progressively blocked. Three
calibrated cameras triangulate from whichever views still see the point and stay
at **~0.003** throughout: **48× → 79× better, and the gap widens with the dose.**
The object moves through a depth range across 21 frames; the occluder grows on
camera 1's sightline; error is mean 3D distance to ground truth under 0.5px
keypoint noise.

**One command** (drives the [multicam-sim](https://github.com/bamdadd/multicam-sim)
producer, runs the numpy-only pipeline, redraws the figure):

```bash
make demo          # == bench/run_sweep.py (sweep + numbers) then the plot
```

| occlusion level | single-view error | multi-view error | multi-view is |
| --------------: | ----------------: | ---------------: | ------------: |
|            0.0% |            0.1338 |          0.00276 |     48× better |
|            4.8% |            0.1446 |          0.00278 |     52× better |
|            7.9% |            0.1664 |          0.00306 |     54× better |
|           14.3% |            0.2426 |          0.00307 |     79× better |

*Seeds: sweep + noise seed `20260716` (fully deterministic — pure NumPy, fixed
geometry, no RNG anywhere else). Hardware: results are hardware-independent by
construction; the exact figures above were generated on Apple Silicon (arm64),
macOS 15, Python 3.13 / NumPy 2.5. CI (Python 3.11) does not rerun the sweep — it
verifies the qualitative result (multi-view beats single-view, single-view climbs
with occlusion) on the committed fixtures. No GPU, no renderer.*

---

> **How much does a second or third camera actually buy you when an object is
> occluded in one view?**

When a point is hidden in one camera, a single view can never recover its 3D
position — one image constrains the point only to a viewing ray, so its depth
is unobservable. Add a second calibrated camera and the rays intersect: the
point is recovered exactly. Add more, and the estimate stops caring that any one
view was lost to occlusion. **This benchmark measures that trade-off** — how
single-view recovery fails, how multi-view recovery succeeds, and how recovery
error under realistic pixel noise falls as you add cameras.

![recovery error vs number of cameras](docs/recovery_vs_views.png)

*With 0.5px Gaussian pixel noise: more cameras over-determine the triangulation
and drive error down, and the cost of losing one view to occlusion (dashed)
shrinks toward zero as the ring grows. Regenerate with
`uv run python docs/plot_recovery_vs_views.py`.*

## Hero: single view fails, multi-view recovers the occluded point

Project one known 3D point into six synthetic ring cameras, occlude it in three
of them, and triangulate from the surviving views (`uv run python
examples/hero_demo.py`, **exact synthetic geometry, no pixel noise**):

```
ground truth point : [ 0.3 -0.4  0.8]
visible views      : [1, 4, 5] (3 of 6)
single view        : REFUSED -- need >= 2 visible views to triangulate
multi-view recovery: [ 0.3 -0.4  0.8]
max abs error      : 3.33e-16
```

A single view is *refused* — it cannot recover depth. Three surviving views
recover the occluded point to **machine precision (~3e-16)**. The figure above
shows what happens once you add realistic pixel noise: recovery is no longer
exact, and every extra camera measurably tightens it.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). The core is Blender-free and depends
only on NumPy.

```bash
uv sync                                   # install core + dev tools (numpy only)
uv run pytest -q                          # passing tests, incl. the dose-response
uv run python examples/hero_demo.py       # print the hero block below
```

The dose-response numbers run on committed fixtures with **no extra
dependencies** — `pytest` triangulates the real analytic manifests under
`tests/fixtures/sweep/`. Regenerating the sweep or the figures is optional:

```bash
make demo                                 # sweep (needs multicam-sim) + redraw figure
uv run --group docs python docs/plot_recovery_vs_views.py   # the vs-views figure
```

Triangulate a point yourself:

```python
import numpy as np
from multicam_occlusion import build_ring_cameras, project_points, triangulate_dlt

cams = build_ring_cameras(n_cameras=6)                 # (6, 3, 4) projection matrices
gt = np.array([0.3, -0.4, 0.8])
pts = np.vstack([project_points(p, gt)[0] for p in cams])

mask = np.ones(6, dtype=bool)
mask[[0, 2, 5]] = False                                # occlude 3 views
print(triangulate_dlt(cams, pts, mask=mask))           # ~[0.3, -0.4, 0.8]
```

## What runs today vs what's planned

**Runs today** — a self-contained, deterministic core:

- `build_ring_cameras` — synthetic calibrated pinhole cameras on a ring
  (`P = K[R|t]`, OpenCV convention).
- `triangulate_dlt` — linear Direct Linear Transformation triangulation from any
  subset of visible views; refuses degenerate (<2-view) solves.
- `drop_k_mask` / `occlude` — deterministic, seed-reproducible occlusion masks.
- **`ObservationManifest`** — a typed loader for multicam-sim's *analytic
  observation manifest* (per-frame per-camera `uv` + `visible` + `xyz_gt`),
  distinct from the pixel `FrameSource` seam. It rebuilds `P = K[R|t]` and
  self-checks that every stored pixel reprojects to ~1e-6 px.
- **`recover_trajectory`** — the dose-response pipeline: mask on `visible`,
  triangulate multi-view, and a monocular depth-prior baseline, with MPJPE-style
  error vs ground truth.
- The **occlusion dose-response sweep** above (`make demo`), the **hero smoke
  test**, and the **recovery-vs-views figure**, all from real runs.

**Planned (roadmap)** — photorealistic scene generation:

- **Kubric/Blender N-camera occlusion scenes**: a camera ring around rendered
  scenes with controllable occluders and per-view visibility masks, so the
  metric runs on rendered imagery instead of exact projections.
- Robust / RANSAC triangulation for outlier observations.
- Sweeping camera *count* (not just occlusion level) and >1 occluded camera on
  larger rings, with an aggregated multi-axis report and a CI artifact.

These are tracked as [good first issues](https://github.com/bamdadd/multicam-occlusion/issues).
See [DESIGN.md](DESIGN.md) for the full benchmark design and evaluation protocol.

## Three camera-relationship modes

Multiple cameras relate to one event in three geometric regimes; the divider is
**view overlap**. Each is a separate, typed, open-closed mode over the same
`multicam-sim` analytic manifest — complementary, not competing. **All three now
run on real `multicam-sim` producer output**, each committed as a numpy-only
fixture so CI never imports the sim.

- **Triangulate** (overlapping views, *the hero above*) — recover a 3D point from
  the views that still see it under occlusion. Real 3-camera occlusion sweep:
  single-view **~0.13 → ~0.24**, multi-view flat at **~0.003** (48× → 79×).
  `triangulation.py` / `recovery.py`.
- **Handoff / MTMC** (non-overlapping views, a blind gap) — keep one identity as
  an entity crosses between stations. Real non-overlap two-station rig
  (`make mtmc-scene`): handoff **IDF1 1.0 / 0 ID-switches** vs a no-handoff
  baseline **0.625 / 2**. `mtmc/`, [docs/mtmc-design.md](docs/mtmc-design.md).
- **Fusion** (complementary/asymmetric views) — one camera sees the operator's
  action, another the item/assembly state; fuse them into a joint "who did what
  to which item" and an order-verification status. Real assembly-station scene
  (`make fusion-scene`) with producer-synced operator `actions[]`: association
  **precision = recall = F1 = 1.0** (3 places, lag 0.0 s), all ordered parts
  **fulfilled**. `fusion/`, [docs/fusion-design.md](docs/fusion-design.md).

Framing is domain-neutral (assembly / order fulfilment) so it reads across
warehouse, manufacturing, and logistics. Perfect identity and association scores
are a property of a controllable benchmark with exact ground truth, not a claim
of real-world SOTA; the harder-regime knobs (detector noise, longer gaps,
appearance ambiguity) are tracked as open issues
([#22](https://github.com/bamdadd/multicam-occlusion/issues/22),
[#23](https://github.com/bamdadd/multicam-occlusion/issues/23),
[#24](https://github.com/bamdadd/multicam-occlusion/issues/24)). The full
three-mode writeup — methods, results table, and limitations — is in
[docs/note.md](docs/note.md).

## Method & references

Uses only public multi-view-geometry methods:

- Hartley & Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed. —
  the DLT and Chapter 12 triangulation.
- [Kubric](https://github.com/google-research/kubric) — planned synthetic scene
  generation.

## License

Apache-2.0 — see [LICENSE](LICENSE).
