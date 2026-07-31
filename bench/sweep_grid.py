"""Sweep the full occlusion grid from DESIGN.md and publish table + figure.

Runs every ``(n_views, k_drop, seed)`` cell of the evaluation protocol in
DESIGN.md — camera counts 3–8, ``k_drop`` from 0 up to ``n-2`` per ring, each
averaged over a fixed seed range — and writes two artifacts:

* ``docs/recovery_grid.json`` — the machine-readable table: a config block plus
  one row per ``(n_views, k_drop)`` cell with the mean and std of the max-abs
  recovery error over the seeds.
* ``docs/recovery_grid.png`` — one ``o-`` curve per camera count, error against
  ``k_drop``, the same metric the other recovery figures plot.

Neither output is committed: CI regenerates both on every run and uploads them
as a build artifact (see the ``sweep-figure`` job in ``.github/workflows/ci.yml``),
so the published figure can never drift from the code that made it.

Determinism: every cell draws its occluded views with ``drop_k_mask(seed=s)``
and its pixel noise from ``default_rng(N_SEEDS + s)``, so the two streams are
disjoint and the whole grid is a pure function of the constants below. Both
outputs are byte-stable across runs.

matplotlib lives in the optional ``docs`` dependency group, so it is imported
inside :func:`render_figure` rather than at module scope — that keeps this
module importable (and testable) under the numpy-only dev environment CI's
``check`` job installs::

    uv run --group docs python bench/sweep_grid.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from multicam_occlusion import (
    build_ring_cameras,
    drop_k_mask,
    project_points,
    triangulate_dlt,
)

ROOT = Path(__file__).resolve().parent.parent

# ---- evaluation protocol, verbatim from DESIGN.md (not tuned to the result) -- #
GT = np.array([0.3, -0.4, 0.8])
PIXEL_NOISE = 0.5  # px Gaussian keypoint noise (DESIGN.md default)
#: Seeds are ``range(N_SEEDS)``; the noise stream of seed ``s`` is ``N_SEEDS + s``,
#: so this one constant fixes both the seed set and the stream offset.
N_SEEDS = 200
#: Start at 3: after occluding one view, ``n-1 >= 2`` must remain to triangulate.
VIEW_COUNTS = list(range(3, 9))

JSON_OUT = ROOT / "docs" / "recovery_grid.json"
PNG_OUT = ROOT / "docs" / "recovery_grid.png"


def k_drops_for(n_views: int) -> list[int]:
    """Occlusion levels swept on an ``n_views`` ring: 0 up to ``n-2`` inclusive."""
    return list(range(n_views - 1))


class GridCell(BaseModel):
    """One ``(n_views, k_drop)`` benchmark condition, aggregated over the seeds."""

    n_views: int
    k_drop: int
    n_visible: int  # views that survive the occlusion, i.e. n_views - k_drop
    mean_error: float  # mean over seeds of the max-abs recovery error
    std_error: float


class GridConfig(BaseModel):
    """The pinned configuration the grid was run under."""

    n_seeds: int
    pixel_noise: float
    gt: list[float]
    view_counts: list[int]
    k_drops_by_view_count: dict[int, list[int]]


class GridResult(BaseModel):
    """The full sweep: the config it ran under plus every grid cell."""

    config: GridConfig
    cells: list[GridCell]


def project_all(proj_mats: npt.NDArray[np.float64], point3d: np.ndarray) -> np.ndarray:
    """Project one 3D point into every camera; returns ``(n_views, 2)``."""
    return np.vstack([project_points(p, point3d)[0] for p in proj_mats])


def cell_errors(
    proj_mats: npt.NDArray[np.float64],
    clean: np.ndarray,
    n_views: int,
    k_drop: int,
) -> list[float]:
    """Max-abs recovery error for each of the ``N_SEEDS`` trials of one cell."""
    errors = []
    for seed in range(N_SEEDS):
        mask = drop_k_mask(n_views=n_views, n_points=1, k_drop=k_drop, seed=seed)[0]
        # Offset the noise stream so it is disjoint from drop_k_mask's stream:
        # seeding both with the same integer would couple mask and noise to one
        # generator.
        rng = np.random.default_rng(N_SEEDS + seed)
        noisy = clean + rng.normal(0.0, PIXEL_NOISE, clean.shape)
        recovered = triangulate_dlt(proj_mats, noisy, mask=mask)
        errors.append(float(np.max(np.abs(recovered - GT))))
    return errors


def sweep() -> GridResult:
    """Run every cell of the grid and aggregate it into a :class:`GridResult`."""
    cells: list[GridCell] = []
    for n_views in VIEW_COUNTS:
        proj_mats = build_ring_cameras(n_cameras=n_views)
        clean = project_all(proj_mats, GT)
        for k_drop in k_drops_for(n_views):
            errors = cell_errors(proj_mats, clean, n_views, k_drop)
            cells.append(
                GridCell(
                    n_views=n_views,
                    k_drop=k_drop,
                    n_visible=n_views - k_drop,
                    mean_error=float(np.mean(errors)),
                    std_error=float(np.std(errors)),
                )
            )
    config = GridConfig(
        n_seeds=N_SEEDS,
        pixel_noise=PIXEL_NOISE,
        gt=[float(x) for x in GT],
        view_counts=list(VIEW_COUNTS),
        k_drops_by_view_count={n: k_drops_for(n) for n in VIEW_COUNTS},
    )
    return GridResult(config=config, cells=cells)


def render_figure(result: GridResult, out_path: Path) -> None:
    """Draw one ``o-`` curve of mean error vs ``k_drop`` per camera count."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for n_views in result.config.view_counts:
        cells = [c for c in result.cells if c.n_views == n_views]
        ax.plot(
            [c.k_drop for c in cells],
            [c.mean_error for c in cells],
            "o-",
            label=f"{n_views} cameras",
        )
    ax.set_xlabel("number of occluded views (k_drop)")
    ax.set_ylabel(f"mean recovery error (max abs, {result.config.pixel_noise}px noise)")
    ax.set_title("Recovery error across the full occlusion grid")
    ax.set_xticks(sorted({c.k_drop for c in result.cells}))
    ax.grid(True, alpha=0.3)
    # Outside the axes: every curve rises to the right, so an in-axes legend
    # covers the widest ring's own data points.
    ax.legend(title="ring size", loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    result = sweep()
    JSON_OUT.write_text(result.model_dump_json(indent=2))
    render_figure(result, PNG_OUT)

    print(f"{'n_views':>8} {'k_drop':>7} {'n_visible':>10} {'mean_err':>12} {'std_err':>12}")
    for cell in result.cells:
        print(
            f"{cell.n_views:8d} {cell.k_drop:7d} {cell.n_visible:10d} "
            f"{cell.mean_error:12.6f} {cell.std_error:12.6f}"
        )
    print(f"\nwrote {JSON_OUT.relative_to(ROOT)}")
    print(f"wrote {PNG_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
