"""Render recovery-error vs number-of-views under pixel noise.

Produces ``docs/recovery_vs_views.png`` with two curves:

  * "all views visible" -- error triangulating from every camera;
  * "one view occluded" -- error after dropping a single view to occlusion,
    i.e. triangulating from ``n - 1`` cameras.

Both are averaged over many seeds of Gaussian pixel noise. The point: with
noise-free geometry recovery is exact regardless of camera count, so the honest
question is robustness *under noise*. More cameras over-determine the linear DLT
system and absorb noise -- and absorb a dropped (occluded) view.

Not run in CI (matplotlib lives in the optional ``docs`` dependency group)::

    uv sync --group docs
    uv run python docs/plot_recovery_vs_views.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from multicam_occlusion import build_ring_cameras, project_points, triangulate_dlt

GT = np.array([0.3, -0.4, 0.8])
PIXEL_NOISE = 0.5
N_SEEDS = 200
# Start at 3: with one view occluded we still need >= 2 to triangulate.
VIEW_COUNTS = list(range(3, 9))


def project_all(proj_mats: np.ndarray, point3d: np.ndarray) -> np.ndarray:
    """Project one 3D point into every camera; returns ``(n_views, 2)``."""
    return np.vstack([project_points(p, point3d)[0] for p in proj_mats])


def mean_error(n_views: int, drop_one: bool) -> float:
    """Mean max-abs recovery error over ``N_SEEDS`` noisy trials."""
    errors = []
    for seed in range(N_SEEDS):
        proj_mats = build_ring_cameras(n_cameras=n_views)
        clean = project_all(proj_mats, GT)
        rng = np.random.default_rng(seed)
        noisy = clean + rng.normal(0.0, PIXEL_NOISE, clean.shape)
        mask = np.ones(n_views, dtype=bool)
        if drop_one:
            mask[rng.integers(n_views)] = False
        recovered = triangulate_dlt(proj_mats, noisy, mask=mask)
        errors.append(float(np.max(np.abs(recovered - GT))))
    return float(np.mean(errors))


def main() -> None:
    all_visible = [mean_error(n, drop_one=False) for n in VIEW_COUNTS]
    one_occluded = [mean_error(n, drop_one=True) for n in VIEW_COUNTS]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(VIEW_COUNTS, all_visible, "o-", label="all views visible")
    ax.plot(VIEW_COUNTS, one_occluded, "s--", label="one view occluded (n-1 used)")
    ax.set_xlabel("number of cameras on the ring")
    ax.set_ylabel(f"mean recovery error (max abs, {PIXEL_NOISE}px noise)")
    ax.set_title("More cameras buy occlusion robustness")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = "docs/recovery_vs_views.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")
    for n, a, o in zip(VIEW_COUNTS, all_visible, one_occluded, strict=True):
        print(f"  n={n}: all={a:.5f}  one_occluded={o:.5f}")


if __name__ == "__main__":
    main()
