"""Render recovery-error vs number of occluded views under pixel noise.

Produces ``docs/recovery_vs_occlusion.png``: with a fixed ring of ``N_CAMERAS``
cameras, occlude ``k_drop`` of them (swept from 0 to ``N_CAMERAS - 2``) and
measure how the DLT recovery error grows as fewer views contribute to the solve.
Averaged over ``N_SEEDS`` seeds of 0.5px Gaussian pixel noise, with the occluded
views drawn per seed by ``drop_k_mask``.

Not run in CI (matplotlib lives in the optional ``docs`` dependency group)::

    uv run --group docs python docs/plot_recovery_vs_occlusion.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from multicam_occlusion import (
    build_ring_cameras,
    drop_k_mask,
    project_points,
    triangulate_dlt,
)

HERE = Path(__file__).resolve().parent

GT = np.array([0.3, -0.4, 0.8])
PIXEL_NOISE = 0.5
N_SEEDS = 200
N_CAMERAS = 8
# Stop at n-2: at least two views must stay visible to triangulate.
K_DROPS = list(range(0, N_CAMERAS - 1))


def project_all(proj_mats: np.ndarray, point3d: np.ndarray) -> np.ndarray:
    """Project one 3D point into every camera; returns ``(n_views, 2)``."""
    return np.vstack([project_points(p, point3d)[0] for p in proj_mats])


def mean_error(proj_mats: np.ndarray, clean: np.ndarray, k_drop: int) -> float:
    """Mean max-abs recovery error over ``N_SEEDS`` noisy trials."""
    errors = []
    for seed in range(N_SEEDS):
        mask = drop_k_mask(n_views=N_CAMERAS, n_points=1, k_drop=k_drop, seed=seed)[0]
        # Offset the noise stream so it is disjoint from drop_k_mask's stream:
        # seeding both with the same integer would couple mask and noise to one
        # generator.
        rng = np.random.default_rng(N_SEEDS + seed)
        noisy = clean + rng.normal(0.0, PIXEL_NOISE, clean.shape)
        recovered = triangulate_dlt(proj_mats, noisy, mask=mask)
        errors.append(float(np.max(np.abs(recovered - GT))))
    return float(np.mean(errors))


def main() -> None:
    proj_mats = build_ring_cameras(n_cameras=N_CAMERAS)
    clean = project_all(proj_mats, GT)
    means = [mean_error(proj_mats, clean, k) for k in K_DROPS]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(K_DROPS, means, "o-", label=f"{N_CAMERAS}-camera ring")
    ax.set_xlabel("number of occluded views (k_drop)")
    ax.set_ylabel(f"mean recovery error (max abs, {PIXEL_NOISE}px noise)")
    ax.set_title(f"Recovery error grows as views are occluded ({N_CAMERAS} cameras)")
    ax.set_xticks(K_DROPS)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = HERE / "recovery_vs_occlusion.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")
    for k, m in zip(K_DROPS, means, strict=True):
        print(f"  k_drop={k}: visible={N_CAMERAS - k}  mean={m:.6f}")


if __name__ == "__main__":
    main()
