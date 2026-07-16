"""Render the occlusion dose-response hero figure from the committed sweep.

Reads ``docs/occlusion_dose_response.json`` (produced by ``bench/run_sweep.py``)
and draws recovery error vs occlusion level on a log-y axis: the single-view
baseline climbing as its one camera loses the point, the multi-view estimator
staying near flat because the surviving cameras cover for the occluded one.

Only depends on matplotlib (the optional ``docs`` group) — it consumes the
committed JSON, so it does NOT need multicam-sim. Regenerate the PNG with::

    uv run --group docs python docs/plot_occlusion_dose_response.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def main() -> None:
    data = json.loads((HERE / "occlusion_dose_response.json").read_text())
    points = data["points"]
    cfg = data["config"]

    occ = [p["occlusion_rate"] * 100.0 for p in points]
    single = [p["single_mpjpe"] for p in points]
    multi = [p["multi_mpjpe"] for p in points]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(
        occ,
        single,
        "o-",
        color="#c0392b",
        linewidth=2.2,
        markersize=6,
        label=f"single view (cam {cfg['single_cam']} + depth prior)",
    )
    ax.plot(
        occ,
        multi,
        "s-",
        color="#1e6f3c",
        linewidth=2.2,
        markersize=6,
        label=f"multi view ({cfg['n_cams']} cams, DLT on visible)",
    )

    ax.set_yscale("log")
    ax.set_xlabel("occlusion level  (% of camera-frame observations occluded)")
    ax.set_ylabel("mean 3D recovery error  (world units, log scale)")
    ax.set_title(
        "Occlusion dose-response: one view can't recover depth, more views can\n"
        f"3-camera scene, {cfg['num_frames']} frames, {cfg['pixel_noise']}px keypoint "
        f"noise, seed {cfg['seed']}"
    )
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="center right", frameon=True)

    # Annotate the widening gap at the highest dose.
    hi = max(range(len(occ)), key=lambda i: occ[i])
    ratio = single[hi] / multi[hi]
    ax.annotate(
        f"{ratio:.0f}x better",
        xy=(occ[hi], multi[hi]),
        xytext=(occ[hi] - 4.0, multi[hi] * 3.0),
        fontsize=9,
        color="#1e6f3c",
        arrowprops={"arrowstyle": "->", "color": "#1e6f3c"},
    )

    fig.tight_layout()
    out = HERE / "occlusion_dose_response.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
