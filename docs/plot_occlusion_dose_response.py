"""Render the two-panel occlusion dose-response hero figure.

Reads ``docs/occlusion_dose_response.json`` (from ``bench/run_sweep.py``) and draws,
against the occlusion dose ``phi`` (mean ``1 - visible_fraction`` over joint x camera):

* LEFT: mean per-joint 3D error on RECOVERABLE joints, single vs multi, log-y, with
  3-seed error bars. This is MPJPE-on-recoverable, so a flat multi curve can be
  survivorship.
* RIGHT: coverage, the fraction of joint-frames each estimator can actually solve
  (multi: >= 2 visible cameras; single: its best camera saw the joint). This is where
  the survivorship in the left panel shows up.

The two panels together are the honest statement: never MPJPE without its coverage.

Only depends on matplotlib (the ``docs`` group); it consumes the committed JSON and
does NOT need multicam-sim. Regenerate::

    uv run --group docs python docs/plot_occlusion_dose_response.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SINGLE = "#c0392b"
MULTI = "#1e6f3c"


def main() -> None:
    data = json.loads((HERE / "occlusion_dose_response.json").read_text())
    cfg = data["config"]
    bins = [b for b in data["bins"] if b["n_frames"] > 0]

    phi = [b["phi_mid"] for b in bins]
    fig, (ax_err, ax_cov) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # LEFT: error on recoverable joints (log-y), with seed error bars.
    err_bins = [b for b in bins if b["multi_mpjpe_mean"] is not None]
    ephi = [b["phi_mid"] for b in err_bins]
    ax_err.errorbar(
        ephi,
        [b["single_mpjpe_mean"] for b in err_bins],
        yerr=[b["single_mpjpe_std"] for b in err_bins],
        fmt="o-",
        color=SINGLE,
        linewidth=2.2,
        markersize=6,
        capsize=3,
        label="single view (per-joint best cam + depth prior)",
    )
    ax_err.errorbar(
        ephi,
        [b["multi_mpjpe_mean"] for b in err_bins],
        yerr=[b["multi_mpjpe_std"] for b in err_bins],
        fmt="s-",
        color=MULTI,
        linewidth=2.2,
        markersize=6,
        capsize=3,
        label="multi view (DLT on visible cams)",
    )
    ax_err.set_yscale("log")
    ax_err.set_xlabel(r"occlusion dose  $\varphi$  = mean$(1-$visible_fraction$)$")
    ax_err.set_ylabel("3D error on recoverable joints (world units, log)")
    ax_err.set_title("Per-joint error: multi stays low, single is depth-blind")
    ax_err.grid(True, which="both", linestyle=":", alpha=0.5)
    ax_err.legend(loc="center left", frameon=True, fontsize=8)

    # RIGHT: coverage (fraction of joint-frames solvable).
    ax_cov.errorbar(
        phi,
        [b["single_observed_mean"] for b in bins],
        yerr=[b["single_observed_std"] for b in bins],
        fmt="o-",
        color=SINGLE,
        linewidth=2.2,
        markersize=6,
        capsize=3,
        label="single view (best camera saw joint)",
    )
    ax_cov.errorbar(
        phi,
        [b["multi_recoverable_mean"] for b in bins],
        yerr=[b["multi_recoverable_std"] for b in bins],
        fmt="s-",
        color=MULTI,
        linewidth=2.2,
        markersize=6,
        capsize=3,
        label="multi view (>= 2 cameras saw joint)",
    )
    ax_cov.set_ylim(0.0, 1.03)
    ax_cov.set_xlabel(r"occlusion dose  $\varphi$  = mean$(1-$visible_fraction$)$")
    ax_cov.set_ylabel("coverage (fraction of joint-frames solvable)")
    ax_cov.set_title("Coverage: the flat error curve hides dropped joints")
    ax_cov.grid(True, linestyle=":", alpha=0.5)
    ax_cov.legend(loc="lower left", frameon=True, fontsize=8)

    fig.suptitle(
        "Occlusion dose-response: single-view vs multi-view 3D pose "
        f"(COCO-17, {cfg['n_cams']} cams, {cfg['num_frames']} frames, "
        f"{cfg['pixel_noise']}px noise, seeds {cfg['seeds']})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = HERE / "occlusion_dose_response.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
