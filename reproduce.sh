#!/usr/bin/env bash
# Regenerate the occlusion dose-response finding end to end, deterministically.
#
#   ./reproduce.sh
#
# Drives the multicam-sim producer to build the posed hand-sweep manifest, runs
# the numpy-only recovery over seeds {0,1,2}, writes docs/occlusion_dose_response.json
# and the committed test fixture, then redraws the two-panel hero figure. Fixed
# seeds + geometry with no wall-clock or RNG in the scene, so the outputs are
# byte-stable across runs on the same architecture.
set -euo pipefail
cd "$(dirname "$0")"

echo ">> sweep (producer + recovery, seeds 0/1/2)"
uv run --group bench python bench/run_sweep.py

echo ">> plot (two-panel hero: MPJPE + coverage)"
uv run --group docs python docs/plot_occlusion_dose_response.py

echo ">> done: docs/occlusion_dose_response.png"
