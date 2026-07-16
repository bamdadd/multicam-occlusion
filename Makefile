.PHONY: demo sweep plot mtmc-scene test check

# One-command hero: generate the occlusion sweep (drives multicam-sim), run the
# numpy-only recovery pipeline, print single-vs-multi numbers, and redraw the
# figure. Needs the optional bench + docs groups (multicam-sim, matplotlib).
demo: sweep plot
	@echo "hero figure -> docs/occlusion_dose_response.png"

# Regenerate the sweep: rebuild the analytic manifests and the curve JSON, and
# refresh the committed numpy-only test fixtures. Optional bench dependency.
sweep:
	uv run --group bench python bench/run_sweep.py

# Regenerate the committed MTMC handoff fixture from real multicam-sim output
# (non-overlapping stations rig -> build_manifest). Optional bench dependency; CI
# consumes the committed JSON and never imports multicam-sim.
mtmc-scene:
	uv run --group bench python bench/gen_mtmc_scene.py

# Redraw the figure from the committed curve JSON (no multicam-sim needed).
plot:
	uv run --group docs python docs/plot_occlusion_dose_response.py

# The fast, numpy-only gate: recovery pipeline over committed fixtures. No
# multicam-sim, no matplotlib — exactly what CI runs.
test:
	uv run pytest -q

# Full local gate, mirroring CI.
check:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src
	uv run pytest -q
