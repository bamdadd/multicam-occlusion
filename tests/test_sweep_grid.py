"""Pin the occlusion sweep grid produced by ``bench/sweep_grid.py``.

Numpy-only by construction: the harness defers its matplotlib import into
``render_figure``, so importing its compute path here works in the dev-only
environment CI's ``check`` job installs. ``bench/`` is not a package and pytest
does not collect it, so the module is loaded by path.

Two things are asserted, both of which a regression would break:

* the DESIGN.md evaluation protocol — the grid's shape and config — and the
  aggregated error of three cells, pinned as literals measured from a run and
  pasted in, sharing no constant with the harness;
* byte-stability: two invocations of the sweep serialise identically, which is
  what lets CI publish the JSON/PNG as an artifact instead of committing them.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

MODULE_PATH = Path(__file__).resolve().parent.parent / "bench" / "sweep_grid.py"

TOL = 1e-12

# Golden aggregates, measured once and pasted in. Independent of the harness's
# own constants: if the grid, the seed streams, the noise level or the
# triangulation change, these stop matching.
GOLDEN: dict[tuple[int, int], tuple[float, float]] = {
    # (n_views, k_drop): (mean_error, std_error)
    (3, 0): (0.002496702151184283, 0.0010825297143640749),
    (8, 0): (0.0015825508878552078, 0.0007088486802471763),
    (8, 6): (0.00375640234198234, 0.0024068743195286915),
}

# The full (n_views, k_drop) grid DESIGN.md prescribes: rings of 3..8 cameras,
# k_drop from 0 up to n-2 so at least two views always survive.
EXPECTED_GRID = [
    (3, 0),
    (3, 1),
    (4, 0),
    (4, 1),
    (4, 2),
    (5, 0),
    (5, 1),
    (5, 2),
    (5, 3),
    (6, 0),
    (6, 1),
    (6, 2),
    (6, 3),
    (6, 4),
    (7, 0),
    (7, 1),
    (7, 2),
    (7, 3),
    (7, 4),
    (7, 5),
    (8, 0),
    (8, 1),
    (8, 2),
    (8, 3),
    (8, 4),
    (8, 5),
    (8, 6),
]


def _load_sweep_grid() -> Any:
    """Import ``bench/sweep_grid.py`` by path (``bench/`` is not a package)."""
    spec = importlib.util.spec_from_file_location("sweep_grid", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sweep_grid = _load_sweep_grid()


@pytest.fixture(scope="module")
def runs() -> tuple[Any, Any]:
    """Two independent invocations of the sweep, same config."""
    return sweep_grid.sweep(), sweep_grid.sweep()


def test_config_matches_the_design_protocol(runs: tuple[Any, Any]) -> None:
    """The pinned config is the DESIGN.md evaluation protocol, not something else."""
    config = runs[0].config
    assert config.n_seeds == 200
    assert config.pixel_noise == pytest.approx(0.5, abs=TOL)
    assert config.gt == pytest.approx([0.3, -0.4, 0.8], abs=TOL)
    assert config.view_counts == [3, 4, 5, 6, 7, 8]
    assert config.k_drops_by_view_count == {
        3: [0, 1],
        4: [0, 1, 2],
        5: [0, 1, 2, 3],
        6: [0, 1, 2, 3, 4],
        7: [0, 1, 2, 3, 4, 5],
        8: [0, 1, 2, 3, 4, 5, 6],
    }


def test_grid_covers_every_cell_once(runs: tuple[Any, Any]) -> None:
    """Every ``(n, k_drop)`` condition appears exactly once, in a stable order."""
    cells = runs[0].cells
    assert [(c.n_views, c.k_drop) for c in cells] == EXPECTED_GRID
    for cell in cells:
        assert cell.n_visible == cell.n_views - cell.k_drop
        assert cell.n_visible >= 2  # below two views the DLT has no solution


def test_golden_cell_aggregates(runs: tuple[Any, Any]) -> None:
    """Three cells are pinned to measured literals: the seeded pipeline must not drift."""
    by_key = {(c.n_views, c.k_drop): c for c in runs[0].cells}
    for key, (mean_error, std_error) in GOLDEN.items():
        cell = by_key[key]
        assert cell.mean_error == pytest.approx(mean_error, abs=TOL), key
        assert cell.std_error == pytest.approx(std_error, abs=TOL), key


def test_more_cameras_beat_fewer_at_equal_occlusion(runs: tuple[Any, Any]) -> None:
    """The benchmark's thesis: at the same k_drop, a bigger ring recovers better."""
    by_key = {(c.n_views, c.k_drop): c.mean_error for c in runs[0].cells}
    for k_drop in range(7):
        errors = [by_key[(n, k_drop)] for n in range(3, 9) if (n, k_drop) in by_key]
        assert errors == sorted(errors, reverse=True), k_drop


def test_two_runs_serialise_identically(runs: tuple[Any, Any]) -> None:
    """The published JSON is byte-stable, so the CI artifact is reproducible."""
    first, second = runs
    assert first.model_dump_json(indent=2) == second.model_dump_json(indent=2)
