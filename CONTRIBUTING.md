# Contributing

Thanks for helping improve multicam-occlusion.

## Dev setup

```bash
uv sync                # install the project + dev tools
uv run pytest -q       # run the test suite
```

Run `pre-commit install` (or `uv run ruff format .` before pushing) — CI enforces ruff format and will fail otherwise.

The full local gate mirrors CI:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -q
```

## Where to start

- Architecture and the three camera-relationship modes: [DESIGN.md](DESIGN.md).
- Open work and good first issues: <https://github.com/bamdadd/multicam-occlusion/issues>.
