# palletjack - Agent Onboarding Guide

## Project Structure

Palletjack updates ArcGIS Online feature services from external data using an extract-transform-load workflow. Keep source-specific loading in `extract.py`, DataFrame cleanup and geocoding in `transform.py`, and ArcGIS service updates in `load.py`. Put shared helpers, such as retry behavior, in `utils.py`.

Preserve pandas DataFrame interfaces. Use type hints and Google-style docstrings for public APIs. Follow the existing module-level logging pattern: `logger = logging.getLogger(__name__)`.

## Python Environment and Validation

Use the Conda environment named `palletjack` for all Python commands, tests, and build validation. Python 3.11 through 3.14 are supported.

Run focused tests for changed behavior with `pytest tests/test_<module>.py`, then run `pytest` when the change affects shared or public behavior. Run `ruff check .` before completion. Do not introduce unconfigured formatters or type checkers.

Tests use `pytest-mock`; prefer `mocker.patch(..., autospec=True)` where practical. Use standard assertions and `pandas.testing` helpers when comparing DataFrames.

## Documentation and Commits

Keep public documentation in Google-style docstrings, which are rendered with pdoc3. Update README or docs examples when changing the public API.

Use conventional commits as outlined here: https://github.com/agrc/release-composite-action/blob/main/README.md#commits. Use a relevant scope when helpful, such as `fix(extract):`, `feat(load):`, or `docs:`.
