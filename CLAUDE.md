# CLAUDE.md

## Project

LOB Regime Scanner: regime detection on cryptocurrency order books. Level 2 snapshots are turned into order flow and liquidity features, a Gaussian hidden Markov model infers latent regimes, and a Plotly Dash dashboard renders the result. An order-book engine written in C++17 is exposed through pybind11; the Python code falls back to a pure-Python path if the extension is not importable.

## Setup

```bash
make install-dev
source .venv/bin/activate
make test
```

## Test

```bash
make test
# or directly:
.venv/bin/pytest tests/ -v
```

## Lint and format

```bash
make lint      # ruff check
make format    # ruff format, then ruff check --fix
```

## Structure

- `src/`: core library (data_loader, book_reconstructor, features, hmm_model, backtest)
- `src/cpp/`: C++17 order-book engine and pybind11 bindings, built by `setup.py`
- `data/`: download and synthetic-data scripts; output goes in `data/raw/` (gitignored)
- `dashboard/`: Plotly Dash app
- `tests/`: pytest unit tests
- `benchmarks/`: throughput benchmark for the C++ engine (`make bench`)
- `notebooks/`: exploratory Jupyter notebooks

## Notes

- The install, test, lint, format, and bench targets in the Makefile use the interpreter in `.venv/`, which `make install-dev` creates.
- `pyproject.toml` holds the dependency list used by the Makefile and by CI; `requirements.txt` is not consumed by either.
- Data files (`*.parquet`, `*.csv.gz`, `data/raw/`) are gitignored.
