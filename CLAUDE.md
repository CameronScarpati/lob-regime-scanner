# CLAUDE.md

## Project

LOB Regime Scanner — HMM-based market microstructure analytics for cryptocurrency order books.

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

## Structure

- `src/` — Core library (data_loader, book_reconstructor, features, hmm_model, backtest)
- `data/` — Download scripts; raw files go in `data/raw/` (gitignored)
- `dashboard/` — Plotly Dash app
- `tests/` — pytest unit tests
- `notebooks/` — Exploratory Jupyter notebooks

## Rules

- **NEVER add Claude as a co-author on any commit.** Do not use `Co-authored-by` trailers referencing Claude, Anthropic, or any AI assistant.
- **All commits must be authored as the repo owner.** Before committing, always set repo-level git identity:
  ```bash
  git config user.name "CameronScarpati"
  git config user.email "138163850+CameronScarpati@users.noreply.github.com"
  ```
  Never commit as "Claude", "Anthropic", or any AI-related identity.
- Use the virtual environment (`.venv/`) for all Python operations. Never install with bare `pip` outside the venv.
- Dependencies are managed in `pyproject.toml`, not `requirements.txt`.
- Run `make test` before committing to verify nothing is broken.
- Data files (`*.parquet`, `*.csv.gz`, `data/raw/`) are gitignored — never commit them.
