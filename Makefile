VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: venv install install-dev test lint format clean

venv:
	python3 -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -e .

install-dev: venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

test:
	$(VENV)/bin/pytest tests/ -v

lint:
	$(VENV)/bin/ruff check src/ dashboard/ data/ tests/

format:
	$(VENV)/bin/ruff format src/ dashboard/ data/ tests/
	$(VENV)/bin/ruff check --fix src/ dashboard/ data/ tests/

clean:
	rm -rf $(VENV) *.egg-info build dist __pycache__
