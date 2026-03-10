VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: venv install install-dev test clean

venv:
	python3 -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -e .

install-dev: venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

test: install-dev
	$(VENV)/bin/pytest tests/ -v

clean:
	rm -rf $(VENV) *.egg-info build dist __pycache__
