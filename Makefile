PYTHON := python3
PIP := pip

.PHONY: help install install-dev precommit format lint typecheck test run docker-build docker-run

help:
	@echo "Common targets:"
	@echo "  make install        # install package"
	@echo "  make install-dev    # install with dev extras"
	@echo "  make precommit      # install pre-commit hooks"
	@echo "  make format         # run black and isort"
	@echo "  make lint           # flake8"
	@echo "  make typecheck      # mypy"
	@echo "  make test           # pytest"
	@echo "  make run            # run uvicorn locally"
	@echo "  make docker-build   # build container"
	@echo "  make docker-run     # run container"

install:
	$(PIP) install -U pip
	$(PIP) install .

install-dev:
	$(PIP) install -U pip
	$(PIP) install '.[dev]'

precommit:
	pre-commit install

format:
	black src tests scripts
	isort src tests scripts

lint:
	flake8 src tests scripts

typecheck:
	mypy src tests

test:
	pytest -q

run:
	uvicorn housing_service.app:app --reload --port 8000

docker-build:
	docker build -t housing-service:latest .

docker-run:
	docker run --rm -p 8000:8000 -v $(PWD)/model:/app/model -v $(PWD)/data:/app/data housing-service:latest
