# Development Guide

## Python environment (no Conda)
- Use Python 3.11 (preferred). 3.10–3.12 should work.
- Install package and dev tools:

```bash
python -m pip install -U pip
pip install '.[dev]'
```

- Install pre-commit hooks:

```bash
pre-commit install
```

## Common tasks
Using `Makefile`:

- Install runtime: `make install`
- Install dev: `make install-dev`
- Format: `make format`
- Lint: `make lint`
- Type check: `make typecheck`
- Tests: `make test`
- Run API locally: `make run` (uvicorn on :8000)

## Running the API
Ensure model artifacts exist under `model/` with `model.pkl` and `model_features.json` and data CSVs exist under `data/`.

```bash
uvicorn housing_service.app:app --reload --port 8000
# Health check
curl http://127.0.0.1:8000/health
```

### Sample client
Send first few rows from `data/future_unseen_examples.csv`:

```bash
python scripts/client.py --n 3
```

## Docker
Build and run:

```bash
make docker-build
make docker-run
# or directly:
docker build -t housing-service:latest .
docker run --rm -p 8000:8000 \
  -v "$PWD"/model:/app/model \
  -v "$PWD"/data:/app/data \
  housing-service:latest
```

## CI
GitHub Actions workflow `.github/workflows/ci.yml` runs on pushes/PRs:
- flake8, black --check, isort --check-only
- mypy
- pytest

## Project Layout
- `src/housing_service/` – FastAPI app (`app.py`)
- `data/` – provided CSVs
- `model/` – trained artifacts (not committed by default)
- `scripts/` – helper scripts (e.g., `client.py`)
- `tests/` – unit tests
- `.pre-commit-config.yaml` – code quality hooks
- `pyproject.toml` – packaging and tool configuration
- `Makefile`, `Dockerfile` – reproducible workflows

## Notes
- Do not commit large model binaries. Mount them at runtime or upload to artifact storage.
- Adjust `mypy.ini` and flake8 rules as the codebase grows.
