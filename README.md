# AI-Trader

AI-Trader is an early trading research scaffold checked in August 2025. It contains a FastAPI API layout for asset and market-data records, LSTM/GRU/Transformer model definitions, training and prediction services, and a backtesting engine. The repository does not provide a working broker execution system: its trading control and position endpoints return placeholder values. Treat it as an incomplete software project for inspection and development, not a production trading platform.

## The problem it explores

The code sketches a workflow for collecting historical price data, training time-series models, producing directional signals, and evaluating strategies against historical data. The intended shape is useful for learning how those pieces might fit together. The code does not establish profitability, reliability, live execution, or suitability for financial decisions.

## What works in the checked-in code

- The FastAPI app mounts health, asset, market-data, trading, and ML routers under `/api/v1` (`api/app/main.py`).
- Asset CRUD handlers and market-data ingestion and retrieval handlers are implemented in source (`api/app/routers/assets.py`, `api/app/routers/market_data.py`). They depend on the database and, for ingestion, external market-data APIs.
- The ML router defines training, prediction, model listing/deployment, and backtest handlers (`api/app/routers/ml.py`). The project includes model, feature engineering, training, prediction, and strategy/backtest modules.
- The health router has basic, detailed, readiness, and liveness endpoints (`api/app/routers/health.py`).
- The tests define health endpoint checks and ML test cases (`api/tests/`). On 2026-11-23, `python -m compileall -q app tests` passed, but `pytest` collection failed before tests ran because required settings were absent and `app.models` is missing.

The checked-in API has no `api/app/models/` directory in this checkout, despite imports of `app.models` throughout the API, and the only migration file creates only ML model and backtest tables. Those gaps mean several described flows are not runnable as-is without repair.

## Architecture

```text
AI-Trader/
├── README.md
├── package.json
├── pnpm-workspace.yaml
├── docker-compose.yml
└── api/
    ├── Dockerfile
    ├── env.example
    ├── pyproject.toml
    ├── alembic.ini
    ├── migrations/
    │   └── versions/001_add_ml_models.py
    ├── app/
    │   ├── main.py
    │   ├── config.py
    │   ├── database.py
    │   ├── routers/
    │   │   ├── assets.py
    │   │   ├── health.py
    │   │   ├── market_data.py
    │   │   ├── ml.py
    │   │   └── trading.py
    │   ├── services/
    │   │   ├── data_ingestion.py
    │   │   └── market_data.py
    │   ├── ml/
    │   │   ├── feature_engineering.py
    │   │   ├── models.py
    │   │   ├── prediction.py
    │   │   └── training.py
    │   └── backtesting/
    │       ├── engine.py
    │       ├── performance.py
    │       └── strategies.py
    ├── tests/
    │   ├── test_health.py
    │   └── test_ml.py
    └── package.json
```

The requested `web/` directory, frontend sources, and frontend manifest are absent from this checkout. `docker-compose.yml` refers to `./web`, `init-db.sql`, and service code that are not present in the inspected tree.

## Tech stack

| Area | Tools declared in the repository |
|---|---|
| API | Python 3.11+, FastAPI, Uvicorn, Pydantic, Pydantic Settings |
| Persistence | SQLAlchemy async sessions, PostgreSQL drivers, Alembic |
| Market data | `httpx`, Alpha Vantage and Polygon.io client code |
| ML | PyTorch model definitions, NumPy, pandas, scikit-learn, `ta` |
| Backtesting | pandas-based engine, mean-reversion, momentum, and ML strategy classes |
| Background jobs | Redis and Celery listed in project dependencies and Compose |
| Tests | pytest and pytest-asyncio; coverage settings in `api/pyproject.toml` |
| Workspace | pnpm workspace files, but no `web/` package in this checkout |

The Python package metadata lists the project as Alpha. The repository history shows its initial checked-in snapshot dated 2025-08-07.

## Quickstart

The API settings require environment variables for a database URL, sync database URL, secret key, and Alpha Vantage key. `api/env.example` documents the expected variables. Use a disposable local database and replace the example secret before starting the service.

```bash
git clone https://github.com/Ottailo/AI-Trader.git
cd AI-Trader

# Create and activate a Python 3.11 virtual environment.
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

# Install the API package and its dependencies.
cd api
pip install -e '.[dev]'
```

The repository includes a root-level `docker-compose.yml` for PostgreSQL, Redis, API, web, Celery worker, and Celery Beat, but this checkout lacks the referenced web project and database initialization SQL. Compose startup is therefore not a verified quickstart and is expected to fail without those files. The direct API server command is declared in `api/package.json`:

```bash
cd api
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Configuration is loaded from `api/.env` or process environment. `api/env.example` is a template, not a ready-to-use secret file. `ALPHA_VANTAGE_API_KEY` and valid database settings are required at application import/startup. No installer or container command was run for this draft.

## Usage

If the missing model and schema modules are supplied or implemented and database configuration is valid, the API routes are intended to support asset records, market-data ingestion, predictions, model training, and historical backtests. Begin with the documented health check:

```bash
curl http://127.0.0.1:8000/api/v1/health
```

The ML training and backtest routes schedule work using FastAPI background tasks and return a record identifier. They are not a command-line trading interface. The root `package.json` exposes pnpm workspace scripts, but the web workspace package is missing from this checkout.

## API surface

The routes below are declared in source. They were not live-tested; missing model modules and other integration gaps can prevent the API from starting.

| Method | Path | Purpose in source |
|---|---|---|
| `GET` | `/api/v1/health`, `/health/live`, `/health/ready`, `/health/detailed` | Health, liveness, readiness, and dependency checks |
| `POST`, `GET`, `PUT`, `DELETE` | `/api/v1/assets` and `/api/v1/assets/{symbol}` | Create, list, update, soft-delete, and inspect assets |
| `POST` | `/api/v1/market-data/ingest` and `/ingest/batch` | Request market-data fetching and storage |
| `GET` | `/api/v1/market-data/{symbol}`, `/range`, `/latest`, `/summary` | Read stored market data |
| `POST` | `/api/v1/ml/train` | Create an ML model record and schedule training |
| `POST` | `/api/v1/ml/predict` | Request a model-generated price signal |
| `POST` | `/api/v1/ml/backtest` | Create a backtest record and schedule a backtest |
| `GET` | `/api/v1/ml/models`, `/models/{model_id}` | List model records and request model details |
| `POST` | `/api/v1/ml/models/{model_id}/deploy` | Mark a trained model as deployed |
| `GET` | `/api/v1/ml/backtests`, `/backtests/{backtest_id}` | List or inspect backtest records |
| `GET`, `POST` | `/api/v1/trading/status`, `/enable`, `/disable`, `/positions`, `/history` | Trading controls and status placeholders only |

FastAPI docs are configured at `/docs` in development mode. The route definitions live in `api/app/routers/`.

## Data model

No `api/app/models/` Python package is present in the inspected tree, although routers, database initialization, and ML services import from `app.models`. The migration `api/migrations/versions/001_add_ml_models.py` defines `ml_models` and `backtest_results` tables, including model status and training metadata, backtest configuration, performance fields, equity-curve JSON, and trade-log JSON. It does not define the asset, historical-data, or other model tables required by the route code.

The `api/app/services/data_ingestion.py` code expects asset records and OHLCV history. `api/app/routers/ml.py` expects `MLModel`, `BacktestResult`, `ModelStatus`, and `BacktestStatus`. Treat those as expected contracts, not a complete current schema.

## Tests

The project declares pytest under the `dev` optional dependency group and configures `testpaths = ["tests"]` in `api/pyproject.toml`. It includes `api/tests/test_health.py` and `api/tests/test_ml.py`.

The ML test file in this checkout is empty. The health tests use `TestClient` and import `app.main`, but import-time settings require environment variables and the application imports model modules that are absent here. Verified on 2026-11-23: `python -m compileall -q app tests` passes; `pytest` collection fails before tests run because required settings are absent and `app.models` is missing. No passing test result is claimed.

The `api/package.json` declares `test` as `pytest`, but that is a package script invoking a Python tool, not a pnpm frontend test suite. The root test script recursively runs workspace scripts; the missing `web` package leaves that workflow unverified.

## Deployment

`api/Dockerfile` builds the Python API image on Python 3.11. `docker-compose.yml` describes PostgreSQL, Redis, API, web, Celery worker, and Celery Beat services. This is a development-oriented composition: the API is configured with `DEBUG=true`, a placeholder secret, and a demo Alpha Vantage key fallback. It also references absent files and directories (`web/`, `init-db.sql`). Do not deploy this Compose configuration as written.

There is no production deployment configuration or verified hosted service in this repository. Live broker access and financial execution are not implemented by the trading endpoints.

## Design decisions

- Use FastAPI routers separated by assets, health, market data, trading, and machine-learning concerns.
- Use async SQLAlchemy sessions for API requests and an async engine for PostgreSQL.
- Store trained model files separately from database metadata; `api/app/ml/prediction.py` loads model weights, scaler, and JSON configuration from local paths.
- Train LSTM, GRU, or Transformer-style sequence models defined in `api/app/ml/models.py`.
- Keep historical strategy evaluation in a local backtesting module rather than connecting that engine directly to a broker.

These describe code organization, not verified operating behavior.

## Limitations

- This is an August 2025 early scaffold, not a production trading system.
- There is no `web/` frontend in the checkout, despite README and Compose references to Next.js.
- `api/app/models/` is absent while many modules import it. The checked-in migration covers only ML model and backtest records.
- Trading status, enable/disable, positions, and history routes return fixed or empty responses and contain comments identifying them as future placeholders (`api/app/routers/trading.py`). No Alpaca order submission is implemented in those routes.
- Alpha Vantage and Polygon request code exists, but credentials, provider behavior, rate limits, and ingestion results were not tested.
- The training, prediction, and backtest routes depend on missing model classes and data schema. Their declared route contracts do not prove they run successfully.
- The prediction signal's confidence is derived from expected-return magnitude relative to a threshold, not a calibrated probability.
- The backtesting engine and trade-statistics implementation have not been validated against known reference cases. No performance or profitability claim is made.
- The environment template has placeholder secrets and example trading parameters. Do not use them with a funded brokerage account.
- The checked-in test suite cannot collect without the required environment settings and the missing `app.models` package.
- No license file is present in this checkout, despite the package metadata and old README claiming MIT. License terms were not verified.

## Roadmap

1. Add the missing model package and complete migrations for every database entity used by the API.
2. Make the API importable and testable with isolated settings and a disposable database; add assertions for ingestion, training, prediction, and backtesting.
3. Remove or clearly label the trading-control routes until a tested broker integration and risk controls exist.
4. Restore the frontend only if it is part of this repository's scope, then align the pnpm workspace and Compose files with the actual files.
5. Document paper-only operation, external provider setup, data assumptions, and deployment only after those paths are exercised and tested.
