# fraud-detection-ml-pipeline

Config-driven **MLOps training pipeline** for credit-card fraud detection. It
trains a Random Forest on the [Kaggle Credit Card Fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)
dataset, tracks experiments in MLflow, enforces quality gates, and packages the
model as a SageMaker-compatible `model.tar.gz` — all orchestrated through GitHub
Actions.

This repo is the **offline training half** of a larger system. Real-time serving
lives in [`fraud-detection-system`](../fraud-detection-system) (Kafka + FastAPI)
and natural-language investigation in [`fraud-rag`](../fraud-rag).

## Pipeline stages

```
   creditcard.csv (S3)
          │
          ▼
  ┌──────────────┐   run_id    ┌──────────────┐  passed?   ┌──────────────┐
  │    train     │────────────▶│   evaluate   │───────────▶│   package    │
  │  (RF+SMOTE)  │             │ quality gates│            │ model.tar.gz │
  └──────────────┘             └──────────────┘            └──────┬───────┘
          │                                                       ▼
          ▼                                              s3://.../models/
   MLflow (Databricks):
   model + preprocessor + metrics
```

| Stage | Script | What it does |
|-------|--------|--------------|
| Train | [`src/models/train.py`](src/models/train.py) | Stratified split, shared feature engineering, optional SMOTE (train split only), fits RandomForest, logs model **and** the fitted preprocessor to MLflow. |
| Evaluate | [`src/models/evaluate.py`](src/models/evaluate.py) | Loads model + preprocessor by run ID, scores the held-out test set, enforces `quality_gates`, emits `passed=true\|false`. |
| Package | [`src/deploy/package_model.py`](src/deploy/package_model.py) | Bundles `model.joblib` + `preprocessor.joblib` + inference handler + shared feature module + pinned requirements into `model.tar.gz`, uploads to S3. |
| Tune (optional) | [`src/models/tune.py`](src/models/tune.py) | `RandomizedSearchCV` over `TimeSeriesSplit` with MLflow autolog. |
| Validate (optional) | [`src/models/validate_model.py`](src/models/validate_model.py) | Independent-test-set validation with detailed error analysis (false-negative / false-alarm breakdown). |

## Feature engineering — single source of truth

All stages (train, evaluate, tune, validate, **and** the SageMaker inference
handler) call one module, [`src/features/feature_engineering.py`](src/features/feature_engineering.py),
so the exact same transformation runs everywhere. This prevents train/serve
skew. Engineered features:

- `hour_of_day`, `day_period` (int-coded), `time_since_start`
- `log_amount`, `amount_scaled`

The `StandardScaler` and the training-time `Time` maximum are **fitted on the
training split only**, persisted to MLflow as `preprocessor.joblib`, and reused
verbatim at evaluation and serving time — never refit on new data.

## Tech stack

- **Python** 3.10
- **ML**: scikit-learn 1.7.2 (`RandomForestClassifier`), imbalanced-learn (`SMOTE`), pandas, numpy
- **Tracking / registry**: MLflow 3.8.1 (Databricks backend, with local `file:./mlruns` fallback)
- **Cloud**: AWS S3 (data + model artifacts), boto3
- **CI/CD**: GitHub Actions (`workflow_dispatch`)
- **Config**: YAML with a base + override merge pattern

## Configuration

Config is a base file plus an optional override merged recursively by
[`src/utils/config_manager.py`](src/utils/config_manager.py):

- [`config/train_config.yaml`](config/train_config.yaml) — data, model params, feature toggles, SMOTE, quality gates
- [`config/hpo_config.yaml`](config/hpo_config.yaml) — search space and CV settings
- [`config/deploy_config.yaml`](config/deploy_config.yaml) — target deployment config (see *Not yet implemented*)

The `Training` GitHub Actions workflow exposes every hyperparameter, feature
toggle, and quality-gate threshold as a `workflow_dispatch` input and patches the
override file with `yq` before running.

### Quality gates

`evaluate.py` checks metrics against `quality_gates` in the config (defaults:
recall ≥ 0.90, precision ≥ 0.85, PR-AUC ≥ 0.95, F1 ≥ 0.87) and emits
`passed=true|false`. The workflow **fails the job and blocks packaging** when the
gates are not met. Set `quality_gates.experimental_mode: true` (or the
`skip_quality_gates` workflow input) to bypass gates for experimentation.

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train (local MLflow file store)
python src/models/train.py \
  --data-path ./data/creditcard.csv \
  --base-config config/train_config.yaml \
  --mlflow-tracking-uri "file:./mlruns"

# Evaluate the run that train.py wrote to run_id.txt
python src/models/evaluate.py \
  --model-run-id "$(cat run_id.txt)" \
  --data-path ./data/creditcard_test.csv \
  --config config/train_config.yaml \
  --output metrics.json \
  --mlflow-tracking-uri "file:./mlruns"

# Package for serving
python src/deploy/package_model.py \
  --mlflow-run-id "$(cat run_id.txt)" \
  --output-file model.tar.gz \
  --mlflow-tracking-uri "file:./mlruns"
```

Use `--mlflow-tracking-uri databricks` (with `DATABRICKS_HOST` / `DATABRICKS_TOKEN`
/ `DATABRICKS_USER` set) to track against Databricks, as CI does.

## Tests

```bash
pytest tests/ -q
```

`tests/test_features.py` guards the feature-engineering contract (fit/transform
parity, integer `day_period`, scaler reuse rather than refit).

## Repository layout

```
src/
  features/feature_engineering.py   # shared transform (single source of truth)
  models/train.py                   # training + preprocessor logging
  models/evaluate.py                # metrics + quality gates
  models/tune.py                    # hyper-parameter search
  models/validate_model.py          # independent-test-set validation
  models/inference.py               # SageMaker handler (used inside model.tar.gz)
  deploy/package_model.py           # build model.tar.gz
  utils/config_manager.py           # config merge + MLflow URI resolution
config/                             # base configs
.github/workflows/                  # Training / HPO / Validation pipelines
tests/                              # unit tests
```

## Not yet implemented

These are stubbed/empty and tracked as future work — the README does not claim
them as working:

- `src/deploy/deploy_sagemaker.py` — automated SageMaker endpoint deployment (`config/deploy_config.yaml` describes the intended blue-green / autoscaling setup)
- `src/data/prepare_data.py` — standalone data-preparation entry point
- Real-time / online inference (served by the sibling `fraud-detection-system` repo)

## Secrets

Credentials are supplied via GitHub Secrets in CI (`AWS_*`, `DATABRICKS_*`,
`MLFLOW_TRACKING_URI`, `S3_*`) and via a local, git-ignored `.env` for
development. Never commit secrets.
