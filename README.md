# Fraud Detection ML Pipeline

End-to-end MLOps pipeline for credit card fraud detection with automated training, evaluation, and deployment. Built with Random Forest, SMOTE oversampling, MLflow experiment tracking, Databricks integration, and GitHub Actions CI/CD.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Pipeline Stages](#pipeline-stages)
- [CI/CD Workflows](#cicd-workflows)
- [Quality Gates](#quality-gates)
- [Results](#results)
- [Roadmap](#roadmap)

---

## Overview

Credit card fraud is a significant problem — fraudulent transactions are extremely rare (typically < 0.2% of all transactions) but carry high financial impact. This project builds a production-grade ML pipeline that:

- **Trains** a Random Forest classifier with configurable hyperparameters
- **Handles class imbalance** using SMOTE oversampling and class weighting
- **Engineers features** from raw transaction data (time-based patterns, amount transformations)
- **Tracks experiments** with MLflow on Databricks for full reproducibility
- **Evaluates** models against configurable quality gates (recall, precision, F1, PR-AUC)
- **Validates** on independent hold-out test data
- **Packages** approved models for deployment to AWS SageMaker
- **Automates** the entire workflow through GitHub Actions CI/CD

The pipeline is designed around the principle that **recall matters most** — missing a fraudulent transaction (false negative) is far more costly than flagging a legitimate one (false positive).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      GitHub Actions CI/CD                       │
├─────────────┬──────────────┬──────────────┬────────────────────┤
│   Training  │  Evaluation  │  Validation  │  Packaging/Deploy  │
│             │              │              │                    │
│ train.py    │ evaluate.py  │ validate_    │ package_model.py   │
│ tune.py     │              │ model.py     │ deploy_sagemaker   │
└──────┬──────┴──────┬───────┴──────┬───────┴────────┬───────────┘
       │             │              │                │
       ▼             ▼              ▼                ▼
┌────────────┐ ┌──────────┐ ┌────────────┐ ┌──────────────────┐
│ S3 Bucket  │ │ MLflow / │ │ Quality    │ │ AWS SageMaker    │
│ (Raw Data) │ │Databricks│ │ Gates      │ │ (Model Serving)  │
└────────────┘ └──────────┘ └────────────┘ └──────────────────┘
```

**Data Flow:**
1. Raw transaction CSV is stored in S3
2. Training pipeline loads data, engineers features, applies SMOTE, and trains a Random Forest
3. All parameters, metrics, and artifacts are logged to MLflow on Databricks
4. Evaluation pipeline loads the trained model and scores it on test data
5. Quality gates check if metrics meet minimum thresholds
6. If gates pass, the model is packaged as a `.tar.gz` and uploaded to S3
7. (Planned) Deployment to AWS SageMaker endpoint

---

## Project Structure

```
fraud-detection-ml-pipeline/
├── .github/
│   └── workflows/
│       ├── train_pipeline.yaml          # Full train → evaluate → package workflow
│       ├── hpo_pipeline.yaml            # Hyperparameter optimization workflow
│       └── model_validation_pipeline.yaml # Independent validation workflow
├── config/
│   ├── train_config.yaml                # Base training configuration
│   ├── hpo_config.yaml                  # HPO search space configuration
│   └── deploy_config.yaml              # Deployment configuration
├── src/
│   ├── data/
│   │   └── prepare_data.py             # Data preparation (planned)
│   ├── models/
│   │   ├── train.py                    # Model training with SMOTE + MLflow
│   │   ├── evaluate.py                 # Model evaluation + quality gates
│   │   ├── validate_model.py           # Independent test set validation
│   │   ├── tune.py                     # Hyperparameter tuning with RandomizedSearchCV
│   │   └── inference.py                # Model inference (planned)
│   ├── deploy/
│   │   ├── package_model.py            # Model packaging for SageMaker
│   │   └── deploy_sagemaker.py         # SageMaker deployment (planned)
│   └── utils/
│       ├── config_manager.py           # YAML config loading with override support
│       └── mlflow_utils.py             # MLflow helper utilities
├── tests/
│   ├── test_model.py                   # Model tests (planned)
│   └── test_inference.py               # Inference tests (planned)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Tech Stack

| Category | Tools |
|---|---|
| **ML Framework** | scikit-learn (Random Forest), imbalanced-learn (SMOTE) |
| **Experiment Tracking** | MLflow, Databricks |
| **Data Storage** | AWS S3 |
| **CI/CD** | GitHub Actions |
| **Deployment Target** | AWS SageMaker |
| **Language** | Python 3.10+ |
| **Visualization** | matplotlib |
| **Config Management** | PyYAML with base/override pattern |

---

## Getting Started

### Prerequisites

- Python 3.10+
- AWS CLI configured with access to your S3 buckets
- (Optional) Databricks workspace for MLflow tracking

### Local Setup

```bash
# Clone the repository
git clone https://github.com/phucvhd/fraud-detection-ml-pipeline.git
cd fraud-detection-ml-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Locally

```bash
# Train with local MLflow tracking
python src/models/train.py \
  --data-path ./data/creditcard.csv \
  --base-config config/train_config.yaml

# Evaluate the trained model
python src/models/evaluate.py \
  --model-run-id <MLFLOW_RUN_ID> \
  --data-path ./data/creditcard_test.csv \
  --config config/train_config.yaml \
  --output metrics.json

# Validate on independent test data
python src/models/validate_model.py \
  --model-run-id <MLFLOW_RUN_ID> \
  --test-data-path ./data/creditcard_test.csv \
  --output validation_metrics.json

# Run hyperparameter tuning
python src/models/tune.py \
  --data-path ./data/creditcard.csv \
  --base-config config/hpo_config.yaml
```

### Running with Databricks MLflow

```bash
# Set environment variables
export DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
export DATABRICKS_TOKEN=your-token
export DATABRICKS_USER=your-email@example.com

# Train with Databricks tracking
python src/models/train.py \
  --data-path ./data/creditcard.csv \
  --base-config config/train_config.yaml \
  --mlflow-tracking-uri databricks
```

---

## Configuration

The pipeline uses a **base + override** configuration pattern. The base config (`config/train_config.yaml`) defines all defaults, and an override config can selectively replace values.

### Key Configuration Options

**Model Parameters:**
- `n_estimators`: Number of trees (default: 300)
- `max_depth`: Maximum tree depth (default: 15)
- `min_samples_split` / `min_samples_leaf`: Regularization controls
- `class_weight`: Manual class weights for imbalance handling

**SMOTE Settings:**
- `sampling_strategy`: Target minority-to-majority ratio (default: 0.5)
- `k_neighbors`: Nearest neighbors for synthetic sample generation (default: 5)

**Feature Engineering:**
- `time_features`: Extracts `hour_of_day`, `day_period`, `time_since_start`
- `amount_features`: Adds `log_amount` and `amount_scaled`

**Quality Gates:**
- Configurable thresholds for recall, precision, F1, and PR-AUC
- `experimental_mode`: Skip gates during development

See [`config/train_config.yaml`](config/train_config.yaml) for the full configuration reference.

---

## Pipeline Stages

### 1. Training (`src/models/train.py`)

Loads raw transaction data, applies feature engineering (time and amount transformations), optionally applies SMOTE oversampling to address class imbalance, and trains a Random Forest classifier. All parameters, metrics, feature importances, and the trained model are logged to MLflow.

### 2. Evaluation (`src/models/evaluate.py`)

Loads the trained model from MLflow, applies the same feature engineering to test data, generates predictions, and computes metrics: precision, recall, F1, ROC-AUC, PR-AUC, specificity, and confusion matrix. Generates PR curve, ROC curve, and confusion matrix plots. Runs quality gate checks against configurable thresholds.

### 3. Hyperparameter Tuning (`src/models/tune.py`)

Uses `RandomizedSearchCV` with `TimeSeriesSplit` cross-validation to search over a configurable parameter grid. Results are tracked in MLflow with autologging enabled.

### 4. Validation (`src/models/validate_model.py`)

Performs independent validation on a completely separate hold-out test set. Generates detailed error analysis (false negatives, false positives, false alarm rates) and saves predictions to CSV for further investigation.

### 5. Packaging (`src/deploy/package_model.py`)

Downloads the trained model from MLflow and packages it as a `.tar.gz` archive suitable for AWS SageMaker deployment. Uploads the package to S3.

---

## CI/CD Workflows

### Training Pipeline (`.github/workflows/train_pipeline.yaml`)

Triggered via `workflow_dispatch` with configurable inputs for all hyperparameters, feature flags, and quality gate thresholds.

**Jobs:** `train` → `evaluate` → `package` → `summary`

- Downloads data from S3
- Generates override config from workflow inputs using `yq`
- Trains model with Databricks MLflow tracking
- Evaluates and runs quality gates
- If passed, packages and uploads model to S3
- Creates GitHub Actions summary with results

### HPO Pipeline (`.github/workflows/hpo_pipeline.yaml`)

Runs hyperparameter optimization with configurable search parameters.

### Validation Pipeline (`.github/workflows/model_validation_pipeline.yaml`)

Validates a specific model (by MLflow run ID) against an independent test dataset.

---

## Quality Gates

The pipeline enforces minimum quality thresholds before a model can be packaged for deployment:

| Metric | Default Threshold | Rationale |
|---|---|---|
| **Recall** | ≥ 0.90 | Must catch at least 90% of fraud cases |
| **Precision** | ≥ 0.85 | At least 85% of fraud alerts should be correct |
| **F1 Score** | ≥ 0.87 | Balanced precision-recall performance |
| **PR-AUC** | ≥ 0.95 | Strong overall ranking quality |

All thresholds are configurable via the config YAML or GitHub Actions workflow inputs.

---

## Results

The pipeline produces the following artifacts per run (logged to MLflow):

- **Metrics**: Precision, Recall, F1, ROC-AUC, PR-AUC, Specificity, Confusion Matrix
- **Plots**: Precision-Recall Curve, ROC Curve, Confusion Matrix visualization
- **Feature Importances**: Ranked feature importance CSV
- **Model**: Serialized scikit-learn model artifact

---

## Roadmap

- [ ] Implement `prepare_data.py` for automated data preprocessing
- [ ] Implement `inference.py` for batch/real-time prediction serving
- [ ] Complete `deploy_sagemaker.py` for automated SageMaker endpoint deployment
- [ ] Add unit tests for feature engineering, config loading, and metric calculation
- [ ] Extract shared feature engineering into a reusable module to prevent train-serve skew
- [ ] Persist fitted scaler alongside the model for consistent transformations
- [ ] Add data validation checks (Great Expectations or similar)
- [ ] Implement model monitoring and drift detection

---

## License

This project is for educational and portfolio purposes.

---

## Author

**Steve Vu** — [GitHub](https://github.com/phucvhd)
