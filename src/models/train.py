import argparse
import os
import mlflow
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.utils.config_manager import load_config, get_mlflow_tracking_uri
from src.features.feature_engineering import engineer_features

def train_model(data_path: str, base_config_path: str,
                override_config_path: str, mlflow_tracking_uri: str = None):

    config = load_config(base_config_path, override_config_path)

    tracking_uri, uri_source = get_mlflow_tracking_uri(mlflow_tracking_uri)

    experiment_name = config['experiment']['name']

    if tracking_uri == "databricks":
        user_email = os.environ.get('DATABRICKS_USER')
        experiment_name = f"/Users/{user_email}/fraud-detection/{experiment_name}"
        print(f"Using Databricks experiment path: {experiment_name}")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    print("RANDOM FOREST TRAINING WITH SMOTE")
    print(f"Experiment: {experiment_name}")
    print(f"Description: {config['experiment']['description']}")
    print(f"Model Type: {config['model']['type']}")
    print(f"Imbalance Method: {config['imbalance']['method']}")
    print(f"MLflow Tracking: {tracking_uri} from {uri_source}")

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    if config['data'].get('sample_size'):
        sample_size = config['data']['sample_size']
        print(f"Using sample size: {sample_size}")
        df = df.sample(n=min(sample_size, len(df)), random_state=config['data']['random_state'])

    print(f"Dataset shape: {df.shape}")
    print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
    print(f"Normal cases: {(df['Class']==0).sum()} ({(df['Class']==0).mean()*100:.3f}%)")

    raw_x = df.drop('Class', axis=1)
    y = df['Class']

    print(f"Splitting data (stratified hold-out)...")
    raw_x_train, raw_x_val, y_train, y_val = train_test_split(
        raw_x, y,
        test_size=config['data']['splits']['test_size'],
        stratify=y if config['data']['splits']['stratify'] else None,
        random_state=config['data']['random_state']
    )
    print(f"Training set: {len(raw_x_train):,} samples ({y_train.sum():,} fraud)")
    print(f"Validation set: {len(raw_x_val):,} samples ({y_val.sum():,} fraud)")

    print(f"Feature Engineering...")
    # Fit the scaler / time-normaliser on the training split ONLY, then reuse the
    # fitted preprocessor for the validation split — no leakage, no per-set refit.
    data_x_train, preprocessor = engineer_features(raw_x_train, config=config, fit=True)
    data_x_val, _ = engineer_features(raw_x_val, preprocessor=preprocessor, fit=False)
    print(f"Total features: {data_x_train.shape[1]}")

    smote_config = config['imbalance']['smote']
    if smote_config.get('enabled', False):
        print(f"Applying SMOTE (training split only)...")
        smote = SMOTE(
            sampling_strategy=smote_config['sampling_strategy'],
            k_neighbors=smote_config['k_neighbors'],
            random_state=smote_config['random_state']
        )
        print(f"Sampling strategy: {smote_config['sampling_strategy']}")
        print(f"K neighbors: {smote_config['k_neighbors']}")
        print(f"Before SMOTE: {len(data_x_train):,} samples ({y_train.sum():,} fraud)")

        data_x_train_resampled, y_train_resampled = smote.fit_resample(data_x_train, y_train)
        y_train_resampled = y_train_resampled.astype(int)

        print(f"After SMOTE: {len(data_x_train_resampled):,} samples ({y_train_resampled.sum():,} fraud)")
        print(f"New fraud rate: {y_train_resampled.mean()*100:.2f}%")
    else:
        print("SMOTE disabled — training on original class distribution "
              "(imbalance handled via class_weight).")
        data_x_train_resampled, y_train_resampled = data_x_train, y_train.astype(int)

    with mlflow.start_run() as run:
        print(f"Starting training...")
        print(f"MLflow Run ID: {run.info.run_id}")

        mlflow.log_params({
            'experiment_name': experiment_name,
            'model_type': config['model']['type'],
            'imbalance_method': config['imbalance']['method'],
            'sample_size': config['data'].get('sample_size', 'full'),
            'smote_enabled': smote_config.get('enabled', False),
            'smote_sampling_strategy': smote_config['sampling_strategy'],
            'smote_k_neighbors': smote_config['k_neighbors'],
            'dataset_original_fraud_rate': float(y.mean()),
            'train_fraud_rate': float(y_train_resampled.mean()),
            'num_features': data_x_train_resampled.shape[1],
            **{f"rf_{k}": v for k, v in config['model']['params'].items()}
        })

        model_params = config['model']['params'].copy()

        if model_params['max_features'] == 'None':
            model_params['max_features'] = None

        print(f"Training Random Forest...")
        print(f"Number of trees: {model_params['n_estimators']}")
        print(f"Max depth: {model_params['max_depth']}")
        print(f"Min samples split: {model_params['min_samples_split']}")
        print(f"Min samples leaf: {model_params['min_samples_leaf']}")
        print(f"Max features: {model_params['max_features']}")
        print(f"Bootstrap: {model_params['bootstrap']}")

        model = RandomForestClassifier(
            **model_params,
            random_state=config['data']['random_state'],
            verbose=1
        )

        print("Labels in y_train_resampled:", np.unique(y_train_resampled))

        model.fit(data_x_train_resampled, y_train_resampled)

        print("Training completed")

        if model_params.get('oob_score', False) and hasattr(model, 'oob_score_'):
            oob_score = model.oob_score_
            mlflow.log_metric('oob_score', oob_score)
            print(f"OOB Score: {oob_score:.4f}")

        # Validation metrics on the held-out split (real, un-resampled data).
        from sklearn.metrics import recall_score, precision_score, average_precision_score
        val_pred = model.predict(data_x_val)
        val_proba = model.predict_proba(data_x_val)[:, 1]
        val_metrics = {
            'val_recall': float(recall_score(y_val, val_pred, zero_division=0)),
            'val_precision': float(precision_score(y_val, val_pred, zero_division=0)),
            'val_pr_auc': float(average_precision_score(y_val, val_proba)),
        }
        mlflow.log_metrics(val_metrics)
        print(f"Validation — recall={val_metrics['val_recall']:.4f} "
              f"precision={val_metrics['val_precision']:.4f} "
              f"pr_auc={val_metrics['val_pr_auc']:.4f}")

        if config['mlflow'].get('log_feature_importance', False):
            feature_importance = pd.DataFrame({
                'feature': data_x_train_resampled.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            print(f"Top 10 Feature Importances:")
            for idx, row in feature_importance.head(10).iterrows():
                print(f"{row['feature']:20s}: {row['importance']:.6f}")

            feature_importance.to_csv('feature_importance.csv', index=False)
            mlflow.log_artifact('feature_importance.csv')

        # Persist the fitted preprocessor alongside the model so evaluation and
        # serving reuse the identical scaler / time-normaliser (no train/serve skew).
        joblib.dump(preprocessor, 'preprocessor.joblib')
        mlflow.log_artifact('preprocessor.joblib', artifact_path='preprocessor')
        print("Preprocessor logged to MLflow")

        if config['mlflow']['log_models']:
            mlflow.sklearn.log_model(model, "model")
            print(f"Model logged to MLflow")

        with open('run_id.txt', 'w') as f:
            f.write(run.info.run_id)

        if 'GITHUB_ENV' in os.environ:
            with open(os.environ['GITHUB_ENV'], 'a') as f:
                f.write(f"RUN_ID={run.info.run_id}\n")
                f.write(f"MODEL_URI=runs:/{run.info.run_id}/model\n")

        print(f"Training pipeline completed successfully")
        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"Model URI: runs:/{run.info.run_id}/model")

        return run.info.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Random Forest with SMOTE')
    parser.add_argument('--data-path', required=True, help='Path to training data CSV')
    parser.add_argument('--base-config', required=True, help='Path to base config YAML')
    parser.add_argument('--override-config', required=False, help='Path to override config YAML')
    parser.add_argument('--mlflow-tracking-uri', required=False, default=None,
                        help='MLflow tracking URI (use "databricks" for Databricks)')

    args = parser.parse_args()

    train_model(
        args.data_path,
        args.base_config,
        args.override_config,
        args.mlflow_tracking_uri
    )