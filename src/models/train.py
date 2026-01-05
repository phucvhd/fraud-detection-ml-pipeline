import argparse
import os
import mlflow
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.utils.config_manager import load_config

def setup_databricks_auth():
    databricks_host = os.environ.get('DATABRICKS_HOST')
    databricks_token = os.environ.get('DATABRICKS_TOKEN')

    if databricks_host and databricks_token:
        os.environ['DATABRICKS_HOST'] = databricks_host
        os.environ['DATABRICKS_TOKEN'] = databricks_token

        print(f"Databricks authentication configured")
        print(f"  Host: {databricks_host}")
        return True

    return False


def get_mlflow_tracking_uri(config: dict, cli_uri: str = None) -> tuple:
    if cli_uri and cli_uri.lower() == 'databricks':
        if setup_databricks_auth():
            return "databricks", "Databricks workspace"
        else:
            print("WARNING: Databricks URI requested but credentials not found")
            print("Falling back to local tracking")
            return "file:./mlruns", "default (local)"

    if cli_uri and cli_uri.strip():
        return cli_uri, "command line argument"

    if os.environ.get('MLFLOW_TRACKING_URI'):
        uri = os.environ.get('MLFLOW_TRACKING_URI')
        if uri.lower() == 'databricks':
            if setup_databricks_auth():
                return "databricks", "environment variable (Databricks)"
        return uri, "environment variable"

    if config.get('mlflow', {}).get('tracking_uri'):
        return config['mlflow']['tracking_uri'], "config file"

    return "file:./mlruns", "default (local)"

def train_model(data_path: str, base_config_path: str,
                override_config_path: str, mlflow_tracking_uri: str = None):

    config = load_config(base_config_path, override_config_path)

    tracking_uri, uri_source = get_mlflow_tracking_uri(config, mlflow_tracking_uri)

    experiment_name = config['experiment']['name']

    if tracking_uri == "databricks":
        user_email = os.environ.get('DATABRICKS_USER', 'github-actions@pipeline.com')
        experiment_name = f"/Users/{user_email}/{experiment_name}"
        print(f"Using Databricks experiment path: {experiment_name}")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    print("="*70)
    print("RANDOM FOREST TRAINING WITH SMOTE")
    print("="*70)
    print(f"Experiment: {experiment_name}")
    print(f"Description: {config['experiment']['description']}")
    print(f"Model Type: {config['model']['type']}")
    print(f"Imbalance Method: {config['imbalance']['method']}")
    print(f"MLflow Tracking: {tracking_uri}")
    print(f"  (from {uri_source})")
    print("="*70)

    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)

    if config['data'].get('sample_size'):
        sample_size = config['data']['sample_size']
        print(f"  Using sample size: {sample_size}")
        df = df.sample(n=min(sample_size, len(df)), random_state=config['data']['random_state'])

    print(f"  Dataset shape: {df.shape}")
    print(f"  Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
    print(f"  Normal cases: {(df['Class']==0).sum()} ({(df['Class']==0).mean()*100:.3f}%)")

    X = df.drop('Class', axis=1)
    y = df['Class']

    print(f"\nFeature Engineering...")

    if config['features']['time_features']['enabled']:
        print("  Adding time-based features...")

        if config['features']['time_features'].get('hour_of_day', False):
            X['hour_of_day'] = (X['Time'] / 3600) % 24
            print("    hour_of_day")

        if config['features']['time_features'].get('day_period', False):
            hour = (X['Time'] / 3600) % 24
            X['day_period'] = pd.cut(hour, bins=[0, 6, 12, 18, 24],
                                     labels=[0, 1, 2, 3], include_lowest=True)
            print("    day_period")

        if config['features']['time_features'].get('time_since_start', False):
            X['time_since_start'] = X['Time'] / X['Time'].max()
            print("    time_since_start")

    if config['features']['amount_features']['enabled']:
        print("  Adding amount-based features...")

        if config['features']['amount_features'].get('log_transform', False):
            X['log_amount'] = np.log1p(X['Amount'])
            print("    log_amount")

        if config['features']['amount_features'].get('standardize', False):
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X['amount_scaled'] = scaler.fit_transform(X[['Amount']])
            print("    amount_scaled")

    print(f"  Total features: {X.shape[1]}")

    print(f"\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['data']['splits']['test_size'],
        stratify=y if config['data']['splits']['stratify'] else None,
        random_state=config['data']['random_state']
    )

    print(f"  Training set: {len(X_train):,} samples ({y_train.sum():,} fraud)")
    print(f"  Test set: {len(X_test):,} samples ({y_test.sum():,} fraud)")

    print(f"\nApplying SMOTE...")
    smote_config = config['imbalance']['smote']

    smote = SMOTE(
        sampling_strategy=smote_config['sampling_strategy'],
        k_neighbors=smote_config['k_neighbors'],
        random_state=smote_config['random_state']
    )

    print(f"  Sampling strategy: {smote_config['sampling_strategy']}")
    print(f"  K neighbors: {smote_config['k_neighbors']}")
    print(f"  Before SMOTE: {len(X_train):,} samples ({y_train.sum():,} fraud)")

    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"  After SMOTE: {len(X_train_resampled):,} samples ({y_train_resampled.sum():,} fraud)")
    print(f"  New fraud rate: {y_train_resampled.mean()*100:.2f}%")

    with mlflow.start_run() as run:
        print(f"\nStarting training...")
        print(f"  MLflow Run ID: {run.info.run_id}")

        mlflow.log_params({
            'experiment_name': experiment_name,
            'model_type': config['model']['type'],
            'imbalance_method': config['imbalance']['method'],
            'sample_size': config['data'].get('sample_size', 'full'),
            'smote_sampling_strategy': smote_config['sampling_strategy'],
            'smote_k_neighbors': smote_config['k_neighbors'],
            'dataset_original_fraud_rate': float(y.mean()),
            'dataset_after_smote_fraud_rate': float(y_train_resampled.mean()),
            'num_features': X_train_resampled.shape[1],
            **{f"rf_{k}": v for k, v in config['model']['params'].items()}
        })

        model_params = config['model']['params'].copy()

        if model_params['max_features'] == 'None':
            model_params['max_features'] = None

        print(f"\nTraining Random Forest...")
        print(f"  Number of trees: {model_params['n_estimators']}")
        print(f"  Max depth: {model_params['max_depth']}")
        print(f"  Min samples split: {model_params['min_samples_split']}")
        print(f"  Min samples leaf: {model_params['min_samples_leaf']}")
        print(f"  Max features: {model_params['max_features']}")
        print(f"  Bootstrap: {model_params['bootstrap']}")

        model = RandomForestClassifier(
            **model_params,
            random_state=config['data']['random_state'],
            verbose=1
        )

        model.fit(X_train_resampled, y_train_resampled)

        print("Training completed")

        if model_params.get('oob_score', False) and hasattr(model, 'oob_score_'):
            oob_score = model.oob_score_
            mlflow.log_metric('oob_score', oob_score)
            print(f"  OOB Score: {oob_score:.4f}")

        if config['mlflow'].get('log_feature_importance', False):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            print(f"\nTop 10 Feature Importances:")
            for idx, row in feature_importance.head(10).iterrows():
                print(f"  {row['feature']:20s}: {row['importance']:.6f}")

            feature_importance.to_csv('feature_importance.csv', index=False)
            mlflow.log_artifact('feature_importance.csv')

        print(f"\nQuick evaluation on test set...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        from sklearn.metrics import recall_score, precision_score, f1_score

        test_recall = recall_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)

        print(f"  Recall: {test_recall:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  F1 Score: {test_f1:.4f}")

        print("\n" + "="*70)
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
        print("="*70)

        if config['mlflow']['log_models']:
            mlflow.sklearn.log_model(model, "model")
            print(f"\nModel logged to MLflow")

        with open('run_id.txt', 'w') as f:
            f.write(run.info.run_id)

        if 'GITHUB_ENV' in os.environ:
            with open(os.environ['GITHUB_ENV'], 'a') as f:
                f.write(f"RUN_ID={run.info.run_id}\n")
                f.write(f"MODEL_URI=runs:/{run.info.run_id}/model\n")

        print(f"\nTraining pipeline completed successfully")
        print(f"   MLflow Run ID: {run.info.run_id}")
        print(f"   Model URI: runs:/{run.info.run_id}/model")

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