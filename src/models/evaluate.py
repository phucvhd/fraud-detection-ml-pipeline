import argparse
import json
import os
import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.utils.config_manager import load_config


def evaluate_model(model_run_id: str, data_path: str,
                   base_config_path: str, override_config_path: str,
                   output_path: str):
    """Evaluate trained Random Forest model and check quality gates."""

    config = load_config(base_config_path, override_config_path)

    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    print(f"MLflow Run ID: {model_run_id}")
    print("=" * 70)

    print(f"\nLoading model from MLflow...")
    model_uri = f"runs:/{model_run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"  Model loaded successfully")
    print(f"  Model type: {type(model).__name__}")

    print(f"\nLoading evaluation data...")
    df = pd.read_csv(data_path)

    test_size = int(len(df) * config['data']['splits']['test_size'])
    df_test = df.tail(test_size)

    X_test = df_test.drop('Class', axis=1)
    y_test = df_test['Class']

    print(f"  Test set: {len(X_test):,} samples")
    print(f"  Fraud cases: {y_test.sum():,} ({y_test.mean() * 100:.3f}%)")

    print(f"\nApplying feature engineering...")

    if config['features']['time_features']['enabled']:
        if config['features']['time_features'].get('hour_of_day', False):
            X_test['hour_of_day'] = (X_test['Time'] / 3600) % 24

        if config['features']['time_features'].get('day_period', False):
            hour = (X_test['Time'] / 3600) % 24
            X_test['day_period'] = pd.cut(hour, bins=[0, 6, 12, 18, 24],
                                          labels=[0, 1, 2, 3], include_lowest=True)

        if config['features']['time_features'].get('time_since_start', False):
            X_test['time_since_start'] = X_test['Time'] / df['Time'].max()

    if config['features']['amount_features']['enabled']:
        if config['features']['amount_features'].get('log_transform', False):
            X_test['log_amount'] = np.log1p(X_test['Amount'])

        if config['features']['amount_features'].get('standardize', False):
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(df[['Amount']])
            X_test['amount_scaled'] = scaler.transform(X_test[['Amount']])

    print(f"  Features: {X_test.shape[1]}")

    print(f"\nMaking predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print(f"\nCalculating metrics...")

    metrics = {
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
        'pr_auc': float(average_precision_score(y_test, y_pred_proba))
    }

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics['true_positives'] = int(tp)
    metrics['false_positives'] = int(fp)
    metrics['true_negatives'] = int(tn)
    metrics['false_negatives'] = int(fn)
    metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Recall (Sensitivity):    {metrics['recall']:.4f}  <- Fraud Detection Rate")
    print(f"Precision:               {metrics['precision']:.4f}  <- Accuracy of Fraud Predictions")
    print(f"F1 Score:                {metrics['f1']:.4f}  <- Harmonic Mean")
    print(f"Specificity:             {metrics['specificity']:.4f}  <- Normal Detection Rate")
    print(f"ROC-AUC:                 {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:                  {metrics['pr_auc']:.4f}  <- Best for Imbalanced Data")
    print("=" * 70)

    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Normal    Fraud")
    print(f"Actual  Normal   {tn:6d}    {fp:6d}")
    print(f"        Fraud    {fn:6d}    {tp:6d}")
    print("=" * 70)

    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))

    print(f"\nLogging metrics to MLflow...")
    with mlflow.start_run(run_id=model_run_id):
        mlflow.log_metrics(metrics)

        if config['mlflow'].get('log_plots', False):
            print(f"  Creating visualization plots...")

            precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(recall_vals, precision_vals, linewidth=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve (AUC = {metrics["pr_auc"]:.4f})')
            plt.grid(True)
            plt.savefig('pr_curve.png', dpi=150, bbox_inches='tight')
            mlflow.log_artifact('pr_curve.png')
            plt.close()

            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, linewidth=2)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve (AUC = {metrics["roc_auc"]:.4f})')
            plt.grid(True)
            plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
            mlflow.log_artifact('roc_curve.png')
            plt.close()

            from sklearn.metrics import ConfusionMatrixDisplay
            fig, ax = plt.subplots(figsize=(8, 6))
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred,
                                                    display_labels=['Normal', 'Fraud'],
                                                    cmap='Blues', ax=ax)
            plt.title('Confusion Matrix')
            plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
            mlflow.log_artifact('confusion_matrix.png')
            plt.close()

            print(f"  Plots saved and logged")

    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {output_path}")

    if 'GITHUB_OUTPUT' in os.environ:
        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            f.write(f"recall={metrics['recall']}\n")
            f.write(f"precision={metrics['precision']}\n")
            f.write(f"pr_auc={metrics['pr_auc']}\n")
            f.write(f"f1={metrics['f1']}\n")

    quality_gates = config['quality_gates']

    if quality_gates.get('experimental_mode', False):
        print("\nEXPERIMENTAL MODE: Quality gates skipped")
        return metrics

    print("\n" + "=" * 70)
    print("QUALITY GATES CHECK")
    print("=" * 70)

    passed = True

    for metric_name, gate_config in quality_gates['metrics'].items():
        if metric_name not in metrics:
            continue

        threshold = gate_config['threshold']
        operator = gate_config.get('operator', '>=')
        value = metrics[metric_name]

        if operator == '>=':
            check_passed = value >= threshold
        elif operator == '>':
            check_passed = value > threshold
        elif operator == '<=':
            check_passed = value <= threshold
        elif operator == '<':
            check_passed = value < threshold
        else:
            check_passed = value == threshold

        status = "PASS" if check_passed else "FAIL"
        print(f"{status}  {metric_name:15s}: {value:.4f} {operator} {threshold:.4f}")

        if not check_passed:
            passed = False

    print("=" * 70)

    if passed:
        print("ALL QUALITY GATES PASSED")
    else:
        print("QUALITY GATES FAILED - Model does not meet requirements")

    print("=" * 70)

    if 'GITHUB_OUTPUT' in os.environ:
        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            f.write(f"passed={'true' if passed else 'false'}\n")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate fraud detection model')
    parser.add_argument('--model-run-id', required=True, help='MLflow run ID')
    parser.add_argument('--data-path', required=True, help='Path to evaluation data')
    parser.add_argument('--config', required=True, help='Path to base config')
    parser.add_argument('--override-config', required=False, help='Path to override config')
    parser.add_argument('--output', required=True, help='Output path for metrics JSON')

    args = parser.parse_args()

    evaluate_model(
        args.model_run_id,
        args.data_path,
        args.config,
        args.override_config,
        args.output
    )