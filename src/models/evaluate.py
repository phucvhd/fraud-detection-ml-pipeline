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

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.utils.config_manager import load_config, get_mlflow_tracking_uri

def evaluate_model(model_run_id: str, data_path: str,
                   base_config_path: str, override_config_path: str,
                   output_path: str, mlflow_tracking_uri: str = None):
    config = load_config(base_config_path, override_config_path)

    tracking_uri, uri_source = get_mlflow_tracking_uri(mlflow_tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)

    print("MODEL EVALUATION")
    print(f"MLflow Run ID: {model_run_id}")
    print(f"MLflow Tracking: {tracking_uri} from {uri_source}")

    print(f"Loading model from MLflow...")
    model_uri = f"runs:/{model_run_id}/model"
    try:
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Model loaded successfully")
        print(f"Model type: {type(model).__name__}")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        print(f"  Troubleshooting:")
        print(f"  - Check run ID is correct: {model_run_id}")
        print(f"  - Check Databricks credentials are set")
        print(f"  - Verify model was logged to MLflow")
        raise

    print(f"Loading evaluation data...")
    df = pd.read_csv(data_path)

    test_size = int(len(df) * config["data"]["splits"]["test_size"])
    df_test = df.tail(test_size)

    data_x_test = df_test.drop("Class", axis=1)
    data_y_test = df_test["Class"]

    print(f"Test set: {len(data_x_test):,} samples")
    print(f"Fraud cases: {data_y_test.sum():,} ({data_y_test.mean() * 100:.3f}%)")

    print(f"Applying feature engineering...")

    if config["features"]["time_features"]["enabled"]:
        if config["features"]["time_features"].get("hour_of_day", False):
            data_x_test["hour_of_day"] = (data_x_test["Time"] / 3600) % 24

        if config["features"]["time_features"].get("day_period", False):
            hour = (data_x_test["Time"] / 3600) % 24
            data_x_test["day_period"] = pd.cut(hour, bins=[0, 6, 12, 18, 24],
                                          labels=[0, 1, 2, 3], include_lowest=True)

        if config["features"]["time_features"].get("time_since_start", False):
            data_x_test["time_since_start"] = data_x_test["Time"] / df["Time"].max()

    if config["features"]["amount_features"]["enabled"]:
        if config["features"]["amount_features"].get("log_transform", False):
            data_x_test["log_amount"] = np.log1p(data_x_test["Amount"])

        if config["features"]["amount_features"].get("standardize", False):
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(df[["Amount"]])
            data_x_test["amount_scaled"] = scaler.transform(data_x_test[["Amount"]])

    print(f"Features: {data_x_test.shape[1]}")

    print(f"Making predictions...")
    data_y_pred = model.predict(data_x_test)
    data_y_pred_proba = model.predict_proba(data_x_test)[:, 1]

    print(f"Calculating metrics...")

    metrics = {
        "precision": float(precision_score(data_y_test, data_y_pred, zero_division=0)),
        "recall": float(recall_score(data_y_test, data_y_pred, zero_division=0)),
        "f1": float(f1_score(data_y_test, data_y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(data_y_test, data_y_pred_proba)),
        "pr_auc": float(average_precision_score(data_y_test, data_y_pred_proba))
    }

    tn, fp, fn, tp = confusion_matrix(data_y_test, data_y_pred).ravel()
    metrics["true_positives"] = int(tp)
    metrics["false_positives"] = int(fp)
    metrics["true_negatives"] = int(tn)
    metrics["false_negatives"] = int(fn)
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    print("EVALUATION RESULTS")
    print(f"Recall (Sensitivity):    {metrics['recall']:.4f}")
    print(f"Precision:               {metrics['precision']:.4f}")
    print(f"F1 Score:                {metrics['f1']:.4f}")
    print(f"Specificity:             {metrics['specificity']:.4f}")
    print(f"ROC-AUC:                 {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:                  {metrics['pr_auc']:.4f}")

    print(f"Confusion Matrix:")
    print(f"tn={tn}")
    print(f"fn={fn}")
    print(f"tp={tp}")
    print(f"fp={fp}")

    print(f"Detailed Classification Report:")
    print(classification_report(data_y_test, data_y_pred, target_names=["Normal", "Fraud"]))

    print(f"Logging metrics to MLflow...")
    with mlflow.start_run(run_id=model_run_id):
        mlflow.log_metrics(metrics)

        if config["mlflow"].get("log_plots", False):
            print(f"Creating visualization plots...")

            precision_vals, recall_vals, _ = precision_recall_curve(data_y_test, data_y_pred_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(recall_vals, precision_vals, linewidth=2)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall Curve (AUC = {metrics['pr_auc']:.4f})")
            plt.grid(True)
            plt.savefig("pr_curve.png", dpi=150, bbox_inches="tight")
            mlflow.log_artifact("pr_curve.png")
            plt.close()

            fpr, tpr, _ = roc_curve(data_y_test, data_y_pred_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, linewidth=2)
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve (AUC = {metrics['roc_auc']:.4f})")
            plt.grid(True)
            plt.savefig("roc_curve.png", dpi=150, bbox_inches="tight")
            mlflow.log_artifact("roc_curve.png")
            plt.close()

            from sklearn.metrics import ConfusionMatrixDisplay
            fig, ax = plt.subplots(figsize=(8, 6))
            ConfusionMatrixDisplay.from_predictions(data_y_test, data_y_pred,
                                                    display_labels=["Normal", "Fraud"],
                                                    cmap="Blues", ax=ax)
            plt.title("Confusion Matrix")
            plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
            mlflow.log_artifact("confusion_matrix.png")
            plt.close()

            print(f"Plots saved and logged")

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {output_path}")

    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"recall={metrics['recall']}\n")
            f.write(f"precision={metrics['precision']}\n")
            f.write(f"pr_auc={metrics['pr_auc']}\n")
            f.write(f"f1={metrics['f1']}\n")

    quality_gates = config["quality_gates"]

    if quality_gates.get("experimental_mode", False):
        print("Edata_xPERIMENTAL MODE: Quality gates skipped")
        return metrics

    print("QUALITY GATES CHECK")

    passed = True

    for metric_name, gate_config in quality_gates["metrics"].items():
        if metric_name not in metrics:
            continue

        threshold = gate_config["threshold"]
        operator = gate_config.get("operator", ">=")
        value = metrics[metric_name]

        if operator == ">=":
            check_passed = value >= threshold
        elif operator == ">":
            check_passed = value > threshold
        elif operator == "<=":
            check_passed = value <= threshold
        elif operator == "<":
            check_passed = value < threshold
        else:
            check_passed = value == threshold

        status = "PASS" if check_passed else "FAIL"
        print(f"{status}  {metric_name:15s}: {value:.4f} {operator} {threshold:.4f}")

        if not check_passed:
            passed = False

    if passed:
        print("ALL QUALITY GATES PASSED")
    else:
        print("QUALITY GATES FAILED - Model does not meet requirements")

    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"passed={'true' if passed else 'false'}\n")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fraud detection model")
    parser.add_argument("--model-run-id", required=True, help="MLflow run ID")
    parser.add_argument("--data-path", required=True, help="Path to evaluation data")
    parser.add_argument("--config", required=True, help="Path to base config")
    parser.add_argument("--override-config", required=False, help="Path to override config")
    parser.add_argument("--output", required=True, help="Output path for metrics JSON")
    parser.add_argument("--mlflow-tracking-uri", required=False, default=None,
                        help="MLflow tracking URI (use 'databricks' for Databricks)")

    args = parser.parse_args()

    evaluate_model(
        args.model_run_id,
        args.data_path,
        args.config,
        args.override_config,
        args.output,
        args.mlflow_tracking_uri
    )