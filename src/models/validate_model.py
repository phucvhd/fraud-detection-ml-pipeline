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
import joblib
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.utils.config_manager import get_mlflow_tracking_uri
from src.features.feature_engineering import engineer_features


def validate_model(model_run_id: str, test_data_path: str,
                   enable_time_features: bool, enable_amount_features: bool,
                   output_path: str, mlflow_tracking_uri: str = None,
                   save_predictions: str = None):
    tracking_uri, uri_source = get_mlflow_tracking_uri(mlflow_tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)

    print("MODEL VALIDATION - INDEPENDENT TEST DATA")
    print(f"Model Run ID: {model_run_id}")
    print(f"Test Data: {test_data_path}")
    print(f"MLflow Tracking: {tracking_uri} from {uri_source}")

    print(f"Loading model from MLflow...")
    model_uri = f"runs:/{model_run_id}/model"

    try:
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Model loaded successfully: {type(model).__name__}")

        if hasattr(model, "n_estimators"):
            print(f"Number of trees: {model.n_estimators}")
        if hasattr(model, "max_depth"):
            print(f"Max depth: {model.max_depth}")

    except Exception as e:
        print(f"ERROR loading model: {e}")
        print(f"Troubleshooting:")
        print(f"- Check run ID is correct: {model_run_id}")
        print(f"- Check Databricks credentials are set")
        print(f"- Verify model exists in MLflow")
        raise

    print(f"Loading preprocessor from MLflow...")
    preprocessor_path = mlflow.artifacts.download_artifacts(
        run_id=model_run_id, artifact_path="preprocessor/preprocessor.joblib"
    )
    preprocessor = joblib.load(preprocessor_path)
    print("Preprocessor loaded")

    print(f"Loading test data (NO SPLITTING - using entire dataset)...")
    df_test = pd.read_csv(test_data_path)

    print(f"Test set: {len(df_test):,} samples")

    if "Class" not in df_test.columns:
        raise ValueError("Test data must have 'Class' column for validation")

    test_data_y = df_test["Class"]
    data_x_test = df_test.drop("Class", axis=1)

    fraud_count = test_data_y.sum()
    fraud_rate = test_data_y.mean() * 100

    print(f"Fraud cases: {fraud_count:,} ({fraud_rate:.3f}%)")
    print(f"Normal cases: {(test_data_y == 0).sum():,} ({100 - fraud_rate:.3f}%)")

    # Reuse the training-time preprocessor so validation reflects real serving
    # behaviour. The enabled feature groups come from the preprocessor itself;
    # the CLI flags are kept only for backward compatibility.
    print("Applying feature engineering (reusing fitted preprocessor)...")
    data_x_test, _ = engineer_features(data_x_test, preprocessor=preprocessor, fit=False)

    print(f"Total features after engineering: {data_x_test.shape[1]}")

    print(f"Making predictions on entire test dataset...")
    y_pred = model.predict(data_x_test)
    y_pred_proba = model.predict_proba(data_x_test)[:, 1]

    print(f"Predictions completed")
    print(f"Predicted fraud: {y_pred.sum():,}")
    print(f"Predicted normal: {(y_pred == 0).sum():,}")

    print(f"Calculating validation metrics...")

    metrics = {
        "precision": float(precision_score(test_data_y, y_pred, zero_division=0)),
        "recall": float(recall_score(test_data_y, y_pred, zero_division=0)),
        "f1": float(f1_score(test_data_y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(test_data_y, y_pred_proba)),
        "pr_auc": float(average_precision_score(test_data_y, y_pred_proba))
    }

    tn, fp, fn, tp = confusion_matrix(test_data_y, y_pred).ravel()
    metrics["true_positives"] = int(tp)
    metrics["false_positives"] = int(fp)
    metrics["true_negatives"] = int(tn)
    metrics["false_negatives"] = int(fn)
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    metrics["test_samples"] = int(len(test_data_y))
    metrics["fraud_samples"] = int(fraud_count)

    print("VALIDATION RESULTS")
    print(f"Recall (Sensitivity):    {metrics['recall']:.4f}  ({tp}/{fraud_count} frauds detected)")
    print(f"Precision:               {metrics['precision']:.4f}  ({tp}/{tp + fp} fraud predictions correct)")
    print(f"F1 Score:                {metrics['f1']:.4f}")
    print(f"Specificity:             {metrics['specificity']:.4f}  ({tn}/{tn + fp} normals detected)")
    print(f"ROC-AUC:                 {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:                  {metrics['pr_auc']:.4f}")

    print(f"Confusion Matrix:")
    print(f"tn={tn}")
    print(f"fn={fn}")
    print(f"tp={tp}")
    print(f"fp={fp}")

    print(f"Detailed Classification Report:")
    print(classification_report(test_data_y, y_pred, target_names=["Normal", "Fraud"]))

    print(f"Error Analysis:")
    print(f"False Negatives (Missed Fraud): {fn}")
    if fn > 0:
        print(f"- {fn} fraud transactions were incorrectly classified as normal")
        print(f"- This represents {(fn / fraud_count) * 100:.2f}% of all fraud cases")

    print(f"False Positives (False Alarms): {fp}")
    if fp > 0:
        print(f"- {fp} normal transactions were incorrectly flagged as fraud")
        print(f"- False alarm rate: {(fp / (tn + fp)) * 100:.4f}%")

    print(f"Creating visualizations...")

    precision_vals, recall_vals, _ = precision_recall_curve(test_data_y, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (AUC = {metrics['pr_auc']:.4f})")
    plt.grid(True)
    plt.savefig("pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: pr_curve.png")

    fpr, tpr, _ = roc_curve(test_data_y, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"Model (AUC = {metrics['roc_auc']:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: roc_curve.png")

    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(test_data_y, y_pred,
                                            display_labels=["Normal", "Fraud"],
                                            cmap="Blues", ax=ax)
    plt.title("Confusion Matrix - Independent Test Data")
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: confusion_matrix.png")

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {output_path}")

    if save_predictions:
        predictions_df = pd.DataFrame({
            "actual": test_data_y.values,
            "predicted": y_pred,
            "fraud_probability": y_pred_proba,
            "correct": (test_data_y.values == y_pred)
        })
        predictions_df.to_csv(save_predictions, index=False)
        print(f"Predictions saved to: {save_predictions}")

        print(f"Prediction Summary:")
        print(f"Correct predictions: {predictions_df['correct'].sum():,} ({predictions_df['correct'].mean() * 100:.2f}%)")
        print(f"Incorrect predictions: {(~predictions_df['correct']).sum():,}")

    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"recall={metrics['recall']}\n")
            f.write(f"precision={metrics['precision']}\n")
            f.write(f"pr_auc={metrics['pr_auc']}\n")
            f.write(f"f1={metrics['f1']}\n")
            f.write(f"roc_auc={metrics['roc_auc']}\n")

    print("VALIDATION COMPLETED SUCCESSFULLY")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate trained model on independent data")
    parser.add_argument("--model-run-id", required=True, help="MLflow run ID")
    parser.add_argument("--test-data-path", required=True, help="Path to test data CSV")
    parser.add_argument("--enable-time-features", type=lambda x: x.lower() == "true",
                        default=True, help="Enable time-based features")
    parser.add_argument("--enable-amount-features", type=lambda x: x.lower() == "true",
                        default=True, help="Enable amount-based features")
    parser.add_argument("--output", required=True, help="Output path for metrics JSON")
    parser.add_argument("--mlflow-tracking-uri", required=False, default=None,
                        help="MLflow tracking URI")
    parser.add_argument("--save-predictions", required=False, default=None,
                        help="Path to save predictions CSV")

    args = parser.parse_args()

    validate_model(
        args.model_run_id,
        args.test_data_path,
        args.enable_time_features,
        args.enable_amount_features,
        args.output,
        args.mlflow_tracking_uri,
        args.save_predictions
    )