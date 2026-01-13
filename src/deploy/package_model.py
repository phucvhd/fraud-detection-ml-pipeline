import argparse
import os
import sys

import mlflow
import tarfile
import shutil
import joblib
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.utils.config_manager import get_mlflow_tracking_uri


def package_model(mlflow_run_id: str, output_file: str, mlflow_tracking_uri: str = None):
    print(f"Packaging model from MLflow run {mlflow_run_id}...")

    tracking_uri, uri_source = get_mlflow_tracking_uri(mlflow_tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)

    print(f"MLflow Tracking: {tracking_uri}")
    print(f"(from {uri_source})")

    temp_dir = Path("temp_model")
    temp_dir.mkdir(exist_ok=True)

    try:
        model_uri = f"runs:/{mlflow_run_id}/model"
        print(f"Downloading from: {model_uri}")

        try:
            model = mlflow.sklearn.load_model(model_uri)
            print(f"Model loaded successfully: {type(model).__name__}")
        except Exception as e:
            print(f"ERROR loading model: {e}")
            print(f"Troubleshooting:")
            print(f"- Check run ID is correct: {mlflow_run_id}")
            if tracking_uri == "databricks":
                print(f"- Check Databricks credentials (DATABRICKS_HOST, DATABRICKS_TOKEN)")
            print(f"- Verify model was logged to MLflow")
            raise

        model_path = temp_dir / "model.joblib"
        joblib.dump(model, model_path)
        print(f"Model saved: {model_path}")

        inference_script = Path("src/models/inference.py")
        if inference_script.exists():
            shutil.copy(inference_script, temp_dir / "inference.py")
            print(f"Inference script copied")
        else:
            print("Creating basic inference.py")
            create_basic_inference_script(temp_dir / "inference.py")

        requirements_content = """
            scikit-learn==1.3.0
            numpy==1.24.3
            pandas==2.0.3
            joblib==1.3.0
            """
        with open(temp_dir / "requirements.txt", 'w') as f:
            f.write(requirements_content)
        print(f"Requirements.txt created")

        print(f"Creating model.tar.gz...")
        with tarfile.open(output_file, "w:gz") as tar:
            for item in temp_dir.iterdir():
                tar.add(item, arcname=item.name)

        file_size_mb = os.path.getsize(output_file) / 1024 / 1024
        print(f"Model packaged successfully")
        print(f"File: {output_file}")
        print(f"Size: {file_size_mb:.2f} MB")

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def create_basic_inference_script(output_path: Path):
    script_content = """
        import json
        import joblib
        import numpy as np
        import os
        
        def model_fn(model_dir):
            model_path = os.path.join(model_dir, 'model.joblib')
            model = joblib.load(model_path)
            return model
        
        def input_fn(request_body, content_type):
            if content_type == 'application/json':
                data = json.loads(request_body)
        
                features = [
                    data['Time'],
                    data['V1'], data['V2'], data['V3'], data['V4'], data['V5'],
                    data['V6'], data['V7'], data['V8'], data['V9'], data['V10'],
                    data['V11'], data['V12'], data['V13'], data['V14'], data['V15'],
                    data['V16'], data['V17'], data['V18'], data['V19'], data['V20'],
                    data['V21'], data['V22'], data['V23'], data['V24'], data['V25'],
                    data['V26'], data['V27'], data['V28'],
                    data['Amount']
                ]
        
                return np.array([features])
            raise ValueError(f"Unsupported content type: {content_type}")
        
        def predict_fn(input_data, model):
            prediction = model.predict(input_data)
        
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(input_data)[:, 1]
                return {
                    'prediction': int(prediction[0]),
                    'fraud_probability': float(probability[0]),
                    'is_fraud': bool(prediction[0] == 1)
                }
            return {'prediction': int(prediction[0])}
        
        def output_fn(prediction, accept):
            if accept == 'application/json':
                return json.dumps(prediction), accept
            raise ValueError(f"Unsupported accept type: {accept}")
        """
    with open(output_path, 'w') as f:
        f.write(script_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Package model for SageMaker')
    parser.add_argument('--mlflow-run-id', required=True, help='MLflow run ID')
    parser.add_argument('--output-file', required=True, help='Output tar.gz file')
    parser.add_argument('--mlflow-tracking-uri', required=False, default=None,
                        help='MLflow tracking URI (use "databricks" for Databricks)')

    args = parser.parse_args()

    package_model(args.mlflow_run_id, args.output_file, args.mlflow_tracking_uri)