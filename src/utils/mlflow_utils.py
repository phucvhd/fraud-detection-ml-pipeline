import argparse
import os

import mlflow
from mlflow.tracking import MlflowClient


def setup_databricks_auth():
    databricks_host = os.environ.get('DATABRICKS_HOST')
    databricks_token = os.environ.get('DATABRICKS_TOKEN')

    if databricks_host and databricks_token:
        os.environ['DATABRICKS_HOST'] = databricks_host
        os.environ['DATABRICKS_TOKEN'] = databricks_token
        return True
    return False


def get_mlflow_tracking_uri(mlflow_tracking_uri: str = None) -> str:
    if mlflow_tracking_uri and mlflow_tracking_uri.lower() == 'databricks':
        if setup_databricks_auth():
            return "databricks"
        else:
            return "file:./mlruns"

    if mlflow_tracking_uri and mlflow_tracking_uri.strip():
        return mlflow_tracking_uri

    if os.environ.get('MLFLOW_TRACKING_URI'):
        uri = os.environ.get('MLFLOW_TRACKING_URI')
        if uri.lower() == 'databricks':
            if setup_databricks_auth():
                return "databricks"
        return uri

    return "file:./mlruns"

def register_model(run_id: str, model_name: str, stage: str = "None", mlflow_tracking_uri: str = None):
    print(f"Registering model from run {run_id}...")

    tracking_uri = get_mlflow_tracking_uri(mlflow_tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)

    print(f"Registering model from run {run_id}...")
    print(f"  MLflow Tracking: {tracking_uri}")

    client = MlflowClient()

    model_uri = f"runs:/{run_id}/model"

    try:
        model_details = mlflow.register_model(model_uri, model_name)

        print(f"  Model registered: {model_name}")
        print(f"  Version: {model_details.version}")

        if stage != "None":
            client.transition_model_version_stage(
                name=model_name,
                version=model_details.version,
                stage=stage
            )
            print(f"  Stage transitioned to: {stage}")

        return model_details

    except Exception as e:
        print(f"  ERROR registering model: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MLflow utilities')
    subparsers = parser.add_subparsers(dest='command')

    register_parser = subparsers.add_parser('register-model')
    register_parser.add_argument('--run-id', required=True)
    register_parser.add_argument('--model-name', required=True)
    register_parser.add_argument('--stage', default='None')
    register_parser.add_argument('--mlflow-tracking-uri', required=False, default=None)

    args = parser.parse_args()

    if args.command == 'register-model':
        register_model(args.run_id, args.model_name, args.stage, args.mlflow_tracking_uri)