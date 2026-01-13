import argparse

import mlflow
from mlflow.tracking import MlflowClient

from src.utils.config_manager import get_mlflow_tracking_uri

def register_model(run_id: str, model_name: str, stage: str = "None", mlflow_tracking_uri: str = None):
    tracking_uri, uri_source= get_mlflow_tracking_uri(mlflow_tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)

    if tracking_uri == "databricks":
        mlflow.set_registry_uri("databricks")
        print(f"Using legacy Databricks Workspace Model Registry")

    print(f"Registering model from run {run_id}...")
    print(f"MLflow Tracking: {tracking_uri} from {uri_source}")
    print(f"Model Name: {model_name}")

    client = MlflowClient()

    model_uri = f"runs:/{run_id}/model"

    try:
        model_details = mlflow.register_model(model_uri, model_name)

        print(f"Model registered: {model_name}")
        print(f"Version: {model_details.version}")

        if stage != "None":
            client.transition_model_version_stage(
                name=model_name,
                version=model_details.version,
                stage=stage
            )
            print(f"Stage transitioned to: {stage}")

        return model_details

    except Exception as e:
        print(f"ERROR registering model: {e}")
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