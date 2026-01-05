import argparse
import mlflow
from mlflow.tracking import MlflowClient


def register_model(run_id: str, model_name: str, stage: str = "None"):
    print(f"Registering model from run {run_id}...")

    client = MlflowClient()

    model_uri = f"runs:/{run_id}/model"

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MLflow utilities')
    subparsers = parser.add_subparsers(dest='command')

    register_parser = subparsers.add_parser('register-model')
    register_parser.add_argument('--run-id', required=True)
    register_parser.add_argument('--model-name', required=True)
    register_parser.add_argument('--stage', default='None')

    args = parser.parse_args()

    if args.command == 'register-model':
        register_model(args.run_id, args.model_name, args.stage)