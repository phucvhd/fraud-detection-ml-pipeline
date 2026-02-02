import os
import yaml
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import recall_score, precision_score, f1_score, precision_recall_curve, auc
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.utils.config_manager import load_config, get_mlflow_tracking_uri


class HPOTuner:
    def __init__(self, data_path: str, base_config_path: str,
                override_config_path: str, mlflow_tracking_uri: str = None):
        self.config = load_config(base_config_path, override_config_path)
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.data_path = data_path

    def load_and_prep_data(self):
        df = pd.read_csv(self.data_path)
        df['hour_of_day'] = (df['Time'] // 3600) % 24

        X = df.drop(['Class'], axis=1)
        y = df['Class']

        return X, y

    def get_param_grid(self):
        rf_dist = self.config['tuner']['param_dist']['random_forest']

        multipliers = rf_dist.pop('class_weight_multiplier')
        rf_dist['class_weight'] = [{0: 1, 1: m} for m in multipliers]

        return rf_dist

    def run(self):
        X, y = self.load_and_prep_data()
        param_grid = self.get_param_grid()

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.config['experiment']['name'])

        tscv = TimeSeriesSplit(n_splits=self.config['tuner']['cv_splits'])

        mlflow.sklearn.autolog(log_models=True, log_input_examples=True)

        with mlflow.start_run(run_name=f"HPO_RF_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as parent_run:
            rf = RandomForestClassifier(random_state=self.config['tuner']['random_state'])

            search = RandomizedSearchCV(
                estimator=rf,
                param_distributions=param_grid,
                n_iter=self.config['tuner']['n_iter'],
                cv=tscv,
                scoring=self.config['tuner']['scoring'],
                n_jobs=self.config['tuner']['n_jobs'],
                verbose=2,
                random_state=self.config['tuner']['random_state']
            )

            search.fit(X, y)

            mlflow.log_params(search.best_params_)
            mlflow.log_metric("best_cv_recall", search.best_score_)

            results = {
                "parent_run_id": parent_run.info.run_id,
                "best_run_id": mlflow.active_run().info.run_id,
                "best_params": search.best_params_,
                "best_cv_recall": search.best_score_
            }

            with open(self.config['output']['results_json'], 'w') as f:
                json.dump(results, f, indent=4)

            print(f"HPO Complete. Best CV Recall: {search.best_score_}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Random Forest with SMOTE')
    parser.add_argument('--data-path', required=True, help='Path to training data CSV')
    parser.add_argument('--base-config', required=True, help='Path to base config YAML')
    parser.add_argument('--override-config', required=False, help='Path to override config YAML')
    parser.add_argument('--mlflow-tracking-uri', required=False, default=None,
                        help='MLflow tracking URI (use "databricks" for Databricks)')

    args = parser.parse_args()

    tuner = HPOTuner(
        args.data_path,
        args.base_config,
        args.override_config,
        args.mlflow_tracking_uri
    )
    tuner.run()