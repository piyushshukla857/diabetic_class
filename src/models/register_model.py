import json
import mlflow
import logging
import os
import dagshub


import dagshub

dagshub_token = os.environ.get('DAGSHUB_PAT')
if not dagshub_token:
    raise ValueError("DAGSHUB_PAT environment variable is not set")

# dagshub.auth.add_app_token(dagshub_token)
# dagshub.init(repo_owner='piyushshukla857', repo_name='diabetic_class', mlflow=True)

# mlflow.set_tracking_uri('https://dagshub.com/piyushshukla857/diabetic_class.mlflow')

mlflow.set_tracking_uri(f"https://dagshub.com/piyushshukla857/diabetic_class.mlflow")
os.environ['MLFLOW_TRACKING_USERNAME'] = 'piyushshukla857'
os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token




def load_model_info(file_path: str) -> dict:
    with open(file_path, 'r') as file:
            model_info = json.load(file)
    return model_info
    

def register_model(model_name: str, model_info: dict):

    model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
    model_version = mlflow.register_model(model_uri, model_name)
    print('regd')
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Staging"
    )
    

def main():

    model_info_path = 'reports/experiment_info.json'
    model_info = load_model_info(model_info_path)
    
    model_name = "my_model"
    register_model(model_name, model_info)

if __name__ == '__main__':
    main()