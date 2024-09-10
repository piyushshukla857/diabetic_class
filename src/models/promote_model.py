# promote model

import os
import mlflow

def promote_model():
    # Set up DagsHub credentials for MLflow tracking
    dagshub_token = os.environ.get('DAGSHUB_PAT')
    if not dagshub_token:
        raise ValueError("DAGSHUB_PAT environment variable is not set")

    # dagshub.auth.add_app_token(dagshub_token)
    # dagshub.init(repo_owner='piyushshukla857', repo_name='diabetic_class', mlflow=True)

    # mlflow.set_tracking_uri('https://dagshub.com/piyushshukla857/diabetic_class.mlflow')

    mlflow.set_tracking_uri(f"https://dagshub.com/piyushshukla857/diabetic_class.mlflow")
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'piyushshukla857'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token

    client = mlflow.MlflowClient()

    model_name = "my_model"
    # Get the latest version in staging
    latest_version_staging = client.get_latest_versions(model_name, stages=["Staging"])[0].version

    # Archive the current production model
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    for version in prod_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Archived"
        )

    # Promote the new model to production
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version_staging,
        stage="Production"
    )
    print(f"Model version {latest_version_staging} promoted to Production")

if __name__ == "__main__":
    promote_model()