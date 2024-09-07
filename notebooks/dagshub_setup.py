import dagshub
dagshub.init(repo_owner='piyushshukla857', repo_name='diabetic_class', mlflow=True)

import mlflow
mlflow.set_tracking_uri('https://dagshub.com/piyushshukla857/diabetic_class.mlflow')
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)