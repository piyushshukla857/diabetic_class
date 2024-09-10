import mlflow
import mlflow.sklearn
import pandas as pd
import json
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split
import pickle

import dagshub
dagshub.init(repo_owner='piyushshukla857', repo_name='diabetic_class', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/piyushshukla857/diabetic_class.mlflow')



# Load dataset
df = pd.read_csv('data/processed/processed_data.csv')
X = df.drop(columns='diabetes')
y = df['diabetes']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# Load model parameters
with open('reports/model_params.json', 'r') as f:
    model_params = json.load(f)

# Reload the model with the parameters
# model = XGBClassifier(**model_params)
# model.fit(X_train, y_train)

# # Predictions and evaluation
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
   
    model_info = {'run_id': run_id, 'model_path': model_path}
    with open(file_path, 'w') as file:
        json.dump(model_info, file, indent=4)
       

# Set MLflow experiment
mlflow.set_experiment("XGBoost dvc pipeline")

# with mlflow.start_run():
#     # Log metrics
#     mlflow.log_metric("accuracy", accuracy)
#     mlflow.log_metric("precision", precision)
#     mlflow.log_metric("recall", recall)
#     mlflow.log_metric("f1_score", f1)

#     # Save metrics to JSON
#     metrics = {
#         "accuracy": accuracy,
#         "precision": precision,
#         "recall": recall,
#         "f1_score": f1
#     }
#     with open('reports/metrics.json', 'w') as f:
#         json.dump(metrics, f)
#     mlflow.log_artifact('reports/metrics.json')

#     # Generate and save ROC curve plot
#     fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
#     roc_auc = auc(fpr, tpr)

#     plt.figure()
#     plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic')
#     plt.legend(loc='lower right')
#     plt.savefig('reports/figures/roc_curve.png')
#     mlflow.log_artifact('reports/figures/roc_curve.png')


with mlflow.start_run() as run:  # Start an MLflow run
        
            with open('./models/xgboost_model.pkl', 'rb') as file:
                clf = pickle.load(file)
            
            

            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            metrics = {
                            "accuracy": accuracy,
                            "precision": precision,
                            "recall": recall,
                            "f1_score": f1
                        }
            
            with open('reports/metrics.json', 'w') as f:
                json.dump(metrics, f)
            
            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model parameters to MLflow
            if hasattr(clf, 'get_params'):
                params = clf.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
            
            # Log model to MLflow
            mlflow.xgboost.log_model(clf, "model")

            
            
            # Save model info
            save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')
            
            # Log the metrics file to MLflow
            mlflow.log_artifact('reports/metrics.json')

            # Log the model info file to MLflow
            # mlflow.log_artifact('reports/model_info.json')

            # Log the evaluation errors log file to MLflow
            # mlflow.log_artifact('model_evaluation_errors.log')
