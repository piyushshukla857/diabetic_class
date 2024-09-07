import mlflow
import mlflow.sklearn
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import yaml
import json
import joblib

# Load dataset
df = pd.read_csv('data/processed/processed_data.csv')
X = df.drop(columns='diabetes')
y = df['diabetes']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# with open('C:\Users\HP\OneDrive\Desktop\mlops\diabetic_class\params.yaml') as f:
#     params = yaml.safe_load(f)

# Set MLflow experiment
mlflow.set_experiment("XGBoost Hyperparameter Tuning")

with mlflow.start_run():
    # XGBoost model with provided hyperparameters
    model = XGBClassifier(
        n_estimators=816,
        learning_rate=0.1474427544605193,
        max_depth=10,
        min_child_weight=1,
        gamma=0.18718463889381676,
        subsample=0.9074737007412226,
        colsample_bytree=0.8030099712163372,
        scale_pos_weight=15.184869934787415
    )
    model.fit(X_train, y_train)
    joblib.dump(model, 'models/xgboost_model.pkl')

    # Save the model
    mlflow.sklearn.log_model(model, "model")

    # Save the model parameters for the evaluation script
    with open('reports/model_params.json', 'w') as f:
        json.dump(model.get_params(), f)
    mlflow.log_artifact('reports/model_params.json')
