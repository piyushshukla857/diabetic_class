import mlflow
import mlflow.sklearn
import mlflow.xgboost
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

model = XGBClassifier(
        n_estimators=816,
        learning_rate=0.1474427544605193,
        max_depth=10,
        min_child_weight=2,
        gamma=0.18,
        subsample=0.90,
        colsample_bytree=0.80,
        scale_pos_weight=15
    )
model.fit(X_train, y_train)
joblib.dump(model, 'models/xgboost_model.pkl')

