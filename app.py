from flask import Flask, request, jsonify
import pandas as pd
import pickle
import xgboost as xgb
from flask_cors import CORS
import mlflow
import dagshub
import os

app = Flask(__name__)
# CORS(app)
CORS(app, resources={r"/predict": {"origins": "http://localhost:8000"}})

# Load encoders and model
with open('models/gender_ohe_columns.pkl', 'rb') as f:
    gender_ohe_columns = pickle.load(f)

with open('models/smoking_ordinal_encoder.pkl', 'rb') as f:
    smoking_encoder = pickle.load(f)
# os.environ['DAGSHUB_PAT'] = "4a305524f676f1f43ad80dbf0be73c84eb4920ff"
# dagshub_token = os.environ.get('DAGSHUB_PAT')
# if not dagshub_token:
#     raise ValueError("DAGSHUB_PAT environment variable is not set")

# # dagshub.auth.add_app_token(dagshub_token)
# # dagshub.init(repo_owner='piyushshukla857', repo_name='diabetic_class', mlflow=True)

# # mlflow.set_tracking_uri('https://dagshub.com/piyushshukla857/diabetic_class.mlflow')

# mlflow.set_tracking_uri(f"https://dagshub.com/piyushshukla857/diabetic_class.mlflow")
# os.environ['MLFLOW_TRACKING_USERNAME'] = 'piyushshukla857'
# os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token

# def get_latest_model_version(model_name):
#     client = mlflow.MlflowClient()
#     latest_version = client.get_latest_versions(model_name, stages=["Production"])
#     if not latest_version:
#         latest_version = client.get_latest_versions(model_name, stages=["None"])
#     return latest_version[0].version if latest_version else None

# model_name = "my_model"
# model_version = get_latest_model_version(model_name)

# model_uri = f'models:/{model_name}/{model_version}'
# model = mlflow.pyfunc.load_model(model_uri)
with open('models/xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        data = request.json
        df = pd.DataFrame([data])
        
        # Check for required columns
        required_columns = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'gender', 'smoking_history']
        for col in required_columns:
            if col not in df.columns:
                return jsonify({'error': f'Missing column: {col}'}), 400
        
        # Encode gender
        gender_ohe = pd.get_dummies(df['gender'], prefix='gender')
        df = pd.concat([df, gender_ohe], axis=1)
        df.drop(columns=['gender'], inplace=True)
        
        # Ensure all dummy variables are present
        for col in gender_ohe_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Encode smoking history
        df['smoking_history_encoded'] = smoking_encoder.transform(df[['smoking_history']])
        df.drop(columns=['smoking_history'], inplace=True)
        
        # Fill missing values
        df.fillna({
            'bmi': df['bmi'].median(),
            'HbA1c_level': df['HbA1c_level'].median(),
            'blood_glucose_level': df['blood_glucose_level'].median()
        }, inplace=True)
        
        # Prepare data for prediction
        X = df[['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level'] + gender_ohe_columns + ['smoking_history_encoded']]
        
        # Predict using the model
        prediction = model.predict(X)
        
        # Return prediction
        return jsonify({'prediction': int(prediction[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(port=5001,debug=True, host="0.0.0.0")






# from flask import Flask, request, jsonify

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return "Flask is working!"

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json()
#         return jsonify({"message": "Received data", "data": data})
#     except Exception as e:
#         return jsonify({'error': str(e)})

# if __name__ == '__main__':
#     app.run(port=5001,debug=True)
