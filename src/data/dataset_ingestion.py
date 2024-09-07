import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
# Load the data
df = pd.read_csv('data/external/dataset.csv')

# Preprocess the dataset
gender_ohe = pd.get_dummies(df['gender'], prefix='gender')
df = pd.concat([df, gender_ohe], axis=1)
df.drop(columns=['gender'], inplace=True)


smoking_ordinal = OrdinalEncoder(categories=[['never', 'former', 'not current', 'current', 'ever', 'No Info']])
df['smoking_history_encoded'] = smoking_ordinal.fit_transform(df[['smoking_history']])
df.drop(columns=['smoking_history'], inplace=True)


df['bmi'].fillna(df['bmi'].median(), inplace=True)
df['HbA1c_level'].fillna(df['HbA1c_level'].median(), inplace=True)
df['blood_glucose_level'].fillna(df['blood_glucose_level'].median(), inplace=True)

print(df.head())


df.to_csv('data/processed/processed_data.csv', index=False)
