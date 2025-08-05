import pandas as pd
from sklearn.pipeline import Pipeline
import joblib
from src.config import MODELS_DIR


# Input
data = pd.DataFrame([{
    'Age': 50,
    'Height': 159,
    'Weight': 100,
    'Systolic': 120,
    'Diastolic': 80,
    'S_Ethnicity': 'White',
    'Family_KD': 'Definitely not',
    'Gender': 'Male',
    'Has_Hpt': False,
    'Has_Diabetes': True,
}])

# Load models
preprocessor = joblib.load(MODELS_DIR / 'preprocessor.pkl')
model = joblib.load(MODELS_DIR / 'train.pkl')

# Run pipeline
full_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('model', model)
])

# Prediction
prediction = full_pipeline.predict(data)
print("Prediction:", prediction)