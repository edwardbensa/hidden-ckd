import pandas as pd
from sklearn.pipeline import Pipeline
import joblib
from src.config import MODELS_DIR


# Input
data = pd.DataFrame([{
    'Age': 70,
    'Height': 159,
    'Weight': 120,
    'Systolic': 120,
    'Diastolic': 80,
    'S_Ethnicity': 'Black',
    'Family_KD': 'Definitely not',
    'Gender': 'Female',
    'Has_KD': False,
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