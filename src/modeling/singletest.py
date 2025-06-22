import pandas as pd
from sklearn.pipeline import Pipeline
import joblib
from src.config import MODELS_DIR


# Input
data = pd.DataFrame([{
    'Age': 46,
    'Height': 178,
    'Weight': 80,
    'Systolic': 180,
    'Diastolic': 120,
    'S_Ethnicity': 'Black',
    'Family_KD': 'Not sure',
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