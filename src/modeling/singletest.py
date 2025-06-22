import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from import_helper import config

import pandas as pd

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
import joblib
preprocessor = joblib.load(config.MODELS_DIR / 'preprocessor.pkl')
model = joblib.load(config.MODELS_DIR / 'train.pkl')

# Run pipeline
from sklearn.pipeline import Pipeline
full_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('model', model)
])

# Prediction
prediction = full_pipeline.predict(data)
print("Prediction:", prediction)
