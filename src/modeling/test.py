# Import modules
import pandas as pd
from src.config import MODEL_DATA_DIR
from src.utils import load_model, evaluate_model

# Load test data
X_test = pd.read_csv(MODEL_DATA_DIR / 'test.csv').iloc[:, :-1]
y_test = pd.read_csv(MODEL_DATA_DIR / 'test.csv').iloc[:, -1]
target_mapping = {'Low': 0, 'Moderate': 1, 'High': 2}

# Load model
model_filename = 'train.pkl'
model = load_model(model_filename)

# Evaluate model
evaluate_model(model, X_test, y_test, target_mapping)