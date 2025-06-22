import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from import_helper import config
from import_helper import utils
from utils import load_model, evaluate_model
import pandas as pd

# Load test data
X_test = pd.read_csv(config.MODEL_DATA_DIR / 'test.csv').iloc[:, :-1]
y_test = pd.read_csv(config.MODEL_DATA_DIR / 'test.csv').iloc[:, -1]
target_mapping = {'Low': 0, 'Moderate': 1, 'High': 2}

# Load model
model_filename = 'train.pkl'
model = load_model(model_filename)

# Evaluate model
evaluate_model(model, X_test, y_test, target_mapping)
