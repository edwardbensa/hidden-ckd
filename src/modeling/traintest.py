# Import modules
import pandas as pd
from src.config import MODEL_DATA_DIR, PROCESSED_DATA_DIR
from src.utils.model_utils import (preprocess_data, stratified_split, random_split,
                                   load_model, evaluate_model)
from src.utils.models import train_xg, train_rf


# Choose model
model_choices = ['xgboost', 'randomforest']
print("Choose a model:")
for i, model in enumerate(model_choices):
    print(f"{i}: {model}")
choice = input("Enter the number of the model to train: ").strip()

try:
    model_name = model_choices[int(choice)]
except (IndexError, ValueError):
    print("Invalid choice. Defaulting to xgboost.")
    model_name = 'xgboost'


# Load features, target, and target encoding
X = pd.read_csv(MODEL_DATA_DIR / 'features.csv')
y = pd.read_csv(PROCESSED_DATA_DIR / "hidden_ckd_processed.csv")['CKD_Risk']
target_mapping = {'Low': 0, 'Moderate': 1, 'High': 2}

# Correct class imbalance
X_over, y_over = preprocess_data(X, y, target_mapping)

# Split data into training and testing sets
split_methods = ['stratified split', 'random split']
print("Choose a data splitting method:")
for i, s_method in enumerate(split_methods):
    print(f"{i}: {s_method}")
choice = input("Enter the number of the splitting method: ").strip()

try:
    s_method = split_methods[int(choice)]
except (IndexError, ValueError):
    print("Invalid choice. Defaulting to stratified split.")
    s_method = 'stratified split'

if s_method == 'stratified split':
    X_train, X_test, y_train, y_test = stratified_split(X_over, y_over)
elif s_method == 'random split':
    X_train, X_test, y_train, y_test = random_split(X_over, y_over)

# Save test data
test_df = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)], axis=1)
test_df.to_csv(MODEL_DATA_DIR / 'test.csv', index=False)

# Train Model
if model_name == 'xgboost':
    train_xg(X_train, y_train)
elif model_name == 'randomforest':
    train_rf(X_train, X_test, y_train, y_test)

# Load test data
X_test = pd.read_csv(MODEL_DATA_DIR / 'test.csv').iloc[:, :-1]
y_test = pd.read_csv(MODEL_DATA_DIR / 'test.csv').iloc[:, -1]
target_mapping = {'Low': 0, 'Moderate': 1, 'High': 2}

# Load model
model_filename = 'train.pkl'
model = load_model(model_filename)

# Evaluate model
evaluate_model(model, X_test, y_test, target_mapping)
