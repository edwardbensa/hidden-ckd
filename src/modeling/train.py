# Import modules
import pandas as pd
from src.config import MODEL_DATA_DIR, PROCESSED_DATA_DIR
from src.utils import preprocess_data, stratified_split, random_split, train_xg, train_rf


# Load features, target, and target encoding
X = pd.read_csv(MODEL_DATA_DIR / 'features.csv')
y = pd.read_csv(PROCESSED_DATA_DIR / "hidden_ckd_processed.csv")['CKD_Risk']
target_mapping = {'Low': 0, 'Moderate': 1, 'High': 2}

# Correct class imbalance
X_over, y_over = preprocess_data(X, y, target_mapping)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = stratified_split(X_over, y_over)
#X_train, X_test, y_train, y_test = random_split(X_over, y_over)

# Save test data
test_df = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)], axis=1)
test_df.to_csv(MODEL_DATA_DIR / 'test.csv', index=False)

# Train Model
train_xg(X_train, y_train)
#train_rf(X_train, X_test, y_train, y_test)