import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from import_helper import config

# Importing modules
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Specifying variables
ml_vars = [
    'Age',
    'Height',
    'Weight',
    'Systolic',
    'Diastolic',
    'S_Ethnicity',
    'Family_KD',
    'Gender',
    'Has_KD',
    'Has_Diabetes',
    'uACR'
]

# Importing processed data
data = pd.read_csv(config.PROCESSED_DATA_DIR / "hidden_ckd_processed.csv")[ml_vars]



# Loading preprocessing model
preprocessor_filename = 'preprocessor-02.pkl'
preprocessor = joblib.load(config.MODELS_DIR / preprocessor_filename) 

# Transforming data and making a dataframe out of the transformed data
data = preprocessor.fit_transform(data)
data = pd.DataFrame(data=data)

# Splitting data into features and targets for oversampling
X = data.iloc[:, 0:-1]
y = data.iloc[:, -1]

# Oversampling minority classes
oversample = SMOTEENN(sampling_strategy='not majority', random_state=42)
X, y = oversample.fit_resample(X, y)

# Splitting the oversampled data in test and train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# Loading prediction model
clf_filename = 'train-02.pkl'
clf = joblib.load(config.MODELS_DIR / clf_filename) 

# Use the loaded model to make predictions 
clf.predict(X_test)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)