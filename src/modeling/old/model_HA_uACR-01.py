from src.config import MODEL_DATA_DIR, MODELS_DIR

# Importing modules
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import joblib

# Import data
X = pd.read_csv(MODEL_DATA_DIR / "features.csv")
y = pd.read_csv(MODEL_DATA_DIR / "targets.csv")
y = y.iloc[:,0]

# Oversample minority class
oversample = RandomOverSampler(sampling_strategy='not majority', random_state=42)
X_over, y_over = oversample.fit_resample(X, y)

# Split the oversampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.25, random_state=42)

# Define the pipeline with undersampling and model
steps = [('under', RandomUnderSampler(random_state=42)), ('model', RandomForestClassifier(n_estimators=25, random_state=42))]
pipeline = Pipeline(steps=steps)

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)



# Save the model as pickle file
model_filename = 'HA_uACR-01.pkl'
joblib.dump(pipeline, MODELS_DIR / model_filename) 

# Load the model from the file 
model = joblib.load(MODELS_DIR / model_filename) 

# Use the loaded model to make predictions 
model.predict(X_test)