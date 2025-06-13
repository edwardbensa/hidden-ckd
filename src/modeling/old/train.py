from src.config import MODELS_DIR, PROCESSED_DATA_DIR

# Importing modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn import metrics
import joblib


# Specifying variables
ml_vars = [
    'Age',
    'Height',
    'Weight',
    'Systolic',
    'Diastolic',
    'Gender',
    'Ethnicity_Black',
    'Has_KD',
    'Has_Diabetes',
    'Family_KD',
    'uACR'
]

# Importing processed data
data = pd.read_csv(PROCESSED_DATA_DIR / "hidden_ckd_processed.csv")[ml_vars]

# Remapping family history of kidney disease to definite family history of kidney disease
data['Family_KD'] = data['Family_KD'].apply(lambda x: 'yes' in x)



# Loading preprocessing model
preprocessor_filename = 'preprocessor.pkl'
preprocessor = joblib.load(MODELS_DIR / preprocessor_filename) 

# Transforming data and making a dataframe out of the transformed data
data = preprocessor.fit_transform(data)
data = pd.DataFrame(data=data, columns=ml_vars)

# Splitting data into features and targets for oversampling
X = data.drop('uACR', axis=1)
y = data['uACR']

# Oversampling minority classes
oversample = SMOTEENN(sampling_strategy='not majority', random_state=42)
X_over, y_over = oversample.fit_resample(X, y)

# Shuffling dataframes and resetting indices after shuffling
X_over = X_over.sample(frac=1, random_state=42).reset_index(drop=True)
y_over = y_over.sample(frac=1, random_state=42).reset_index(drop=True)

# Splitting the oversampled data in test and train sets
X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.25, random_state=42)


# initialize lists to store accuracies for training and test data we start with 50% accuracy
train_accuracies = []
test_accuracies = []

# iterate over a few depth values
for n in range(1, 100):

    # Define parameters for random forest model
    clf = RandomForestClassifier(n_estimators=n, random_state=42)

    # Fit the model to the training data
    clf.fit(X_train, y_train)

    #generate predictions on the training set
    train_predictions = clf.predict(X_train)

    # generate predictions on the test set
    test_predictions = clf.predict(X_test)

    # calculate the accuracy of predictions on training data set
    train_accuracy = metrics.accuracy_score(
    y_train, train_predictions
    )

    # calculate the accuracy of predictions on test data set
    test_accuracy = metrics.accuracy_score(
        y_test, test_predictions
    )

    # append accuracies
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

# create two plots using matplotlib and seaborn
plt.figure(figsize=(13, 5))
sns.set_style("whitegrid")
plt.plot(train_accuracies, label="train accuracy")
plt.plot(test_accuracies, label="test accuracy")
plt.legend(loc="upper left", prop={'size': 15})
plt.xticks(range(0, 101, 10))
plt.xlabel("n_estimators", size=20)
plt.ylabel("accuracy", size=20)
plt.show()

# find max test accuracy
max_accuracy = max(test_accuracies)
max_n_estimators = test_accuracies.index(max(test_accuracies))+1

print(f"Max test accuracy: {max_accuracy} at {max_n_estimators} estimators")


# Define parameters for random forest model
clf = RandomForestClassifier(n_estimators=max_n_estimators, random_state=42)

# Fit the model to the training data
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)



# Save the model as pickle file
model_filename = 'train.pkl'
joblib.dump(clf, MODELS_DIR / model_filename) 

# Load the model from the file 
model = joblib.load(MODELS_DIR / model_filename) 

# Use the loaded model to make predictions 
model.predict(X_test)