# src/utils.py

import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from import_helper import config

def preprocess_data(X, y, target_mapping):
    '''Applies target encoding and resamples data.'''

    # Encode target
    y = y.map(target_mapping)

    # Balance the data with SMOTEENN
    oversample = SMOTEENN(sampling_strategy='not majority', random_state=42)
    X_over, y_over = oversample.fit_resample(X, y)
    return X_over, y_over

def stratified_split(X_over, y_over, test_size=0.25, random_state=42):
    '''Splits data into stratified train and test sets.'''
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_idx, test_idx in split.split(X_over, y_over):
        X_train, X_test = X_over.iloc[train_idx], X_over.iloc[test_idx]
        y_train, y_test = y_over.iloc[train_idx], y_over.iloc[test_idx]
    return X_train, X_test, y_train, y_test

def random_split(X_over, y_over, test_size=0.25, random_state=42):
    '''Splits data into randomly sampled train and test sets.'''
    # Shuffle dataframes and resetting indices after shuffling
    X_over = X_over.sample(frac=1, random_state=42).reset_index(drop=True)
    y_over = y_over.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split oversampled data into test and train sets
    X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

def train_xg(X_train, y_train, model_filename = 'train.pkl'):
    '''Trains XGBoost with grid search and saves the best model.'''
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }

    xgb_clf = XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        eval_metric='mlogloss',
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=xgb_clf,
        param_grid=param_grid,
        scoring='recall_macro',
        cv=10,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    # Save best model
    best_model = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)
    model_path = config.MODELS_DIR / model_filename
    joblib.dump(best_model, model_path)

    return best_model, grid_search.best_params_

def train_rf(X_train, X_test, y_train, y_test, model_filename = 'train.pkl'):
    '''Trains random forest optimising the number of estimators to maximise recall.'''
    # Initialize lists to store recall scores
    train_recalls = []
    test_recalls = []

    # Evaluate RandomForest performance across different tree counts
    for n in range(1, 100):
        clf = RandomForestClassifier(n_estimators=n, random_state=42)
        clf.fit(X_train, y_train)

        train_preds = clf.predict(X_train)
        test_preds = clf.predict(X_test)

        train_recall = recall_score(y_train, train_preds, average='macro')
        test_recall = recall_score(y_test, test_preds, average='macro')

        train_recalls.append(train_recall)
        test_recalls.append(test_recall)

    # Plotting results
    plt.figure(figsize=(13, 5))
    sns.set_style("whitegrid")
    plt.plot(train_recalls, label="Train Recall", linewidth=2)
    plt.plot(test_recalls, label="Test Recall", linewidth=2)
    plt.xlabel("n_estimators", fontsize=16)
    plt.ylabel("Macro Recall", fontsize=16)
    plt.title("Random Forest Recall vs n_estimators", fontsize=18)
    plt.legend(loc="lower right", fontsize=14)
    plt.xticks(range(0, 101, 10))
    plt.tight_layout()
    plt.show()

    # Find the best performing estimator
    max_recall = max(test_recalls)
    optimal_n = test_recalls.index(max_recall) + 1

    print(f"Max test recall: {max_recall:.4f} at {optimal_n} estimators")

    # Define parameters for random forest model
    clf = RandomForestClassifier(n_estimators=optimal_n, random_state=42)

    # Fit the model to the training data
    clf.fit(X_train, y_train)

    # Save the model
    joblib.dump(clf, config.MODELS_DIR / model_filename)

def load_model(model_filename):
    """Loads a trained model from disk."""
    return joblib.load(config.MODELS_DIR / model_filename)

def evaluate_model(model, X_test, y_test, target_mapping):
    """Evaluates the model's performance and displays classification results."""
    # Create ordered label names
    label_names = [label for label, _ in sorted(target_mapping.items(), key=lambda item: item[1])]

    # Predict
    y_pred = model.predict(X_test)

    # Print metrics
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()
