from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import recall_score
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
from src import config


# XGBoost model
def train_xg(X_train, y_train, model_filename = 'train.pkl'):
    '''Trains XGBoost with gridsearch and saves the best model.'''
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }

    xgb_clf = XGBClassifier(
        objective='multi:softprob',
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

    return best_model


# Random forest model
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

    return clf


# KNN model
def train_knn(X_train, y_train, model_filename = 'train.pkl'):
    '''Trains KNN classifier with gridsearch and saves the best model.'''
    
    param_grid = {
        'n_neighbors': list(range(1, 31)),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }

    # GridSearchCV with recall scoring
    grid_search = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        cv=10,
        scoring='recall_weighted',
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    # Save best model
    best_model = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)
    model_path = config.MODELS_DIR / model_filename
    joblib.dump(best_model, model_path)

    return best_model


# Logistic regression model
def train_lr(X_train, y_train, model_filename = 'train.pkl'):
    '''Trains logistic regression with gridsearch and saves the best model.'''
    
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
    }

    # GridSearchCV with recall scoring
    grid_search = GridSearchCV(
        LogisticRegression(max_iter=5000, random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='recall_weighted',
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    # Save best model
    best_model = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)
    model_path = config.MODELS_DIR / model_filename
    joblib.dump(best_model, model_path)

    return best_model


# Gaussian naive bayes model
def train_gnb(X_train, y_train, model_filename = 'train.pkl'):
    '''Trains gaussian naive bayes model with gridsearch and saves the best model.'''

    # Create lists of priors
    options = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    all_combinations = np.array(list(product(options, repeat=3)))
    valid_combinations = all_combinations[np.isclose(all_combinations.sum(axis=1), 1.0)]
    priors = [list(map(float, combo)) for combo in valid_combinations]

    param_grid = {
    'var_smoothing': np.logspace(-9, -1, 100),
    'priors': priors}

    # GridSearchCV with recall scoring
    grid_search = GridSearchCV(
        GaussianNB(),
        param_grid,
        cv=5,
        scoring='recall_weighted',
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    # Save best model
    best_model = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)
    model_path = config.MODELS_DIR / model_filename
    joblib.dump(best_model, model_path)

    return best_model


# Ensemble classifier 01 (XGBoost + GNB)
def train_ensemble1(X_train, y_train, model_filename = 'train.pkl'):
    '''Trains ensemble classifier (XGBoost + GNB) and saves it.'''
    xgb_clf = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        random_state=42
    )

    gnb_clf = GaussianNB()

    # Probability-based soft voting ensemble classifier
    ensemble = VotingClassifier(
        estimators=[('xgb', xgb_clf), ('gnb', gnb_clf)],
        voting='soft', weights=[2,1])

    # Create lists of priors
    options = np.array([0.2, 0.3, 0.4, 0.5])
    all_combinations = np.array(list(product(options, repeat=3)))
    valid_combinations = all_combinations[np.isclose(all_combinations.sum(axis=1), 1.0)]
    priors = [list(map(float, combo)) for combo in valid_combinations]

    param_grid = {
        'xgb__n_estimators': [50, 100, 200],
        'xgb__max_depth': [3, 6],
        'xgb__learning_rate': [0.01, 0.1],
        'xgb__subsample': [0.8, 1.0],
        'gnb__var_smoothing': np.logspace(-3, -1, 10),
        'gnb__priors': priors,
    }

    # Grid search setup
    grid_search = GridSearchCV(
        estimator=ensemble,
        param_grid=param_grid,
        cv=5,
        scoring='recall_macro',
        n_jobs=-1,
        verbose=1
    )

    # Fit to training data
    grid_search.fit(X_train, y_train)

    # Results
    print("Best Params:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)

    # Save model
    best_model = grid_search.best_estimator_
    model_path = config.MODELS_DIR / model_filename
    joblib.dump(best_model, model_path)

    return best_model


# Ensemble classifier 02 (XGB + KNN)
def train_ensemble2(X_train, y_train, model_filename = 'train.pkl'):
    '''Trains ensemble classifier (XGB + KNN) and saves it.'''

    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    # GridSearchCV with recall scoring
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=rf_param_grid,
        cv=5,
        scoring='recall_macro',
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    # Best model
    best_rf = grid_search.best_estimator_
    print("Best RF Parameters:", grid_search.best_params_)

    xg_param_grid = {
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
        param_grid=xg_param_grid,
        scoring='recall_macro',
        cv=10,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    # Best model
    best_xg = grid_search.best_estimator_
    print("Best XGBoost Parameters:", grid_search.best_params_)

    # Create lists of priors
    options = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    all_combinations = np.array(list(product(options, repeat=3)))
    valid_combinations = all_combinations[np.isclose(all_combinations.sum(axis=1), 1.0)]
    priors = [list(map(float, combo)) for combo in valid_combinations]

    gnb_param_grid = {
    'var_smoothing': np.logspace(-9, -1, 100),
    'priors': priors
    }

    # GridSearchCV with recall scoring
    grid_search = GridSearchCV(
        GaussianNB(),
        param_grid=gnb_param_grid,
        cv=5,
        scoring='recall_macro',
        verbose=1,
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    # Save best model
    best_gnb = grid_search.best_estimator_
    print("Best GNB Parameters:", grid_search.best_params_)

    # Probability-based soft voting ensemble classifier
    ensemble = VotingClassifier(
        estimators=[('xgb', best_xg), ('rf', best_rf), ('gnb', best_gnb)],
        voting='soft', weights=[3,2,1])
    ensemble.fit(X_train, y_train)

    # Save model
    model_path = config.MODELS_DIR / model_filename
    joblib.dump(ensemble, model_path)

    return ensemble


def train_regressor(X_train, y_train, model_filename = 'regressor.pkl'):
    '''Trains a multi output regressor'''

    xgb_model = XGBRegressor(objective='reg:squarederror',
                             random_state=42)

    param_grid = {
        'n_estimators': [10, 15, 25, 50, 100],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [2, 4, 6, 8],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=10,
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train['Diastolic'])
    best_params = grid_search.best_params_
    print("Best hyperparameters:", best_params)
    tuned_regressor = grid_search.best_estimator_

    model = MultiOutputRegressor(tuned_regressor)
    model.fit(X_train, y_train)

    # Save model
    model_path = config.MODELS_DIR / model_filename
    joblib.dump(model, model_path)

    return model


# Ensemble classifier 02 (XGB + KNN)
def train_ensemble3(X_train, y_train, model_filename = 'train.pkl'):
    '''Trains ensemble classifier (XGB + KNN) and saves it.'''

    rf_param_grid = {
        'n_estimators': range(1, 100),
        'max_depth': [None, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # GridSearchCV with recall scoring
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=rf_param_grid,
        cv=5,
        scoring='recall_macro',
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    # Best model
    best_rf = grid_search.best_estimator_
    print("Best RF Parameters:", grid_search.best_params_)

    xg_param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }

    xgb_clf = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=xgb_clf,
        param_grid=xg_param_grid,
        scoring='recall_macro',
        cv=10,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    # Best model
    best_xg = grid_search.best_estimator_
    print("Best XGBoost Parameters:", grid_search.best_params_)

    # Probability-based soft voting ensemble classifier
    ensemble = VotingClassifier(
        estimators=[('xgb', best_xg), ('rf', best_rf)],
        voting='soft', weights=[2,1])
    ensemble.fit(X_train, y_train)

    # Save model
    model_path = config.MODELS_DIR / model_filename
    joblib.dump(ensemble, model_path)

    return ensemble
