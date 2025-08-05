import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import joblib
from src import config

def resample_data(X, y, target_mapping):
    '''Applies target encoding and resamples data.'''

    # Encode target
    y = y.map(target_mapping)

    # Balance the data with SMOTEENN
    resampler = SMOTE(sampling_strategy='all', random_state=42)
    X_res, y_res = resampler.fit_resample(X, y)

    # Balance the data with SMOTEENN
    #resampler = SMOTE(sampling_strategy='all')
    #X_res, y_res = resampler.fit_resample(X_res, y_res)
    return X_res, y_res


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
    X_train, X_test, y_train, y_test = train_test_split(X_over, y_over,
                                                        test_size=test_size,
                                                        random_state=random_state)

    return X_train, X_test, y_train, y_test


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

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(cmap='Blues', ax=axs[1])
    axs[1].set_title('Confusion Matrix')
    axs[1].grid(False)

    # Classification report heatmap
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    df_metrics = df_report[['precision', 'recall', 'f1-score']].iloc[:-3]
    df_metrics.index = label_names
    df_metrics.loc['Overall'] = df_report.loc['weighted avg'][['precision', 'recall', 'f1-score']]

    sns.heatmap(df_metrics, annot=True, fmt=".3f", cmap='Blues', ax=axs[0])
    for spine in axs[0].spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(0.75)
    axs[0].set_title('Classification Metrics Heatmap')
    axs[0].set_xlabel('Metrics')
    axs[0].set_ylabel('Classes')
    axs[0].set_yticklabels(axs[0].get_yticklabels(), rotation=0)
    axs[0].set_aspect(0.755) 

    plt.tight_layout()
    plt.show()
