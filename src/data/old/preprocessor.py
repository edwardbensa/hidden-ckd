from src.config import PROCESSED_DATA_DIR, MODELS_DIR

# Importing modules
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import PowerTransformer
import joblib


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

# Creating preprocessing pipelines for both numeric and categorical data.
num_features = ['Age', 'Height', 'Weight', 'Systolic', 'Diastolic']
num_transformer = Pipeline(steps=[
    ('scaler', PowerTransformer())])

cat_features = ['Gender', 'Ethnicity_Black', 'Has_KD', 'Has_Diabetes', 'Family_KD', 'uACR']

gender_values = [['Female', 'Male']]
uACR_values = [['Normal', 'Abnormal', 'High Abnormal']]
binary_values = [[False, True]]

cat_transformer = make_column_transformer(
    (OrdinalEncoder(categories=gender_values), ['Gender']),
    (OrdinalEncoder(categories=binary_values), ['Ethnicity_Black']),
    (OrdinalEncoder(categories=binary_values), ['Has_KD']),
    (OrdinalEncoder(categories=binary_values), ['Has_Diabetes']),
    (OrdinalEncoder(categories=binary_values), ['Family_KD']),
    (OrdinalEncoder(categories=uACR_values), ['uACR']),
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)])


# Save preprocessor as pickle file
preprocessor_filename = 'preprocessor.pkl'
joblib.dump(preprocessor, MODELS_DIR / preprocessor_filename)