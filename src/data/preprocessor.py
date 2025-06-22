import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from import_helper import config

# Importing modules
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.impute import SimpleImputer
import joblib


features = [
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
]

# Importing processed data
data = pd.read_csv(config.PROCESSED_DATA_DIR / "hidden_ckd_processed.csv")[features]


# Creating preprocessing pipelines for both numeric and nominal and ordinal data.
num_features = ['Age', 'Height', 'Weight', 'Systolic', 'Diastolic']
num_transformer = Pipeline(steps=[
    ('power_transform', PowerTransformer(method='yeo-johnson'))])

nom_features = ['S_Ethnicity', 'Family_KD']
nom_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
    #('ordinal_encode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ])

ord_features = ['Gender', 'Has_KD', 'Has_Diabetes']

gender_values = [['Female', 'Male']]
binary_values = [[False, True]]

ord_transformer = make_column_transformer(
    (OrdinalEncoder(categories=gender_values), ['Gender']),
    (OrdinalEncoder(categories=binary_values), ['Has_KD']),
    (OrdinalEncoder(categories=binary_values), ['Has_Diabetes']),
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('nom', nom_transformer, nom_features),
        ('ord', ord_transformer, ord_features)])


# Save preprocessor as pickle file
preprocessor.fit(data)
preprocessor_filename = 'preprocessor.pkl'
joblib.dump(preprocessor, config.MODELS_DIR / preprocessor_filename)

# Save to CSV
data = preprocessor.transform(data)
data = pd.DataFrame(data=data)
data.to_csv(config.MODEL_DATA_DIR / 'features.csv', index=False)