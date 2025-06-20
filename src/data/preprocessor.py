from src.config import PROCESSED_DATA_DIR, MODELS_DIR

# Importing modules
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.impute import SimpleImputer
import joblib


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
data = pd.read_csv(PROCESSED_DATA_DIR / "hidden_ckd_processed.csv")[ml_vars]


# Creating preprocessing pipelines for both numeric and nominal and ordinal data.
num_features = ['Age', 'Height', 'Weight', 'Systolic', 'Diastolic']
num_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

nom_features = ['S_Ethnicity', 'Family_KD']
nom_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

ord_features = ['Gender', 'Has_KD', 'Has_Diabetes','uACR']

gender_values = [['Female', 'Male']]
binary_values = [[False, True]]
uACR_values = [['Normal', 'Abnormal', 'High Abnormal']]

ord_transformer = make_column_transformer(
    (OrdinalEncoder(categories=gender_values), ['Gender']),
    (OrdinalEncoder(categories=binary_values), ['Has_KD']),
    (OrdinalEncoder(categories=binary_values), ['Has_Diabetes']),
    (OrdinalEncoder(categories=uACR_values), ['uACR']),
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('nom', nom_transformer, nom_features),
        ('ord', ord_transformer, ord_features)])


# Save preprocessor as pickle file
preprocessor_filename = 'preprocessor-02.pkl'
joblib.dump(preprocessor, MODELS_DIR / preprocessor_filename)

# Alternate preprocessing to convert 'uACR' to binary variable
data['uACR'] = data['uACR'].replace({1:0, 2:1})

# Save alternate preprocessor as pickle file
preprocessor_filename = 'preprocessor-01.pkl'
joblib.dump(preprocessor, MODELS_DIR / preprocessor_filename)