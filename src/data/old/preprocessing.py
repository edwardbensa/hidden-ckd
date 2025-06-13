from src.config import PROCESSED_DATA_DIR, MODEL_DATA_DIR

# Importing modules
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer

# Importing processed data
data = pd.read_csv(PROCESSED_DATA_DIR / "hidden_ckd_processed.csv")

numeric_vars = ['Age', 'Height', 'Weight', 'Systolic', 'Diastolic']
nv = np.array(data[numeric_vars])

scaler = PowerTransformer()

# Standardizing numeric variables 
scaled_data = scaler.fit_transform(nv) 
data[numeric_vars] = pd.DataFrame(scaled_data, columns=numeric_vars)

# Encode data
data['Male'] = data['Gender'].map({'Male' : 1, 'Female' : 0})

# Encode uACR values
uACR_mapping = {
    'Normal' : 0,
    'Abnormal' : 1,
    'High Abnormal' : 2
}

data['uACR'] = data['uACR'].map(uACR_mapping)

# Create a High Abnormal uACR column since this show the most variation when compared with other variables (especially BP)
HA_uACR_mapping = {
    0 : 0,
    1 : 0,
    2 : 1
}

data['HA_uACR'] = data['uACR'].map(HA_uACR_mapping)

# Make list of nominal categorical variables
nominal_vars = ['S_Ethnicity', 'Family_KD']

# Apply one hot encoding
data = pd.get_dummies(data = data,
                         prefix = nominal_vars,
                         columns = nominal_vars)

# Select train features/columns
cols = [
    'Age',
    'Male',
    'S_Ethnicity_Black',
    'S_Ethnicity_Indian',
    'S_Ethnicity_Mixed',
    'S_Ethnicity_SE Asian',
    'S_Ethnicity_White',
    'Height',
    'Weight',
    'Systolic',
    'Diastolic',
    'Has_KD',
    'Has_Diabetes',
    'Family_KD_Definitely yes',
    'Family_KD_Definitely not'
]

# Feature variables
X = data[cols]
X.to_csv(MODEL_DATA_DIR / "features.csv", index = False)

# Target variables (HA_uACR and encoded uACR)
y = data[['HA_uACR', 'uACR']]
y.to_csv(MODEL_DATA_DIR / "targets.csv", index = False)