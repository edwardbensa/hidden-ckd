from import_helper import config

# Importing libraries
import numpy as np
import pandas as pd

raw_data = pd.read_csv(config.RAW_DATA_DIR / "hidden_ckd_raw.csv")

# Filtering out all rows with uACR results that are not Normal, Abnormal or High Abnormal
raw_data = raw_data[(raw_data['uACR'] == "Normal") | (raw_data['uACR'] == "Abnormal") | (raw_data['uACR'] == "High abnormal")]

# Renaming specific row values
raw_data['uACR']=raw_data['uACR'].replace({'High abnormal': 'High Abnormal'})
raw_data['Ethnicity'] = raw_data['Ethnicity'].replace({'Black African ' : 'Black African (unspecified)',
                                                       'Black African' : 'Black African (unspecified)'})

# Adding a "Simplified Ethnicity" column
raw_data['S_Ethnicity'] = raw_data['Ethnicity'].replace({
    'Black African' : 'Black',
    'Black African (Central Africa)' : 'Black',
    'Black African (East Africa)' : 'Black',
    'Black African (North Africa)' : 'Black',
    'Black African (South Africa)' : 'Black',
    'Black African (West Africa)' : 'Black',
    'Black African (unspecified)' : 'Black',
    'Black Caribbean' : 'Black',
    'Black other' : 'Black',
    'Indian' : 'Indian',
    'Mixed White/Asian' : 'Mixed',
    'Mixed White/Black African' : 'Mixed',
    'Mixed White/Black Caribbean' : 'Mixed',
    'Mixed other' : 'Mixed',
    'Pakistani' : 'Indian',
    'White British' : 'White',
    'White Gypsy/Traveller' : 'White',
    'White Irish' : 'White',
    'White other' : 'White',
    'Any other' : 'Other',
    'Asian other' : 'SE Asian',
    'Bangladeshi' : 'Indian'})

# Adding another column that simply shows whether the ethicity is black or not
raw_data['Ethnicity_Black'] = raw_data['Ethnicity'].str.contains('Black')

# Dropping the one gender outlier 'Prefer not to say' (sample size of gender='Prefer not to say' = 1. This may interfere with analysis)
raw_data.drop(raw_data[raw_data.Gender == 'Prefer not to say'].index, inplace=True)

# Dropping an outlier whose date of birth was recorded as 10/09/2023
raw_data.drop(raw_data[raw_data['D.O.B.'] == "10/09/2023"].index, inplace=True)

# Dropping the one BMI outlier whose BMI Category is 'Underweight' (sample size = 1. May interfere with analysis)
raw_data.drop(raw_data[raw_data['BMI Category'] == "UNDERWEIGHT"].index, inplace=True)

# Adding a column that calculates pulse pressure
raw_data['Pulse_Pressure'] = raw_data['Systolic'] - raw_data['Diastolic']

# Splitting the "Medical Conditions" column into the constituent medical conditions
raw_data['Has_High_BP'] = raw_data['Medical Conditions'].apply(lambda x: 'High blood pressure' in x)
raw_data['Has_Diabetes'] = raw_data['Medical Conditions'].apply(lambda x: 'Diabetes' in x)
raw_data['Has_KD'] = raw_data['Medical Conditions'].apply(lambda x: 'Kidney disease' in x)
raw_data['Has_HD'] = raw_data['Medical Conditions'].apply(lambda x: 'Heart disease (heart attack, angina, heart failure)' in x)
raw_data['Has_Other'] = raw_data['Medical Conditions'].apply(lambda x: 'Other' in x)

# Splitting the "What medications/tablets are you currently taking? -" column into the constituent forms of medication
raw_data['BP_Meds'] = raw_data['What medications/tablets are you currently taking? - '].apply(lambda x: 'Blood pressure medication' in x)
raw_data['Diabetes_Meds'] = raw_data['What medications/tablets are you currently taking? - '].apply(lambda x: 'Diabetes medication' in x)
raw_data['Cholesterol_Meds'] = raw_data['What medications/tablets are you currently taking? - '].apply(lambda x: 'Cholesterol medication (e.g. Statin)' in x)
raw_data['Other_Meds'] = raw_data['What medications/tablets are you currently taking? - '].apply(lambda x: 'Other' in x)

# Creating processed dataframe
data = raw_data.rename({"Date of event": "Date",
                        "D.O.B.": "DOB", 
                        "Height (cm)": "Height",
                        "Weight(kg)": "Weight",
                        "BMI Category": "BMI_Category",
                        "BP Category": "BP_Category",
                        "Do you have a family history of  kidney disease?":"Family_KD"},
                       axis=1)

# Calculating age by subtracting date of birth from date of event
data['Age'] = data['Age'] = (pd.to_datetime(data['Date'], dayfirst = True) - pd.to_datetime(data['DOB'], dayfirst = True)) / np.timedelta64(1, 'D') / 365
data['Age'] = data['Age'].round(1)

# Creating an 'Age Category' column
data['Age_Category'] = pd.cut(data['Age'], bins=[0, 25, 40, 55, 70, float('inf')], labels=['<25', '25-40', '41-55', '56-70', '>70']).astype(str)

# Final processed dataframe
data = data[['Date',
             'Gender',
             'Ethnicity',
             'S_Ethnicity',
             'Ethnicity_Black',
             'DOB',
             'Age',
             'Age_Category',
             'Height',
             'Weight',
             'BMI',
             'BMI_Category',
             'Systolic',
             'Diastolic',
             'Pulse_Pressure',
             'BP_Category',
             'Has_High_BP',
             'Has_Diabetes',
             'Has_KD',
             'Has_HD',
             'BP_Meds',
             'Diabetes_Meds',
             'Cholesterol_Meds',
             'Other_Meds',
             'Family_KD',
             'uACR']]

# Exporting dataframe to csv
data.to_csv(config.PROCESSED_DATA_DIR / 'hidden_ckd_processed.csv', index = False)