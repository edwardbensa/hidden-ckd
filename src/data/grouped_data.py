from src.config import PROCESSED_DATA_DIR

# Importing libraries
import pandas as pd

data = pd.read_csv(PROCESSED_DATA_DIR / "hidden_ckd_processed.csv")

# Creating a function to group variables

def group_stats(column_name):
    '''
    Groups the data and runs some basic stats
    '''
    table = {
        column_name: data[column_name].sort_values().unique(),
        'Mean Age': data.groupby(column_name)['Age'].mean().round(2),
        'Mean Height (cm)': data.groupby(column_name)['Height'].mean().round(2),
        'Mean Weight (kg)': data.groupby(column_name)['Weight'].mean().round(2),
        'Mean BMI': data.groupby(column_name)['BMI'].mean().round(2),
        'Mean Systolic': data.groupby(column_name)['Systolic'].mean().round(2),
        'Mean Diastolic': data.groupby(column_name)['Diastolic'].mean().round(2),
        'Count': data.groupby(column_name).size(),
        'Normal uACR %': round(data[(data['uACR'] == 'Normal')].groupby([column_name]).size()/data.groupby(column_name).size()*100, 2),
        'Abnormal uACR %': round(data[(data['uACR'] == 'Abnormal')].groupby([column_name]).size()/data.groupby(column_name).size()*100, 2),
        'High Abnormal uACR %': round(data[(data['uACR'] == 'High Abnormal')].groupby([column_name]).size()/data.groupby(column_name).size()*100, 2),
        }
    table = pd.DataFrame(table).set_index(column_name).fillna(0)

    return table


# Grouping data by ethnicity
eth_data = group_stats('Ethnicity')

# Grouping data by simplified ethnicity
s_eth_data = group_stats('S_Ethnicity')

# Grouping data by whether or not the particpant is black
eth_black_data = group_stats('Ethnicity_Black')

# Grouping data by gender
gender_data = group_stats('Gender')

# Grouping data by age category
age_data = group_stats('Age_Category').iloc[[3,0,1,2,4], :]

# Grouping data by BP category
bp_cat_data = group_stats('BP_Category').iloc[[3,4,0,1,2], :]

# Grouping data by BMI category
bmi_data = group_stats('BMI_Category').iloc[[0,2,1], :]

# Grouping data by whether or not the particpant has a family history of kidney disease
fam_data = group_stats('Family_KD')

# Grouping data by whether or not the participant has kidney disease
hkd_data = group_stats('Has_KD')

# Grouping data by whether or not the participant has diabetes
hdiabetes_data = group_stats('Has_Diabetes')