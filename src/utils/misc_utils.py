# Import modules
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors

# Function to group variables
def group_stats(df, column_name):
    '''
    Groups the df and runs some basic stats
    '''
    table = {
        column_name: df[column_name].sort_values().unique(),
        'Mean Age': df.groupby(column_name)['Age'].mean().round(2),
        'Mean Height (cm)': df.groupby(column_name)['Height'].mean().round(2),
        'Mean Weight (kg)': df.groupby(column_name)['Weight'].mean().round(2),
        'Mean BMI': df.groupby(column_name)['BMI'].mean().round(2),
        'Mean Systolic': df.groupby(column_name)['Systolic'].mean().round(2),
        'Mean Diastolic': df.groupby(column_name)['Diastolic'].mean().round(2),
        'Mean MAP': df.groupby(column_name)['MAP'].mean().round(2),
        'Count': df.groupby(column_name).size(),
        'Normal uACR %': round(df[(df['uACR'] == 'Normal')].groupby([column_name]).size()/df.groupby(column_name).size()*100, 2),
        'Abnormal uACR %': round(df[(df['uACR'] == 'Abnormal')].groupby([column_name]).size()/df.groupby(column_name).size()*100, 2),
        'High Abnormal uACR %': round(df[(df['uACR'] == 'High Abnormal')].groupby([column_name]).size()/df.groupby(column_name).size()*100, 2),
        'Low Risk %': round(df[(df['CKD_Risk'] == 'Low')].groupby([column_name]).size()/df.groupby(column_name).size()*100, 2),
        'Moderate Risk %': round(df[(df['CKD_Risk'] == 'Moderate')].groupby([column_name]).size()/df.groupby(column_name).size()*100, 2),
        'High Risk %': round(df[(df['CKD_Risk'] == 'High')].groupby([column_name]).size()/df.groupby(column_name).size()*100, 2),
        }
    table = pd.DataFrame(table).set_index(column_name).fillna(0)

    return table


# Multistop gradient function
def multi_stop_gradient(colours, n):
    '''Creates continuous gradient out of provided list of colours and breaks them up into new colours based on th number provided'''
    colours_rgb = [np.array(mcolors.to_rgb(colour)) for colour in colours]
    result = []
    
    for i in range(n):
        # Map i to the continuous range [0, len(colours)-1]
        pos = i * (len(colours) - 1) / (n - 1)
        
        # Find the segment
        segment = int(pos)
        if segment >= len(colours) - 1:
            segment = len(colours) - 2
        
        # Interpolate within the segment
        ratio = pos - segment
        interpolated = colours_rgb[segment] + (colours_rgb[segment + 1] - colours_rgb[segment]) * ratio
        result.append(mcolors.to_hex(interpolated))

    return result

def create_cmap(colours):
    cmap = mcolors.LinearSegmentedColormap.from_list("", colours)
    return cmap