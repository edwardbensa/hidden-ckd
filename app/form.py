import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
from sklearn.pipeline import Pipeline
import joblib
import csv
from datetime import datetime
import os

# Assuming MODELS_DIR and APP_DIR are defined in a config similar to your Tkinter app
# For demonstration, I'll define them here. In a real app, manage these paths carefully.
# You might need to adjust these paths based on where your models and CSV will be stored
# relative to your Dash app's execution location.
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models') # Assuming 'models' directory in the same place as your app.py
APP_DIR = os.path.dirname(__file__) # Assuming 'responses.csv' will be in the same place as your app.py

# Load models outside the app layout/callbacks to avoid reloading on every request
preprocessor = None
model = None

try:
    preprocessor = joblib.load(os.path.join(MODELS_DIR, 'preprocessor.pkl'))
    model = joblib.load(os.path.join(MODELS_DIR, 'train.pkl'))
except FileNotFoundError:
    print("Error: Model files not found. Ensure preprocessor.pkl and train.pkl are in the correct directory.")
except Exception as e:
    print(f"Error loading models: {str(e)}")

ethnicity_map = {
    'Black African (Central Africa)': 'Black',
    'Black African (East Africa)': 'Black',
    'Black African (North Africa)': 'Black',
    'Black African (South Africa)': 'Black',
    'Black African (West Africa)': 'Black',
    'Black African (Other)': 'Black',
    'Black Caribbean': 'Black',
    'Black (Other)': 'Black',
    'Mixed White/Asian': 'Mixed',
    'Mixed White/Black African': 'Mixed',
    'Mixed White/Black Caribbean': 'Mixed',
    'Mixed (Other)': 'Mixed',
    'White British': 'White',
    'White Gypsy/Traveller': 'White',
    'White Irish': 'White',
    'White (Other)': 'White',
    'Pakistani': 'South Asian',
    'Indian': 'South Asian',
    'Bangladeshi': 'South Asian',
    'East Asian': 'Other',
    'South East Asian': 'Other',
    'Other': 'Other',
}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H1("CKD Risk Prediction System", className="text-center my-4"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Gender:"),
            dcc.Dropdown(
                id='gender-input',
                options=[{'label': 'Male', 'value': 'Male'},
                         {'label': 'Female', 'value': 'Female'},
                         {'label': 'Prefer not to say', 'value': 'Prefer not to say'}],
                placeholder="Select Gender",
                style={'width': '100%'}
            ),
        ], width=6),
        dbc.Col([
            dbc.Label("Age:"),
            dbc.Input(id='age-input', type='number', placeholder='Enter Age (1-120)'),
        ], width=6),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Height (cm):"),
            dbc.Input(id='height-input', type='number', placeholder='Enter Height (cm)'),
        ], width=6),
        dbc.Col([
            dbc.Label("Weight (kg):"),
            dbc.Input(id='weight-input', type='number', placeholder='Enter Weight (kg)'),
        ], width=6),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Systolic BP:"),
            dbc.Input(id='systolic-input', type='number', placeholder='Enter Systolic BP'),
        ], width=6),
        dbc.Col([
            dbc.Label("Diastolic BP:"),
            dbc.Input(id='diastolic-input', type='number', placeholder='Enter Diastolic BP'),
        ], width=6),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Ethnicity:"),
            dcc.Dropdown(
                id='ethnicity-input',
                options=[{'label': k, 'value': k} for k in ethnicity_map.keys()],
                placeholder="Select Ethnicity",
                style={'width': '100%'}
            ),
        ], width=6),
        dbc.Col([
            dbc.Label("Do you have a family history of kidney disease:"),
            dcc.Dropdown(
                id='family-kd-input',
                options=[
                    {'label': 'Definitely yes', 'value': 'Definitely yes'},
                    {'label': 'Definitely not', 'value': 'Definitely not'},
                    {'label': 'Not sure', 'value': 'Not sure'}
                ],
                placeholder="Select Option",
                style={'width': '100%'}
            ),
        ], width=6),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Have you ever been diagnosed with kidney disease?"),
            dbc.Checklist(
                options=[{"label": "Yes", "value": True}],
                value=[],
                id="has-kd-input",
                switch=True,
            ),
        ], width=6),
        dbc.Col([
            dbc.Label("Have you been diagnosed with diabetes?"),
            dbc.Checklist(
                options=[{"label": "Yes", "value": True}],
                value=[],
                id="has-diabetes-input",
                switch=True,
            ),
        ], width=6),
    ], className="mb-4"),

    dbc.Button("Get Results", id='predict-button', color="primary", className="me-2"),
    dbc.Button("Clear Results", id='clear-button', color="secondary", className="ms-2"),

    html.Hr(),

    dbc.Card([
        dbc.CardHeader("Risk Prediction and Recommendation"),
        dbc.CardBody([
            html.Pre(id='results-output', style={'whiteSpace': 'pre-wrap', 'wordBreak': 'break-all'})
        ])
    ], className="mt-4"),

    dcc.Store(id='prediction-store') # To store the prediction value across callbacks
])

def validate_inputs(age, height, weight, systolic, diastolic):
    """Validate user inputs and return error message if any."""
    if not all([age, height, weight, systolic, diastolic]):
        return "All numerical fields must be filled."

    if not (1 <= age <= 120):
        return "Age must be between 1 and 100."
    if not (1 <= height <= 200):
        return "Height must be between 1 and 200 cm."
    if not (1 <= weight <= 250):
        return "Weight must be between 1 and 250 kg."
    if not (1 <= systolic <= 300):
        return "Systolic BP must be between 1 and 300."
    if not (1 <= diastolic <= 200):
        return "Diastolic BP must be between 1 and 200."
    return None

@app.callback(
    [Output('results-output', 'children'),
     Output('prediction-store', 'data')],
    [Input('predict-button', 'n_clicks')],
    [State('gender-input', 'value'),
     State('age-input', 'value'),
     State('height-input', 'value'),
     State('weight-input', 'value'),
     State('systolic-input', 'value'),
     State('diastolic-input', 'value'),
     State('ethnicity-input', 'value'),
     State('family-kd-input', 'value'),
     State('has-kd-input', 'value'),
     State('has-diabetes-input', 'value')]
)
def make_prediction(n_clicks, gender, age, height, weight, systolic, diastolic,
                    ethnicity, family_kd, has_kd, has_diabetes):
    if n_clicks is None:
        return "", None

    validation_error = validate_inputs(age, height, weight, systolic, diastolic)
    if validation_error:
        return validation_error, None

    if preprocessor is None or model is None:
        return "Error: Models not loaded properly.", None

    try:
        # Dash Checklists return a list, convert to boolean
        has_kd_bool = bool(has_kd)
        has_diabetes_bool = bool(has_diabetes)

        data = pd.DataFrame([{
            'Age': float(age),
            'Height': float(height),
            'Weight': float(weight),
            'Systolic': float(systolic),
            'Diastolic': float(diastolic),
            'S_Ethnicity': ethnicity_map.get(ethnicity),
            'Family_KD': family_kd,
            'Gender': gender,
            'Has_KD': has_kd_bool,
            'Has_Diabetes': has_diabetes_bool,
        }])

        full_pipeline = Pipeline(steps=[
            ('preprocessing', preprocessor),
            ('model', model)
        ])

        prediction = full_pipeline.predict(data)[0] # Get the single prediction value

        # Convert prediction to a readable message
        if prediction == 0:
            message = "CKD Risk: Low Risk \nTake it easy"
        elif prediction == 1:
            message = "CKD Risk: Moderate Risk \nWe recommend a few lifestyle changes to mitigate your risk"
        elif prediction == 2:
            message = "CKD Risk: High \nWe recommend a GP visit immediately"
        else:
            message = "Unknown prediction result"

        return message, prediction

    except Exception as e:
        return f"Prediction Error: {str(e)}", None

@app.callback(
    Output('results-output', 'children', allow_duplicate=True), # Allow duplicate output for clearing
    [Input('clear-button', 'n_clicks')],
    prevent_initial_call=True
)
def clear_results(n_clicks):
    if n_clicks:
        return ""
    return dash.no_update

@app.callback(
    Output('dummy-output-for-saving', 'children'), # A dummy output since we are performing a side effect
    [Input('prediction-store', 'data')],
    [State('gender-input', 'value'),
     State('age-input', 'value'),
     State('height-input', 'value'),
     State('weight-input', 'value'),
     State('systolic-input', 'value'),
     State('diastolic-input', 'value'),
     State('ethnicity-input', 'value'),
     State('family-kd-input', 'value'),
     State('has-kd-input', 'value'),
     State('has-diabetes-input', 'value')],
    prevent_initial_call=True
)
def save_to_csv(prediction_data, gender, age, height, weight, systolic, diastolic,
                ethnicity, family_kd, has_kd, has_diabetes):
    if prediction_data is None:
        return ""

    try:
        has_kd_bool = bool(has_kd)
        has_diabetes_bool = bool(has_diabetes)

        entry_data = {
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Gender': gender,
            'Ethnicity': ethnicity,
            'S_Ethnicity': ethnicity_map.get(ethnicity),
            'Age': age,
            'Height': height,
            'Weight': weight,
            'Systolic': systolic,
            'Diastolic': diastolic,
            'Family_KD': family_kd,
            'Has_KD': has_kd_bool,
            'Has_Diabetes': has_diabetes_bool,
            'Prediction': prediction_data
        }
        
        # Create a DataFrame for consistent structure with your original code
        entry_df = pd.DataFrame([entry_data])

        filename = os.path.join(APP_DIR, 'responses.csv')
        cols = [
            'Timestamp', 'Gender', 'Ethnicity', 'S_Ethnicity', 'Age', 'Height',
            'Weight', 'Systolic', 'Diastolic', 'Family_KD', 'Has_KD', 'Has_Diabetes', 'Prediction'
        ]
        
        # Check if the file exists to write headers if needed
        file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0

        with open(filename, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=cols)

            if not file_exists:
                writer.writeheader()  # write header only once

            writer.writerow(entry_df.iloc[0].to_dict())
        return "Data saved successfully!"
    except Exception as e:
        return f"Error saving data: {str(e)}"


if __name__ == '__main__':
    app.run(debug=True)