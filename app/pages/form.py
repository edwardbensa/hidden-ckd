import os
import csv
from datetime import datetime
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
from sklearn.pipeline import Pipeline
from src.config import APP_DIR
from src.utils.model_utils import load_model

basic_preprocessor = None
scaler = None
classifier_bp = None
classifier_nobp = None

try:
    basic_preprocessor = load_model('basic_preprocessor_bp.pkl')
    scaler = load_model('scaler_bp.pkl')
    classifier_bp = load_model('classifier_bp.pkl')
    classifier_nobp = load_model('classifier_nobp.pkl')
except FileNotFoundError:
    print("Error: Model files not found. Ensure models are in the correct directory.")
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

layout = dbc.Container([
    html.H1("CKD Risk Prediction", className="text-left my-4"),

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
        ], width=4),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Age:"),
            dbc.Input(id='age-input', type='number', placeholder='Enter Age'),
        ], width=4),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Weight:"),
            dbc.Input(id='weight-input-visible', type='number', placeholder='Enter Weight'),
        ], width=2),
        dbc.Col([
            dbc.Label("Unit:"),
            dbc.RadioItems(
                options=[
                    {'label': 'kg', 'value': 'kg'},
                    {'label': 'stone', 'value': 'st'},
                    {'label': 'lbs', 'value': 'lbs'}
                ],
                value='kg',
                id='weight-unit-radio',
                inline=True
            ),
        ], width=2),
        dbc.Input(id='weight-input-kg', type='hidden'),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Height:"),
            dbc.Input(id='height-input-visible', type='number', placeholder='Enter Height'),
        ], width=2),
        dbc.Col([
            dbc.Label("Unit:"),
            dbc.RadioItems(
                options=[
                    {'label': 'cm', 'value': 'cm'},
                    {'label': 'ft', 'value': 'ft'}
                ],
                value='cm',
                id='height-unit-radio',
                inline=True
            ),
        ], width=2),
        dbc.Input(id='height-input-cm', type='hidden'),
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
        ], width=4),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Do you have recent BP readings?"),
            dbc.Checklist(
                options=[{"label": "Yes", "value": True}],
                value=[],
                id="bp-readings-switch",
                switch=True,
            ),
        ], width=12),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Systolic BP:"),
            dbc.Input(id='systolic-input', type='number', placeholder='Enter Systolic BP'),
        ], width=4),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Diastolic BP:"),
            dbc.Input(id='diastolic-input', type='number', placeholder='Enter Diastolic BP'),
        ], width=4),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Do you have a family history of kidney disease?:"),
            dcc.Dropdown(
                id='family-kd-input',
                options=[
                    {'label': 'Yes', 'value': 'Definitely yes'},
                    {'label': 'No', 'value': 'Definitely not'},
                    {'label': 'Not sure', 'value': 'Not sure'}
                ],
                placeholder="Select Option",
                style={'width': '100%'}
            ),
        ], width=4),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Have you been diagnosed with any of the following?"),
        ], width=6),
    ], className="mb-2"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Diabetes:  "),
            html.Span(id="diabetes-label-text"),
            dbc.Checklist(
                options=[{"label": "", "value": True}],
                value=[],
                id="has-diabetes-input",
                switch=True,
            ),
        ], width=2),
        dbc.Col([
            dbc.Label("Hypertension:  "),
            html.Span(id="hpt-label-text"),
            dbc.Checklist(
                options=[{"label": "", "value": True}],
                value=[],
                id="has-hpt-input",
                switch=True,
            ),
        ], width=2),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Button(
                "Optional Research Information",
                id="collapse-button",
                className="mb-3",
                color="info",
                n_clicks=0,
            ),
        ], width=12),
    ]),

    dbc.Collapse(
        id="collapse",
        is_open=False,
        children=[
            dbc.Card(
                dbc.CardBody([
                    html.H4("Optional Information for Future Research", className="text-center my-4"),
                    html.P(
                        "This data will not be used in the current prediction but will aid in future research. Your participation is completely voluntary.",
                        className="text-center"
                    ),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Have you been diagnosed with heart disease?"),
                            dbc.Checklist(
                                options=[{"label": "Yes", "value": True}],
                                value=[],
                                id="has-heart-disease-input",
                                switch=True,
                            ),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Have you been diagnosed with kidney disease?"),
                            dbc.Checklist(
                                options=[{"label": "Yes", "value": True}],
                                value=[],
                                id="has-kidney-disease-input",
                                switch=True,
                            ),
                        ], width=6),
                    ], className="mb-4"),

                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Are you currently taking medication for your blood pressure?"),
                            dbc.Checklist(
                                options=[{"label": "Yes", "value": True}],
                                value=[],
                                id="taking-bp-meds-input",
                                switch=True,
                            ),
                        ], width=4),
                        dbc.Col([
                            dbc.Label("Are you currently taking medication for diabetes?"),
                            dbc.Checklist(
                                options=[{"label": "Yes", "value": True}],
                                value=[],
                                id="taking-diabetes-meds-input",
                                switch=True,
                            ),
                        ], width=4),
                        dbc.Col([
                            dbc.Label("Are you currently taking medication for high cholesterol?"),
                            dbc.Checklist(
                                options=[{"label": "Yes", "value": True}],
                                value=[],
                                id="taking-cholesterol-meds-input",
                                switch=True,
                            ),
                        ], width=4),
                    ], className="mb-4"),

                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Are you currently taking any other medication?"),
                            dbc.Checklist(
                                options=[{"label": "Yes", "value": True}],
                                value=[],
                                id="taking-other-meds-input",
                                switch=True,
                            ),
                        ], width=12),
                    ], className="mb-4"),
                ]),
            )
        ],
    ),
    
    dbc.Row([
        dbc.Col([
            dbc.Button("Get Results", id='predict-button', color="primary", className="me-2"),
            dbc.Button("Clear Results", id='clear-button', color="secondary", className="ms-2"),
        ], width=12, className="mt-3"),
    ]),

    html.Hr(),

    dbc.Card([
        dbc.CardHeader("Risk Prediction and Recommendation"),
        dbc.CardBody([
            html.Pre(id='results-output', style={'whiteSpace': 'pre-wrap', 'wordBreak': 'break-all'})
        ])
    ], className="mt-4"),

    dcc.Store(id='prediction-store'),

])

# Callback for height conversion
@callback(
    Output('height-input-cm', 'value'),
    Input('height-input-visible', 'value'),
    Input('height-unit-radio', 'value')
)
def convert_height(height_value, unit):
    if height_value is None:
        return None
    if unit == 'ft':
        return float(height_value) * 30.48
    return float(height_value)

# Callback for weight conversion
@callback(
    Output('weight-input-kg', 'value'),
    Input('weight-input-visible', 'value'),
    Input('weight-unit-radio', 'value')
)
def convert_weight(weight_value, unit):
    if weight_value is None:
        return None
    if unit == 'st':
        return float(weight_value) * 6.35029
    elif unit == 'lbs':
        return float(weight_value) * 0.453592
    return float(weight_value)

# Callback for toggling BP inputs
@callback(
    [Output('systolic-input', 'disabled'),
     Output('diastolic-input', 'disabled')],
    [Input('bp-readings-switch', 'value')]
)
def toggle_bp_inputs(bp_switch_value):
    is_disabled = not bp_switch_value
    return is_disabled, is_disabled

# Callback for updating diabetes label
@callback(
    Output("diabetes-label-text", "children"),
    Input("has-diabetes-input", "value")
)
def update_diabetes_label(value):
    if value:
        return " Yes"
    return " No"

# Callback for updating hypertension label
@callback(
    Output("hpt-label-text", "children"),
    Input("has-hpt-input", "value")
)
def update_hpt_label(value):
    if value:
        return " Yes"
    return " No"

# Callback for opening Optional Section
@callback(
    Output("collapse", "is_open"),
    Input("collapse-button", "n_clicks"),
    State("collapse", "is_open"),
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

# The main prediction callback
@callback(
    [Output('results-output', 'children'),
     Output('prediction-store', 'data')],
    [Input('predict-button', 'n_clicks')],
    [State('gender-input', 'value'),
     State('age-input', 'value'),
     State('height-input-cm', 'value'),
     State('weight-input-kg', 'value'),
     State('systolic-input', 'value'),
     State('diastolic-input', 'value'),
     State('ethnicity-input', 'value'),
     State('family-kd-input', 'value'),
     State('has-hpt-input', 'value'),
     State('has-diabetes-input', 'value'),
     State('bp-readings-switch', 'value')]
)
def make_prediction(n_clicks, gender, age, height, weight, systolic, diastolic,
                    ethnicity, family_kd, has_kd, has_diabetes, bp_switch_value):
    if n_clicks is None:
        return "", None

    # --- Combined Validation Logic ---
    validation_error = None
    if bp_switch_value:
        # Validate all fields if BP readings are provided
        if not all([age, height, weight, systolic, diastolic]):
            validation_error = "All numerical fields must be filled."
        elif not (1 <= age <= 120):
            validation_error = "Age must be between 1 and 100."
        elif not (1 <= height <= 250):
            validation_error = "Height must be between 1 and 250 cm."
        elif not (1 <= weight <= 300):
            validation_error = "Weight must be between 1 and 300 kg."
        elif not (1 <= systolic <= 300):
            validation_error = "Systolic BP must be between 1 and 300."
        elif not (1 <= diastolic <= 200):
            validation_error = "Diastolic BP must be between 1 and 200."
    else:
        # Validate only core fields if BP readings are not provided
        if not all([age, height, weight]):
            validation_error = "Age, Height, and Weight must be filled."
        elif not (1 <= age <= 120):
            validation_error = "Age must be between 1 and 120."
        elif not (1 <= height <= 250):
            validation_error = "Height must be between 1 and 250 cm."
        elif not (1 <= weight <= 300):
            validation_error = "Weight must be between 1 and 300 kg."

    if validation_error:
        return validation_error, None

    # --- Model Selection and Data Preparation ---
    if bp_switch_value:
        selected_model = classifier_bp
        data = pd.DataFrame([{
            'Age': float(age), 'Height': float(height), 'Weight': float(weight),
            'Systolic': float(systolic), 'Diastolic': float(diastolic),
            'S_Ethnicity': ethnicity_map.get(ethnicity), 'Family_KD': family_kd,
            'Gender': gender, 'Has_KD': bool(has_kd), 'Has_Diabetes': bool(has_diabetes),
        }])
    else:
        selected_model = classifier_nobp
        data = pd.DataFrame([{
            'Age': float(age), 'Height': float(height), 'Weight': float(weight),
            'S_Ethnicity': ethnicity_map.get(ethnicity), 'Family_KD': family_kd,
            'Gender': gender, 'Has_KD': bool(has_kd), 'Has_Diabetes': bool(has_diabetes),
        }])

    # --- Prediction and Result Formatting ---
    if selected_model is None:
        return "Error: Models not loaded properly.", None

    try:
        full_pipeline = Pipeline(steps=[
            ('basic_preprocessor', basic_preprocessor),
            ('scaler', scaler),
            ('model', classifier_bp)
        ])
        prediction = full_pipeline.predict(data)[0]

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

@callback(
    Output('results-output', 'children', allow_duplicate=True), # Allow duplicate output for clearing
    [Input('clear-button', 'n_clicks')],
    prevent_initial_call=True
)
def clear_results(n_clicks):
    if n_clicks:
        return ""
    return dash.no_update

@callback(
    Output('dummy-output-for-saving', 'children'),
    [Input('prediction-store', 'data')],
    [State('gender-input', 'value'),
     State('age-input', 'value'),
     State('height-input-cm', 'value'),
     State('weight-input-kg', 'value'),
     State('systolic-input', 'value'),
     State('diastolic-input', 'value'),
     State('ethnicity-input', 'value'),
     State('family-kd-input', 'value'),
     State('has-hpt-input', 'value'),
     State('has-diabetes-input', 'value'),
     State('has-heart-disease-input', 'value'),
     State('has-kidney-disease-input', 'value'),
     State('taking-bp-meds-input', 'value'),
     State('taking-diabetes-meds-input', 'value'),
     State('taking-cholesterol-meds-input', 'value'),
     State('taking-other-meds-input', 'value')],
    prevent_initial_call=True
    )

def save_to_csv(prediction_data, gender, age, height, weight, systolic, diastolic,
                ethnicity, family_kd, has_hpt, has_diabetes,
                has_heart_disease, has_kidney_disease, taking_bp_meds,
                taking_diabetes_meds, taking_cholesterol_meds, taking_other_meds):
    if prediction_data is None:
        return ""

    try:
        # Convert checklist values to boolean
        has_hpt_bool = bool(has_hpt)
        has_diabetes_bool = bool(has_diabetes)
        has_heart_disease_bool = bool(has_heart_disease)
        has_kidney_disease_bool = bool(has_kidney_disease)
        taking_bp_meds_bool = bool(taking_bp_meds)
        taking_diabetes_meds_bool = bool(taking_diabetes_meds)
        taking_cholesterol_meds_bool = bool(taking_cholesterol_meds)
        taking_other_meds_bool = bool(taking_other_meds)

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
            'Has_HPT': has_hpt_bool,
            'Has_Diabetes': has_diabetes_bool,
            'Has_Heart_Disease': has_heart_disease_bool,
            'Has_Kidney_Disease': has_kidney_disease_bool,
            'Taking_BP_Meds': taking_bp_meds_bool,
            'Taking_Diabetes_Meds': taking_diabetes_meds_bool,
            'Taking_Cholesterol_Meds': taking_cholesterol_meds_bool,
            'Taking_Other_Meds': taking_other_meds_bool,
            'Prediction': prediction_data
        }
        
        entry_df = pd.DataFrame([entry_data])

        filename = os.path.join(APP_DIR, 'responses.csv')
        # Update the columns list to include the new fields
        cols = [
            'Timestamp', 'Gender', 'Ethnicity', 'S_Ethnicity', 'Age', 'Height',
            'Weight', 'Systolic', 'Diastolic', 'Family_KD', 'Has_HPT', 'Has_Diabetes',
            'Has_Heart_Disease', 'Has_Kidney_Disease', 'Taking_BP_Meds',
            'Taking_Diabetes_Meds', 'Taking_Cholesterol_Meds', 'Taking_Other_Meds',
            'Prediction'
        ]
        
        file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0

        with open(filename, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=cols)

            if not file_exists:
                writer.writeheader()

            writer.writerow(entry_df.iloc[0].to_dict())
        return "Data saved successfully!"
    except Exception as e:
        return f"Error saving data: {str(e)}"
