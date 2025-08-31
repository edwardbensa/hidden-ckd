import csv
from datetime import datetime
from dash import dcc, html, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc
import pandas as pd
from src.config import APP_DIR
from src.utils.model_utils import load_model

# Load models
try:
    basic_preprocessor_bp = load_model('basic_preprocessor_bp.pkl')
    basic_preprocessor_nobp = load_model('basic_preprocessor_nobp.pkl')
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
            ),
        ], md=6),
        dbc.Col([
            dbc.Label("Age:"),
            dbc.Input(id='age-input', type='number', placeholder='Enter Age'),
        ], md=6),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Weight:"),
            dbc.Input(id='weight-input-visible', type='number', placeholder='Enter Weight'),
            dbc.RadioItems(
                options=[
                    {'label': 'kg', 'value': 'kg'},
                    {'label': 'stone', 'value': 'st'},
                    {'label': 'lbs', 'value': 'lbs'}
                ],
                value='kg',
                id='weight-unit-radio',
                inline=True,
            ),
        ], md=6),
        dbc.Col([
            dbc.Label("Height:"),
            dbc.Input(id='height-input-visible', type='number', placeholder='Enter Height'),
            dbc.RadioItems(
                options=[
                    {'label': 'cm', 'value': 'cm'},
                    {'label': 'ft', 'value': 'ft'}
                ],
                value='cm',
                id='height-unit-radio',
                inline=True,
            ),
        ], md=6),
        dbc.Input(id='weight-input-kg', type='hidden'),
        dbc.Input(id='height-input-cm', type='hidden'),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Ethnicity:"),
            dcc.Dropdown(
                id='ethnicity-input',
                options=[{'label': k, 'value': k} for k in ethnicity_map.keys()],
                placeholder="Select Ethnicity",
            ),
        ], md=6),
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
            ),
        ], md=6),
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
        ], md=6),
        dbc.Col([
            dbc.Label("Diastolic BP:"),
            dbc.Input(id='diastolic-input', type='number', placeholder='Enter Diastolic BP'),
        ], md=6),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Have you been diagnosed with any of the following?"),
        ], width=12, className="mb-2"),
    ]),

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
        ], md=6),
        dbc.Col([
            dbc.Label("Hypertension:  "),
            html.Span(id="hpt-label-text"),
            dbc.Checklist(
                options=[{"label": "", "value": True}],
                value=[],
                id="has-hpt-input",
                switch=True,
            ),
        ], md=6),
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
                        ], md=6),
                        dbc.Col([
                            dbc.Label("Have you been diagnosed with kidney disease?"),
                            dbc.Checklist(
                                options=[{"label": "Yes", "value": True}],
                                value=[],
                                id="has-kidney-disease-input",
                                switch=True,
                            ),
                        ], md=6),
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
                        ], md=4),
                        dbc.Col([
                            dbc.Label("Are you currently taking medication for diabetes?"),
                            dbc.Checklist(
                                options=[{"label": "Yes", "value": True}],
                                value=[],
                                id="taking-diabetes-meds-input",
                                switch=True,
                            ),
                        ], md=4),
                        dbc.Col([
                            dbc.Label("Are you currently taking medication for high cholesterol?"),
                            dbc.Checklist(
                                options=[{"label": "Yes", "value": True}],
                                value=[],
                                id="taking-cholesterol-meds-input",
                                switch=True,
                            ),
                        ], md=4),
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

    html.Div(id='dummy-output-for-saving', style={'display': 'none'}),
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
                    ethnicity, family_kd, has_hpt, has_diabetes, bp_switch_value):
    if n_clicks is None:
        return "", None

    # --- Combined Validation Logic ---
    validation_error = None
    if bp_switch_value:
        # Validate all fields if BP readings are provided
        if not all([age, height, weight, systolic, diastolic]):
            validation_error = "All numerical fields must be filled."
        elif not (1 <= age <= 100):
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
        elif not (1 <= age <= 100):
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
            'Gender': gender, 'Has_Hpt': bool(has_hpt), 'Has_Diabetes': bool(has_diabetes),
        }])
        X_transformed = basic_preprocessor_bp.transform(data)
    else:
        selected_model = classifier_nobp
        data = pd.DataFrame([{
            'Age': float(age), 'Height': float(height), 'Weight': float(weight),
            'S_Ethnicity': ethnicity_map.get(ethnicity), 'Family_KD': family_kd,
            'Gender': gender, 'Has_Hpt': bool(has_hpt), 'Has_Diabetes': bool(has_diabetes),
        }])
        X_transformed = basic_preprocessor_nobp.transform(data)

    # --- Prediction and Result Formatting ---
    if selected_model is None:
        return "Error: Models not loaded properly.", None

    try:
        prediction = selected_model.predict(X_transformed)[0]
        probabilities = selected_model.predict_proba(X_transformed)[0]

        # Define messages and recommendations based on the prediction
        risk_level = ""
        recommendation = ""
        if prediction == 0:
            risk_level = "Low"
            recommendation = "Maintain your current healthy lifestyle and continue to monitor your health."
        elif prediction == 1:
            risk_level = "Moderate"
            recommendation = "We recommend a few lifestyle changes to mitigate your risk, such as regular exercise and a balanced diet. Consider consulting a healthcare professional for a more personalized plan."
        elif prediction == 2:
            risk_level = "High"
            recommendation = "We recommend a GP visit immediately for a full check-up and further medical advice."
        else:
            return "Unknown prediction result", None

        # Format the probabilities into a readable string
        prob_report = "\n".join([
            f"  - Low Risk: {probabilities[0]:.2%}",
            f"  - Moderate Risk: {probabilities[1]:.2%}",
            f"  - High Risk: {probabilities[2]:.2%}"
        ])

        # Combine all parts into a single output string
        final_message = (
            f"Predicted CKD Risk: {risk_level}\n\n"
            f"Risk Probabilities:\n"
            f"{prob_report}\n\n"
            f"Recommendation:\n"
            f"  - {recommendation}"
        )
        
        return final_message, prediction
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
    return no_update

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
    """Saves prediction data and user inputs to a CSV file."""
    if prediction_data is None:
        return ""

    try:
        # Prepare the data dictionary
        entry_data = {
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Gender': gender,
            'Ethnicity': ethnicity,
            'S_Ethnicity': ethnicity_map.get(ethnicity, 'Other'),
            'Age': age,
            'Height': height,
            'Weight': weight,
            'Systolic': systolic if systolic is not None else '',
            'Diastolic': diastolic if diastolic is not None else '',
            'Family_KD': family_kd,
            'Has_Hpt': bool(has_hpt),
            'Has_Diabetes': bool(has_diabetes),
            'Has_Heart_Disease': bool(has_heart_disease),
            'Has_Kidney_Disease': bool(has_kidney_disease),
            'Taking_BP_Meds': bool(taking_bp_meds),
            'Taking_Diabetes_Meds': bool(taking_diabetes_meds),
            'Taking_Cholesterol_Meds': bool(taking_cholesterol_meds),
            'Taking_Other_Meds': bool(taking_other_meds),
            'Prediction': prediction_data
        }

        # Use pathlib to create the file path
        filename = APP_DIR / 'responses.csv'
        cols = list(entry_data.keys())

        file_exists = filename.exists()

        with open(filename, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=cols)

            if not file_exists or filename.stat().st_size == 0:
                writer.writeheader()

            writer.writerow(entry_data)
        return "Data saved successfully!"

    except Exception as e:
        print(f"Error saving data to CSV: {e}")
        return "Error saving data. Please check the server logs."
