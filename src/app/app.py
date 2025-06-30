from flask import Flask, render_template, request, jsonify, flash
import pandas as pd
from sklearn.pipeline import Pipeline
import joblib
from src.config import MODELS_DIR
import traceback

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a random secret key

# Global variables to store models
preprocessor = None
model = None

def load_models():
    """Load the trained models"""
    global preprocessor, model
    try:
        preprocessor = joblib.load(MODELS_DIR / 'preprocessor.pkl')
        model = joblib.load(MODELS_DIR / 'train.pkl')
        print("Models loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return False

@app.route('/')
def index():
    """Main page with the prediction form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        data = {
            'Age': float(request.form['age']),
            'Height': float(request.form['height']),
            'Weight': float(request.form['weight']),
            'Systolic': float(request.form['systolic']),
            'Diastolic': float(request.form['diastolic']),
            'S_Ethnicity': request.form['ethnicity'],
            'Family_KD': request.form['family_kd'],
            'Gender': request.form['gender'],
            'Has_KD': request.form.get('has_kd') == 'on',
            'Has_Diabetes': request.form.get('has_diabetes') == 'on',
        }
        
        # Validate inputs
        validation_error = validate_inputs(data)
        if validation_error:
            flash(validation_error, 'error')
            return render_template('index.html', **data)
        
        # Create DataFrame
        df = pd.DataFrame([data])
        
        # Make prediction
        if preprocessor is None or model is None:
            flash('Models not loaded properly. Please restart the application.', 'error')
            return render_template('index.html', **data)
        
        # Create and run pipeline
        full_pipeline = Pipeline(steps=[
            ('preprocessing', preprocessor),
            ('model', model)
        ])
        
        prediction = full_pipeline.predict(df)
        
        # Prepare results
        result = {
            'prediction': prediction[0],
            'prediction_type': str(type(prediction[0]).__name__),
            'input_data': data
        }
        
        # Add interpretation for binary predictions
        if isinstance(prediction[0], (bool, int)) and prediction[0] in [0, 1, True, False]:
            result['interpretation'] = "Positive" if prediction[0] else "Negative"
        
        return render_template('result.html', result=result)
        
    except ValueError as e:
        flash(f'Invalid input: {str(e)}', 'error')
        return render_template('index.html')
    except Exception as e:
        flash(f'Prediction error: {str(e)}', 'error')
        print(f"Error details: {traceback.format_exc()}")
        return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions (JSON)"""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['Age', 'Height', 'Weight', 'Systolic', 'Diastolic', 
                          'S_Ethnicity', 'Family_KD', 'Gender', 'Has_KD', 'Has_Diabetes']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Create DataFrame
        df = pd.DataFrame([data])
        
        # Make prediction
        if preprocessor is None or model is None:
            return jsonify({'error': 'Models not loaded properly'}), 500
        
        full_pipeline = Pipeline(steps=[
            ('preprocessing', preprocessor),
            ('model', model)
        ])
        
        prediction = full_pipeline.predict(df)
        
        result = {
            'prediction': prediction[0].item() if hasattr(prediction[0], 'item') else prediction[0],
            'prediction_type': str(type(prediction[0]).__name__),
            'success': True
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

def validate_inputs(data):
    """Validate input data"""
    try:
        if data['Age'] <= 0 or data['Age'] > 150:
            return "Age must be between 1 and 150"
        if data['Height'] <= 0 or data['Height'] > 300:
            return "Height must be between 1 and 300 cm"
        if data['Weight'] <= 0 or data['Weight'] > 500:
            return "Weight must be between 1 and 500 kg"
        if data['Systolic'] <= 0 or data['Systolic'] > 300:
            return "Systolic BP must be between 1 and 300"
        if data['Diastolic'] <= 0 or data['Diastolic'] > 200:
            return "Diastolic BP must be between 1 and 200"
        
        return None
    except (ValueError, KeyError) as e:
        return f"Invalid input: {str(e)}"

if __name__ == '__main__':
    # Load models on startup
    if load_models():
        print("Starting Flask application...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load models. Please check your model files.")