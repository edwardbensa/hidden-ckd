import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sklearn.pipeline import Pipeline
import joblib
import csv
from datetime import datetime
from src.config import MODELS_DIR, APP_DIR

class CKDPredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CKD Risk Prediction")
        self.root.geometry("500x700")
        self.root.resizable(True, True)
        
        # Initialize models
        self.preprocessor = None
        self.model = None
        self.load_models()
        
        self.create_widgets()
        
    def load_models(self):
        """Load the trained models"""
        try:
            self.preprocessor = joblib.load(MODELS_DIR / 'preprocessor.pkl')
            self.model = joblib.load(MODELS_DIR / 'train.pkl')
        except FileNotFoundError:
            messagebox.showerror("Error", "Model files not found. Please ensure preprocessor.pkl and train.pkl are in the correct directory.")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading models: {str(e)}")
    
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="CKD Risk Prediction System", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        row = 1
        
        # Gender
        ttk.Label(main_frame, text="Gender:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.gender_var = tk.StringVar()
        gender_combo = ttk.Combobox(main_frame, textvariable=self.gender_var,
                                   values=["Male", "Female"],
                                   state="readonly", width=12)
        gender_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        row += 1

        # Age
        ttk.Label(main_frame, text="Age:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.age_var = tk.StringVar()
        age_entry = ttk.Entry(main_frame, textvariable=self.age_var, width=15)
        age_entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        row += 1
        
        # Height
        ttk.Label(main_frame, text="Height (cm):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.height_var = tk.StringVar()
        height_entry = ttk.Entry(main_frame, textvariable=self.height_var, width=15)
        height_entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        row += 1
        
        # Weight
        ttk.Label(main_frame, text="Weight (kg):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.weight_var = tk.StringVar()
        weight_entry = ttk.Entry(main_frame, textvariable=self.weight_var, width=15)
        weight_entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        row += 1
        
        # Systolic BP
        ttk.Label(main_frame, text="Systolic BP:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.systolic_var = tk.StringVar()
        systolic_entry = ttk.Entry(main_frame, textvariable=self.systolic_var, width=15)
        systolic_entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        row += 1
        
        # Diastolic BP
        ttk.Label(main_frame, text="Diastolic BP:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.diastolic_var = tk.StringVar()
        diastolic_entry = ttk.Entry(main_frame, textvariable=self.diastolic_var, width=15)
        diastolic_entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        row += 1
        
        # Ethnicity
        self.ethnicity_map = {
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

        ttk.Label(main_frame, text="Ethnicity:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.ethnicity_var = tk.StringVar()
        ethnicity_combo = ttk.Combobox(main_frame, textvariable=self.ethnicity_var, 
                                      values=list(self.ethnicity_map.keys()),
                                      state="readonly", width=12)
        ethnicity_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        row += 1
        
        # Family KD
        ttk.Label(main_frame, text="Do you have a family \nhistory of kidney disease?:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.family_kd_var = tk.StringVar()
        family_kd_combo = ttk.Combobox(main_frame, textvariable=self.family_kd_var,
                                      values=["Definitely yes", "Definitely not", "Not sure"],
                                      state="readonly", width=12)
        family_kd_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        row += 1
        
        # Has KD
        ttk.Label(main_frame, text="Have you been diagnosed \nwith kidney disease?:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.has_kd_var = tk.BooleanVar(value=False)
        has_kd_check = ttk.Checkbutton(main_frame, variable=self.has_kd_var)
        has_kd_check.grid(row=row, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        row += 1
        
        # Has Diabetes
        ttk.Label(main_frame, text="Have you been diagnosed \nwith diabetes?:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.has_diabetes_var = tk.BooleanVar(value=False)
        has_diabetes_check = ttk.Checkbutton(main_frame, variable=self.has_diabetes_var)
        has_diabetes_check.grid(row=row, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        row += 1
        
        # Predict Button
        predict_button = ttk.Button(main_frame, text="Get Results", 
                                   command=self.make_prediction, style='Accent.TButton')
        predict_button.grid(row=row, column=0, columnspan=2, pady=20)
        row += 1
        
        # Results Frame
        results_frame = ttk.LabelFrame(main_frame, text="Risk Prediction and Recommendation", padding="10")
        results_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        results_frame.columnconfigure(0, weight=1)
        
        # Results Text
        self.results_text = tk.Text(results_frame, height=6, wrap=tk.WORD, 
                                   font=('Courier', 10))
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Clear Results Button
        clear_button = ttk.Button(results_frame, text="Clear Results", 
                                 command=self.clear_results)
        clear_button.grid(row=1, column=0, pady=(10, 0))
    
    def validate_inputs(self):
        """Validate user inputs"""
        try:
            age = float(self.age_var.get())
            height = float(self.height_var.get())
            weight = float(self.weight_var.get())
            systolic = float(self.systolic_var.get())
            diastolic = float(self.diastolic_var.get())
            
            if age <= 0 or age > 150:
                raise ValueError("Age must be between 1 and 150")
            if height <= 0 or height > 300:
                raise ValueError("Height must be between 1 and 300 cm")
            if weight <= 0 or weight > 500:
                raise ValueError("Weight must be between 1 and 500 kg")
            if systolic <= 0 or systolic > 300:
                raise ValueError("Systolic BP must be between 1 and 300")
            if diastolic <= 0 or diastolic > 200:
                raise ValueError("Diastolic BP must be between 1 and 200")
            
            return True
            
        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))
            return False
    
    def make_prediction(self):
        """Make prediction using the loaded models"""
        if not self.validate_inputs():
            return
        
        if self.preprocessor is None or self.model is None:
            messagebox.showerror("Error", "Models not loaded properly")
            return
        
        try:
            # Create DataFrame with input data
            data = pd.DataFrame([{
                'Age': float(self.age_var.get()),
                'Height': float(self.height_var.get()),
                'Weight': float(self.weight_var.get()),
                'Systolic': float(self.systolic_var.get()),
                'Diastolic': float(self.diastolic_var.get()),
                'S_Ethnicity': self.ethnicity_map.get(self.ethnicity_var.get()),
                'Family_KD': self.family_kd_var.get(),
                'Gender': self.gender_var.get(),
                'Has_KD': self.has_kd_var.get(),
                'Has_Diabetes': self.has_diabetes_var.get(),
            }])
            
            # Create pipeline and make prediction
            full_pipeline = Pipeline(steps=[
                ('preprocessing', self.preprocessor),
                ('model', self.model)
            ])
            
            prediction = full_pipeline.predict(data)
            
            # Display results
            self.display_results(data, prediction)

            # Store prediction
            self.prediction = prediction

            # Save results to CSV
            self.save_to_csv(data)
            
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Error making prediction: {str(e)}")
    
    def display_results(self, data, prediction):
        """Display prediction results"""
        self.results_text.delete(1.0, tk.END)

        # Convert prediction to a readable message
        if prediction == 0:
            message = "CKD Risk: Low Risk \nTake it easy"
        elif prediction == 1:
            message = "CKD Risk: Moderate Risk \nWe recommend a few lifestyle changes to mitigate your risk"
        elif prediction == 2:
            message = "CKD Risk: High \nWe recommend a GP visit immediately"
        else:
            message = "Unknown prediction result"
        
        result_text = message
        
        self.results_text.insert(1.0, result_text)

    def save_to_csv(self, data):
        entry = data.copy()
        entry['Ethnicity'] = self.ethnicity_var.get()
        entry['Prediction'] = self.prediction
        entry['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cols = [
            'Timestamp',
            'Gender',
            'Ethnicity',
            'S_Ethnicity',
            'Age',
            'Height',
            'Weight',
            'Systolic',
            'Diastolic',
            'Family_KD',
            'Has_KD',
            'Has_Diabetes',
            'Prediction']
        entry = entry[cols]

        filename = APP_DIR / 'responses_tk.csv'
        fieldnames = cols

        # Check if the file exists to write headers if needed
        try:
            with open(filename, 'r'):
                file_exists = True
        except FileNotFoundError:
            file_exists = False

        with open(filename, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()  # write header only once

            writer.writerow(entry.iloc[0].to_dict())
    
    def clear_results(self):
        """Clear the results text area"""
        self.results_text.delete(1.0, tk.END)

def main():
    root = tk.Tk()
    app = CKDPredictionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()