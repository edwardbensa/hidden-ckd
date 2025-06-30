import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sklearn.pipeline import Pipeline
import joblib
from src.config import MODELS_DIR

class MedicalPredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical Prediction System")
        self.root.geometry("500x650")
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
        title_label = ttk.Label(main_frame, text="Medical Prediction System", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        row = 1
        
        # Age
        ttk.Label(main_frame, text="Age:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.age_var = tk.StringVar(value="46")
        age_entry = ttk.Entry(main_frame, textvariable=self.age_var, width=15)
        age_entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        row += 1
        
        # Height
        ttk.Label(main_frame, text="Height (cm):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.height_var = tk.StringVar(value="178")
        height_entry = ttk.Entry(main_frame, textvariable=self.height_var, width=15)
        height_entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        row += 1
        
        # Weight
        ttk.Label(main_frame, text="Weight (kg):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.weight_var = tk.StringVar(value="80")
        weight_entry = ttk.Entry(main_frame, textvariable=self.weight_var, width=15)
        weight_entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        row += 1
        
        # Systolic BP
        ttk.Label(main_frame, text="Systolic BP:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.systolic_var = tk.StringVar(value="180")
        systolic_entry = ttk.Entry(main_frame, textvariable=self.systolic_var, width=15)
        systolic_entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        row += 1
        
        # Diastolic BP
        ttk.Label(main_frame, text="Diastolic BP:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.diastolic_var = tk.StringVar(value="120")
        diastolic_entry = ttk.Entry(main_frame, textvariable=self.diastolic_var, width=15)
        diastolic_entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        row += 1
        
        # Ethnicity
        ttk.Label(main_frame, text="Ethnicity:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.ethnicity_var = tk.StringVar(value="Black")
        ethnicity_combo = ttk.Combobox(main_frame, textvariable=self.ethnicity_var, 
                                      values=["Black", "White", "Asian", "Hispanic", "Other"],
                                      state="readonly", width=12)
        ethnicity_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        row += 1
        
        # Family KD
        ttk.Label(main_frame, text="Family Kidney Disease:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.family_kd_var = tk.StringVar(value="Not sure")
        family_kd_combo = ttk.Combobox(main_frame, textvariable=self.family_kd_var,
                                      values=["Yes", "No", "Not sure"],
                                      state="readonly", width=12)
        family_kd_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        row += 1
        
        # Gender
        ttk.Label(main_frame, text="Gender:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.gender_var = tk.StringVar(value="Female")
        gender_combo = ttk.Combobox(main_frame, textvariable=self.gender_var,
                                   values=["Male", "Female"],
                                   state="readonly", width=12)
        gender_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        row += 1
        
        # Has KD
        ttk.Label(main_frame, text="Has Kidney Disease:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.has_kd_var = tk.BooleanVar(value=False)
        has_kd_check = ttk.Checkbutton(main_frame, variable=self.has_kd_var)
        has_kd_check.grid(row=row, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        row += 1
        
        # Has Diabetes
        ttk.Label(main_frame, text="Has Diabetes:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.has_diabetes_var = tk.BooleanVar(value=True)
        has_diabetes_check = ttk.Checkbutton(main_frame, variable=self.has_diabetes_var)
        has_diabetes_check.grid(row=row, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        row += 1
        
        # Predict Button
        predict_button = ttk.Button(main_frame, text="Make Prediction", 
                                   command=self.make_prediction, style='Accent.TButton')
        predict_button.grid(row=row, column=0, columnspan=2, pady=20)
        row += 1
        
        # Results Frame
        results_frame = ttk.LabelFrame(main_frame, text="Prediction Results", padding="10")
        results_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        results_frame.columnconfigure(0, weight=1)
        
        # Results Text
        self.results_text = tk.Text(results_frame, height=6, wrap=tk.WORD, 
                                   font=('Courier', 10))
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbar for results
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, 
                                 command=self.results_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
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
                'S_Ethnicity': self.ethnicity_var.get(),
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
            
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Error making prediction: {str(e)}")
    
    def display_results(self, data, prediction):
        """Display prediction results"""
        self.results_text.delete(1.0, tk.END)
        
        result_text = "PREDICTION RESULTS\n"
        result_text += "=" * 40 + "\n\n"
        
        result_text += "Input Data:\n"
        result_text += "-" * 20 + "\n"
        for column, value in data.iloc[0].items():
            result_text += f"{column}: {value}\n"
        
        result_text += f"\nPrediction: {prediction[0]}\n"
        result_text += f"Prediction Type: {type(prediction[0])}\n"
        
        # Add interpretation if prediction is binary
        if isinstance(prediction[0], (bool, int)) and prediction[0] in [0, 1, True, False]:
            interpretation = "Positive" if prediction[0] else "Negative"
            result_text += f"Interpretation: {interpretation}\n"
        
        self.results_text.insert(1.0, result_text)
    
    def clear_results(self):
        """Clear the results text area"""
        self.results_text.delete(1.0, tk.END)

def main():
    root = tk.Tk()
    app = MedicalPredictionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()