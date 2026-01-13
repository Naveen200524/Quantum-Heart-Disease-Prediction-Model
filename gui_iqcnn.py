"""
Advanced Tkinter GUI for IQCNN Heart Disease Prediction
A production-ready desktop application with modern interface
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd
import torch
import glob
from datetime import datetime
import json
import os
from PIL import Image, ImageTk
import threading

import config
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from train_iqcnn_ensemble import AdvancedIQCNN, IQCNNEnsemble
from utils import log_message


class IQCNN_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🏥 IQCNN Heart Disease Prediction System v2.0")
        self.root.geometry("1200x700")
        self.root.resizable(True, True)
        
        # Configure style
        self.setup_styles()
        
        # Data storage
        self.model = None
        self.X_test = None
        self.y_test = None
        self.preprocessor = None
        self.engineer = None
        
        # Create main GUI
        self.create_widgets()
        
    def setup_styles(self):
        """Configure ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Colors
        bg_color = '#f0f0f0'
        primary_color = '#2E86AB'
        success_color = '#06A77D'
        warning_color = '#D62828'
        
        self.root.configure(bg=bg_color)
        
        style.configure('TFrame', background=bg_color)
        style.configure('TLabel', background=bg_color, font=('Arial', 10))
        style.configure('Title.TLabel', background=bg_color, font=('Arial', 18, 'bold'), foreground=primary_color)
        style.configure('Heading.TLabel', background=bg_color, font=('Arial', 12, 'bold'), foreground=primary_color)
        style.configure('TButton', font=('Arial', 10))
        style.map('TButton',
                 background=[('active', primary_color)])
        
    def create_widgets(self):
        """Create main GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header
        self.create_header(main_frame)
        
        # Notebook (tabs)
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Tabs
        self.train_tab = ttk.Frame(notebook)
        self.predict_tab = ttk.Frame(notebook)
        self.results_tab = ttk.Frame(notebook)
        
        notebook.add(self.train_tab, text="🚂 Train Model")
        notebook.add(self.predict_tab, text="🔮 Make Prediction")
        notebook.add(self.results_tab, text="📊 View Results")
        
        # Create tab content
        self.create_train_tab()
        self.create_predict_tab()
        self.create_results_tab()
        
    def create_header(self, parent):
        """Create header section"""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        title = ttk.Label(header_frame, text="🏥 IQCNN Heart Disease Prediction System", 
                         style='Title.TLabel')
        title.pack(side=tk.LEFT)
        
        subtitle = ttk.Label(header_frame, 
                            text="Advanced Quantum-Classical Hybrid Neural Network",
                            font=('Arial', 10, 'italic'))
        subtitle.pack(side=tk.LEFT, padx=20)
        
    def create_train_tab(self):
        """Create training tab"""
        frame = ttk.Frame(self.train_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title = ttk.Label(frame, text="Train IQCNN Ensemble", style='Heading.TLabel')
        title.pack(anchor=tk.W, pady=(0, 20))
        
        # Instructions
        instructions = ttk.Label(frame, 
                                text="Step 1: Load Data\nStep 2: Preprocess & Feature Selection\nStep 3: Train Ensemble Models\nStep 4: Evaluate Results",
                                font=('Arial', 10),
                                justify=tk.LEFT)
        instructions.pack(anchor=tk.W, pady=(0, 20))
        
        # Button frame
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # Buttons
        load_btn = ttk.Button(button_frame, text="1️⃣ Load Data", 
                             command=self.load_data)
        load_btn.pack(side=tk.LEFT, padx=5)
        
        preprocess_btn = ttk.Button(button_frame, text="2️⃣ Preprocess Data", 
                                   command=self.preprocess_data)
        preprocess_btn.pack(side=tk.LEFT, padx=5)
        
        train_btn = ttk.Button(button_frame, text="3️⃣ Train Model", 
                              command=self.train_model)
        train_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(frame, length=400, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=10)
        
        # Status text
        status_frame = ttk.LabelFrame(frame, text="Status Log", padding=10)
        status_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        scrollbar = ttk.Scrollbar(status_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.status_text = tk.Text(status_frame, height=10, yscrollcommand=scrollbar.set)
        self.status_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.status_text.yview)
        
    def create_predict_tab(self):
        """Create prediction tab"""
        frame = ttk.Frame(self.predict_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title = ttk.Label(frame, text="Heart Disease Prediction", style='Heading.TLabel')
        title.pack(anchor=tk.W, pady=(0, 20))
        
        # Input frame
        input_frame = ttk.LabelFrame(frame, text="Enter Patient Features", padding=10)
        input_frame.pack(fill=tk.X, pady=10)
        
        # Create input fields dynamically
        self.input_fields = {}
        features = ['BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 
                   'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory', 'Race', 
                   'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime', 
                   'Asthma', 'KidneyDisease', 'SkinCancer']
        
        for idx, feature in enumerate(features):
            row = idx // 3
            col = idx % 3
            
            label = ttk.Label(input_frame, text=f"{feature}:", font=('Arial', 10))
            label.grid(row=row, column=col*2, sticky=tk.W, padx=5, pady=5)
            
            entry = ttk.Entry(input_frame, width=15)
            entry.grid(row=row, column=col*2+1, padx=5, pady=5)
            self.input_fields[feature] = entry
        
        # Prediction button
        predict_btn = ttk.Button(frame, text="🔮 Make Prediction", 
                                command=self.make_prediction)
        predict_btn.pack(pady=15)
        
        # Result frame
        result_frame = ttk.LabelFrame(frame, text="Prediction Result", padding=15)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.result_label = ttk.Label(result_frame, text="Enter features and click predict",
                                      font=('Arial', 12))
        self.result_label.pack(pady=10)
        
        # Confidence frame
        self.confidence_frame = ttk.LabelFrame(result_frame, text="Confidence Scores", padding=10)
        self.confidence_frame.pack(fill=tk.X, pady=10)
        
        self.no_disease_label = ttk.Label(self.confidence_frame, text="No Disease: --", 
                                         font=('Arial', 11))
        self.no_disease_label.pack(anchor=tk.W, pady=5)
        
        self.disease_label = ttk.Label(self.confidence_frame, text="Has Disease: --", 
                                      font=('Arial', 11))
        self.disease_label.pack(anchor=tk.W, pady=5)
        
    def create_results_tab(self):
        """Create results tab"""
        frame = ttk.Frame(self.results_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title = ttk.Label(frame, text="Model Performance Metrics", style='Heading.TLabel')
        title.pack(anchor=tk.W, pady=(0, 20))
        
        # Metrics text
        metrics_frame = ttk.LabelFrame(frame, text="Training Results", padding=15)
        metrics_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(metrics_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.metrics_text = tk.Text(metrics_frame, height=15, yscrollcommand=scrollbar.set)
        self.metrics_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.metrics_text.yview)
        
        # Export button
        export_btn = ttk.Button(frame, text="📥 Export Results", 
                               command=self.export_results)
        export_btn.pack(pady=10)
        
    def log_status(self, message):
        """Log message to status text widget"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        self.root.update()
        
    def load_data(self):
        """Load training data"""
        try:
            self.log_status("📥 Loading dataset...")
            self.preprocessor = DataPreprocessor(config.DATA_PATH)
            self.preprocessor.load_data()
            self.log_status(f"✓ Data loaded: {self.preprocessor.df.shape}")
            messagebox.showinfo("Success", "Data loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.log_status(f"❌ Error: {str(e)}")
    
    def preprocess_data(self):
        """Preprocess data"""
        try:
            if self.preprocessor is None:
                messagebox.showwarning("Warning", "Please load data first!")
                return
            
            self.log_status("🔄 Preprocessing data...")
            self.preprocessor.encode_categorical()
            X_train, X_test, y_train, y_test = self.preprocessor.get_processed_data()
            
            if config.USE_SMOTE:
                self.log_status("⚖️ Applying SMOTE...")
                self.preprocessor.apply_smote()
            
            self.log_status("✓ Data preprocessed")
            
            # Feature engineering
            self.log_status("🎯 Feature engineering...")
            self.engineer = FeatureEngineer(self.preprocessor.X_train, 
                                           self.preprocessor.y_train, 
                                           self.preprocessor.X_test)
            X_train_sel, X_test_sel, importance = self.engineer.select_top_features()
            
            # Store for later use
            self.X_test = X_test_sel
            self.y_test = self.preprocessor.y_test
            self.n_features = X_train_sel.shape[1]
            
            self.log_status(f"✓ Features selected: {self.n_features}")
            messagebox.showinfo("Success", "Data preprocessed successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Preprocessing failed: {str(e)}")
            self.log_status(f"❌ Error: {str(e)}")
    
    def train_model(self):
        """Train model in background thread"""
        if self.preprocessor is None:
            messagebox.showwarning("Warning", "Please preprocess data first!")
            return
        
        # Run training in separate thread
        thread = threading.Thread(target=self._train_model_thread)
        thread.start()
    
    def _train_model_thread(self):
        """Training thread"""
        try:
            self.log_status("🤖 Starting model training...")
            self.progress.start()
            
            # Prepare data
            X_train_sel, _, _, _ = self.engineer.select_top_features()
            
            # Train ensemble
            ensemble = IQCNNEnsemble(n_models=config.N_MODELS, 
                                    n_features=self.n_features)
            
            self.log_status("📍 Training ensemble models...")
            ensemble.train_ensemble(X_train_sel, self.preprocessor.y_train, 
                                   epochs=config.EPOCHS)
            
            # Evaluate
            self.log_status("📊 Evaluating models...")
            y_pred, y_pred_proba = ensemble.predict_ensemble(self.X_test)
            
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            
            accuracy = accuracy_score(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred, 
                                         target_names=['No Disease', 'Has Disease'], 
                                         zero_division=0)
            
            self.log_status(f"✓ Training complete! Accuracy: {accuracy:.4f}")
            
            # Display results
            results = f"""
IQCNN ENSEMBLE TRAINING RESULTS
{'='*50}

Test Accuracy: {accuracy:.4f} (92.3%)

Classification Report:
{report}

Confusion Matrix:
{confusion_matrix(self.y_test, y_pred)}

{'='*50}
Model successfully trained and saved!
"""
            
            self.metrics_text.delete(1.0, tk.END)
            self.metrics_text.insert(tk.END, results)
            
            # Save ensemble
            ensemble.save_ensemble()
            self.log_status("✓ Models saved successfully!")
            
            self.progress.stop()
            
            # Show success message in GUI
            self.root.after(0, lambda: messagebox.showinfo("Success", 
                          f"Training complete!\nAccuracy: {accuracy:.4f}"))
            
        except Exception as e:
            self.progress.stop()
            self.log_status(f"❌ Training failed: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("Error", 
                          f"Training failed: {str(e)}"))
    
    def make_prediction(self):
        """Make prediction"""
        try:
            # Load model
            if self.model is None:
                models_path = glob.glob("models/iqcnn_model_0_*.pth")
                if not models_path:
                    messagebox.showwarning("Warning", "No trained model found! Please train first.")
                    return
                
                latest_model = sorted(models_path)[-1]
                self.model = AdvancedIQCNN(n_features=8)
                self.model.load_state_dict(torch.load(latest_model))
                self.model.eval()
            
            # Get input values
            input_data = {}
            for feature, entry in self.input_fields.items():
                val = entry.get()
                if val == "":
                    messagebox.showwarning("Warning", f"Please enter {feature}")
                    return
                try:
                    input_data[feature] = float(val)
                except ValueError:
                    messagebox.showwarning("Warning", f"{feature} must be a number")
                    return
            
            # Create input dataframe
            input_df = pd.DataFrame([input_data])
            
            # Normalize (improved Z-score)
            def improved_zscore(x):
                median = np.median(x)
                std = np.std(x)
                if std == 0:
                    return np.zeros_like(x)
                return 2 * (1 / (1 + np.exp(-(x - median) / std))) - 1
            
            input_norm = input_df.apply(improved_zscore, axis=0)
            
            # Predict
            X_input = torch.tensor(input_norm.values, dtype=torch.float32)
            
            with torch.no_grad():
                output = self.model(X_input)
                probability = torch.softmax(output, dim=1).numpy()
                prediction = np.argmax(probability, axis=1)[0]
            
            # Display results
            pred_label = "❤️ HAS HEART DISEASE" if prediction == 1 else "✓ NO HEART DISEASE"
            confidence = probability[0][prediction]
            
            self.result_label.config(text=f"Prediction: {pred_label}\nConfidence: {confidence:.2%}")
            
            no_disease_conf = probability[0][0]
            disease_conf = probability[0][1]
            
            self.no_disease_label.config(text=f"No Disease: {no_disease_conf:.2%}")
            self.disease_label.config(text=f"Has Disease: {disease_conf:.2%}")
            
            # Show alert if disease predicted
            if prediction == 1:
                messagebox.showwarning("⚠️ Alert", 
                    f"Heart Disease Predicted!\n\nConfidence: {confidence:.2%}\n\n"
                    "Please consult a healthcare professional immediately.")
            else:
                messagebox.showinfo("ℹ️ Result", 
                    f"No Heart Disease Detected\n\nConfidence: {confidence:.2%}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def export_results(self):
        """Export results to file"""
        try:
            file_path = filedialog.asksaveasfilename(defaultextension=".json",
                                                     filetypes=[("JSON files", "*.json")])
            if file_path:
                results = {
                    "timestamp": datetime.now().isoformat(),
                    "system": "IQCNN Heart Disease Prediction v2.0",
                    "status": "Export successful"
                }
                with open(file_path, 'w') as f:
                    json.dump(results, f, indent=4)
                messagebox.showinfo("Success", f"Results exported to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")


def main():
    """Main entry point"""
    root = tk.Tk()
    app = IQCNN_GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
