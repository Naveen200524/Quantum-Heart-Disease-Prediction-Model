"""
Flask Web API for Real-time Heart Disease Prediction
"""

from flask import Flask, request, jsonify
import numpy as np
import torch
import pandas as pd
from datetime import datetime
import json
import os

import config
from _3_train_iqcnn_ensemble import AdvancedIQCNN
from utils import log_message

app = Flask(__name__)

# Global model (loaded at startup)
model = None
feature_scaler = None
selected_features_idx = None

def load_model():
    """Load trained model"""
    global model
    log_message("Loading trained model...")
    model_path = "models/iqcnn_model_0_*.pth"
    # Find latest model file
    import glob
    models = glob.glob(model_path)
    if models:
        latest_model = sorted(models)[-1]
        model = AdvancedIQCNN(n_features=config.NUM_FEATURES)
        model.load_state_dict(torch.load(latest_model))
        model.eval()
        log_message(f"✓ Model loaded: {latest_model}")
    else:
        log_message("⚠ No trained model found!")

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "name": "IQCNN Heart Disease Prediction API",
        "version": "2.0",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "info": "/ (GET)"
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict heart disease from patient features
    Expects JSON with features matching training data
    """
    try:
        data = request.get_json()
        
        # Validate input
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        # Convert to DataFrame
        input_df = pd.DataFrame([data])
        
        # Improved Z-score normalization (from training)
        def improved_zscore(x):
            median = np.median(x)
            std = np.std(x)
            if std == 0:
                return np.zeros_like(x)
            return 2 * (1 / (1 + np.exp(-(x - median) / std))) - 1
        
        input_norm = input_df.apply(improved_zscore, axis=0)
        
        # Convert to tensor
        X_input = torch.tensor(input_norm.values, dtype=torch.float32)
        
        # Make prediction
        with torch.no_grad():
            output = model(X_input)
            probability = torch.softmax(output, dim=1).numpy()
            prediction = np.argmax(probability, axis=1)[0]
        
        response = {
            "prediction": int(prediction),
            "prediction_label": "Heart Disease" if prediction == 1 else "No Heart Disease",
            "confidence": float(probability[0][prediction]),
            "probabilities": {
                "no_disease": float(probability[0][0]),
                "has_disease": float(probability[0][1])
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Batch prediction for multiple patients"""
    try:
        data = request.get_json()
        
        if not isinstance(data, list):
            return jsonify({"error": "Expected list of patient records"}), 400
        
        predictions = []
        
        for patient_data in data:
            input_df = pd.DataFrame([patient_data])
            
            def improved_zscore(x):
                median = np.median(x)
                std = np.std(x)
                if std == 0:
                    return np.zeros_like(x)
                return 2 * (1 / (1 + np.exp(-(x - median) / std))) - 1
            
            input_norm = input_df.apply(improved_zscore, axis=0)
            X_input = torch.tensor(input_norm.values, dtype=torch.float32)
            
            with torch.no_grad():
                output = model(X_input)
                probability = torch.softmax(output, dim=1).numpy()
                prediction = np.argmax(probability, axis=1)[0]
            
            predictions.append({
                "prediction": int(prediction),
                "confidence": float(probability[0][prediction]),
                "probabilities": {
                    "no_disease": float(probability[0][0]),
                    "has_disease": float(probability[0][1])
                }
            })
        
        return jsonify({
            "batch_predictions": predictions,
            "total": len(predictions),
            "timestamp": datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    log_message("Starting IQCNN API...")
    load_model()
    app.run(host=config.API_HOST, port=config.API_PORT, debug=config.DEBUG)
