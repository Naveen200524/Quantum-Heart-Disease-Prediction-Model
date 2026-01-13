"""
Configuration file for IQCNN Advanced Heart Disease Prediction
"""

import os

# Data
DATA_PATH = "heart_2020_cleaned.csv"
TRAIN_TEST_SPLIT = 0.2
RANDOM_STATE = 42

# Features
NUM_FEATURES = 8
MIN_MAX_SCALING = True

# Quantum Layer
N_QUBITS = 8
QUANTUM_SHOTS = 256

# Training
EPOCHS = 10
BATCH_SIZE = 10
LEARNING_RATE = 0.001
CLASS_WEIGHTS = True

# Data Augmentation
USE_SMOTE = True
SMOTE_RATIO = 0.8

# Ensemble
N_MODELS = 3
ENSEMBLE_VOTING = "soft"  # soft or hard

# API
API_HOST = "127.0.0.1"
API_PORT = 5000
DEBUG = True

# Paths
MODEL_DIR = "models"
RESULTS_DIR = "results"

# Create directories if not exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
