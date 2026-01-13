"""
Advanced IQCNN Model Training with Ensemble
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import os
from datetime import datetime

import config
from utils import log_message, create_report, plot_metrics
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer

# Initialize quantum simulator
backend = AerSimulator()

class AdvancedQuantumLayer(nn.Module):
    """Multi-layer Quantum Circuit"""
    def __init__(self, n_qubits, depth=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.thetas = nn.ParameterList([
            nn.Parameter(0.01 * torch.randn(n_qubits)) 
            for _ in range(depth)
        ])
        
    def forward(self, x):
        outputs = []
        for xi in x:
            out = self.simulate_quantum_circuit(
                xi.detach().cpu().numpy(),
                [theta.detach().cpu().numpy() for theta in self.thetas]
            )
            outputs.append(out)
        return torch.tensor(outputs, dtype=torch.float32)
    
    def simulate_quantum_circuit(self, x, thetas):
        """Simulate quantum circuit with multiple layers"""
        n_qubits = len(x)
        qc = QuantumCircuit(n_qubits, n_qubits)
        
        # Encoding layer
        for i, val in enumerate(x):
            qc.ry(float(val), i)
        
        # Multiple variational layers
        for theta in thetas:
            # Entanglement
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)
            # Variational rotations
            for i in range(n_qubits):
                qc.ry(float(theta[i]), i)
        
        qc.measure(range(n_qubits), range(n_qubits))
        
        # Execute
        job = backend.run(qc, shots=config.QUANTUM_SHOTS)
        result = job.result()
        counts = result.get_counts()
        
        # Extract expectation
        output = sum([int(key[-1]) * val for key, val in counts.items()]) / config.QUANTUM_SHOTS
        return [output]


class AdvancedIQCNN(nn.Module):
    """Advanced IQCNN with Multiple Quantum Layers"""
    def __init__(self, n_features, n_classes=2):
        super().__init__()
        
        # Classical preprocessing
        self.fc1 = nn.Linear(n_features, n_features * 2)
        self.bn1 = nn.BatchNorm1d(n_features * 2)
        
        self.fc2 = nn.Linear(n_features * 2, n_features)
        self.bn2 = nn.BatchNorm1d(n_features)
        
        # Quantum layers (2 quantum circuits)
        self.quantum1 = AdvancedQuantumLayer(n_features, depth=2)
        self.quantum2 = AdvancedQuantumLayer(n_features, depth=1)
        
        # Classification layers
        self.fc3 = nn.Linear(1, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, n_classes)
        
        self.dropout = nn.Dropout(0.4)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Classical preprocessing
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Quantum layers
        xq1 = self.quantum1(x)
        xq2 = self.quantum2(x)
        
        # Combine quantum outputs
        xq = (xq1 + xq2) / 2
        
        # Classification
        out = self.relu(self.fc3(xq))
        out = self.dropout(out)
        out = self.relu(self.fc4(out))
        out = self.fc5(out)
        
        return out


class IQCNNEnsemble:
    """Ensemble of Multiple IQCNN Models"""
    def __init__(self, n_models=config.N_MODELS, n_features=config.NUM_FEATURES):
        self.n_models = n_models
        self.models = [AdvancedIQCNN(n_features=n_features) for _ in range(n_models)]
        self.optimizers = [optim.Adam(m.parameters(), lr=config.LEARNING_RATE) 
                          for m in self.models]
        self.criterion = nn.CrossEntropyLoss()
        
    def train_ensemble(self, X_train, y_train, epochs=config.EPOCHS):
        """Train all models"""
        log_message(f"\n🤖 Training ensemble of {self.n_models} models...")
        
        X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_tensor = torch.tensor(y_train.values, dtype=torch.long)
        
        # Use subset for faster training
        subset_size = min(1000, X_tensor.shape[0])
        indices = torch.randperm(X_tensor.shape[0])[:subset_size]
        X_subset = X_tensor[indices]
        y_subset = y_tensor[indices]
        
        log_message(f"Training on {subset_size} samples (subset for speed)")
        
        for model_idx, model in enumerate(self.models):
            log_message(f"\n📍 Model {model_idx + 1}/{self.n_models}")
            model.train()
            
            for epoch in range(epochs):
                running_loss = 0.0
                correct = 0
                total = 0
                
                num_batches = len(X_subset) // config.BATCH_SIZE
                
                for batch_idx in tqdm(range(num_batches), 
                                     desc=f"Model {model_idx + 1} Epoch {epoch + 1}/{epochs}"):
                    start = batch_idx * config.BATCH_SIZE
                    end = start + config.BATCH_SIZE
                    
                    xi = X_subset[start:end]
                    yi = y_subset[start:end]
                    
                    self.optimizers[model_idx].zero_grad()
                    output = model(xi)
                    loss = self.criterion(output, yi)
                    loss.backward()
                    self.optimizers[model_idx].step()
                    
                    running_loss += loss.item()
                    _, pred = torch.max(output.data, 1)
                    total += yi.size(0)
                    correct += (pred == yi).sum().item()
                
                avg_loss = running_loss / num_batches
                acc = 100 * correct / total
                log_message(f"  Epoch {epoch + 1}: Loss={avg_loss:.4f}, Acc={acc:.2f}%")
    
    def predict_ensemble(self, X_test):
        """Make ensemble predictions"""
        X_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        all_probs = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                probs = []
                batch_size = 10
                
                for i in range(0, len(X_tensor), batch_size):
                    batch = X_tensor[i:i+batch_size]
                    output = model(batch)
                    prob = torch.softmax(output, dim=1)
                    probs.extend(prob.numpy())
                
                all_probs.append(np.array(probs))
        
        # Average predictions
        ensemble_prob = np.mean(all_probs, axis=0)
        ensemble_pred = np.argmax(ensemble_prob, axis=1)
        
        return ensemble_pred, ensemble_prob
    
    def save_ensemble(self, path="models"):
        """Save trained models"""
        os.makedirs(path, exist_ok=True)
        for idx, model in enumerate(self.models):
            model_path = os.path.join(path, 
                f"iqcnn_model_{idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
            torch.save(model.state_dict(), model_path)
            log_message(f"✓ Model {idx} saved: {model_path}")


# Run this step
if __name__ == "__main__":
    log_message("=" * 70)
    log_message("STEP 3: TRAINING ADVANCED IQCNN ENSEMBLE")
    log_message("=" * 70)
    
    try:
        # Load data
        log_message("\n📥 Loading and preprocessing data...")
        preprocessor = DataPreprocessor(config.DATA_PATH)
        X_train, X_test, y_train, y_test = preprocessor.get_processed_data()
        
        # Feature engineering
        log_message("\n🎯 Feature engineering...")
        engineer = FeatureEngineer(X_train, y_train, X_test)
        X_train_sel, X_test_sel, _ = engineer.select_top_features()
        
        # Train ensemble
        log_message("\n" + "=" * 70)
        ensemble = IQCNNEnsemble(n_models=config.N_MODELS, 
                                n_features=X_train_sel.shape[1])
        ensemble.train_ensemble(X_train_sel, y_train, epochs=config.EPOCHS)
        
        # Evaluate
        log_message("\n📊 Evaluating ensemble...")
        y_pred, y_pred_proba = ensemble.predict_ensemble(X_test_sel)
        
        accuracy = accuracy_score(y_test, y_pred)
        log_message(f"\n🎯 Test Accuracy: {accuracy:.4f}")
        
        log_message("\n📈 Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['No Disease', 'Has Disease'], 
                                   zero_division=0))
        
        # Save results
        report = create_report(y_test, y_pred, y_pred_proba, "IQCNN_Ensemble")
        plot_metrics(y_test, y_pred_proba, "IQCNN_Ensemble")
        
        # Save models
        ensemble.save_ensemble()
        
        log_message("\n" + "=" * 70)
        log_message("✓ TRAINING COMPLETE!")
        log_message("=" * 70)
        
    except Exception as e:
        log_message(f"\n❌ ERROR: {str(e)}", level="ERROR")
        import traceback
        traceback.print_exc()
