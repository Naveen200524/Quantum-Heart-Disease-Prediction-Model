"""
Explainability using SHAP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import torch
from sklearn.preprocessing import StandardScaler
import config
from utils import log_message

class ExplainabilityAnalyzer:
    def __init__(self, model, X_train, X_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.explainer = None
        self.shap_values = None
        
    def prepare_for_shap(self):
        """Prepare data for SHAP"""
        log_message("Preparing data for SHAP analysis...")
        
        # Use a subset for faster computation
        subset_size = min(100, len(self.X_train))
        self.X_background = self.X_train.sample(n=subset_size, random_state=42)
        
        # Select test samples
        self.X_explain = self.X_test.sample(n=min(50, len(self.X_test)), random_state=42)
        
        log_message(f"Background set: {self.X_background.shape}, Explain set: {self.X_explain.shape}")
    
    def create_model_function(self):
        """Create prediction function for SHAP"""
        def model_predict(X):
            X_tensor = torch.tensor(X, dtype=torch.float32)
            self.model.eval()
            with torch.no_grad():
                output = self.model(X_tensor)
                proba = torch.softmax(output, dim=1)
            return proba.numpy()
        
        return model_predict
    
    def compute_shap_values(self):
        """Compute SHAP values"""
        log_message("Computing SHAP values...")
        
        model_func = self.create_model_function()
        
        # Create SHAP explainer
        self.explainer = shap.KernelExplainer(
            model_func,
            self.X_background.values
        )
        
        # Compute SHAP values
        self.shap_values = self.explainer.shap_values(self.X_explain.values)
        
        log_message(f"✓ SHAP values computed: shape {np.array(self.shap_values).shape}")
        return self.shap_values
    
    def plot_shap_summary(self, class_idx=1, save_dir="results"):
        """Plot SHAP summary"""
        log_message("Plotting SHAP summary...")
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(self.shap_values[class_idx], self.X_explain.values,
                         feature_names=self.X_explain.columns, show=False)
        plt.title(f"SHAP Summary Plot - Class {class_idx}")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/shap_summary.png", dpi=300, bbox_inches='tight')
        log_message("✓ SHAP summary saved")
        plt.close()
    
    def plot_shap_dependence(self, feature_idx, class_idx=1, save_dir="results"):
        """Plot SHAP dependence"""
        log_message(f"Plotting SHAP dependence for feature {feature_idx}...")
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature_idx, self.shap_values[class_idx],
                            self.X_explain.values,
                            feature_names=self.X_explain.columns, show=False)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/shap_dependence_feature_{feature_idx}.png", dpi=300, bbox_inches='tight')
        log_message("✓ SHAP dependence plot saved")
        plt.close()
    
    def get_feature_importance(self, class_idx=1):
        """Get feature importance from SHAP"""
        mean_abs_shap = np.mean(np.abs(self.shap_values[class_idx]), axis=0)
        feature_importance = pd.DataFrame({
            'feature': self.X_explain.columns,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)
        
        log_message("\nFeature Importance (SHAP):")
        print(feature_importance)
        
        return feature_importance


# Main execution
if __name__ == "__main__":
    log_message("=== EXPLAINABILITY ANALYSIS ===")
    log_message("(Run after training)")
