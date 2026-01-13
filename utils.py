"""
Utility functions for IQCNN project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    brier_score_loss
)
from sklearn.calibration import calibration_curve  # ✓ FIXED: Import from sklearn.calibration
from datetime import datetime
import json
import os

def create_report(y_true, y_pred, y_pred_proba, model_name, save_dir="results"):
    """Generate comprehensive evaluation report"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_pred_proba[:, 1])) if y_pred_proba.shape[1] > 1 else 0.5,
        "brier_score": float(brier_score_loss(y_true, y_pred_proba[:, 1])) if y_pred_proba.shape[1] > 1 else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    }
    
    os.makedirs(save_dir, exist_ok=True)
    report_path = os.path.join(save_dir, f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"\n✓ Report saved: {report_path}")
    return report

def plot_metrics(y_true, y_pred_proba, model_name, save_dir="results"):
    """Plot ROC curve, calibration curve, and other metrics"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
    roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
    axes[0, 0].plot(fpr, tpr, label=f'ROC Curve (AUC={roc_auc:.3f})', linewidth=2, color='blue')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Calibration Curve
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba[:, 1], n_bins=10)
        axes[0, 1].plot(prob_pred, prob_true, 'o-', label='IQCNN', linewidth=2, markersize=8)
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        axes[0, 1].set_xlabel('Mean Predicted Probability')
        axes[0, 1].set_ylabel('Fraction of Positives')
        axes[0, 1].set_title('Calibration Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    except Exception as e:
        print(f"Warning: Could not plot calibration curve: {e}")
        axes[0, 1].text(0.5, 0.5, f"Calibration plot error:\n{str(e)}", 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_proba.argmax(axis=1))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0], cbar=False)
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('True')
    axes[1, 0].set_title('Confusion Matrix')
    
    # Prediction Distribution
    axes[1, 1].hist(y_pred_proba[y_true == 0, 1], bins=30, alpha=0.6, label='No Disease', color='blue')
    axes[1, 1].hist(y_pred_proba[y_true == 1, 1], bins=30, alpha=0.6, label='Has Disease', color='red')
    axes[1, 1].set_xlabel('Predicted Probability of Disease')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Prediction Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"{model_name}_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Metrics plot saved: {plot_path}")
    plt.close()

def log_message(message, level="INFO"):
    """Log messages with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")
