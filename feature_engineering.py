"""
Advanced Feature Engineering
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import config
from utils import log_message
from data_preprocessing import DataPreprocessor  # ✓ FIXED IMPORT

class FeatureEngineer:
    def __init__(self, X_train, y_train, X_test=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.selected_features = None
        self.feature_importance = None
        
    def improved_entropy(self, col):
        """Calculate improved entropy (from paper)"""
        value, counts = np.unique(col, return_counts=True)
        prob = counts / counts.sum()
        b = 2
        entropy = 0
        for p in prob:
            if p > 0:
                bel = p + 1 * p
                entropy -= p * np.log2(p / b) / np.exp(2 * (p + 1))
        return entropy
    
    def select_top_features(self, k=None):
        """Select top k features using Information Gain"""
        if k is None:
            k = config.NUM_FEATURES
        
        log_message(f"\n🎯 Selecting top {k} features using Information Gain...")
        
        # Calculate mutual information for each feature
        mi = mutual_info_classif(self.X_train, self.y_train, 
                                random_state=config.RANDOM_STATE)
        
        # Get top k features
        idx = np.argsort(mi)[-k:]
        
        # Create importance dataframe
        self.feature_importance = pd.DataFrame({
            'Feature': self.X_train.columns,
            'Importance': mi
        }).sort_values('Importance', ascending=False)
        
        log_message(f"✓ Top {k} features selected:\n")
        print(self.feature_importance.head(k))
        
        # Select features
        X_train_sel = self.X_train.iloc[:, idx]
        X_test_sel = self.X_test.iloc[:, idx] if self.X_test is not None else None
        
        self.selected_features = idx
        return X_train_sel, X_test_sel, self.feature_importance


# Run this step
if __name__ == "__main__":
    log_message("=" * 60)
    log_message("STEP 2: FEATURE ENGINEERING")
    log_message("=" * 60)
    
    # Load preprocessed data
    preprocessor = DataPreprocessor(config.DATA_PATH)
    X_train, X_test, y_train, y_test = preprocessor.get_processed_data()
    
    # Feature engineering
    engineer = FeatureEngineer(X_train, y_train, X_test)
    X_train_sel, X_test_sel, importance = engineer.select_top_features()
    
    log_message("\n" + "=" * 60)
    log_message("✓ FEATURE ENGINEERING COMPLETE!")
    log_message("=" * 60)
