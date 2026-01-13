"""
Advanced Data Preprocessing Pipeline
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import config
from utils import log_message

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoders = {}
        
    def load_data(self):
        """Load dataset"""
        log_message("📥 Loading data...")
        self.df = pd.read_csv(self.data_path)
        log_message(f"Dataset shape: {self.df.shape}")
        log_message(f"Features: {self.df.columns.tolist()}")
        print(f"\nClass Distribution:\n{self.df['HeartDisease'].value_counts()}")
        return self.df
    
    def encode_categorical(self):
        """Encode categorical features to numbers"""
        log_message("\n🔄 Encoding categorical features...")
        for col in self.df.select_dtypes(include="object").columns:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
        log_message("✓ Encoding complete")
    
    def improved_zscore_normalization(self, x):
        """Paper-based improved Z-score normalization formula"""
        median = np.median(x)
        std = np.std(x)
        if std == 0:
            return np.zeros_like(x)
        # Formula: IY = 2 * (1 / (1 + exp(-(Y - median) / std))) - 1
        return 2 * (1 / (1 + np.exp(-(x - median) / std))) - 1
    
    def normalize_features(self):
        """Apply improved Z-score normalization"""
        log_message("\n📊 Normalizing features (improved Z-score)...")
        X = self.df.drop("HeartDisease", axis=1)
        y = self.df["HeartDisease"]
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TRAIN_TEST_SPLIT, 
            random_state=config.RANDOM_STATE, stratify=y
        )
        
        # Apply normalization
        X_train_norm = X_train.apply(self.improved_zscore_normalization, axis=0)
        X_test_norm = X_test.apply(self.improved_zscore_normalization, axis=0)
        
        self.X_train = X_train_norm
        self.X_test = X_test_norm
        self.y_train = y_train.reset_index(drop=True)
        self.y_test = y_test.reset_index(drop=True)
        
        log_message(f"✓ Train samples: {self.X_train.shape[0]}, Test samples: {self.X_test.shape[0]}")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def apply_smote(self):
        """Apply SMOTE to balance classes"""
        log_message("\n⚖️ Balancing with SMOTE...")
        smote = SMOTE(sampling_strategy=config.SMOTE_RATIO, 
                     random_state=config.RANDOM_STATE)
        X_train_smote, y_train_smote = smote.fit_resample(self.X_train, self.y_train)
        
        self.X_train = pd.DataFrame(X_train_smote, columns=self.X_train.columns)
        self.y_train = pd.Series(y_train_smote)
        
        print(f"\nAfter SMOTE:")
        print(f"Train samples: {self.X_train.shape[0]}")
        print(f"Class distribution:\n{self.y_train.value_counts()}")
        
        return self.X_train, self.y_train
    
    def get_processed_data(self):
        """Run entire preprocessing pipeline"""
        self.load_data()
        self.encode_categorical()
        self.normalize_features()
        if config.USE_SMOTE:
            self.apply_smote()
        return self.X_train, self.X_test, self.y_train, self.y_test


# Run this step
if __name__ == "__main__":
    log_message("=" * 60)
    log_message("STEP 1: DATA PREPROCESSING")
    log_message("=" * 60)
    
    preprocessor = DataPreprocessor(config.DATA_PATH)
    X_train, X_test, y_train, y_test = preprocessor.get_processed_data()
    
    log_message("\n" + "=" * 60)
    log_message("✓ PREPROCESSING COMPLETE!")
    log_message("=" * 60)
