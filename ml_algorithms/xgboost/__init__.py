# =====================================
# File: ml_algorithms/xgboost/__init__.py
# XGBoost Package Initialization - FIXED
# =====================================

"""
XGBoost Package for Financial Analysis
High-performance gradient boosting framework
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("🔥 Initializing XGBoost Package...")

# Initialize availability flags
XGBOOST_AVAILABLE = False
NUMPY_PANDAS_AVAILABLE = False
SKLEARN_AVAILABLE = False

# Try to import XGBoost
try:
    import xgboost as xgb  
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost library available")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("❌ XGBoost not available. Install with: pip install xgboost")

# Try to import other dependencies
try:
    import numpy as np  
    import pandas as pd  
    NUMPY_PANDAS_AVAILABLE = True
    print("✅ NumPy/Pandas available")
except ImportError:
    NUMPY_PANDAS_AVAILABLE = False
    print("❌ NumPy/Pandas not available")

try:
    from sklearn.model_selection import train_test_split  
    from sklearn.metrics import mean_squared_error, accuracy_score  
    SKLEARN_AVAILABLE = True
    print("✅ Scikit-learn available")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("❌ Scikit-learn not available")

# Initialize model classes as None
XGBRegressorModel = None
XGBClassifierModel = None
XGBConfig = None

# Import XGBoost modules if dependencies are available
if XGBOOST_AVAILABLE and NUMPY_PANDAS_AVAILABLE:
    try:
        # Try to import custom XGBoost modules
        from .xgb_regressor import XGBRegressorModel
        print("✅ XGBRegressorModel imported")
    except ImportError as e:
        print(f"⚠️ XGBRegressorModel not found: {e}")
        # Create a simple placeholder
        if XGBOOST_AVAILABLE and SKLEARN_AVAILABLE:
            class XGBRegressorModel:
                """Simple XGBoost Regressor wrapper"""
                def __init__(self, **kwargs):
                    self.model = xgb.XGBRegressor(**kwargs)
                    self.is_trained = False
                
                def train(self, X, y):
                    self.model.fit(X, y)
                    self.is_trained = True
                    return True
                
                def predict(self, X):
                    if not self.is_trained:
                        raise ValueError("Model not trained")
                    return self.model.predict(X)
                
                def evaluate(self, X, y):
                    y_pred = self.predict(X)
                    return {'mse': mean_squared_error(y, y_pred)}
            print("✅ Created simple XGBRegressorModel")
    
    try:
        from .xgb_classifier import XGBClassifierModel
        print("✅ XGBClassifierModel imported")
    except ImportError as e:
        print(f"⚠️ XGBClassifierModel not found: {e}")
        # Create a simple placeholder
        if XGBOOST_AVAILABLE and SKLEARN_AVAILABLE:
            class XGBClassifierModel:
                """Simple XGBoost Classifier wrapper"""
                def __init__(self, **kwargs):
                    self.model = xgb.XGBClassifier(**kwargs)
                    self.is_trained = False
                
                def train(self, X, y):
                    self.model.fit(X, y)
                    self.is_trained = True
                    return True
                
                def predict(self, X):
                    if not self.is_trained:
                        raise ValueError("Model not trained")
                    return self.model.predict(X)
                
                def evaluate(self, X, y):
                    y_pred = self.predict(X)
                    return {'accuracy': accuracy_score(y, y_pred)}
            print("✅ Created simple XGBClassifierModel")
    
    try:
        from .xgb_config import XGBConfig
        print("✅ XGBConfig imported")
    except ImportError as e:
        print(f"⚠️ XGBConfig not found: {e}")
        # Create a simple config class
        class XGBConfig:
            """Simple XGBoost Configuration"""
            DEFAULT_REGRESSOR_PARAMS = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
            
            DEFAULT_CLASSIFIER_PARAMS = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            }
            
            @classmethod
            def get_financial_regressor_config(cls):
                return cls.DEFAULT_REGRESSOR_PARAMS.copy()
            
            @classmethod
            def get_financial_classifier_config(cls):
                return cls.DEFAULT_CLASSIFIER_PARAMS.copy()
        print("✅ Created simple XGBConfig")

# Set up exports based on what's available
__all__ = ['XGBOOST_AVAILABLE', 'NUMPY_PANDAS_AVAILABLE', 'SKLEARN_AVAILABLE']

if XGBRegressorModel is not None:
    __all__.append('XGBRegressorModel')

if XGBClassifierModel is not None:
    __all__.append('XGBClassifierModel')

if XGBConfig is not None:
    __all__.append('XGBConfig')

# Package metadata
__version__ = "1.0.0"
__author__ = "Financial ML Team"
__description__ = "XGBoost implementation for financial risk assessment"

def get_xgboost_info():
    """Get XGBoost package information"""
    info = {
        'xgboost_available': XGBOOST_AVAILABLE,
        'numpy_pandas_available': NUMPY_PANDAS_AVAILABLE,
        'sklearn_available': SKLEARN_AVAILABLE,
        'package_version': __version__,
        'models_available': {
            'regressor': XGBRegressorModel is not None,
            'classifier': XGBClassifierModel is not None,
            'config': XGBConfig is not None
        }
    }
    
    if XGBOOST_AVAILABLE:
        try:
            info['xgboost_version'] = xgb.__version__
        except:
            info['xgboost_version'] = 'unknown'
    
    return info

def create_xgboost_regressor(**kwargs):
    """Factory function to create XGBoost regressor"""
    if XGBRegressorModel is None:
        raise ImportError("XGBRegressorModel not available")
    return XGBRegressorModel(**kwargs)

def create_xgboost_classifier(**kwargs):
    """Factory function to create XGBoost classifier"""
    if XGBClassifierModel is None:
        raise ImportError("XGBClassifierModel not available")
    return XGBClassifierModel(**kwargs)

# Print status
print("="*50)
print("📊 XGBoost Package Status:")
print(f"✅ XGBoost Library: {XGBOOST_AVAILABLE}")
print(f"✅ NumPy/Pandas: {NUMPY_PANDAS_AVAILABLE}")
print(f"✅ Scikit-learn: {SKLEARN_AVAILABLE}")
print(f"✅ Regressor Model: {XGBRegressorModel is not None}")
print(f"✅ Classifier Model: {XGBClassifierModel is not None}")
print(f"✅ Config Class: {XGBConfig is not None}")
print("="*50)

if __name__ == "__main__":
    print("🚀 XGBoost Package Information:")
    info = get_xgboost_info()
    for key, value in info.items():
        print(f"   {key}: {value}")