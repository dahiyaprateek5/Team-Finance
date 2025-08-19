"""
Random Forest Package for Financial Analysis - FIXED
Ensemble learning with decision trees
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

print("🌲 Initializing Random Forest Package...")

# Try to import scikit-learn
try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    SKLEARN_AVAILABLE = True
    print("✅ Scikit-learn available")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("❌ Scikit-learn not available")

# Try to import Random Forest modules
RF_REGRESSOR_AVAILABLE = False
RF_CLASSIFIER_AVAILABLE = False
RF_CONFIG_AVAILABLE = False

try:
    from .rf_regressor import RandomForestRegressorModel
    RF_REGRESSOR_AVAILABLE = True
    print("✅ Random Forest Regressor loaded")
except ImportError as e:
    print(f"⚠️ Random Forest Regressor not available: {e}")
    RandomForestRegressorModel = None

try:
    from .rf_classifier import RandomForestClassifierModel # type: ignore
    RF_CLASSIFIER_AVAILABLE = True
    print("✅ Random Forest Classifier loaded")
except ImportError as e:
    print(f"⚠️ Random Forest Classifier not available: {e}")
    RandomForestClassifierModel = None

try:
    from .rf_config import RandomForestConfig
    RF_CONFIG_AVAILABLE = True
    print("✅ Random Forest Config loaded")
except ImportError as e:
    print(f"⚠️ Random Forest Config not available: {e}")
    RandomForestConfig = None

# Package status
RANDOM_FOREST_AVAILABLE = SKLEARN_AVAILABLE and RF_REGRESSOR_AVAILABLE and RF_CLASSIFIER_AVAILABLE and RF_CONFIG_AVAILABLE

if RANDOM_FOREST_AVAILABLE:
    print("🌲 Random Forest Package: ✅ Fully Available")
else:
    print("🌲 Random Forest Package: ❌ Limited")

print("="*60)

# Package exports
__all__ = [
    'RandomForestRegressorModel',
    'RandomForestClassifierModel', 
    'RandomForestConfig',
    'RANDOM_FOREST_AVAILABLE',
    'SKLEARN_AVAILABLE'
]

__version__ = "1.0.0"