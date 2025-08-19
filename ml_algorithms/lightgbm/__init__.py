"""
LightGBM Package for Financial Analysis
High-performance gradient boosting framework
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

print("⚡ Initializing LightGBM Package...")

# Check LightGBM availability
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    print(f"✅ LightGBM {lgb.__version__} available")
except ImportError as e:
    print(f"❌ LightGBM not available: {e}")
    print("📦 Install with: pip install lightgbm")
    LIGHTGBM_AVAILABLE = False

# Import LightGBM models
if LIGHTGBM_AVAILABLE:
    try:
        from .lgb_regressor import LGBRegressorModel
        from .lgb_classifier import LGBClassifierModel
        from .lgb_config import LGBConfig
        
        print("✅ LightGBM Regressor loaded")
        print("✅ LightGBM Classifier loaded")
        print("✅ LightGBM Configuration loaded")
        
    except ImportError as e:
        print(f"❌ LightGBM models import failed: {e}")
        LGBRegressorModel = None
        LGBClassifierModel = None
        LGBConfig = None
else:
    # Create dummy classes if LightGBM not available
    class LGBRegressorModel:
        def __init__(self, **kwargs):
            raise ImportError("LightGBM not available")
    
    class LGBClassifierModel:
        def __init__(self, **kwargs):
            raise ImportError("LightGBM not available")
    
    class LGBConfig:
        @staticmethod
        def get_default_config():
            return {}

# Package exports
__all__ = [
    'LGBRegressorModel',
    'LGBClassifierModel', 
    'LGBConfig',
    'LIGHTGBM_AVAILABLE'
]

print(f"⚡ LightGBM Package Status: {'✅ Ready' if LIGHTGBM_AVAILABLE else '❌ Not Available'}")
print("=" * 50)