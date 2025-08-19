"""
Model Evaluation Package
Comprehensive evaluation tools for ML models
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

print("📊 Initializing Model Evaluation Package...")

# Import evaluation modules
try:
    from .metrics import ModelMetrics, RegressionMetrics, ClassificationMetrics
    from .cross_validation import CrossValidator
    from .performance_tracker import PerformanceTracker
    
    EVALUATION_AVAILABLE = True
    print("✅ All evaluation modules loaded")
    
except ImportError as e:
    print(f"⚠️ Some evaluation modules not available: {e}")
    EVALUATION_AVAILABLE = False
    
    # Create dummy classes if not available
    class ModelMetrics:
        pass
    class RegressionMetrics:
        pass
    class ClassificationMetrics:
        pass
    class CrossValidator:
        pass
    class PerformanceTracker:
        pass

# Package exports
__all__ = [
    'ModelMetrics',
    'RegressionMetrics', 
    'ClassificationMetrics',
    'CrossValidator',
    'PerformanceTracker',
    'EVALUATION_AVAILABLE'
]

print(f"📊 Model Evaluation Package: {'✅ Ready' if EVALUATION_AVAILABLE else '❌ Limited'}")
print("=" * 60)