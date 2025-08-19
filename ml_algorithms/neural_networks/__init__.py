# =====================================
# File: ml_algorithms/neural_networks/__init__.py
# Neural Networks Package Initialization - Sklearn-based
# =====================================

"""
Neural Networks Package
Financial Analysis with Scikit-learn Based Deep Learning Models
Lightweight alternative to TensorFlow for financial risk assessment
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

print("🧠 Initializing Neural Networks Package (Sklearn-based)...")

# Initialize availability flags
SKLEARN_AVAILABLE = False
NEURAL_NETWORKS_AVAILABLE = False
NUMPY_AVAILABLE = False
PANDAS_AVAILABLE = False

# Check core dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
    print("✅ NumPy available")
except ImportError:
    NUMPY_AVAILABLE = False
    print("❌ NumPy not available")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
    print("✅ Pandas available")
except ImportError:
    PANDAS_AVAILABLE = False
    print("❌ Pandas not available")

# Check Scikit-learn availability
try:
    from sklearn.neural_network import MLPRegressor, MLPClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
    print("✅ Scikit-learn available with MLPRegressor and MLPClassifier")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("❌ Scikit-learn not available. Install with: pip install scikit-learn")

# Initialize model classes as None
NeuralNetworkRegressor = None
NeuralNetworkClassifier = None
NeuralNetworkConfig = None

# Import neural network modules
try:
    from .nn_regressor import NeuralNetworkRegressor
    print("✅ NeuralNetworkRegressor loaded")
    NEURAL_NETWORKS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ NeuralNetworkRegressor not found: {e}")
    # Create a simple neural network regressor if we have sklearn
    if SKLEARN_AVAILABLE and NUMPY_AVAILABLE:
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_squared_error, r2_score
        import numpy as np
        
        class NeuralNetworkRegressor:
            """Simple Neural Network Regressor using sklearn MLPRegressor"""
            def __init__(self, hidden_layer_sizes=(100, 50), activation='relu', 
                         solver='adam', alpha=0.0001, learning_rate_init=0.001,
                         max_iter=200, random_state=42, **kwargs):
                self.hidden_layer_sizes = hidden_layer_sizes
                self.activation = activation
                self.solver = solver
                self.alpha = alpha
                self.learning_rate_init = learning_rate_init
                self.max_iter = max_iter
                self.random_state = random_state
                
                self.model = None
                self.scaler = StandardScaler()
                self.is_trained = False
                self.feature_names = None
                
                print(f"✅ Created sklearn Neural Network Regressor with layers: {hidden_layer_sizes}")
            
            def train(self, X, y, verbose=True):
                """Train the neural network regressor"""
                try:
                    if verbose:
                        print("🚀 Training Neural Network Regressor...")
                    
                    # Handle pandas DataFrames
                    if hasattr(X, 'columns'):
                        self.feature_names = list(X.columns)
                        X = X.values
                    if hasattr(y, 'values'):
                        y = y.values
                    
                    # Scale features
                    X_scaled = self.scaler.fit_transform(X)
                    
                    # Create and train model
                    self.model = MLPRegressor(
                        hidden_layer_sizes=self.hidden_layer_sizes,
                        activation=self.activation,
                        solver=self.solver,
                        alpha=self.alpha,
                        learning_rate_init=self.learning_rate_init,
                        max_iter=self.max_iter,
                        random_state=self.random_state,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=20
                    )
                    
                    self.model.fit(X_scaled, y)
                    self.is_trained = True
                    
                    if verbose:
                        print("✅ Neural Network Regressor training completed!")
                        print(f"📊 Training iterations: {self.model.n_iter_}")
                        print(f"📊 Final loss: {self.model.loss_:.6f}")
                    
                    return True
                except Exception as e:
                    print(f"❌ Training failed: {e}")
                    return False
            
            def predict(self, X):
                """Make predictions"""
                if not self.is_trained:
                    raise ValueError("Model not trained")
                
                if hasattr(X, 'values'):
                    X = X.values
                
                X_scaled = self.scaler.transform(X)
                return self.model.predict(X_scaled)
            
            def evaluate(self, X, y):
                """Evaluate model performance"""
                y_pred = self.predict(X)
                if hasattr(y, 'values'):
                    y = y.values
                
                return {
                    'mse': float(mean_squared_error(y, y_pred)),
                    'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
                    'r2': float(r2_score(y, y_pred)),
                    'mae': float(np.mean(np.abs(y - y_pred)))
                }
            
            def get_feature_importance(self):
                """Get approximate feature importance"""
                if not self.is_trained or not hasattr(self.model, 'coefs_'):
                    return {}
                
                # Use first layer weights
                first_layer_weights = self.model.coefs_[0]
                importance_scores = np.sum(np.abs(first_layer_weights), axis=1)
                importance_scores = importance_scores / np.sum(importance_scores)
                
                if self.feature_names:
                    return dict(zip(self.feature_names, importance_scores))
                else:
                    return {f'feature_{i}': score for i, score in enumerate(importance_scores)}
        
        NEURAL_NETWORKS_AVAILABLE = True
    else:
        # Fallback without sklearn
        class NeuralNetworkRegressor:
            """Placeholder Neural Network Regressor"""
            def __init__(self, *args, **kwargs):
                self.is_trained = False
                print("⚠️ Neural Networks require scikit-learn. Install with: pip install scikit-learn")
            
            def train(self, X, y, verbose=True):
                print("❌ Cannot train: scikit-learn not available")
                return False
            
            def predict(self, X):
                print("❌ Cannot predict: scikit-learn not available")
                return []
            
            def evaluate(self, X, y):
                print("❌ Cannot evaluate: scikit-learn not available")
                return {}

try:
    from .nn_classifier import NeuralNetworkClassifier
    print("✅ NeuralNetworkClassifier loaded")
except ImportError as e:
    print(f"⚠️ NeuralNetworkClassifier not found: {e}")
    # Create a simple classifier if we have sklearn
    if SKLEARN_AVAILABLE and NUMPY_AVAILABLE:
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.metrics import accuracy_score, classification_report
        import numpy as np
        
        class NeuralNetworkClassifier:
            """Simple Neural Network Classifier using sklearn MLPClassifier"""
            def __init__(self, hidden_layer_sizes=(100, 50), activation='relu', 
                         solver='adam', alpha=0.0001, learning_rate_init=0.001,
                         max_iter=200, random_state=42, class_weight=None, **kwargs):
                self.hidden_layer_sizes = hidden_layer_sizes
                self.activation = activation
                self.solver = solver
                self.alpha = alpha
                self.learning_rate_init = learning_rate_init
                self.max_iter = max_iter
                self.random_state = random_state
                self.class_weight = class_weight
                
                self.model = None
                self.scaler = StandardScaler()
                self.label_encoder = LabelEncoder()
                self.is_trained = False
                self.feature_names = None
                self.class_names = None
                
                print(f"✅ Created sklearn Neural Network Classifier with layers: {hidden_layer_sizes}")
            
            def train(self, X, y, class_names=None, verbose=True):
                """Train the neural network classifier"""
                try:
                    if verbose:
                        print("🚀 Training Neural Network Classifier...")
                    
                    # Handle pandas DataFrames
                    if hasattr(X, 'columns'):
                        self.feature_names = list(X.columns)
                        X = X.values
                    if hasattr(y, 'values'):
                        y = y.values
                    
                    # Encode labels
                    y_encoded = self.label_encoder.fit_transform(y)
                    self.class_names = class_names or [f'Class_{i}' for i in self.label_encoder.classes_]
                    
                    # Scale features
                    X_scaled = self.scaler.fit_transform(X)
                    
                    # Create and train model
                    self.model = MLPClassifier(
                        hidden_layer_sizes=self.hidden_layer_sizes,
                        activation=self.activation,
                        solver=self.solver,
                        alpha=self.alpha,
                        learning_rate_init=self.learning_rate_init,
                        max_iter=self.max_iter,
                        random_state=self.random_state,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=20
                    )
                    
                    self.model.fit(X_scaled, y_encoded)
                    self.is_trained = True
                    
                    if verbose:
                        print("✅ Neural Network Classifier training completed!")
                        print(f"📊 Classes: {self.class_names}")
                        print(f"📊 Training iterations: {self.model.n_iter_}")
                        print(f"📊 Final loss: {self.model.loss_:.6f}")
                    
                    return True
                except Exception as e:
                    print(f"❌ Training failed: {e}")
                    return False
            
            def predict(self, X):
                """Make predictions"""
                if not self.is_trained:
                    raise ValueError("Model not trained")
                
                if hasattr(X, 'values'):
                    X = X.values
                
                X_scaled = self.scaler.transform(X)
                predictions_encoded = self.model.predict(X_scaled)
                return self.label_encoder.inverse_transform(predictions_encoded)
            
            def predict_proba(self, X):
                """Get prediction probabilities"""
                if not self.is_trained:
                    raise ValueError("Model not trained")
                
                if hasattr(X, 'values'):
                    X = X.values
                
                X_scaled = self.scaler.transform(X)
                return self.model.predict_proba(X_scaled)
            
            def evaluate(self, X, y):
                """Evaluate model performance"""
                y_pred = self.predict(X)
                if hasattr(y, 'values'):
                    y = y.values
                
                accuracy = accuracy_score(y, y_pred)
                
                return {
                    'accuracy': float(accuracy),
                    'n_samples': len(y),
                    'n_classes': len(self.class_names)
                }
            
            def get_feature_importance(self):
                """Get approximate feature importance"""
                if not self.is_trained or not hasattr(self.model, 'coefs_'):
                    return {}
                
                # Use first layer weights
                first_layer_weights = self.model.coefs_[0]
                importance_scores = np.sum(np.abs(first_layer_weights), axis=1)
                importance_scores = importance_scores / np.sum(importance_scores)
                
                if self.feature_names:
                    return dict(zip(self.feature_names, importance_scores))
                else:
                    return {f'feature_{i}': score for i, score in enumerate(importance_scores)}
    else:
        # Fallback without sklearn
        class NeuralNetworkClassifier:
            """Placeholder Neural Network Classifier"""
            def __init__(self, *args, **kwargs):
                self.is_trained = False
                print("⚠️ Neural Networks require scikit-learn. Install with: pip install scikit-learn")
            
            def train(self, X, y, class_names=None, verbose=True):
                print("❌ Cannot train: scikit-learn not available")
                return False
            
            def predict(self, X):
                print("❌ Cannot predict: scikit-learn not available")
                return []
            
            def predict_proba(self, X):
                print("❌ Cannot predict probabilities: scikit-learn not available")
                return []
            
            def evaluate(self, X, y):
                print("❌ Cannot evaluate: scikit-learn not available")
                return {}

try:
    from .nn_config import NeuralNetworkConfig
    print("✅ NeuralNetworkConfig loaded")
except ImportError as e:
    print(f"⚠️ NeuralNetworkConfig not found: {e}")
    # Create a simple config class for sklearn-based models
    class NeuralNetworkConfig:
        """Neural Network Configuration for Sklearn-based models"""
        
        # Default parameters for sklearn MLPRegressor/MLPClassifier
        DEFAULT_HIDDEN_LAYERS = (128, 64, 32)
        DEFAULT_ACTIVATION = 'relu'
        DEFAULT_SOLVER = 'adam'
        DEFAULT_ALPHA = 0.0001  # L2 regularization
        DEFAULT_LEARNING_RATE = 0.001
        DEFAULT_MAX_ITER = 500
        DEFAULT_BATCH_SIZE = 'auto'
        
        # Financial-specific configurations
        FINANCIAL_REGRESSION_CONFIG = {
            'hidden_layer_sizes': (128, 64, 32),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.01,
            'learning_rate_init': 0.001,
            'max_iter': 400,
            'early_stopping': True,
            'validation_fraction': 0.15,
            'n_iter_no_change': 20
        }
        
        FINANCIAL_CLASSIFICATION_CONFIG = {
            'hidden_layer_sizes': (256, 128, 64),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.01,
            'learning_rate_init': 0.001,
            'max_iter': 500,
            'class_weight': 'balanced',
            'early_stopping': True,
            'validation_fraction': 0.15,
            'n_iter_no_change': 25
        }
        
        BANKRUPTCY_PREDICTION_CONFIG = {
            'hidden_layer_sizes': (128, 64, 32),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.01,
            'learning_rate_init': 0.001,
            'max_iter': 400,
            'class_weight': 'balanced',
            'early_stopping': True,
            'validation_fraction': 0.15,
            'n_iter_no_change': 25
        }
        
        RISK_CATEGORIZATION_CONFIG = {
            'hidden_layer_sizes': (256, 128, 64, 32),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.05,
            'learning_rate_init': 0.0005,
            'max_iter': 600,
            'class_weight': 'balanced',
            'early_stopping': True,
            'validation_fraction': 0.2,
            'n_iter_no_change': 30
        }
        
        FAST_TRAINING_CONFIG = {
            'hidden_layer_sizes': (64, 32),
            'activation': 'relu',
            'solver': 'lbfgs',
            'alpha': 0.001,
            'max_iter': 200
        }
        
        @classmethod
        def get_financial_regressor_config(cls):
            """Get optimized config for financial regression"""
            return cls.FINANCIAL_REGRESSION_CONFIG.copy()
        
        @classmethod
        def get_financial_classifier_config(cls):
            """Get optimized config for financial classification"""
            return cls.FINANCIAL_CLASSIFICATION_CONFIG.copy()
        
        @classmethod
        def get_bankruptcy_prediction_config(cls):
            """Get optimized config for bankruptcy prediction"""
            return cls.BANKRUPTCY_PREDICTION_CONFIG.copy()
        
        @classmethod
        def get_risk_categorization_config(cls):
            """Get optimized config for risk categorization"""
            return cls.RISK_CATEGORIZATION_CONFIG.copy()
        
        @classmethod
        def get_fast_training_config(cls):
            """Get fast training config for prototyping"""
            return cls.FAST_TRAINING_CONFIG.copy()
        
        @staticmethod
        def get_config_by_scenario(scenario):
            """Get config by scenario"""
            configs = {
                'financial_regression': NeuralNetworkConfig.FINANCIAL_REGRESSION_CONFIG,
                'financial_classification': NeuralNetworkConfig.FINANCIAL_CLASSIFICATION_CONFIG,
                'bankruptcy_prediction': NeuralNetworkConfig.BANKRUPTCY_PREDICTION_CONFIG,
                'risk_categorization': NeuralNetworkConfig.RISK_CATEGORIZATION_CONFIG,
                'fast_training': NeuralNetworkConfig.FAST_TRAINING_CONFIG,
                'default': {
                    'hidden_layer_sizes': (64, 32),
                    'learning_rate_init': 0.001,
                    'max_iter': 200
                }
            }
            return configs.get(scenario, configs['default']).copy()
        
        @staticmethod
        def validate_config(config):
            """Validate configuration parameters"""
            errors = []
            
            # Check required parameters
            if 'hidden_layer_sizes' not in config:
                errors.append("hidden_layer_sizes is required")
            elif not config['hidden_layer_sizes']:
                errors.append("hidden_layer_sizes cannot be empty")
            
            # Check activation function
            valid_activations = ['identity', 'logistic', 'tanh', 'relu']
            if config.get('activation') not in valid_activations:
                errors.append(f"activation must be one of {valid_activations}")
            
            # Check solver
            valid_solvers = ['lbfgs', 'sgd', 'adam']
            if config.get('solver') not in valid_solvers:
                errors.append(f"solver must be one of {valid_solvers}")
            
            # Check learning rate
            if config.get('learning_rate_init', 0) <= 0:
                errors.append("learning_rate_init must be positive")
            
            # Check alpha (regularization)
            if config.get('alpha', 0) < 0:
                errors.append("alpha must be non-negative")
            
            return errors
    
    print("✅ Created sklearn-based NeuralNetworkConfig")

# Set up exports based on what's available
__all__ = [
    'NeuralNetworkRegressor',
    'NeuralNetworkClassifier', 
    'NeuralNetworkConfig',
    'NEURAL_NETWORKS_AVAILABLE',
    'SKLEARN_AVAILABLE',
    'NUMPY_AVAILABLE',
    'PANDAS_AVAILABLE'
]

# Package metadata
__version__ = "2.0.0"
__author__ = "Financial ML Team"
__description__ = "Sklearn-based Neural Networks implementation for financial risk assessment"

def get_neural_networks_info():
    """Get Neural Networks package information"""
    info = {
        'sklearn_available': SKLEARN_AVAILABLE,
        'numpy_available': NUMPY_AVAILABLE,
        'pandas_available': PANDAS_AVAILABLE,
        'neural_networks_available': NEURAL_NETWORKS_AVAILABLE,
        'package_version': __version__,
        'backend': 'scikit-learn',
        'models_available': {
            'regressor': NeuralNetworkRegressor is not None,
            'classifier': NeuralNetworkClassifier is not None,
            'config': NeuralNetworkConfig is not None
        }
    }
    
    if SKLEARN_AVAILABLE:
        try:
            import sklearn
            info['sklearn_version'] = sklearn.__version__
        except:
            info['sklearn_version'] = 'unknown'
    
    return info

def create_neural_network_regressor(config_type='default', **kwargs):
    """Factory function to create Neural Network regressor"""
    if NeuralNetworkRegressor is None:
        raise ImportError("NeuralNetworkRegressor not available. Install scikit-learn.")
    
    # Get base config
    if config_type == 'financial':
        base_config = NeuralNetworkConfig.get_financial_regressor_config()
    elif config_type == 'fast':
        base_config = NeuralNetworkConfig.get_fast_training_config()
    else:
        base_config = {}
    
    # Override with custom kwargs
    base_config.update(kwargs)
    
    return NeuralNetworkRegressor(**base_config)

def create_neural_network_classifier(config_type='default', **kwargs):
    """Factory function to create Neural Network classifier"""
    if NeuralNetworkClassifier is None:
        raise ImportError("NeuralNetworkClassifier not available. Install scikit-learn.")
    
    # Get base config
    if config_type == 'financial':
        base_config = NeuralNetworkConfig.get_financial_classifier_config()
    elif config_type == 'bankruptcy':
        base_config = NeuralNetworkConfig.get_bankruptcy_prediction_config()
    elif config_type == 'risk_categorization':
        base_config = NeuralNetworkConfig.get_risk_categorization_config()
    elif config_type == 'fast':
        base_config = NeuralNetworkConfig.get_fast_training_config()
    else:
        base_config = {}
    
    # Override with custom kwargs
    base_config.update(kwargs)
    
    return NeuralNetworkClassifier(**base_config)

def create_financial_risk_classifier(risk_type='balanced'):
    """Create pre-configured financial risk classifier"""
    configs = {
        'balanced': {
            'hidden_layer_sizes': (128, 64),
            'max_iter': 300,
            'learning_rate_init': 0.001,
            'alpha': 0.01,
            'class_weight': 'balanced'
        },
        'fast': {
            'hidden_layer_sizes': (64, 32),
            'max_iter': 200,
            'learning_rate_init': 0.01,
            'alpha': 0.001,
            'solver': 'lbfgs'
        },
        'accurate': {
            'hidden_layer_sizes': (256, 128, 64),
            'max_iter': 500,
            'learning_rate_init': 0.0005,
            'alpha': 0.05,
            'class_weight': 'balanced'
        },
        'deep': {
            'hidden_layer_sizes': (512, 256, 128, 64),
            'max_iter': 800,
            'learning_rate_init': 0.0001,
            'alpha': 0.1,
            'class_weight': 'balanced'
        }
    }
    
    config = configs.get(risk_type, configs['balanced'])
    return create_neural_network_classifier(**config)

def check_dependencies():
    """Check and report dependency status"""
    print("🔍 Checking Neural Networks Dependencies:")
    print(f"   ✅ NumPy: {'Available' if NUMPY_AVAILABLE else 'Missing'}")
    print(f"   ✅ Pandas: {'Available' if PANDAS_AVAILABLE else 'Missing'}")
    print(f"   ✅ Scikit-learn: {'Available' if SKLEARN_AVAILABLE else 'Missing'}")
    print(f"   ✅ Neural Networks: {'Available' if NEURAL_NETWORKS_AVAILABLE else 'Missing'}")
    
    missing_deps = []
    if not NUMPY_AVAILABLE:
        missing_deps.append('numpy')
    if not PANDAS_AVAILABLE:
        missing_deps.append('pandas')
    if not SKLEARN_AVAILABLE:
        missing_deps.append('scikit-learn')
    
    if missing_deps:
        print(f"\n⚠️ Missing dependencies: {', '.join(missing_deps)}")
        print(f"📦 Install with: pip install {' '.join(missing_deps)}")
    else:
        print("\n✅ All dependencies satisfied!")
    
    return len(missing_deps) == 0

# Print status
print("="*70)
print("📊 Neural Networks Package Status (Sklearn-based):")
print(f"✅ NumPy: {NUMPY_AVAILABLE}")
print(f"✅ Pandas: {PANDAS_AVAILABLE}")
print(f"✅ Scikit-learn: {SKLEARN_AVAILABLE}")
print(f"✅ Neural Networks: {NEURAL_NETWORKS_AVAILABLE}")
print(f"✅ Regressor Model: {NeuralNetworkRegressor is not None}")
print(f"✅ Classifier Model: {NeuralNetworkClassifier is not None}")
print(f"✅ Config Class: {NeuralNetworkConfig is not None}")
print(f"🔧 Backend: scikit-learn MLPRegressor/MLPClassifier")
print(f"📦 Version: {__version__}")
print("="*70)

if not SKLEARN_AVAILABLE:
    print("⚠️ WARNING: scikit-learn is required for neural networks")
    print("📦 Install with: pip install scikit-learn pandas numpy")

if __name__ == "__main__":
    print("🧠 Neural Networks Package Information:")
    info = get_neural_networks_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    print("\n🔍 Dependency Check:")
    check_dependencies()
    
    # Test basic functionality if dependencies are available
    if SKLEARN_AVAILABLE and NUMPY_AVAILABLE:
        print("\n🧪 Testing Basic Functionality:")
        try:
            # Test regressor
            regressor = create_neural_network_regressor('fast')
            print(f"✅ Regressor created: {type(regressor).__name__}")
            
            # Test classifier
            classifier = create_neural_network_classifier('fast')
            print(f"✅ Classifier created: {type(classifier).__name__}")
            
            # Test financial risk classifier
            risk_classifier = create_financial_risk_classifier('fast')
            print(f"✅ Financial risk classifier created: {type(risk_classifier).__name__}")
            
            print("✅ All tests passed!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
    else:
        print("⚠️ Cannot run tests - missing dependencies")