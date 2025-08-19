"""
LightGBM Regressor for Financial Health Prediction
High-performance gradient boosting for regression tasks
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)

import numpy as np # type: ignore
from typing import Dict, Any, Optional, List
import time
from datetime import datetime

# Import base model
try:
    from ml_algorithms.base_model import BaseMLModel
except ImportError:
    # Fallback if base model not available
    class BaseMLModel:
        def __init__(self, model_name):
            self.model_name = model_name
            self.model = None
            self.is_trained = False
            self.training_time = 0
            self.performance_metrics = {}
            self.feature_importance = None

# Import LightGBM
try:
    import lightgbm as lgb
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error  # type: ignore
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

class LGBRegressorModel(BaseMLModel):
    """
    LightGBM Regressor optimized for financial health prediction
    
    Features:
    - Fast training and prediction
    - Built-in feature importance
    - Handles missing values
    - Memory efficient
    - Early stopping support
    """
    
    def __init__(self, **kwargs):
        super().__init__("LightGBM_Regressor")
        
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available. Install with: pip install lightgbm")
        
        self.early_stopping_rounds = None
        self.eval_results = {}
        self.best_iteration = None
        
        self.create_model(**kwargs)
    
    def create_model(self, **kwargs):
        """Create LightGBM regressor with optimized financial parameters"""
        
        # Default parameters optimized for financial data
        default_params = {
            # Core parameters
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            
            # Tree parameters
            'num_leaves': 31,
            'max_depth': 8,
            'min_data_in_leaf': 10,
            'min_sum_hessian_in_leaf': 1e-3,
            
            # Learning parameters
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'subsample_freq': 1,
            'colsample_bytree': 0.8,
            
            # Regularization
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            
            # Performance
            'n_jobs': -1,
            'random_state': 42,
            'verbose': -1,
            
            # Advanced parameters for financial data
            'extra_trees': False,
            'feature_fraction_bynode': 0.8,
            'min_gain_to_split': 0.02,
            'max_bin': 255
        }
        
        # Update with user parameters
        default_params.update(kwargs)
        
        # Extract early stopping parameter
        self.early_stopping_rounds = default_params.pop('early_stopping_rounds', 20)
        
        # Create model
        self.model = lgb.LGBMRegressor(**default_params)
        self.model_params = default_params
        
        print(f"✅ {self.model_name} created")
        print(f"   Num leaves: {default_params['num_leaves']}")
        print(f"   Max depth: {default_params['max_depth']}")
        print(f"   Learning rate: {default_params['learning_rate']}")
        print(f"   N estimators: {default_params['n_estimators']}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None,
              eval_set: Optional[List] = None) -> bool:
        """
        Train LightGBM regressor with optional validation
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            eval_set: Custom evaluation set (optional)
        """
        try:
            start_time = time.time()
            
            # Prepare evaluation set
            if eval_set is None:
                if X_val is not None and y_val is not None:
                    eval_set = [(X_val, y_val)]
                    eval_names = ['validation']
                else:
                    eval_set = None
                    eval_names = None
            else:
                eval_names = [f'eval_{i}' for i in range(len(eval_set))]
            
            # Train with early stopping if validation data available
            if eval_set is not None:
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    eval_names=eval_names,
                    eval_metric='rmse',
                    early_stopping_rounds=self.early_stopping_rounds,
                    verbose=False
                )
                
                # Store best iteration
                self.best_iteration = self.model.best_iteration_
                
                # Store evaluation results
                if hasattr(self.model, 'evals_result_'):
                    self.eval_results = self.model.evals_result_
                
            else:
                # Train without validation
                self.model.fit(X_train, y_train)
            
            self.training_time = time.time() - start_time
            self.is_trained = True
            
            # Get feature importance
            self.feature_importance = self.model.feature_importances_
            
            print(f"✅ {self.model_name} trained in {self.training_time:.2f} seconds")
            
            if self.best_iteration:
                print(f"   Best iteration: {self.best_iteration}")
                print(f"   Total trees: {self.model.n_estimators}")
            
            return True
            
        except Exception as e:
            print(f"❌ {self.model_name} training failed: {e}")
            return False
    
    def predict(self, X: np.ndarray, num_iteration: Optional[int] = None) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features for prediction
            num_iteration: Number of trees to use (None for best iteration)
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Use best iteration if available and num_iteration not specified
        if num_iteration is None and self.best_iteration:
            num_iteration = self.best_iteration
        
        return self.model.predict(X, num_iteration=num_iteration)
    
    def predict_with_uncertainty(self, X: np.ndarray, n_iterations: int = 100) -> Dict:
        """
        Predict with uncertainty estimation using different tree subsets
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        try:
            predictions = []
            step_size = max(1, self.model.n_estimators // n_iterations)
            
            for i in range(step_size, self.model.n_estimators + 1, step_size):
                pred = self.model.predict(X, num_iteration=i)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            
            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)
            
            return {
                'mean_prediction': mean_pred,
                'std_prediction': std_pred,
                'confidence_interval_95': {
                    'lower': mean_pred - 1.96 * std_pred,
                    'upper': mean_pred + 1.96 * std_pred
                },
                'uncertainty_score': np.mean(std_pred)
            }
            
        except Exception as e:
            print(f"❌ Uncertainty prediction failed: {e}")
            return {'mean_prediction': self.predict(X)}
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Comprehensive model evaluation"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        try:
            predictions = self.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            # Additional financial metrics
            mape = np.mean(np.abs((y_test - predictions) / (y_test + 1e-8))) * 100
            
            # Prediction ranges
            pred_min, pred_max = np.min(predictions), np.max(predictions)
            actual_min, actual_max = np.min(y_test), np.max(y_test)
            
            self.performance_metrics = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2_score': float(r2),
                'mape': float(mape),
                'model_type': 'regressor',
                'prediction_range': {
                    'min': float(pred_min),
                    'max': float(pred_max)
                },
                'actual_range': {
                    'min': float(actual_min),
                    'max': float(actual_max)
                },
                'best_iteration': self.best_iteration,
                'total_estimators': self.model.n_estimators,
                'training_time': self.training_time
            }
            
            print(f"📊 {self.model_name} Performance:")
            print(f"   R² Score: {r2:.4f}")
            print(f"   RMSE: {rmse:.4f}")
            print(f"   MAE: {mae:.4f}")
            print(f"   MAPE: {mape:.2f}%")
            
            return self.performance_metrics
            
        except Exception as e:
            print(f"❌ {self.model_name} evaluation failed: {e}")
            return {}
    
    def get_feature_importance(self, importance_type: str = 'split') -> Dict:
        """
        Get feature importance with different importance types
        
        Args:
            importance_type: 'split', 'gain', or 'both'
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        try:
            importance_data = {}
            
            if importance_type in ['split', 'both']:
                importance_data['split_importance'] = self.model.feature_importances_.tolist()
            
            if importance_type in ['gain', 'both']:
                # Get gain importance if available
                if hasattr(self.model.booster_, 'feature_importance'):
                    gain_importance = self.model.booster_.feature_importance(importance_type='gain')
                    importance_data['gain_importance'] = gain_importance.tolist()
            
            return importance_data
            
        except Exception as e:
            print(f"❌ Feature importance extraction failed: {e}")
            return {'split_importance': self.feature_importance.tolist() if self.feature_importance is not None else []}
    
    def get_hyperparameters(self) -> Dict:
        """Get current hyperparameters"""
        return {
            'model_params': self.model_params.copy(),
            'early_stopping_rounds': self.early_stopping_rounds,
            'best_iteration': self.best_iteration
        }
    
    def tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray, 
                           X_val: np.ndarray, y_val: np.ndarray,
                           param_grid: Optional[Dict] = None) -> Dict:
        """
        Hyperparameter tuning using validation set
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            param_grid: Custom parameter grid
        """
        try:
            if param_grid is None:
                param_grid = {
                    'num_leaves': [31, 50, 100],
                    'max_depth': [6, 8, 10],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'n_estimators': [100, 200, 300],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            
            best_score = float('-inf')
            best_params = None
            tuning_results = []
            
            print(f"🔧 Tuning hyperparameters for {self.model_name}...")
            
            # Simple grid search (you can replace with more sophisticated methods)
            param_combinations = self._generate_param_combinations(param_grid)
            
            for i, params in enumerate(param_combinations[:20]):  # Limit to 20 combinations
                try:
                    # Create temporary model
                    temp_params = self.model_params.copy()
                    temp_params.update(params)
                    
                    temp_model = lgb.LGBMRegressor(**temp_params)
                    temp_model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=self.early_stopping_rounds,
                        verbose=False
                    )
                    
                    # Evaluate
                    val_pred = temp_model.predict(X_val)
                    val_score = r2_score(y_val, val_pred)
                    
                    tuning_results.append({
                        'params': params.copy(),
                        'score': val_score
                    })
                    
                    if val_score > best_score:
                        best_score = val_score
                        best_params = params.copy()
                    
                    print(f"   Combination {i+1}: R² = {val_score:.4f}")
                    
                except Exception as e:
                    print(f"   Combination {i+1} failed: {e}")
                    continue
            
            if best_params:
                # Update model with best parameters
                self.model_params.update(best_params)
                self.model = lgb.LGBMRegressor(**self.model_params)
                
                print(f"✅ Best parameters found: {best_params}")
                print(f"   Best R² score: {best_score:.4f}")
                
                return {
                    'best_params': best_params,
                    'best_score': best_score,
                    'all_results': tuning_results
                }
            else:
                print("❌ No valid parameter combination found")
                return {}
                
        except Exception as e:
            print(f"❌ Hyperparameter tuning failed: {e}")
            return {}
    
    def _generate_param_combinations(self, param_grid: Dict) -> List[Dict]:
        """Generate parameter combinations from grid"""
        import itertools
        
        keys = param_grid.keys()
        values = param_grid.values()
        
        combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)
        
        return combinations
    
    def save_model(self, filepath: str) -> bool:
        """Save trained model"""
        try:
            if not self.is_trained:
                print("❌ Cannot save untrained model")
                return False
            
            self.model.booster_.save_model(filepath)
            print(f"✅ Model saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Model save failed: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load trained model"""
        try:
            self.model = lgb.Booster(model_file=filepath)
            self.is_trained = True
            print(f"✅ Model loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Model load failed: {e}")
            return False
    
    def get_training_history(self) -> Dict:
        """Get training history and evaluation results"""
        return {
            'eval_results': self.eval_results,
            'best_iteration': self.best_iteration,
            'total_estimators': self.model.n_estimators if self.is_trained else 0,
            'training_time': self.training_time
        }
    
    def plot_feature_importance(self, max_features: int = 20) -> Dict:
        """
        Get data for feature importance plotting
        
        Args:
            max_features: Maximum number of features to include
        """
        if not self.is_trained or self.feature_importance is None:
            return {}
        
        try:
            # Get feature importance
            importance_scores = self.feature_importance
            
            # Create feature names if not available
            feature_names = [f'feature_{i}' for i in range(len(importance_scores))]
            
            # Sort by importance
            sorted_indices = np.argsort(importance_scores)[::-1][:max_features]
            
            plot_data = {
                'feature_names': [feature_names[i] for i in sorted_indices],
                'importance_scores': [float(importance_scores[i]) for i in sorted_indices],
                'total_features': len(importance_scores)
            }
            
            return plot_data
            
        except Exception as e:
            print(f"❌ Feature importance plot data failed: {e}")
            return {}

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing LightGBM Regressor...")
    
    if LIGHTGBM_AVAILABLE:
        try:
            # Generate sample financial data
            np.random.seed(42)
            n_samples = 1000
            n_features = 10
            
            X = np.random.randn(n_samples, n_features)
            y = np.sum(X[:, :5], axis=1) + np.random.randn(n_samples) * 0.1
            
            # Split data
            split_idx = int(0.8 * n_samples)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Create and train model
            model = LGBRegressorModel(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6
            )
            
            print("\n🚀 Training model...")
            success = model.train(X_train, y_train, X_test, y_test)
            
            if success:
                print("\n📊 Evaluating model...")
                performance = model.evaluate(X_test, y_test)
                
                print("\n🎯 Making predictions...")
                predictions = model.predict(X_test[:5])
                print(f"Sample predictions: {predictions}")
                
                print("\n🔍 Feature importance...")
                importance = model.get_feature_importance()
                print(f"Feature importance available: {len(importance) > 0}")
                
                print("\n✅ Test completed successfully!")
            else:
                print("❌ Training failed")
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
    else:
        print("❌ LightGBM not available for testing")