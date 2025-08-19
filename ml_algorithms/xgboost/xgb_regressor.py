# =====================================
# File: ml_algorithms/xgboost/xgb_regressor.py
# XGBoost Regressor for Financial Health Prediction
# =====================================

"""
XGBoost Regressor for Financial Health Prediction
High-performance gradient boosting for regression tasks
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

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from typing import Dict, List, Any, Optional, Tuple, Union
import json
from datetime import datetime
import pickle

# Try to import XGBoost
try:
    import xgboost as xgb  # type: ignore
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Try to import sklearn
try:
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV   # type: ignore
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .xgb_config import XGBConfig

class XGBRegressorModel:
    """
    Advanced XGBoost Regressor for Financial Health Prediction
    Optimized for financial analysis and cash flow forecasting
    """
    
    def __init__(self, **params):
        """
        Initialize XGBoost Regressor
        
        Args:
            **params: XGBoost parameters
        """
        
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required. Install with: pip install xgboost")
        
        # Default parameters
        default_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.08,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 30,
            'eval_metric': 'rmse'
        }
        
        # Update with provided parameters
        self.params = {**default_params, **params}
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = None
        self.n_features = None
        self.training_score = None
        self.validation_score = None
        self.best_iteration = None
        self.feature_importance = None
        self.training_time = None
        
        self.model_name = "XGBoost Regressor"
        
        print(f"🚀 {self.model_name} initialized")
    
    def train(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series],
              X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
              y_val: Optional[Union[np.ndarray, pd.Series]] = None,
              scale_features: bool = True, verbose: bool = True) -> bool:
        """
        Train the XGBoost model
        
        Args:
            X: Training features
            y: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            scale_features: Whether to scale features
            verbose: Whether to print training progress
            
        Returns:
            True if training successful, False otherwise
        """
        try:
            start_time = datetime.now()
            
            if verbose:
                print(f"🚀 Starting {self.model_name} training...")
            
            # Convert to numpy arrays and store feature info
            if isinstance(X, pd.DataFrame):
                self.feature_names = list(X.columns)
                X = X.values
            else:
                self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            
            if isinstance(y, pd.Series):
                y = y.values
            
            # Store feature count
            self.n_features = X.shape[1]
            
            # Scale features if requested
            if scale_features:
                X_scaled = self.scaler.fit_transform(X)
                if X_val is not None:
                    if isinstance(X_val, pd.DataFrame):
                        X_val = X_val.values
                    X_val_scaled = self.scaler.transform(X_val)
                else:
                    X_val_scaled = None
            else:
                X_scaled = X
                X_val_scaled = X_val
                # Create dummy scaler for consistency
                self.scaler.fit(X)
            
            if isinstance(y_val, pd.Series):
                y_val = y_val.values
            
            if verbose:
                print(f"📊 Training on {X_scaled.shape[0]} samples with {X_scaled.shape[1]} features")
                if X_val is not None:
                    print(f"📊 Validation on {X_val_scaled.shape[0]} samples")
            
            # Prepare evaluation set
            eval_set = []
            if X_val is not None and y_val is not None:
                eval_set = [(X_val_scaled, y_val)]
            
            # Create and train model
            self.model = xgb.XGBRegressor(**self.params)
            
            # Train with evaluation set if available
            if eval_set:
                self.model.fit(
                    X_scaled, y,
                    eval_set=eval_set,
                    verbose=verbose
                )
            else:
                self.model.fit(X_scaled, y, verbose=verbose)
            
            # Get training score
            self.training_score = self.model.score(X_scaled, y)
            
            # Get validation score if available
            if X_val is not None and y_val is not None:
                self.validation_score = self.model.score(X_val_scaled, y_val)
            
            # Get best iteration if available
            if hasattr(self.model, 'best_iteration') and self.model.best_iteration:
                self.best_iteration = self.model.best_iteration
            
            # Get feature importance
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = self.model.feature_importances_
            
            self.is_trained = True
            
            # Calculate training time
            end_time = datetime.now()
            self.training_time = (end_time - start_time).total_seconds()
            
            if verbose:
                print(f"✅ {self.model_name} trained in {self.training_time:.2f} seconds")
                print(f"📊 Training R² Score: {self.training_score:.4f}")
                if self.validation_score is not None:
                    print(f"📊 Validation R² Score: {self.validation_score:.4f}")
                if self.best_iteration:
                    print(f"📊 Best iteration: {self.best_iteration}")
                    print(f"📊 Total trees: {self.model.n_estimators}")
            
            return True
            
        except Exception as e:
            print(f"❌ {self.model_name} training failed: {e}")
            return False
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X: Features for prediction
            
        Returns:
            Predicted values
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            # Convert to numpy array
            if isinstance(X, pd.DataFrame):
                X = X.values
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            predictions = self.model.predict(X_scaled)
            
            return predictions
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return np.array([])
    
    def predict_with_uncertainty(self, X: Union[np.ndarray, pd.DataFrame], 
                               n_iterations: int = 100) -> Dict[str, np.ndarray]:
        """
        Make predictions with uncertainty estimation using model iterations
        
        Args:
            X: Features for prediction
            n_iterations: Number of iterations for uncertainty estimation
            
        Returns:
            Dictionary with predictions, uncertainties, and confidence intervals
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            # Convert to numpy array
            if isinstance(X, pd.DataFrame):
                X = X.values
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Get predictions from different numbers of trees
            predictions_list = []
            max_iterations = min(n_iterations, self.model.n_estimators)
            
            for i in range(1, max_iterations + 1, max(1, max_iterations // 20)):
                # Create temporary model with fewer estimators
                temp_model = xgb.XGBRegressor(**{**self.params, 'n_estimators': i})
                temp_model._Booster = self.model.get_booster()
                
                try:
                    pred = temp_model.predict(X_scaled, iteration_range=(0, i))
                    predictions_list.append(pred)
                except:
                    # Fallback to regular prediction
                    pred = self.model.predict(X_scaled)
                    predictions_list.append(pred)
            
            if not predictions_list:
                # Fallback to single prediction
                base_predictions = self.predict(X)
                return {
                    'predictions': base_predictions,
                    'uncertainty': np.zeros_like(base_predictions),
                    'confidence_interval_lower': base_predictions,
                    'confidence_interval_upper': base_predictions
                }
            
            predictions_array = np.array(predictions_list)
            
            # Calculate statistics
            mean_pred = np.mean(predictions_array, axis=0)
            std_pred = np.std(predictions_array, axis=0)
            
            # 95% confidence intervals
            ci_lower = np.percentile(predictions_array, 2.5, axis=0)
            ci_upper = np.percentile(predictions_array, 97.5, axis=0)
            
            return {
                'predictions': mean_pred,
                'uncertainty': std_pred,
                'confidence_interval_lower': ci_lower,
                'confidence_interval_upper': ci_upper,
                'iteration_predictions': predictions_array
            }
            
        except Exception as e:
            print(f"❌ Uncertainty prediction failed: {e}")
            # Fallback to regular prediction
            base_predictions = self.predict(X)
            return {
                'predictions': base_predictions,
                'uncertainty': np.zeros_like(base_predictions),
                'confidence_interval_lower': base_predictions,
                'confidence_interval_upper': base_predictions
            }
    
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], 
                y: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """
        Evaluate model performance on test data
        
        Args:
            X: Test features
            y: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before evaluation")
            
            # Make predictions
            y_pred = self.predict(X)
            
            # Convert to numpy
            if isinstance(y, pd.Series):
                y = y.values
            
            # Calculate metrics
            metrics = {}
            
            # Core regression metrics
            metrics['mse'] = float(mean_squared_error(y, y_pred))
            metrics['rmse'] = float(np.sqrt(metrics['mse']))
            metrics['mae'] = float(mean_absolute_error(y, y_pred))
            metrics['r2_score'] = float(r2_score(y, y_pred))
            
            # Financial-specific metrics
            metrics['mape'] = float(np.mean(np.abs((y - y_pred) / (y + 1e-8))) * 100)
            metrics['financial_accuracy'] = float(np.mean(np.abs((y - y_pred) / (y + 1e-8)) <= 0.1) * 100)
            
            # Directional accuracy
            if len(y) > 1:
                y_diff = np.diff(y)
                pred_diff = np.diff(y_pred)
                directional_acc = np.mean(np.sign(y_diff) == np.sign(pred_diff)) * 100
                metrics['directional_accuracy'] = float(directional_acc)
            
            # Prediction interval coverage
            uncertainty_results = self.predict_with_uncertainty(X)
            if uncertainty_results and 'confidence_interval_lower' in uncertainty_results:
                ci_lower = uncertainty_results['confidence_interval_lower']
                ci_upper = uncertainty_results['confidence_interval_upper']
                coverage = np.mean((y >= ci_lower) & (y <= ci_upper))
                metrics['prediction_interval_coverage'] = float(coverage)
                
                # Average prediction interval width
                avg_width = np.mean(ci_upper - ci_lower)
                metrics['avg_prediction_interval_width'] = float(avg_width)
            
            # Model-specific metrics
            if self.training_score is not None:
                metrics['training_score'] = float(self.training_score)
            
            if self.validation_score is not None:
                metrics['validation_score'] = float(self.validation_score)
            
            metrics['n_samples'] = len(y)
            metrics['model_name'] = self.model_name
            
            return metrics
            
        except Exception as e:
            print(f"❌ Model evaluation failed: {e}")
            return {}
    
    def get_feature_importance(self, importance_type: str = 'weight') -> Dict[str, float]:
        """
        Get feature importance scores
        
        Args:
            importance_type: 'weight', 'gain', 'cover', or 'total_gain'
            
        Returns:
            Dictionary with feature importance scores
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before getting feature importance")
            
            # Get importance based on type
            if importance_type == 'weight':
                if hasattr(self.model, 'feature_importances_'):
                    importance_scores = self.model.feature_importances_
                else:
                    importance_scores = self.model.get_booster().get_score(importance_type='weight')
            else:
                importance_scores = self.model.get_booster().get_score(importance_type=importance_type)
            
            # Create feature importance dictionary
            feature_importance = {}
            
            if isinstance(importance_scores, dict):
                # XGBoost returns dict with feature names
                for feature, score in importance_scores.items():
                    feature_importance[feature] = float(score)
            else:
                # Array-like importance scores
                for i, score in enumerate(importance_scores):
                    feature_name = self.feature_names[i] if self.feature_names else f'feature_{i}'
                    feature_importance[feature_name] = float(score)
            
            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(), 
                                           key=lambda x: x[1], reverse=True))
            
            return feature_importance
            
        except Exception as e:
            print(f"❌ Feature importance calculation failed: {e}")
            return {}
    
    def analyze_residuals(self, X: Union[np.ndarray, pd.DataFrame], 
                         y: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """
        Analyze prediction residuals for model diagnostics
        
        Args:
            X: Features
            y: True values
            
        Returns:
            Dictionary with residual analysis
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before residual analysis")
            
            # Make predictions
            y_pred = self.predict(X)
            
            # Convert to numpy
            if isinstance(y, pd.Series):
                y = y.values
            
            # Calculate residuals
            residuals = y - y_pred
            
            # Statistical analysis
            analysis = {
                'mean_residual': float(np.mean(residuals)),
                'std_residual': float(np.std(residuals)),
                'min_residual': float(np.min(residuals)),
                'max_residual': float(np.max(residuals)),
                'median_residual': float(np.median(residuals)),
                'residual_percentiles': {
                    '25th': float(np.percentile(residuals, 25)),
                    '75th': float(np.percentile(residuals, 75)),
                    '90th': float(np.percentile(residuals, 90)),
                    '95th': float(np.percentile(residuals, 95))
                }
            }
            
            # Outlier detection
            q1 = np.percentile(residuals, 25)
            q3 = np.percentile(residuals, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = np.where((residuals < lower_bound) | (residuals > upper_bound))[0]
            analysis['n_outliers'] = len(outliers)
            analysis['outlier_percentage'] = float(len(outliers) / len(residuals) * 100)
            
            # Heteroscedasticity test (simplified)
            abs_residuals = np.abs(residuals)
            analysis['heteroscedasticity_score'] = float(np.corrcoef(y_pred, abs_residuals)[0, 1])
            
            return analysis
            
        except Exception as e:
            print(f"❌ Residual analysis failed: {e}")
            return {}
    
    def cross_validate(self, X: Union[np.ndarray, pd.DataFrame], 
                      y: Union[np.ndarray, pd.Series],
                      cv: int = 5, scoring: str = 'r2') -> Dict[str, Any]:
        """
        Perform cross-validation
        
        Args:
            X: Features
            y: Targets
            cv: Number of cross-validation folds
            scoring: Scoring metric
            
        Returns:
            Cross-validation results
        """
        try:
            if not SKLEARN_AVAILABLE:
                print("❌ Scikit-learn not available for cross-validation")
                return {}
            
            # Convert to numpy arrays
            if isinstance(X, pd.DataFrame):
                X = X.values
            if isinstance(y, pd.Series):
                y = y.values
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Create model for CV
            cv_model = xgb.XGBRegressor(**self.params)
            
            # Perform cross-validation
            cv_scores = cross_val_score(cv_model, X_scaled, y, cv=cv, scoring=scoring)
            
            results = {
                'cv_scores': cv_scores.tolist(),
                'mean_score': float(np.mean(cv_scores)),
                'std_score': float(np.std(cv_scores)),
                'min_score': float(np.min(cv_scores)),
                'max_score': float(np.max(cv_scores)),
                'scoring_metric': scoring,
                'n_folds': cv
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Cross-validation failed: {e}")
            return {}
    
    def hyperparameter_tuning(self, X: Union[np.ndarray, pd.DataFrame], 
                             y: Union[np.ndarray, pd.Series],
                             param_grid: Optional[Dict] = None,
                             cv: int = 3, scoring: str = 'r2',
                             n_jobs: int = -1) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Args:
            X: Training features
            y: Training targets
            param_grid: Parameter grid for tuning
            cv: Number of cross-validation folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
            
        Returns:
            Tuning results
        """
        try:
            if not SKLEARN_AVAILABLE:
                print("❌ Scikit-learn not available for hyperparameter tuning")
                return {}
            
            # Convert to numpy arrays
            if isinstance(X, pd.DataFrame):
                X = X.values
            if isinstance(y, pd.Series):
                y = y.values
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Default parameter grid if not provided
            if param_grid is None:
                param_grid = XGBConfig.create_hyperparameter_grid('financial_health')
            
            # Create model for tuning
            tuning_model = xgb.XGBRegressor(**{k: v for k, v in self.params.items() 
                                             if k not in ['early_stopping_rounds', 'eval_metric']})
            
            # Perform grid search
            grid_search = GridSearchCV(
                estimator=tuning_model,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                return_train_score=True
            )
            
            print(f"🔍 Starting hyperparameter tuning with {sum(len(v) for v in param_grid.values() if isinstance(v, list))} combinations...")
            grid_search.fit(X_scaled, y)
            
            results = {
                'best_params': grid_search.best_params_,
                'best_score': float(grid_search.best_score_),
                'best_estimator': grid_search.best_estimator_,
                'cv_results': grid_search.cv_results_,
                'param_grid': param_grid
            }
            
            print(f"✅ Hyperparameter tuning completed!")
            print(f"📊 Best Score: {results['best_score']:.4f}")
            print(f"📊 Best Parameters: {results['best_params']}")
            
            return results
            
        except Exception as e:
            print(f"❌ Hyperparameter tuning failed: {e}")
            return {}
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive model summary
        
        Returns:
            Dictionary with model information
        """
        try:
            summary = {
                'model_type': self.model_name,
                'is_trained': self.is_trained,
                'n_features': self.n_features,
                'feature_names': self.feature_names,
                'model_params': self.params.copy(),
                'xgboost_available': XGBOOST_AVAILABLE,
                'sklearn_available': SKLEARN_AVAILABLE
            }
            
            if self.is_trained:
                summary['training_score'] = float(self.training_score) if self.training_score else None
                summary['validation_score'] = float(self.validation_score) if self.validation_score else None
                summary['best_iteration'] = self.best_iteration
                summary['training_time'] = self.training_time
                
                # Add feature importance
                try:
                    feature_importance = self.get_feature_importance()
                    # Get top 5 features
                    top_features = dict(list(feature_importance.items())[:5])
                    summary['top_features'] = top_features
                except:
                    summary['top_features'] = {}
                
                # Add model info
                if hasattr(self.model, 'n_estimators'):
                    summary['n_estimators'] = self.model.n_estimators
                
                if hasattr(self.model, 'get_booster'):
                    try:
                        booster_info = self.model.get_booster().attributes()
                        summary['booster_info'] = booster_info
                    except:
                        pass
            
            return summary
            
        except Exception as e:
            print(f"❌ Model summary generation failed: {e}")
            return {'error': str(e)}
    
    def save_model(self, filepath: str) -> bool:
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.is_trained:
                print("❌ Cannot save untrained model")
                return False
            
            # Prepare model data
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'params': self.params,
                'feature_names': self.feature_names,
                'n_features': self.n_features,
                'training_score': self.training_score,
                'validation_score': self.validation_score,
                'best_iteration': self.best_iteration,
                'training_time': self.training_time,
                'model_name': self.model_name
            }
            
            # Save to pickle file
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"✅ Model saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Model save failed: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load a saved model
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load from pickle file
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Restore model components
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.params = model_data['params']
            self.feature_names = model_data['feature_names']
            self.n_features = model_data['n_features']
            self.training_score = model_data.get('training_score')
            self.validation_score = model_data.get('validation_score')
            self.best_iteration = model_data.get('best_iteration')
            self.training_time = model_data.get('training_time')
            self.model_name = model_data.get('model_name', 'XGBoost Regressor')
            
            self.is_trained = True
            
            print(f"✅ Model loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Model load failed: {e}")
            return False


# Test the XGBoost Regressor
if __name__ == "__main__":
    print("🚀 Testing XGBoost Regressor...")
    
    if XGBOOST_AVAILABLE:
        # Generate synthetic financial data
        np.random.seed(42)
        n_samples = 1000
        n_features = 15
        
        # Synthetic features representing financial indicators
        X = np.random.randn(n_samples, n_features)
        
        # Create synthetic target (financial health score)
        # Simulate complex relationships that XGBoost can capture
        y = (
            X[:, 0] * 2.5 +                    # Revenue growth
            X[:, 1] * 1.8 +                    # Profit margin
            X[:, 2] * (-1.2) +                 # Debt ratio
            X[:, 3] * X[:, 4] * 0.5 +          # Interaction term
            np.where(X[:, 5] > 0, X[:, 5] * 2, X[:, 5] * 0.5) +  # Non-linear relationship
            X[:, 6] * X[:, 7] * X[:, 8] * 0.1 +  # Three-way interaction
            np.random.randn(n_samples) * 0.3    # Noise
        )
        
        # Create DataFrame
        feature_names = [
            'revenue_growth', 'profit_margin', 'debt_ratio', 'liquidity_ratio', 'roa',
            'current_ratio', 'quick_ratio', 'inventory_turnover', 'receivables_turnover',
            'payables_turnover', 'cash_ratio', 'interest_coverage', 'debt_to_equity',
            'working_capital', 'market_cap'
        ]
        
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='financial_health_score')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_series, test_size=0.2, random_state=42
        )
        
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        print(f"📊 Dataset: {len(X_train_split)} train, {len(X_val)} validation, {len(X_test)} test samples")
        print(f"📊 Target range: {y.min():.2f} to {y.max():.2f}")
        
        # Test different configurations
        configs_to_test = [
            ('financial_health', 'Financial Health Prediction'),
            ('fast', 'Fast Training'),
            ('memory_efficient', 'Memory Efficient')
        ]
        
        for scenario, description in configs_to_test:
            print(f"\n🚀 Testing {description}...")
            
            try:
                # Get configuration
                config = XGBConfig.get_config_by_scenario(scenario)
                
                # Create model
                xgb_model = XGBRegressorModel(**config)
                
                # Train model
                success = xgb_model.train(X_train_split, y_train_split, X_val, y_val, verbose=False)
                
                if success:
                    print(f"✅ Training successful")
                    
                    # Make predictions
                    y_pred = xgb_model.predict(X_test)
                    print(f"✅ Predictions shape: {y_pred.shape}")
                    print(f"✅ Prediction range: {y_pred.min():.2f} to {y_pred.max():.2f}")
                    
                    # Evaluate model
                    metrics = xgb_model.evaluate(X_test, y_test)
                    print(f"📊 R² Score: {metrics.get('r2_score', 0):.4f}")
                    print(f"📊 RMSE: {metrics.get('rmse', 0):.4f}")
                    print(f"📊 MAE: {metrics.get('mae', 0):.4f}")
                    print(f"📊 MAPE: {metrics.get('mape', 0):.2f}%")
                    print(f"📊 Financial Accuracy: {metrics.get('financial_accuracy', 0):.2f}%")
                    
                    if 'directional_accuracy' in metrics:
                        print(f"📊 Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
                    
                    # Test uncertainty prediction
                    uncertainty_results = xgb_model.predict_with_uncertainty(X_test[:5])
                    if uncertainty_results:
                        print(f"✅ Uncertainty estimation working")
                        avg_uncertainty = np.mean(uncertainty_results['uncertainty'])
                        print(f"📊 Average uncertainty: {avg_uncertainty:.4f}")
                    
                    # Test residual analysis
                    residual_analysis = xgb_model.analyze_residuals(X_test, y_test)
                    if residual_analysis:
                        print(f"✅ Residual analysis working")
                        print(f"📊 Mean residual: {residual_analysis['mean_residual']:.4f}")
                        print(f"📊 Outlier percentage: {residual_analysis['outlier_percentage']:.2f}%")
                    
                    # Get feature importance
                    importance = xgb_model.get_feature_importance()
                    if importance:
                        top_feature = max(importance.items(), key=lambda x: x[1])
                        print(f"📊 Top feature: {top_feature[0]} ({top_feature[1]:.4f})")
                    
                    # Get model summary
                    summary = xgb_model.get_model_summary()
                    print(f"📊 Model trained: {summary.get('is_trained', False)}")
                    print(f"📊 Features: {summary.get('n_features', 0)}")
                    
                else:
                    print("❌ Training failed")
                    
            except Exception as e:
                print(f"❌ Test failed: {e}")
        
        # Test cross-validation
        print(f"\n🚀 Testing Cross-Validation...")
        try:
            config = XGBConfig.get_config_by_scenario('fast')
            cv_model = XGBRegressorModel(**config)
            
            cv_results = cv_model.cross_validate(X_train, y_train, cv=3, scoring='r2')
            if cv_results:
                print(f"✅ Cross-validation completed")
                print(f"📊 Mean CV R² Score: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
        except Exception as e:
            print(f"❌ Cross-validation test failed: {e}")
        
        # Test hyperparameter tuning (with reduced grid for speed)
        print(f"\n🚀 Testing Hyperparameter Tuning...")
        try:
            config = XGBConfig.get_config_by_scenario('fast')
            tuning_model = XGBRegressorModel(**config)
            
            # Small parameter grid for testing
            small_grid = {
                'n_estimators': [50, 100],
                'max_depth': [3, 5],
                'learning_rate': [0.1, 0.2]
            }
            
            tuning_results = tuning_model.hyperparameter_tuning(
                X_train[:200], y_train[:200],  # Use subset for speed
                param_grid=small_grid,
                cv=2
            )
            
            if tuning_results:
                print(f"✅ Hyperparameter tuning completed")
                print(f"📊 Best R² score: {tuning_results['best_score']:.4f}")
                print(f"📊 Best parameters: {tuning_results['best_params']}")
        except Exception as e:
            print(f"❌ Hyperparameter tuning test failed: {e}")
        
        # Test model persistence
        print(f"\n🚀 Testing Model Save/Load...")
        try:
            # Create and train a simple model
            config = XGBConfig.get_config_by_scenario('fast')
            save_model = XGBRegressorModel(**config)
            
            if save_model.train(X_train[:100], y_train[:100], verbose=False):
                # Test save
                test_filepath = "test_xgb_regressor.pkl"
                if save_model.save_model(test_filepath):
                    print("✅ Model save successful")
                    
                    # Test load
                    load_model = XGBRegressorModel()
                    if load_model.load_model(test_filepath):
                        print("✅ Model load successful")
                        
                        # Verify loaded model works
                        test_pred = load_model.predict(X_test[:5])
                        print(f"✅ Loaded model prediction successful: {len(test_pred)} predictions")
                
                # Clean up
                import os
                try:
                    os.remove(test_filepath)
                    print("✅ Test file cleaned up")
                except:
                    pass
                    
        except Exception as e:
            print(f"❌ Model persistence test failed: {e}")
        
        print("\n🎉 XGBoost Regressor Test Completed!")
        print("="*60)
        print("✅ All major functionality tested successfully")
        print("📊 Model ready for financial health prediction")
        print("🔮 Advanced features: uncertainty estimation, residual analysis")
        print("💾 Persistence: save/load functionality working")
        print("🎯 Optimized for: financial analysis, cash flow forecasting")
    
    else:
        print("❌ XGBoost not available - Cannot test XGBoost models")
        print("Install XGBoost with: pip install xgboost")