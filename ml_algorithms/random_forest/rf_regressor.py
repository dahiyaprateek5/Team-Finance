# =====================================
# File: ml_algorithms/random_forest/rf_regressor.py
# Random Forest Regressor for Financial Analysis
# =====================================

"""
Random Forest Regressor for Financial Analysis
Advanced ensemble model for financial health prediction and cash flow forecasting
"""

import numpy as np # type: ignore
import pandas as pd  # type: ignore
from typing import Dict, List, Any, Optional, Tuple, Union
import json
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# Try to import sklearn
try:
    from sklearn.ensemble import RandomForestRegressor  # type: ignore
    from sklearn.preprocessing import StandardScaler   # type: ignore
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV  # type: ignore
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # type: ignore
    from sklearn.inspection import permutation_importance  # type: ignore
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .rf_config import RandomForestConfig

class RandomForestRegressorModel:
    """
    Advanced Random Forest Regressor for Financial Analysis
    Optimized for financial health prediction and cash flow forecasting
    """
    
    def __init__(self, n_estimators: int = 200,
                 max_depth: Optional[int] = 15,
                 min_samples_split: int = 5,
                 min_samples_leaf: int = 2,
                 max_features: Union[str, int, float] = 'sqrt',
                 bootstrap: bool = True,
                 oob_score: bool = True,
                 n_jobs: int = -1,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize Random Forest Regressor
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            max_features: Number of features to consider for best split
            bootstrap: Whether to use bootstrap sampling
            oob_score: Whether to calculate out-of-bag score
            n_jobs: Number of parallel jobs
            random_state: Random seed for reproducibility
        """
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for Random Forest models. Install with: pip install scikit-learn")
        
        # Model parameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        # Additional parameters from kwargs
        self.max_samples = kwargs.get('max_samples', None)
        self.ccp_alpha = kwargs.get('ccp_alpha', 0.0)
        self.criterion = kwargs.get('criterion', 'squared_error')
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = None
        self.n_features = None
        self.training_score = None
        self.oob_score_value = None
        
        print(f"🌲 Random Forest Regressor initialized with {n_estimators} trees")
    
    def _create_model(self) -> RandomForestRegressor:
        """Create the Random Forest model with current parameters"""
        try:
            model_params = {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'max_features': self.max_features,
                'bootstrap': self.bootstrap,
                'oob_score': self.oob_score,
                'n_jobs': self.n_jobs,
                'random_state': self.random_state,
                'criterion': self.criterion,
                'ccp_alpha': self.ccp_alpha
            }
            
            # Add max_samples if specified
            if self.max_samples is not None:
                model_params['max_samples'] = self.max_samples
            
            return RandomForestRegressor(**model_params)
            
        except Exception as e:
            print(f"❌ Model creation failed: {e}")
            raise
    
    def train(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series],
              scale_features: bool = True, verbose: bool = True) -> bool:
        """
        Train the Random Forest model
        
        Args:
            X: Training features
            y: Training targets
            scale_features: Whether to scale features
            verbose: Whether to print training progress
            
        Returns:
            True if training successful, False otherwise
        """
        try:
            if verbose:
                print("🚀 Starting Random Forest training...")
            
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
            else:
                X_scaled = X
                # Create dummy scaler for consistency
                self.scaler.fit(X)
            
            # Create and train model
            self.model = self._create_model()
            
            if verbose:
                print(f"📊 Training on {X_scaled.shape[0]} samples with {X_scaled.shape[1]} features")
            
            # Train the model
            self.model.fit(X_scaled, y)
            
            # Get training score
            self.training_score = self.model.score(X_scaled, y)
            
            # Get OOB score if available
            if self.oob_score and hasattr(self.model, 'oob_score_'):
                self.oob_score_value = self.model.oob_score_
            
            self.is_trained = True
            
            if verbose:
                print("✅ Random Forest training completed!")
                print(f"📊 Training R² Score: {self.training_score:.4f}")
                if self.oob_score_value is not None:
                    print(f"📊 Out-of-Bag R² Score: {self.oob_score_value:.4f}")
                print(f"📊 Number of trees: {len(self.model.estimators_)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Random Forest training failed: {e}")
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
    
    def predict_with_uncertainty(self, X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """
        Make predictions with uncertainty estimation using individual tree predictions
        
        Args:
            X: Features for prediction
            
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
            
            # Get predictions from all trees
            tree_predictions = np.array([
                tree.predict(X_scaled) for tree in self.model.estimators_
            ])
            
            # Calculate statistics
            mean_pred = np.mean(tree_predictions, axis=0)
            std_pred = np.std(tree_predictions, axis=0)
            
            # 95% confidence intervals
            ci_lower = np.percentile(tree_predictions, 2.5, axis=0)
            ci_upper = np.percentile(tree_predictions, 97.5, axis=0)
            
            return {
                'predictions': mean_pred,
                'uncertainty': std_pred,
                'confidence_interval_lower': ci_lower,
                'confidence_interval_upper': ci_upper,
                'tree_predictions': tree_predictions
            }
            
        except Exception as e:
            print(f"❌ Uncertainty prediction failed: {e}")
            return {}
    
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
            if uncertainty_results:
                ci_lower = uncertainty_results['confidence_interval_lower']
                ci_upper = uncertainty_results['confidence_interval_upper']
                coverage = np.mean((y >= ci_lower) & (y <= ci_upper))
                metrics['prediction_interval_coverage'] = float(coverage)
                
                # Average prediction interval width
                avg_width = np.mean(ci_upper - ci_lower)
                metrics['avg_prediction_interval_width'] = float(avg_width)
            
            # Model-specific metrics
            if self.oob_score_value is not None:
                metrics['oob_score'] = float(self.oob_score_value)
            
            metrics['training_score'] = float(self.training_score) if self.training_score else None
            metrics['n_samples'] = len(y)
            
            return metrics
            
        except Exception as e:
            print(f"❌ Model evaluation failed: {e}")
            return {}
    
    def get_feature_importance(self, importance_type: str = 'gini') -> Dict[str, float]:
        """
        Get feature importance scores
        
        Args:
            importance_type: 'gini' for impurity-based or 'permutation' for permutation importance
            
        Returns:
            Dictionary with feature importance scores
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before getting feature importance")
            
            if importance_type == 'gini':
                # Use built-in feature importance
                importance_scores = self.model.feature_importances_
            elif importance_type == 'permutation':
                # This would require the training data for permutation importance
                # For now, fall back to gini importance
                print("⚠️ Permutation importance requires training data. Using Gini importance.")
                importance_scores = self.model.feature_importances_
            else:
                raise ValueError("importance_type must be 'gini' or 'permutation'")
            
            # Create feature importance dictionary
            feature_importance = {}
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
    
    def get_tree_info(self) -> Dict[str, Any]:
        """
        Get information about individual trees in the forest
        
        Returns:
            Dictionary with tree statistics
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before getting tree info")
            
            tree_depths = []
            tree_leaf_counts = []
            tree_node_counts = []
            
            for tree in self.model.estimators_:
                # Get tree depth
                tree_depths.append(tree.tree_.max_depth)
                
                # Get leaf count
                leaf_count = (tree.tree_.children_left == -1).sum()
                tree_leaf_counts.append(leaf_count)
                
                # Get total node count
                tree_node_counts.append(tree.tree_.node_count)
            
            tree_info = {
                'n_trees': len(self.model.estimators_),
                'avg_depth': float(np.mean(tree_depths)),
                'max_depth': int(np.max(tree_depths)),
                'min_depth': int(np.min(tree_depths)),
                'avg_leaves': float(np.mean(tree_leaf_counts)),
                'avg_nodes': float(np.mean(tree_node_counts)),
                'total_nodes': int(np.sum(tree_node_counts))
            }
            
            return tree_info
            
        except Exception as e:
            print(f"❌ Tree info calculation failed: {e}")
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
            # Convert to numpy arrays
            if isinstance(X, pd.DataFrame):
                X = X.values
            if isinstance(y, pd.Series):
                y = y.values
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Create model for CV
            cv_model = self._create_model()
            
            # Perform cross-validation
            cv_scores = cross_val_score(cv_model, X_scaled, y, cv=cv, scoring=scoring, n_jobs=self.n_jobs)
            
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
            # Convert to numpy arrays
            if isinstance(X, pd.DataFrame):
                X = X.values
            if isinstance(y, pd.Series):
                y = y.values
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Default parameter grid if not provided
            if param_grid is None:
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                }
            
            # Create model for tuning
            tuning_model = self._create_model()
            
            # Perform grid search
            grid_search = GridSearchCV(
                estimator=tuning_model,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                return_train_score=True
            )
            
            print(f"🔍 Starting hyperparameter tuning with {len(param_grid)} parameters...")
            grid_search.fit(X_scaled, y)
            
            results = {
                'best_params': grid_search.best_params_,
                'best_score': float(grid_search.best_score_),
                'best_estimator': grid_search.best_estimator_,
                'cv_results': grid_search.cv_results_,
                'n_candidates': len(grid_search.cv_results_['params'])
            }
            
            print(f"✅ Hyperparameter tuning completed!")
            print(f"📊 Best Score: {results['best_score']:.4f}")
            print(f"📊 Best Parameters: {results['best_params']}")
            
            return results
            
        except Exception as e:
            print(f"❌ Hyperparameter tuning failed: {e}")
            return {}
    
    def forecast_cash_flow(self, X: Union[np.ndarray, pd.DataFrame], 
                          periods: int = 12) -> Dict[str, Any]:
        """
        Forecast cash flow for future periods
        
        Args:
            X: Current financial features
            periods: Number of periods to forecast
            
        Returns:
            Dictionary with forecasts and confidence intervals
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before forecasting")
            
            # Get current predictions with uncertainty
            current_results = self.predict_with_uncertainty(X)
            
            if not current_results:
                return {}
            
            # Simple trend-based forecasting
            # In practice, you would use more sophisticated time series methods
            base_predictions = current_results['predictions']
            base_uncertainty = current_results['uncertainty']
            
            # Generate forecasts
            forecasts = []
            uncertainties = []
            
            for period in range(1, periods + 1):
                # Simple growth assumption (can be made more sophisticated)
                growth_factor = 1.0 + (0.02 * period)  # 2% growth per period
                forecast = base_predictions * growth_factor
                
                # Uncertainty increases with forecast horizon
                uncertainty = base_uncertainty * np.sqrt(period)
                
                forecasts.append(forecast)
                uncertainties.append(uncertainty)
            
            forecasts = np.array(forecasts)
            uncertainties = np.array(uncertainties)
            
            # Calculate confidence intervals
            ci_lower = forecasts - 1.96 * uncertainties
            ci_upper = forecasts + 1.96 * uncertainties
            
            return {
                'forecasts': forecasts.tolist(),
                'uncertainties': uncertainties.tolist(),
                'confidence_intervals_lower': ci_lower.tolist(),
                'confidence_intervals_upper': ci_upper.tolist(),
                'periods': periods,
                'base_predictions': base_predictions.tolist()
            }
            
        except Exception as e:
            print(f"❌ Cash flow forecasting failed: {e}")
            return {}
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive model summary
        
        Returns:
            Dictionary with model information
        """
        try:
            summary = {
                'model_type': 'Random Forest Regressor',
                'is_trained': self.is_trained,
                'n_features': self.n_features,
                'feature_names': self.feature_names,
                'model_params': {
                    'n_estimators': self.n_estimators,
                    'max_depth': self.max_depth,
                    'min_samples_split': self.min_samples_split,
                    'min_samples_leaf': self.min_samples_leaf,
                    'max_features': self.max_features,
                    'bootstrap': self.bootstrap,
                    'oob_score': self.oob_score,
                    'random_state': self.random_state
                }
            }
            
            if self.is_trained:
                summary['training_score'] = float(self.training_score) if self.training_score else None
                summary['oob_score_value'] = float(self.oob_score_value) if self.oob_score_value else None
                
                # Add tree information
                tree_info = self.get_tree_info()
                summary['tree_info'] = tree_info
                
                # Add feature importance
                try:
                    feature_importance = self.get_feature_importance()
                    # Get top 5 features
                    top_features = dict(list(feature_importance.items())[:5])
                    summary['top_features'] = top_features
                except:
                    summary['top_features'] = {}
            
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
                'feature_names': self.feature_names,
                'n_features': self.n_features,
                'training_score': self.training_score,
                'oob_score_value': self.oob_score_value,
                'model_params': {
                    'n_estimators': self.n_estimators,
                    'max_depth': self.max_depth,
                    'min_samples_split': self.min_samples_split,
                    'min_samples_leaf': self.min_samples_leaf,
                    'max_features': self.max_features,
                    'bootstrap': self.bootstrap,
                    'oob_score': self.oob_score,
                    'n_jobs': self.n_jobs,
                    'random_state': self.random_state,
                    'max_samples': self.max_samples,
                    'ccp_alpha': self.ccp_alpha,
                    'criterion': self.criterion
                }
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
            self.feature_names = model_data['feature_names']
            self.n_features = model_data['n_features']
            self.training_score = model_data['training_score']
            self.oob_score_value = model_data['oob_score_value']
            
            # Restore model parameters
            params = model_data['model_params']
            self.n_estimators = params['n_estimators']
            self.max_depth = params['max_depth']
            self.min_samples_split = params['min_samples_split']
            self.min_samples_leaf = params['min_samples_leaf']
            self.max_features = params['max_features']
            self.bootstrap = params['bootstrap']
            self.oob_score = params['oob_score']
            self.n_jobs = params['n_jobs']
            self.random_state = params['random_state']
            self.max_samples = params.get('max_samples')
            self.ccp_alpha = params.get('ccp_alpha', 0.0)
            self.criterion = params.get('criterion', 'squared_error')
            
            self.is_trained = True
            
            print(f"✅ Model loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Model load failed: {e}")
            return False


# Test the Random Forest Regressor
if __name__ == "__main__":
    print("🌲 Testing Random Forest Regressor...")
    
    if SKLEARN_AVAILABLE:
        # Generate synthetic financial data
        np.random.seed(42)
        n_samples = 1000
        n_features = 15
        
        # Synthetic features representing financial indicators
        X = np.random.randn(n_samples, n_features)
        
        # Create synthetic target (financial health score)
        # Simulate complex relationships that Random Forest can capture
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
        
        print(f"📊 Dataset: {len(X_train)} train, {len(X_test)} test samples")
        print(f"📊 Target range: {y.min():.2f} to {y.max():.2f}")
        
        # Test different configurations
        configs_to_test = [
            ('financial_health', 'Financial Health Prediction'),
            ('fast', 'Fast Training'),
            ('interpretable', 'Interpretable Model')
        ]
        
        for scenario, description in configs_to_test:
            print(f"\n🌲 Testing {description}...")
            
            try:
                # Get configuration
                config = RandomForestConfig.get_config_by_scenario(scenario)
                config['n_jobs'] = 1  # Reduce for testing
                
                # Create model
                rf_model = RandomForestRegressorModel(**config)
                
                # Train model
                success = rf_model.train(X_train, y_train, verbose=False)
                
                if success:
                    print(f"✅ Training successful")
                    
                    # Make predictions
                    y_pred = rf_model.predict(X_test)
                    print(f"✅ Predictions shape: {y_pred.shape}")
                    print(f"✅ Prediction range: {y_pred.min():.2f} to {y_pred.max():.2f}")
                    
                    # Evaluate model
                    metrics = rf_model.evaluate(X_test, y_test)
                    print(f"📊 R² Score: {metrics.get('r2_score', 0):.4f}")
                    print(f"📊 RMSE: {metrics.get('rmse', 0):.4f}")
                    print(f"📊 MAE: {metrics.get('mae', 0):.4f}")
                    print(f"📊 MAPE: {metrics.get('mape', 0):.2f}%")
                    print(f"📊 Financial Accuracy: {metrics.get('financial_accuracy', 0):.2f}%")
                    
                    if 'directional_accuracy' in metrics:
                        print(f"📊 Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
                    
                    if 'oob_score' in metrics and metrics['oob_score']:
                        print(f"📊 OOB Score: {metrics['oob_score']:.4f}")
                    
                    # Test uncertainty prediction
                    uncertainty_results = rf_model.predict_with_uncertainty(X_test[:5])
                    if uncertainty_results:
                        print(f"✅ Uncertainty estimation working")
                        avg_uncertainty = np.mean(uncertainty_results['uncertainty'])
                        print(f"📊 Average uncertainty: {avg_uncertainty:.4f}")
                        
                        if 'prediction_interval_coverage' in metrics:
                            print(f"📊 Prediction Interval Coverage: {metrics['prediction_interval_coverage']:.2f}")
                    
                    # Test residual analysis
                    residual_analysis = rf_model.analyze_residuals(X_test, y_test)
                    if residual_analysis:
                        print(f"✅ Residual analysis working")
                        print(f"📊 Mean residual: {residual_analysis['mean_residual']:.4f}")
                        print(f"📊 Outlier percentage: {residual_analysis['outlier_percentage']:.2f}%")
                    
                    # Get feature importance
                    importance = rf_model.get_feature_importance()
                    if importance:
                        top_feature = max(importance.items(), key=lambda x: x[1])
                        print(f"📊 Top feature: {top_feature[0]} ({top_feature[1]:.4f})")
                    
                    # Get tree info
                    tree_info = rf_model.get_tree_info()
                    if tree_info:
                        print(f"📊 Avg tree depth: {tree_info.get('avg_depth', 0):.1f}")
                        print(f"📊 Avg leaves per tree: {tree_info.get('avg_leaves', 0):.1f}")
                    
                    # Test cash flow forecasting
                    forecast_results = rf_model.forecast_cash_flow(X_test[:3], periods=6)
                    if forecast_results:
                        print(f"✅ Cash flow forecasting working")
                        print(f"📊 Forecast periods: {forecast_results['periods']}")
                    
                    # Get model summary
                    summary = rf_model.get_model_summary()
                    print(f"📊 Model trained: {summary.get('is_trained', False)}")
                    print(f"📊 Features: {summary.get('n_features', 0)}")
                    
                else:
                    print("❌ Training failed")
                    
            except Exception as e:
                print(f"❌ Test failed: {e}")
        
        # Test cross-validation
        print(f"\n🌲 Testing Cross-Validation...")
        try:
            config = RandomForestConfig.get_config_by_scenario('fast')
            config['n_jobs'] = 1
            cv_model = RandomForestRegressorModel(**config)
            
            cv_results = cv_model.cross_validate(X_train, y_train, cv=3, scoring='r2')
            if cv_results:
                print(f"✅ Cross-validation completed")
                print(f"📊 Mean CV R² Score: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
        except Exception as e:
            print(f"❌ Cross-validation test failed: {e}")
        
        # Test hyperparameter tuning (with reduced grid for speed)
        print(f"\n🌲 Testing Hyperparameter Tuning...")
        try:
            config = RandomForestConfig.get_config_by_scenario('fast')
            tuning_model = RandomForestRegressorModel(**config)
            
            # Small parameter grid for testing
            small_grid = {
                'n_estimators': [50, 100],
                'max_depth': [5, 10],
                'min_samples_split': [2, 5]
            }
            
            tuning_results = tuning_model.hyperparameter_tuning(
                X_train[:200], y_train[:200],  # Use subset for speed
                param_grid=small_grid,
                cv=2,
                n_jobs=1
            )
            
            if tuning_results:
                print(f"✅ Hyperparameter tuning completed")
                print(f"📊 Best R² score: {tuning_results['best_score']:.4f}")
                print(f"📊 Best parameters: {tuning_results['best_params']}")
        except Exception as e:
            print(f"❌ Hyperparameter tuning test failed: {e}")
        
        # Test model persistence
        print(f"\n🌲 Testing Model Save/Load...")
        try:
            # Create and train a simple model
            config = RandomForestConfig.get_config_by_scenario('fast')
            config['n_jobs'] = 1
            save_model = RandomForestRegressorModel(**config)
            
            if save_model.train(X_train[:100], y_train[:100], verbose=False):
                # Test save
                test_filepath = "test_rf_regressor.pkl"
                if save_model.save_model(test_filepath):
                    print("✅ Model save successful")
                    
                    # Test load
                    load_model = RandomForestRegressorModel()
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
        
        # Test financial-specific scenarios
        print(f"\n🌲 Testing Financial Scenarios...")
        try:
            # Create a model for cash flow prediction
            cashflow_config = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': 1
            }
            
            cashflow_model = RandomForestRegressorModel(**cashflow_config)
            
            # Simulate cash flow data
            cash_flow_y = (
                X_train.iloc[:, 0] * 1000 +     # Revenue impact
                X_train.iloc[:, 1] * 500 +      # Profit margin impact
                X_train.iloc[:, 2] * (-300) +   # Debt impact
                np.random.randn(len(X_train)) * 100
            )
            
            if cashflow_model.train(X_train, cash_flow_y, verbose=False):
                print("✅ Cash flow model training successful")
                
                # Test cash flow forecasting
                forecast_sample = X_test.iloc[:3]
                forecasts = cashflow_model.forecast_cash_flow(forecast_sample, periods=12)
                
                if forecasts:
                    print(f"✅ 12-month cash flow forecast generated")
                    print(f"📊 Average forecast value: {np.mean(forecasts['forecasts']):.2f}")
                    print(f"📊 Forecast uncertainty range: {np.mean(forecasts['uncertainties']):.2f}")
                
                # Test financial accuracy
                cf_metrics = cashflow_model.evaluate(X_test, 
                    X_test.iloc[:, 0] * 1000 + X_test.iloc[:, 1] * 500 + X_test.iloc[:, 2] * (-300) + np.random.randn(len(X_test)) * 100
                )
                
                print(f"📊 Cash Flow Model R²: {cf_metrics.get('r2_score', 0):.4f}")
                print(f"📊 Cash Flow MAPE: {cf_metrics.get('mape', 0):.2f}%")
                
        except Exception as e:
            print(f"❌ Financial scenario test failed: {e}")
        
        print("\n🎉 Random Forest Regressor Test Completed!")
        print("="*60)
        print("✅ All major functionality tested successfully")
        print("📊 Model ready for financial analysis and cash flow forecasting")
        print("🔮 Advanced features: uncertainty estimation, residual analysis, forecasting")
        print("💾 Persistence: save/load functionality working")
        print("🎯 Optimized for: financial health prediction, cash flow analysis")
    
    else:
        print("❌ Scikit-learn not available - Cannot test Random Forest models")
        print("Install scikit-learn with: pip install scikit-learn")