"""
Cross-Validation and Model Validation Tools
"""

import numpy as np # type: ignore
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime
import json

# Import sklearn CV tools
try:
    from sklearn.model_selection import ( # type: ignore
        cross_val_score, cross_validate, StratifiedKFold, 
        KFold, TimeSeriesSplit, train_test_split
    )
    from sklearn.metrics import make_scorer # type: ignore
    SKLEARN_CV_AVAILABLE = True
except ImportError:
    SKLEARN_CV_AVAILABLE = False

class CrossValidator:
    """
    Comprehensive cross-validation and model validation
    """
    
    def __init__(self):
        self.cv_results = []
        self.custom_scorers = {}
    
    def add_custom_scorer(self, name: str, scorer_func: Callable, 
                         greater_is_better: bool = True):
        """Add custom scoring function"""
        if SKLEARN_CV_AVAILABLE:
            self.custom_scorers[name] = make_scorer(scorer_func, greater_is_better=greater_is_better)
        else:
            self.custom_scorers[name] = scorer_func
    
    def cross_validate_model(self, model, X: np.ndarray, y: np.ndarray,
                           cv_strategy: str = 'kfold', n_splits: int = 5,
                           scoring: Optional[List[str]] = None,
                           return_train_score: bool = True) -> Dict[str, Any]:
        """
        Comprehensive cross-validation
        
        Args:
            model: Model to validate
            X: Features
            y: Targets
            cv_strategy: 'kfold', 'stratified', 'timeseries'
            n_splits: Number of CV splits
            scoring: List of scoring metrics
            return_train_score: Whether to return train scores
        
        Returns:
            CV results dictionary
        """
        try:
            if not SKLEARN_CV_AVAILABLE:
                return self._manual_cross_validation(model, X, y, n_splits)
            
            # Set up CV strategy
            cv_splitter = self._get_cv_splitter(cv_strategy, n_splits, y)
            
            # Default scoring
            if scoring is None:
                if hasattr(model, 'predict_proba'):
                    scoring = ['accuracy', 'f1_weighted']
                    try:
                        # Try to add ROC AUC for binary classification
                        if len(np.unique(y)) == 2:
                            scoring.append('roc_auc')
                        else:
                            scoring.append('roc_auc_ovr_weighted')
                    except:
                        pass
                else:
                    scoring = ['neg_mean_squared_error', 'r2', 'neg_mean_absolute_error']
            
            # Perform cross-validation
            cv_results = cross_validate(
                model, X, y,
                cv=cv_splitter,
                scoring=scoring,
                return_train_score=return_train_score,
                n_jobs=1  # Changed from -1 to avoid issues
            )
            
            # Process results
            processed_results = self._process_cv_results(cv_results, scoring)
            
            # Add metadata
            processed_results['cv_strategy'] = cv_strategy
            processed_results['n_splits'] = n_splits
            processed_results['n_samples'] = len(X)
            processed_results['n_features'] = X.shape[1]
            processed_results['timestamp'] = datetime.now().isoformat()
            
            # Store results
            self.cv_results.append(processed_results)
            
            return processed_results
            
        except Exception as e:
            print(f"❌ Cross-validation failed: {e}")
            return self._manual_cross_validation(model, X, y, n_splits)
    
    def _get_cv_splitter(self, strategy: str, n_splits: int, y: np.ndarray):
        """Get appropriate CV splitter"""
        if strategy == 'stratified':
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        elif strategy == 'timeseries':
            return TimeSeriesSplit(n_splits=n_splits)
        else:  # kfold
            return KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    def _process_cv_results(self, cv_results: Dict, scoring: List[str]) -> Dict[str, Any]:
        """Process cross-validation results"""
        processed = {}
        
        for metric in scoring:
            test_key = f'test_{metric}'
            train_key = f'train_{metric}'
            
            if test_key in cv_results:
                test_scores = cv_results[test_key]
                processed[f'{metric}_test_mean'] = float(np.mean(test_scores))
                processed[f'{metric}_test_std'] = float(np.std(test_scores))
                processed[f'{metric}_test_scores'] = test_scores.tolist()
            
            if train_key in cv_results:
                train_scores = cv_results[train_key]
                processed[f'{metric}_train_mean'] = float(np.mean(train_scores))
                processed[f'{metric}_train_std'] = float(np.std(train_scores))
                processed[f'{metric}_train_scores'] = train_scores.tolist()
        
        # Fit times
        if 'fit_time' in cv_results:
            processed['fit_time_mean'] = float(np.mean(cv_results['fit_time']))
            processed['fit_time_std'] = float(np.std(cv_results['fit_time']))
        
        if 'score_time' in cv_results:
            processed['score_time_mean'] = float(np.mean(cv_results['score_time']))
            processed['score_time_std'] = float(np.std(cv_results['score_time']))
        
        return processed
    
    def _manual_cross_validation(self, model, X: np.ndarray, y: np.ndarray, 
                                n_splits: int) -> Dict[str, Any]:
        """Manual cross-validation when sklearn not available"""
        try:
            fold_size = len(X) // n_splits
            scores = []
            
            is_classification = hasattr(model, 'predict_proba') or len(np.unique(y)) < 20
            
            for i in range(n_splits):
                # Create train/test split
                start_idx = i * fold_size
                end_idx = (i + 1) * fold_size if i < n_splits - 1 else len(X)
                
                test_indices = np.arange(start_idx, end_idx)
                train_indices = np.concatenate([np.arange(0, start_idx), np.arange(end_idx, len(X))])
                
                X_train, X_test = X[train_indices], X[test_indices]
                y_train, y_test = y[train_indices], y[test_indices]
                
                # Train and predict
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Calculate score
                if is_classification:
                    score = np.mean(y_test == y_pred)  # Accuracy
                else:
                    score = 1 - np.mean((y_test - y_pred) ** 2) / (np.var(y_test) + 1e-10)  # R2-like
                
                scores.append(score)
            
            return {
                'manual_cv_mean': float(np.mean(scores)),
                'manual_cv_std': float(np.std(scores)),
                'manual_cv_scores': scores,
                'n_splits': n_splits,
                'metric_type': 'classification' if is_classification else 'regression'
            }
            
        except Exception as e:
            print(f"❌ Manual cross-validation failed: {e}")
            return {}
    
    def holdout_validation(self, model, X: np.ndarray, y: np.ndarray,
                          test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """Simple holdout validation"""
        try:
            if SKLEARN_CV_AVAILABLE:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state,
                    stratify=y if len(np.unique(y)) < 20 else None
                )
            else:
                # Manual split
                n_test = int(len(X) * test_size)
                np.random.seed(random_state)
                test_indices = np.random.choice(len(X), n_test, replace=False)
                train_indices = np.setdiff1d(np.arange(len(X)), test_indices)
                
                X_train, X_test = X[train_indices], X[test_indices]
                y_train, y_test = y[train_indices], y[test_indices]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            if hasattr(model, 'predict_proba'):
                from .metrics import ClassificationMetrics
                metrics_calc = ClassificationMetrics()
                results = metrics_calc.calculate_all_metrics(y_test, y_pred, y_prob)
            else:
                from .metrics import RegressionMetrics
                metrics_calc = RegressionMetrics()
                results = metrics_calc.calculate_all_metrics(y_test, y_pred)
            
            results['validation_type'] = 'holdout'
            results['test_size'] = test_size
            results['train_size'] = len(X_train)
            results['test_size_actual'] = len(X_test)
            
            return results
            
        except Exception as e:
            print(f"❌ Holdout validation failed: {e}")
            return {}