# =====================================
# File 2: ml_algorithms/model_evaluation/metrics.py
# =====================================

"""
Comprehensive Metrics Calculator
Regression, Classification, and Custom Financial Metrics
"""

import numpy as np # type: ignore
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json

# Import sklearn metrics
try:
    from sklearn.metrics import ( # type: ignore
        # Regression metrics
        mean_squared_error, mean_absolute_error, r2_score,
        mean_absolute_percentage_error,
        
        # Classification metrics
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report,
        log_loss, roc_curve, precision_recall_curve,
        
        # Additional metrics
        explained_variance_score, max_error
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class ModelMetrics:
    """
    Base class for model evaluation metrics
    """
    
    def __init__(self):
        self.metrics_history = []
        self.custom_metrics = {}
    
    def calculate_basic_stats(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate basic statistical measures"""
        try:
            residuals = y_true - y_pred
            
            return {
                'mean_true': float(np.mean(y_true)),
                'mean_pred': float(np.mean(y_pred)),
                'std_true': float(np.std(y_true)),
                'std_pred': float(np.std(y_pred)),
                'mean_residual': float(np.mean(residuals)),
                'std_residual': float(np.std(residuals)),
                'min_residual': float(np.min(residuals)),
                'max_residual': float(np.max(residuals)),
                'residual_skewness': float(self._calculate_skewness(residuals)),
                'residual_kurtosis': float(self._calculate_kurtosis(residuals))
            }
        except Exception as e:
            print(f"❌ Basic stats calculation failed: {e}")
            return {}
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            return np.mean(((data - mean) / std) ** 3) if std > 0 else 0.0
        except:
            return 0.0
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            return np.mean(((data - mean) / std) ** 4) - 3 if std > 0 else 0.0
        except:
            return 0.0
    
    def add_custom_metric(self, name: str, func: callable):
        """Add custom metric function"""
        self.custom_metrics[name] = func
    
    def calculate_custom_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate all custom metrics"""
        custom_results = {}
        for name, func in self.custom_metrics.items():
            try:
                custom_results[name] = float(func(y_true, y_pred))
            except Exception as e:
                print(f"⚠️ Custom metric {name} failed: {e}")
                custom_results[name] = None
        return custom_results

class RegressionMetrics(ModelMetrics):
    """
    Comprehensive regression metrics calculator
    """
    
    def __init__(self):
        super().__init__()
        
        # Add financial-specific custom metrics
        self.add_custom_metric('financial_accuracy', self._financial_accuracy)
        self.add_custom_metric('directional_accuracy', self._directional_accuracy)
        self.add_custom_metric('prediction_stability', self._prediction_stability)
    
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            sample_weight: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive regression metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values  
            sample_weight: Sample weights (optional)
        
        Returns:
            Dictionary with all regression metrics
        """
        try:
            if not SKLEARN_AVAILABLE:
                return self._fallback_regression_metrics(y_true, y_pred)
            
            metrics = {}
            
            # Core regression metrics
            metrics['mse'] = float(mean_squared_error(y_true, y_pred, sample_weight=sample_weight))
            metrics['rmse'] = float(np.sqrt(metrics['mse']))
            metrics['mae'] = float(mean_absolute_error(y_true, y_pred, sample_weight=sample_weight))
            metrics['r2_score'] = float(r2_score(y_true, y_pred, sample_weight=sample_weight))
            
            # Additional sklearn metrics
            try:
                metrics['mape'] = float(mean_absolute_percentage_error(y_true, y_pred, sample_weight=sample_weight))
            except:
                metrics['mape'] = float(self._manual_mape(y_true, y_pred))
            
            try:
                metrics['explained_variance'] = float(explained_variance_score(y_true, y_pred, sample_weight=sample_weight))
                metrics['max_error'] = float(max_error(y_true, y_pred))
            except:
                metrics['explained_variance'] = None
                metrics['max_error'] = float(np.max(np.abs(y_true - y_pred)))
            
            # Custom regression metrics
            metrics['median_ae'] = float(np.median(np.abs(y_true - y_pred)))
            metrics['mean_percentage_error'] = float(self._mean_percentage_error(y_true, y_pred))
            metrics['symmetric_mape'] = float(self._symmetric_mape(y_true, y_pred))
            metrics['normalized_rmse'] = float(self._normalized_rmse(y_true, y_pred))
            metrics['adjusted_r2'] = float(self._adjusted_r2(y_true, y_pred, n_features=1))
            
            # Statistical measures
            basic_stats = self.calculate_basic_stats(y_true, y_pred)
            metrics.update(basic_stats)
            
            # Custom financial metrics
            custom_metrics = self.calculate_custom_metrics(y_true, y_pred)
            metrics.update(custom_metrics)
            
            # Prediction intervals
            prediction_intervals = self._calculate_prediction_intervals(y_true, y_pred)
            metrics.update(prediction_intervals)
            
            # Add metadata
            metrics['n_samples'] = len(y_true)
            metrics['metric_type'] = 'regression'
            metrics['timestamp'] = datetime.now().isoformat()
            
            # Store in history
            self.metrics_history.append(metrics.copy())
            
            return metrics
            
        except Exception as e:
            print(f"❌ Regression metrics calculation failed: {e}")
            return self._fallback_regression_metrics(y_true, y_pred)
    
    def _fallback_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Fallback metrics when sklearn not available"""
        try:
            residuals = y_true - y_pred
            mse = np.mean(residuals ** 2)
            mae = np.mean(np.abs(residuals))
            
            # Simple R2 calculation
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                'mse': float(mse),
                'rmse': float(np.sqrt(mse)),
                'mae': float(mae),
                'r2_score': float(r2),
                'n_samples': len(y_true),
                'metric_type': 'regression_fallback'
            }
        except Exception as e:
            print(f"❌ Fallback metrics failed: {e}")
            return {}
    
    def _manual_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Manual MAPE calculation"""
        try:
            return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        except:
            return float('inf')
    
    def _mean_percentage_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Percentage Error"""
        try:
            return np.mean((y_true - y_pred) / (y_true + 1e-8)) * 100
        except:
            return 0.0
    
    def _symmetric_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Symmetric Mean Absolute Percentage Error"""
        try:
            return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
        except:
            return 0.0
    
    def _normalized_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Normalized Root Mean Square Error"""
        try:
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            return rmse / (np.max(y_true) - np.min(y_true) + 1e-8)
        except:
            return 0.0
    
    def _adjusted_r2(self, y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
        """Adjusted R-squared"""
        try:
            n = len(y_true)
            if n <= n_features + 1:
                return 0.0
            
            # Simple R2 calculation
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Adjusted R2
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
            return adj_r2
        except:
            return 0.0
    
    def _calculate_prediction_intervals(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate prediction intervals"""
        try:
            residuals = y_true - y_pred
            std_residual = np.std(residuals)
            
            # 95% prediction interval
            pi_lower = y_pred - 1.96 * std_residual
            pi_upper = y_pred + 1.96 * std_residual
            
            # Coverage probability
            coverage = np.mean((y_true >= pi_lower) & (y_true <= pi_upper))
            
            return {
                'prediction_interval_width': float(np.mean(pi_upper - pi_lower)),
                'prediction_coverage_95': float(coverage),
                'residual_std': float(std_residual)
            }
        except Exception as e:
            print(f"⚠️ Prediction intervals calculation failed: {e}")
            return {}
    
    def _financial_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Financial accuracy metric (within 10% tolerance)"""
        try:
            tolerance = 0.1  # 10%
            relative_error = np.abs((y_true - y_pred) / (y_true + 1e-8))
            return np.mean(relative_error <= tolerance) * 100
        except:
            return 0.0
    
    def _directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Directional accuracy (trend prediction)"""
        try:
            if len(y_true) < 2:
                return 0.0
            
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            
            return np.mean(true_direction == pred_direction) * 100
        except:
            return 0.0
    
    def _prediction_stability(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Prediction stability metric"""
        try:
            if len(y_pred) < 2:
                return 0.0
            
            pred_volatility = np.std(np.diff(y_pred))
            true_volatility = np.std(np.diff(y_true))
            
            return 1.0 - abs(pred_volatility - true_volatility) / (true_volatility + 1e-8)
        except:
            return 0.0

class ClassificationMetrics(ModelMetrics):
    """
    Comprehensive classification metrics calculator
    """
    
    def __init__(self):
        super().__init__()
        
        # Add financial-specific custom metrics
        self.add_custom_metric('risk_precision', self._risk_precision)
        self.add_custom_metric('conservative_accuracy', self._conservative_accuracy)
    
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                            y_prob: Optional[np.ndarray] = None,
                            class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            class_names: Class names (optional)
        
        Returns:
            Dictionary with all classification metrics
        """
        try:
            if not SKLEARN_AVAILABLE:
                return self._fallback_classification_metrics(y_true, y_pred)
            
            metrics = {}
            
            # Basic metrics
            metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
            metrics['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
            metrics['precision_weighted'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
            metrics['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
            metrics['recall_weighted'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
            metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
            metrics['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Probability-based metrics
            if y_prob is not None:
                prob_metrics = self._calculate_probability_metrics(y_true, y_prob)
                metrics.update(prob_metrics)
            
            # Binary vs multi-class specific metrics
            if len(np.unique(y_true)) == 2:
                binary_metrics = self._calculate_binary_metrics(y_true, y_pred, y_prob)
                metrics.update(binary_metrics)
            else:
                multiclass_metrics = self._calculate_multiclass_metrics(y_true, y_pred, y_prob)
                metrics.update(multiclass_metrics)
            
            # Class distribution
            unique, counts = np.unique(y_true, return_counts=True)
            metrics['class_distribution'] = dict(zip(unique.tolist(), counts.tolist()))
            
            # Custom metrics
            custom_metrics = self.calculate_custom_metrics(y_true, y_pred)
            metrics.update(custom_metrics)
            
            # Add metadata
            metrics['n_samples'] = len(y_true)
            metrics['n_classes'] = len(np.unique(y_true))
            metrics['metric_type'] = 'classification'
            metrics['timestamp'] = datetime.now().isoformat()
            
            # Store in history
            self.metrics_history.append(metrics.copy())
            
            return metrics
            
        except Exception as e:
            print(f"❌ Classification metrics calculation failed: {e}")
            return self._fallback_classification_metrics(y_true, y_pred)
    
    def _fallback_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Fallback metrics when sklearn not available"""
        try:
            accuracy = np.mean(y_true == y_pred)
            
            return {
                'accuracy': float(accuracy),
                'n_samples': len(y_true),
                'n_classes': len(np.unique(y_true)),
                'metric_type': 'classification_fallback'
            }
        except Exception as e:
            print(f"❌ Fallback classification metrics failed: {e}")
            return {}
    
    def _calculate_probability_metrics(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """Calculate probability-based metrics"""
        try:
            metrics = {}
            
            # Log loss
            try:
                metrics['log_loss'] = float(log_loss(y_true, y_prob))
            except:
                metrics['log_loss'] = None
            
            # Brier score (for binary classification)
            if y_prob.shape[1] == 2:
                brier_score = np.mean((y_prob[:, 1] - y_true) ** 2)
                metrics['brier_score'] = float(brier_score)
            
            return metrics
            
        except Exception as e:
            print(f"⚠️ Probability metrics calculation failed: {e}")
            return {}
    
    def _calculate_binary_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate binary classification specific metrics"""
        try:
            metrics = {}
            
            # Binary AUC
            if y_prob is not None:
                try:
                    auc = roc_auc_score(y_true, y_prob[:, 1])
                    metrics['auc_roc'] = float(auc)
                except:
                    pass
            
            # Sensitivity and Specificity
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                
                metrics['sensitivity'] = float(sensitivity)
                metrics['specificity'] = float(specificity)
                metrics['false_positive_rate'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            
            return metrics
            
        except Exception as e:
            print(f"⚠️ Binary metrics calculation failed: {e}")
            return {}
    
    def _calculate_multiclass_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate multi-class specific metrics"""
        try:
            metrics = {}
            
            # Balanced accuracy
            cm = confusion_matrix(y_true, y_pred)
            per_class_accuracy = np.diag(cm) / (np.sum(cm, axis=1) + 1e-10)
            metrics['balanced_accuracy'] = float(np.mean(per_class_accuracy))
            
            # Multi-class AUC
            if y_prob is not None:
                try:
                    auc_ovr = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
                    metrics['auc_ovr_macro'] = float(auc_ovr)
                except:
                    pass
            
            return metrics
            
        except Exception as e:
            print(f"⚠️ Multi-class metrics calculation failed: {e}")
            return {}
    
    def _risk_precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Risk-weighted precision for financial applications"""
        try:
            unique_classes = np.unique(y_true)
            weights = np.ones(len(unique_classes))
            
            # Higher weights for higher risk classes
            for i in range(len(weights)):
                weights[i] = 1.0 + (i * 0.5)
            
            weighted_correct = 0
            weighted_total = 0
            
            for i, class_label in enumerate(unique_classes):
                mask = y_pred == class_label
                correct = np.sum((y_true == class_label) & mask)
                total = np.sum(mask)
                
                if total > 0:
                    weighted_correct += correct * weights[i]
                    weighted_total += total * weights[i]
            
            return weighted_correct / weighted_total if weighted_total > 0 else 0.0
            
        except:
            return 0.0
    
    def _conservative_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Conservative accuracy (penalizes high-risk misclassifications)"""
        try:
            n_classes = len(np.unique(y_true))
            penalty = np.ones((n_classes, n_classes))
            
            # Higher penalty for predicting low risk when true risk is high
            for i in range(n_classes):
                for j in range(n_classes):
                    if i > j:  # True class higher risk than predicted
                        penalty[i, j] = 2.0 + (i - j) * 0.5
            
            total_penalty = 0
            n_samples = len(y_true)
            
            for k in range(n_samples):
                true_class = int(y_true[k])
                pred_class = int(y_pred[k])
                if true_class < n_classes and pred_class < n_classes:
                    total_penalty += penalty[true_class, pred_class]
            
            # Convert penalty to accuracy (lower penalty = higher accuracy)
            max_penalty = n_samples * np.max(penalty)
            conservative_acc = 1.0 - (total_penalty - n_samples) / (max_penalty - n_samples + 1e-10)
            
            return max(0.0, conservative_acc)
            
        except:
            return 0.0