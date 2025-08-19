"""
LightGBM Classifier for Financial Risk Categorization
High-performance gradient boosting for classification tasks
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
from typing import Dict, Any, Optional, List, Tuple
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

# Import LightGBM and sklearn
try:
    import lightgbm as lgb
    from sklearn.metrics import ( # type: ignore
        accuracy_score, classification_report, confusion_matrix, 
        roc_auc_score, f1_score, precision_score, recall_score, 
        log_loss, roc_curve, precision_recall_curve
    )
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

class LGBClassifierModel(BaseMLModel):
    """
    LightGBM Classifier optimized for financial risk categorization
    
    Features:
    - Multi-class classification support
    - Class imbalance handling
    - Feature importance analysis
    - Probability predictions
    - Early stopping support
    """
    
    def __init__(self, **kwargs):
        super().__init__("LightGBM_Classifier")
        
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available. Install with: pip install lightgbm")
        
        self.early_stopping_rounds = None
        self.eval_results = {}
        self.best_iteration = None
        self.class_names = None
        self.n_classes = None
        
        self.create_model(**kwargs)
    
    def create_model(self, **kwargs):
        """Create LightGBM classifier with optimized financial parameters"""
        
        # Default parameters optimized for financial classification
        default_params = {
            # Core parameters
            'objective': 'multiclass',
            'metric': 'multi_logloss',
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
            
            # Class imbalance handling
            'class_weight': 'balanced',
            'is_unbalance': True,
            
            # Performance
            'n_jobs': -1,
            'random_state': 42,
            'verbose': -1,
            
            # Advanced parameters
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
        self.model = lgb.LGBMClassifier(**default_params)
        self.model_params = default_params
        
        print(f"✅ {self.model_name} created")
        print(f"   Objective: {default_params['objective']}")
        print(f"   Num leaves: {default_params['num_leaves']}")
        print(f"   Max depth: {default_params['max_depth']}")
        print(f"   Learning rate: {default_params['learning_rate']}")
        print(f"   Class balanced: {default_params.get('class_weight') == 'balanced'}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None,
              eval_set: Optional[List[Tuple]] = None,
              class_names: Optional[List[str]] = None) -> bool:
        """
        Train LightGBM classifier with optional validation
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            eval_set: Custom evaluation set (optional)
            class_names: Names for classes (optional)
        """
        try:
            start_time = time.time()
            
            # Store class information
            unique_classes = np.unique(y_train)
            self.n_classes = len(unique_classes)
            
            if class_names and len(class_names) == self.n_classes:
                self.class_names = class_names
            else:
                self.class_names = [f'Class_{i}' for i in range(self.n_classes)]
            
            print(f"🎯 Training for {self.n_classes} classes: {self.class_names}")
            
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
                    eval_metric='multi_logloss',
                    early_stopping_rounds=self.early_stopping_rounds,
                    verbose=False
                )
                
                # Store best iteration
                if hasattr(self.model, 'best_iteration_'):
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
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = self.model.feature_importances_
            
            print(f"✅ {self.model_name} trained in {self.training_time:.2f} seconds")
            
            if self.best_iteration:
                print(f"   Best iteration: {self.best_iteration}")
                print(f"   Total trees: {self.model.n_estimators}")
            
            # Print class distribution
            unique, counts = np.unique(y_train, return_counts=True)
            class_dist = dict(zip(unique, counts))
            print(f"   Class distribution: {class_dist}")
            
            return True
            
        except Exception as e:
            print(f"❌ {self.model_name} training failed: {e}")
            return False
    
    def predict(self, X: np.ndarray, num_iteration: Optional[int] = None) -> np.ndarray:
        """
        Make class predictions
        
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
    
    def predict_proba(self, X: np.ndarray, num_iteration: Optional[int] = None) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            X: Features for prediction
            num_iteration: Number of trees to use (None for best iteration)
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Use best iteration if available and num_iteration not specified
        if num_iteration is None and self.best_iteration:
            num_iteration = self.best_iteration
        
        return self.model.predict_proba(X, num_iteration=num_iteration)
    
    def predict_with_confidence(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Predict with confidence scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        try:
            # Get probabilities
            probabilities = self.predict_proba(X)
            
            # Get predictions
            predictions = self.predict(X)
            
            # Calculate confidence scores
            max_probs = np.max(probabilities, axis=1)
            confidence_scores = max_probs
            
            # Entropy-based uncertainty
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
            uncertainty_scores = entropy / np.log(self.n_classes)  # Normalized entropy
            
            return {
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist(),
                'confidence_scores': confidence_scores.tolist(),
                'uncertainty_scores': uncertainty_scores.tolist(),
                'class_names': self.class_names
            }
            
        except Exception as e:
            print(f"❌ Confidence prediction failed: {e}")
            return {'predictions': self.predict(X).tolist()}
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        try:
            predictions = self.predict(X_test)
            probabilities = self.predict_proba(X_test)
            
            # Basic metrics
            accuracy = accuracy_score(y_test, predictions)
            f1_macro = f1_score(y_test, predictions, average='macro')
            f1_weighted = f1_score(y_test, predictions, average='weighted')
            precision_macro = precision_score(y_test, predictions, average='macro', zero_division=0)
            recall_macro = recall_score(y_test, predictions, average='macro', zero_division=0)
            
            # Loss
            try:
                logloss = log_loss(y_test, probabilities)
            except Exception:
                logloss = None
            
            # Confusion matrix
            conf_matrix = confusion_matrix(y_test, predictions)
            
            # Per-class metrics
            try:
                class_report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
            except Exception:
                class_report = {}
            
            # Multi-class AUC if possible
            auc_score = None
            try:
                if self.n_classes == 2:
                    auc_score = roc_auc_score(y_test, probabilities[:, 1])
                else:
                    auc_score = roc_auc_score(y_test, probabilities, multi_class='ovr', average='macro')
            except Exception as e:
                print(f"⚠️ AUC calculation failed: {e}")
            
            # Class distribution in test set
            unique_test, counts_test = np.unique(y_test, return_counts=True)
            test_distribution = dict(zip(unique_test.tolist(), counts_test.tolist()))
            
            self.performance_metrics = {
                'accuracy': float(accuracy),
                'f1_macro': float(f1_macro),
                'f1_weighted': float(f1_weighted),
                'precision_macro': float(precision_macro),
                'recall_macro': float(recall_macro),
                'log_loss': float(logloss) if logloss is not None else None,
                'auc_score': float(auc_score) if auc_score is not None else None,
                'confusion_matrix': conf_matrix.tolist(),
                'classification_report': class_report,
                'test_distribution': test_distribution,
                'n_classes': self.n_classes,
                'class_names': self.class_names,
                'model_type': 'classifier',
                'best_iteration': self.best_iteration,
                'total_estimators': getattr(self.model, 'n_estimators', 0),
                'training_time': self.training_time
            }
            
            print(f"📊 {self.model_name} Performance:")
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   F1 (macro): {f1_macro:.4f}")
            print(f"   F1 (weighted): {f1_weighted:.4f}")
            if logloss is not None:
                print(f"   Log Loss: {logloss:.4f}")
            if auc_score is not None:
                print(f"   AUC Score: {auc_score:.4f}")
            
            return self.performance_metrics
            
        except Exception as e:
            print(f"❌ {self.model_name} evaluation failed: {e}")
            return {}
    
    def get_feature_importance(self, importance_type: str = 'split') -> Dict[str, Any]:
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
                if self.feature_importance is not None:
                    importance_data['split_importance'] = self.feature_importance.tolist()
            
            if importance_type in ['gain', 'both']:
                # Get gain importance if available
                try:
                    if hasattr(self.model, 'booster_') and self.model.booster_ is not None:
                        gain_importance = self.model.booster_.feature_importance(importance_type='gain')
                        importance_data['gain_importance'] = gain_importance.tolist()
                except Exception as e:
                    print(f"⚠️ Gain importance extraction failed: {e}")
            
            return importance_data
            
        except Exception as e:
            print(f"❌ Feature importance extraction failed: {e}")
            return {'split_importance': self.feature_importance.tolist() if self.feature_importance is not None else []}
    
    def get_class_probabilities_summary(self, X_test: np.ndarray) -> Dict[str, Any]:
        """Get summary statistics for class probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        try:
            probabilities = self.predict_proba(X_test)
            
            summary = {}
            for i, class_name in enumerate(self.class_names):
                class_probs = probabilities[:, i]
                summary[class_name] = {
                    'mean_probability': float(np.mean(class_probs)),
                    'std_probability': float(np.std(class_probs)),
                    'min_probability': float(np.min(class_probs)),
                    'max_probability': float(np.max(class_probs)),
                    'median_probability': float(np.median(class_probs))
                }
            
            # Overall confidence statistics
            max_probs = np.max(probabilities, axis=1)
            summary['overall_confidence'] = {
                'mean_confidence': float(np.mean(max_probs)),
                'std_confidence': float(np.std(max_probs)),
                'low_confidence_samples': int(np.sum(max_probs < 0.6)),
                'high_confidence_samples': int(np.sum(max_probs > 0.8)),
                'total_samples': len(max_probs)
            }
            
            return summary
            
        except Exception as e:
            print(f"❌ Probability summary failed: {e}")
            return {}
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get current hyperparameters"""
        return {
            'model_params': self.model_params.copy(),
            'early_stopping_rounds': self.early_stopping_rounds,
            'best_iteration': self.best_iteration,
            'n_classes': self.n_classes,
            'class_names': self.class_names
        }
    
    def tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray, 
                           X_val: np.ndarray, y_val: np.ndarray,
                           param_grid: Optional[Dict[str, List]] = None,
                           scoring_metric: str = 'f1_macro') -> Dict[str, Any]:
        """
        Hyperparameter tuning using validation set
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            param_grid: Custom parameter grid
            scoring_metric: Metric to optimize ('accuracy', 'f1_macro', 'f1_weighted')
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
            print(f"   Optimizing for: {scoring_metric}")
            
            # Simple grid search
            param_combinations = self._generate_param_combinations(param_grid)
            
            for i, params in enumerate(param_combinations[:20]):  # Limit to 20 combinations
                try:
                    # Create temporary model
                    temp_params = self.model_params.copy()
                    temp_params.update(params)
                    
                    temp_model = lgb.LGBMClassifier(**temp_params)
                    temp_model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=self.early_stopping_rounds,
                        verbose=False
                    )
                    
                    # Evaluate
                    val_pred = temp_model.predict(X_val)
                    
                    if scoring_metric == 'accuracy':
                        val_score = accuracy_score(y_val, val_pred)
                    elif scoring_metric == 'f1_macro':
                        val_score = f1_score(y_val, val_pred, average='macro')
                    elif scoring_metric == 'f1_weighted':
                        val_score = f1_score(y_val, val_pred, average='weighted')
                    else:
                        val_score = accuracy_score(y_val, val_pred)
                    
                    tuning_results.append({
                        'params': params.copy(),
                        'score': val_score
                    })
                    
                    if val_score > best_score:
                        best_score = val_score
                        best_params = params.copy()
                    
                    print(f"   Combination {i+1}: {scoring_metric} = {val_score:.4f}")
                    
                except Exception as e:
                    print(f"   Combination {i+1} failed: {e}")
                    continue
            
            if best_params:
                # Update model with best parameters
                self.model_params.update(best_params)
                self.model = lgb.LGBMClassifier(**self.model_params)
                
                print(f"✅ Best parameters found: {best_params}")
                print(f"   Best {scoring_metric}: {best_score:.4f}")
                
                return {
                    'best_params': best_params,
                    'best_score': best_score,
                    'scoring_metric': scoring_metric,
                    'all_results': tuning_results
                }
            else:
                print("❌ No valid parameter combination found")
                return {}
                
        except Exception as e:
            print(f"❌ Hyperparameter tuning failed: {e}")
            return {}
    
    def _generate_param_combinations(self, param_grid: Dict[str, List]) -> List[Dict[str, Any]]:
        """Generate parameter combinations from grid"""
        import itertools
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
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
            
            if hasattr(self.model, 'booster_') and self.model.booster_ is not None:
                self.model.booster_.save_model(filepath)
                print(f"✅ Model saved to {filepath}")
                return True
            else:
                print("❌ Model booster not available for saving")
                return False
            
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
    
    def get_training_history(self) -> Dict[str, Any]:
        """Get training history and evaluation results"""
        return {
            'eval_results': self.eval_results,
            'best_iteration': self.best_iteration,
            'total_estimators': getattr(self.model, 'n_estimators', 0) if self.is_trained else 0,
            'training_time': self.training_time,
            'n_classes': self.n_classes,
            'class_names': self.class_names
        }
    
    def plot_confusion_matrix_data(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Get data for confusion matrix plotting
        """
        if not self.is_trained:
            return {}
        
        try:
            predictions = self.predict(X_test)
            conf_matrix = confusion_matrix(y_test, predictions)
            
            # Normalize confusion matrix
            conf_matrix_norm = conf_matrix.astype('float') / (conf_matrix.sum(axis=1)[:, np.newaxis] + 1e-10)
            
            return {
                'confusion_matrix': conf_matrix.tolist(),
                'confusion_matrix_normalized': conf_matrix_norm.tolist(),
                'class_names': self.class_names,
                'accuracy_per_class': [conf_matrix_norm[i, i] for i in range(len(self.class_names))]
            }
            
        except Exception as e:
            print(f"❌ Confusion matrix data failed: {e}")
            return {}
    
    def analyze_misclassifications(self, X_test: np.ndarray, y_test: np.ndarray, 
                                  top_n: int = 10) -> Dict[str, Any]:
        """
        Analyze misclassified samples
        
        Args:
            X_test: Test features
            y_test: True labels
            top_n: Number of top misclassifications to analyze
        """
        if not self.is_trained:
            return {}
        
        try:
            predictions = self.predict(X_test)
            probabilities = self.predict_proba(X_test)
            
            # Find misclassified samples
            misclassified_idx = np.where(predictions != y_test)[0]
            
            if len(misclassified_idx) == 0:
                return {'message': 'No misclassifications found!'}
            
            # Get confidence scores for misclassified samples
            misclassified_confidence = np.max(probabilities[misclassified_idx], axis=1)
            
            # Sort by confidence (most confident wrong predictions first)
            sorted_idx = np.argsort(misclassified_confidence)[::-1]
            top_misclassified = misclassified_idx[sorted_idx[:top_n]]
            
            analysis = {
                'total_misclassified': len(misclassified_idx),
                'misclassification_rate': len(misclassified_idx) / len(y_test),
                'top_misclassifications': []
            }
            
            for i, idx in enumerate(top_misclassified):
                try:
                    sample_analysis = {
                        'sample_index': int(idx),
                        'true_class': int(y_test[idx]),
                        'predicted_class': int(predictions[idx]),
                        'true_class_name': self.class_names[y_test[idx]] if y_test[idx] < len(self.class_names) else f'Class_{y_test[idx]}',
                        'predicted_class_name': self.class_names[predictions[idx]] if predictions[idx] < len(self.class_names) else f'Class_{predictions[idx]}',
                        'confidence': float(misclassified_confidence[sorted_idx[i]]),
                        'all_probabilities': probabilities[idx].tolist()
                    }
                    analysis['top_misclassifications'].append(sample_analysis)
                except Exception as e:
                    print(f"⚠️ Error processing misclassification {i}: {e}")
                    continue
            
            return analysis
            
        except Exception as e:
            print(f"❌ Misclassification analysis failed: {e}")
            return {}

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing LightGBM Classifier...")
    
    if LIGHTGBM_AVAILABLE:
        try:
            # Generate sample financial data
            np.random.seed(42)
            n_samples = 1000
            n_features = 10
            n_classes = 4
            
            X = np.random.randn(n_samples, n_features)
            # Create class labels based on feature combinations
            y = ((np.sum(X[:, :3], axis=1) > 0).astype(int) + 
                 (np.sum(X[:, 3:6], axis=1) > 0).astype(int) + 
                 (np.sum(X[:, 6:9], axis=1) > 0).astype(int))
            
            # Ensure all classes are represented
            y = y % n_classes
            
            # Split data
            split_idx = int(0.8 * n_samples)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Create and train model
            class_names = ['Poor', 'Fair', 'Good', 'Excellent']
            model = LGBClassifierModel(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6
            )
            
            print("\n🚀 Training model...")
            success = model.train(X_train, y_train, X_test, y_test, class_names=class_names)
            
            if success:
                print("\n📊 Evaluating model...")
                performance = model.evaluate(X_test, y_test)
                
                print("\n🎯 Making predictions...")
                predictions = model.predict(X_test[:5])
                probabilities = model.predict_proba(X_test[:5])
                print(f"Sample predictions: {predictions}")
                print(f"Sample probabilities shape: {probabilities.shape}")
                
                print("\n🔍 Feature importance...")
                importance = model.get_feature_importance()
                print(f"Feature importance available: {len(importance) > 0}")
                
                print("\n📈 Confidence analysis...")
                confidence_data = model.predict_with_confidence(X_test[:10])
                print(f"Confidence scores available: {'confidence_scores' in confidence_data}")
                
                print("\n📊 Probability summary...")
                prob_summary = model.get_class_probabilities_summary(X_test)
                print(f"Overall confidence mean: {prob_summary.get('overall_confidence', {}).get('mean_confidence', 'N/A')}")
                
                print("\n🔍 Misclassification analysis...")
                misclass_analysis = model.analyze_misclassifications(X_test, y_test, top_n=3)
                print(f"Total misclassified: {misclass_analysis.get('total_misclassified', 'N/A')}")
                
                print("\n✅ Test completed successfully!")
            else:
                print("❌ Training failed")
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ LightGBM not available for testing")