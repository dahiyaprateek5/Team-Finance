# =====================================
# File: ml_algorithms/random_forest/rf_classifier.py
# Random Forest Classifier for Financial Risk Assessment
# =====================================

"""
Random Forest Classifier for Financial Risk Assessment
Advanced ensemble model for risk categorization and bankruptcy prediction
"""

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from typing import Dict, List, Any, Optional, Tuple, Union
import json
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# Try to import sklearn
try:
    from sklearn.ensemble import RandomForestClassifier # type: ignore
    from sklearn.preprocessing import StandardScaler, LabelEncoder  # type: ignore
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV  # type: ignore
    from sklearn.metrics import (   # type: ignore
        accuracy_score, precision_score, recall_score, f1_score,  
        roc_auc_score, confusion_matrix, classification_report
    )
    from sklearn.inspection import permutation_importance   # type: ignore
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .rf_config import RandomForestConfig

class RandomForestClassifierModel:
    """
    Advanced Random Forest Classifier for Financial Risk Assessment
    Optimized for risk categorization and bankruptcy prediction
    """
    
    def __init__(self, n_estimators: int = 300,
                 max_depth: Optional[int] = 20,
                 min_samples_split: int = 10,
                 min_samples_leaf: int = 5,
                 max_features: Union[str, int, float] = 'sqrt',
                 bootstrap: bool = True,
                 oob_score: bool = True,
                 class_weight: Optional[Union[str, Dict]] = 'balanced',
                 n_jobs: int = -1,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize Random Forest Classifier
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            max_features: Number of features to consider for best split
            bootstrap: Whether to use bootstrap sampling
            oob_score: Whether to calculate out-of-bag score
            class_weight: Weights for classes ('balanced', dict, or None)
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
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        # Additional parameters from kwargs
        self.max_samples = kwargs.get('max_samples', None)
        self.ccp_alpha = kwargs.get('ccp_alpha', 0.0)
        self.criterion = kwargs.get('criterion', 'gini')
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.feature_names = None
        self.class_names = None
        self.n_features = None
        self.n_classes = None
        self.training_score = None
        self.oob_score_value = None
        
        print(f"🌲 Random Forest Classifier initialized with {n_estimators} trees")
    
    def _create_model(self) -> RandomForestClassifier:
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
                'class_weight': self.class_weight,
                'n_jobs': self.n_jobs,
                'random_state': self.random_state,
                'criterion': self.criterion,
                'ccp_alpha': self.ccp_alpha
            }
            
            # Add max_samples if specified
            if self.max_samples is not None:
                model_params['max_samples'] = self.max_samples
            
            return RandomForestClassifier(**model_params)
            
        except Exception as e:
            print(f"❌ Model creation failed: {e}")
            raise
    
    def train(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series],
              class_names: Optional[List[str]] = None,
              scale_features: bool = True, verbose: bool = True) -> bool:
        """
        Train the Random Forest classifier
        
        Args:
            X: Training features
            y: Training targets
            class_names: Names of classes
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
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            self.n_classes = len(self.label_encoder.classes_)
            
            # Store class names
            if class_names:
                self.class_names = class_names
            else:
                self.class_names = [f'Class_{cls}' for cls in self.label_encoder.classes_]
            
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
                print(f"📊 Classes: {self.class_names}")
                
                # Print class distribution
                unique, counts = np.unique(y, return_counts=True)
                class_dist = dict(zip(unique, counts))
                print(f"📊 Class distribution: {class_dist}")
            
            # Train the model
            self.model.fit(X_scaled, y_encoded)
            
            # Get training score
            self.training_score = self.model.score(X_scaled, y_encoded)
            
            # Get OOB score if available
            if self.oob_score and hasattr(self.model, 'oob_score_'):
                self.oob_score_value = self.model.oob_score_
            
            self.is_trained = True
            
            if verbose:
                print("✅ Random Forest training completed!")
                print(f"📊 Training Accuracy: {self.training_score:.4f}")
                if self.oob_score_value is not None:
                    print(f"📊 Out-of-Bag Accuracy: {self.oob_score_value:.4f}")
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
            Predicted class labels
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
            predictions_encoded = self.model.predict(X_scaled)
            
            # Convert back to original labels
            predictions = self.label_encoder.inverse_transform(predictions_encoded)
            
            return predictions
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return np.array([])
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            X: Features for prediction
            
        Returns:
            Prediction probabilities
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            # Convert to numpy array
            if isinstance(X, pd.DataFrame):
                X = X.values
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Get probabilities
            probabilities = self.model.predict_proba(X_scaled)
            
            return probabilities
            
        except Exception as e:
            print(f"❌ Probability prediction failed: {e}")
            return np.array([])
    
    def predict_with_confidence(self, X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """
        Make predictions with confidence estimation using tree voting
        
        Args:
            X: Features for prediction
            
        Returns:
            Dictionary with predictions, probabilities, and confidence scores
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
            
            # Get probabilities
            probabilities = self.predict_proba(X)
            
            # Calculate confidence as the agreement between trees
            n_trees = len(self.model.estimators_)
            tree_votes = np.zeros((len(X_scaled), self.n_classes))
            
            for i in range(len(X_scaled)):
                for tree_pred in tree_predictions[:, i]:
                    tree_votes[i, tree_pred] += 1
            
            # Confidence is the proportion of trees voting for the winning class
            confidence_scores = np.max(tree_votes, axis=1) / n_trees
            
            # Get final predictions
            predicted_classes = np.argmax(probabilities, axis=1)
            predicted_labels = self.label_encoder.inverse_transform(predicted_classes)
            
            return {
                'predictions': predicted_labels.tolist(),
                'probabilities': probabilities.tolist(),
                'confidence_scores': confidence_scores.tolist(),
                'tree_votes': tree_votes.tolist(),
                'class_names': self.class_names
            }
            
        except Exception as e:
            print(f"❌ Confidence prediction failed: {e}")
            return {}
    
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], 
                y: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
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
            y_prob = self.predict_proba(X)
            
            # Convert to numpy
            if isinstance(y, pd.Series):
                y = y.values
            
            # Calculate metrics
            metrics = {}
            
            # Basic classification metrics
            metrics['accuracy'] = float(accuracy_score(y, y_pred))
            metrics['precision_macro'] = float(precision_score(y, y_pred, average='macro', zero_division=0))
            metrics['precision_weighted'] = float(precision_score(y, y_pred, average='weighted', zero_division=0))
            metrics['recall_macro'] = float(recall_score(y, y_pred, average='macro', zero_division=0))
            metrics['recall_weighted'] = float(recall_score(y, y_pred, average='weighted', zero_division=0))
            metrics['f1_macro'] = float(f1_score(y, y_pred, average='macro', zero_division=0))
            metrics['f1_weighted'] = float(f1_score(y, y_pred, average='weighted', zero_division=0))
            
            # Confusion matrix
            cm = confusion_matrix(y, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            # ROC AUC
            try:
                if self.n_classes == 2:
                    # Binary classification
                    metrics['roc_auc'] = float(roc_auc_score(y, y_prob[:, 1]))
                else:
                    # Multi-class classification
                    y_encoded = self.label_encoder.transform(y)
                    metrics['roc_auc_ovr'] = float(roc_auc_score(y_encoded, y_prob, average='macro', multi_class='ovr'))
            except Exception as auc_error:
                print(f"⚠️ ROC AUC calculation failed: {auc_error}")
            
            # Financial risk-specific metrics
            if self.n_classes > 2:
                # Risk-weighted accuracy (higher penalty for misclassifying high-risk as low-risk)
                risk_weighted_correct = 0
                total_weight = 0
                
                for true_label, pred_label in zip(y, y_pred):
                    try:
                        true_idx = list(self.label_encoder.classes_).index(
                            self.label_encoder.transform([true_label])[0]
                        )
                        pred_idx = list(self.label_encoder.classes_).index(
                            self.label_encoder.transform([pred_label])[0]
                        )
                        
                        # Weight increases with risk level
                        weight = true_idx + 1
                        
                        # Higher penalty for predicting lower risk than actual
                        if true_idx > pred_idx:
                            penalty = (true_idx - pred_idx) * 0.5
                            weight *= (1 + penalty)
                        
                        if true_label == pred_label:
                            risk_weighted_correct += weight
                        
                        total_weight += weight
                    except (ValueError, IndexError):
                        continue
                
                metrics['risk_weighted_accuracy'] = float(risk_weighted_correct / total_weight) if total_weight > 0 else 0.0
            
            # Model-specific metrics
            if self.oob_score_value is not None:
                metrics['oob_score'] = float(self.oob_score_value)
            
            metrics['training_score'] = float(self.training_score) if self.training_score else None
            
            # Class distribution
            unique, counts = np.unique(y, return_counts=True)
            metrics['class_distribution'] = dict(zip(unique.tolist(), counts.tolist()))
            
            metrics['n_samples'] = len(y)
            metrics['n_classes'] = self.n_classes
            
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
    
    def get_class_probabilities_summary(self, X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """
        Get summary of class probabilities
        
        Args:
            X: Features for prediction
            
        Returns:
            Summary statistics for class probabilities
        """
        try:
            probabilities = self.predict_proba(X)
            
            summary = {
                'class_names': self.class_names,
                'n_samples': len(probabilities),
                'class_statistics': {}
            }
            
            for i, class_name in enumerate(self.class_names):
                if i < probabilities.shape[1]:
                    class_probs = probabilities[:, i]
                    summary['class_statistics'][class_name] = {
                        'mean_probability': float(np.mean(class_probs)),
                        'std_probability': float(np.std(class_probs)),
                        'min_probability': float(np.min(class_probs)),
                        'max_probability': float(np.max(class_probs)),
                        'median_probability': float(np.median(class_probs))
                    }
            
            # Overall confidence
            max_probs = np.max(probabilities, axis=1)
            summary['overall_confidence'] = {
                'mean': float(np.mean(max_probs)),
                'std': float(np.std(max_probs)),
                'low_confidence_samples': int(np.sum(max_probs < 0.6)),
                'high_confidence_samples': int(np.sum(max_probs > 0.8))
            }
            
            return summary
            
        except Exception as e:
            print(f"❌ Probability summary failed: {e}")
            return {}
    
    def analyze_misclassifications(self, X: Union[np.ndarray, pd.DataFrame],
                                 y: Union[np.ndarray, pd.Series],
                                 top_n: int = 10) -> Dict[str, Any]:
        """
        Analyze misclassified samples
        
        Args:
            X: Features
            y: True labels
            top_n: Number of top misclassifications to analyze
            
        Returns:
            Analysis of misclassifications
        """
        try:
            y_pred = self.predict(X)
            y_prob = self.predict_proba(X)
            
            # Convert to numpy
            if isinstance(y, pd.Series):
                y = y.values
            
            # Find misclassifications
            misclassified_mask = y != y_pred
            misclassified_indices = np.where(misclassified_mask)[0]
            
            if len(misclassified_indices) == 0:
                return {'message': 'No misclassifications found!'}
            
            # Get confidence scores for misclassified samples
            misclassified_probs = y_prob[misclassified_indices]
            max_probs = np.max(misclassified_probs, axis=1)
            
            # Sort by confidence (high confidence misclassifications are more concerning)
            sorted_indices = np.argsort(-max_probs)[:top_n]
            
            analysis = {
                'total_misclassifications': len(misclassified_indices),
                'misclassification_rate': float(len(misclassified_indices) / len(y)),
                'top_misclassifications': []
            }
            
            for idx in sorted_indices:
                orig_idx = misclassified_indices[idx]
                
                try:
                    true_class_encoded = self.label_encoder.transform([y[orig_idx]])[0]
                    pred_class_encoded = self.label_encoder.transform([y_pred[orig_idx]])[0]
                    
                    misclass_info = {
                        'sample_index': int(orig_idx),
                        'true_class': str(y[orig_idx]),
                        'predicted_class': str(y_pred[orig_idx]),
                        'true_class_name': self.class_names[true_class_encoded] if true_class_encoded < len(self.class_names) else f'Class_{y[orig_idx]}',
                        'predicted_class_name': self.class_names[pred_class_encoded] if pred_class_encoded < len(self.class_names) else f'Class_{y_pred[orig_idx]}',
                        'confidence': float(max_probs[idx]),
                        'class_probabilities': {
                            self.class_names[i]: float(misclassified_probs[idx, i])
                            for i in range(min(len(self.class_names), misclassified_probs.shape[1]))
                        }
                    }
                    
                    analysis['top_misclassifications'].append(misclass_info)
                except (ValueError, IndexError):
                    continue
            
            # Confusion pairs analysis
            confusion_pairs = {}
            for true_label, pred_label in zip(y[misclassified_indices], y_pred[misclassified_indices]):
                pair = f"{true_label} → {pred_label}"
                confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
            
            # Sort confusion pairs by frequency
            sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
            analysis['common_confusion_pairs'] = sorted_pairs[:5]
            
            return analysis
            
        except Exception as e:
            print(f"❌ Misclassification analysis failed: {e}")
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
                      cv: int = 5, scoring: str = 'accuracy') -> Dict[str, Any]:
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
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Create model for CV
            cv_model = self._create_model()
            
            # Perform cross-validation
            cv_scores = cross_val_score(cv_model, X_scaled, y_encoded, cv=cv, scoring=scoring, n_jobs=self.n_jobs)
            
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
                             cv: int = 3, scoring: str = 'accuracy',
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
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
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
            grid_search.fit(X_scaled, y_encoded)
            
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
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive model summary
        
        Returns:
            Dictionary with model information
        """
        try:
            summary = {
                'model_type': 'Random Forest Classifier',
                'is_trained': self.is_trained,
                'n_features': self.n_features,
                'n_classes': self.n_classes,
                'feature_names': self.feature_names,
                'class_names': self.class_names,
                'model_params': {
                    'n_estimators': self.n_estimators,
                    'max_depth': self.max_depth,
                    'min_samples_split': self.min_samples_split,
                    'min_samples_leaf': self.min_samples_leaf,
                    'max_features': self.max_features,
                    'bootstrap': self.bootstrap,
                    'oob_score': self.oob_score,
                    'class_weight': self.class_weight,
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
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names,
                'class_names': self.class_names,
                'n_features': self.n_features,
                'n_classes': self.n_classes,
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
                    'class_weight': self.class_weight,
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
            self.label_encoder = model_data['label_encoder']
            self.feature_names = model_data['feature_names']
            self.class_names = model_data['class_names']
            self.n_features = model_data['n_features']
            self.n_classes = model_data['n_classes']
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
            self.class_weight = params['class_weight']
            self.n_jobs = params['n_jobs']
            self.random_state = params['random_state']
            self.max_samples = params.get('max_samples')
            self.ccp_alpha = params.get('ccp_alpha', 0.0)
            self.criterion = params.get('criterion', 'gini')
            
            self.is_trained = True
            
            print(f"✅ Model loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Model load failed: {e}")
            return False


# Test the Random Forest Classifier
if __name__ == "__main__":
    print("🌲 Testing Random Forest Classifier...")
    
    if SKLEARN_AVAILABLE:
        # Generate synthetic financial data
        np.random.seed(42)
        n_samples = 2000
        n_features = 15
        
        # Synthetic features representing financial indicators
        X = np.random.randn(n_samples, n_features)
        
        # Create synthetic risk categories (4 classes: Poor, Fair, Good, Excellent)
        # Simulate complex decision boundaries that Random Forest can capture
        risk_score = (
            X[:, 0] * 1.5 +                    # Revenue factor
            X[:, 1] * 1.2 +                    # Profit margin
            X[:, 2] * (-1.0) +                 # Debt ratio (negative impact)
            X[:, 3] * X[:, 4] * 0.3 +          # Interaction term
            np.where(X[:, 5] > 0, X[:, 5] * 2, X[:, 5] * 0.5) +  # Non-linear relationship
            X[:, 6] * X[:, 7] * X[:, 8] * 0.1 +  # Three-way interaction
            np.random.randn(n_samples) * 0.2    # Noise
        )
        
        # Convert continuous score to risk categories
        y = np.zeros(n_samples, dtype=int)
        y[risk_score > 1.0] = 3  # Excellent
        y[(risk_score > 0.0) & (risk_score <= 1.0)] = 2  # Good
        y[(risk_score > -1.0) & (risk_score <= 0.0)] = 1  # Fair
        y[risk_score <= -1.0] = 0  # Poor
        
        # Create DataFrame
        feature_names = [
            'revenue_growth', 'profit_margin', 'debt_ratio', 'liquidity_ratio', 'roa',
            'current_ratio', 'quick_ratio', 'inventory_turnover', 'receivables_turnover',
            'payables_turnover', 'cash_ratio', 'interest_coverage', 'debt_to_equity',
            'working_capital', 'market_cap'
        ]
        
        class_names = ['Poor', 'Fair', 'Good', 'Excellent']
        
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_series, test_size=0.2, random_state=42, stratify=y_series
        )
        
        print(f"📊 Dataset: {len(X_train)} train, {len(X_test)} test samples")
        print(f"📊 Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        # Test different configurations
        configs_to_test = [
            ('risk_classification', 'Risk Classification'),
            ('bankruptcy_prediction', 'Bankruptcy Prediction'),
            ('interpretable', 'Interpretable Model')
        ]
        
        for scenario, description in configs_to_test:
            print(f"\n🌲 Testing {description}...")
            
            try:
                # Get configuration
                config = RandomForestConfig.get_config_by_scenario(scenario)
                config['n_jobs'] = 1  # Reduce for testing
                
                # Create model
                rf_model = RandomForestClassifierModel(**config)
                
                # Train model
                success = rf_model.train(X_train, y_train, class_names=class_names, verbose=False)
                
                if success:
                    print(f"✅ Training successful")
                    
                    # Make predictions
                    y_pred = rf_model.predict(X_test)
                    print(f"✅ Predictions shape: {y_pred.shape}")
                    
                    # Get probabilities
                    y_prob = rf_model.predict_proba(X_test)
                    print(f"✅ Probabilities shape: {y_prob.shape}")
                    
                    # Evaluate model
                    metrics = rf_model.evaluate(X_test, y_test)
                    print(f"📊 Accuracy: {metrics.get('accuracy', 0):.4f}")
                    print(f"📊 F1 Weighted: {metrics.get('f1_weighted', 0):.4f}")
                    
                    if 'risk_weighted_accuracy' in metrics:
                        print(f"📊 Risk Weighted Accuracy: {metrics['risk_weighted_accuracy']:.4f}")
                    
                    if 'oob_score' in metrics and metrics['oob_score']:
                        print(f"📊 OOB Score: {metrics['oob_score']:.4f}")
                    
                    # Test confidence prediction
                    confidence_results = rf_model.predict_with_confidence(X_test[:5])
                    if confidence_results:
                        print(f"✅ Confidence estimation working")
                        avg_confidence = np.mean(confidence_results['confidence_scores'])
                        print(f"📊 Average confidence: {avg_confidence:.4f}")
                    
                    # Analyze probabilities
                    prob_summary = rf_model.get_class_probabilities_summary(X_test)
                    if prob_summary and 'overall_confidence' in prob_summary:
                        print(f"📊 Overall confidence mean: {prob_summary['overall_confidence']['mean']:.4f}")
                    
                    # Analyze misclassifications
                    misclass_analysis = rf_model.analyze_misclassifications(X_test, y_test, top_n=3)
                    if 'total_misclassifications' in misclass_analysis:
                        print(f"📊 Misclassifications: {misclass_analysis['total_misclassifications']}")
                        print(f"📊 Misclassification rate: {misclass_analysis['misclassification_rate']:.4f}")
                    
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
                    
                    # Get model summary
                    summary = rf_model.get_model_summary()
                    print(f"📊 Model trained: {summary.get('is_trained', False)}")
                    print(f"📊 Classes: {summary.get('n_classes', 0)}")
                    
                else:
                    print("❌ Training failed")
                    
            except Exception as e:
                print(f"❌ Test failed: {e}")
        
        # Test binary classification
        print(f"\n🌲 Testing Binary Classification...")
        try:
            # Convert to binary (Poor/Fair vs Good/Excellent)
            y_binary = (y >= 2).astype(int)  # 0: Poor/Fair, 1: Good/Excellent
            y_binary_series = pd.Series(y_binary)
            
            X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
                X_df, y_binary_series, test_size=0.2, random_state=42, stratify=y_binary_series
            )
            
            # Create binary classifier
            binary_config = RandomForestConfig.get_config_by_scenario('bankruptcy_prediction')
            binary_config['n_jobs'] = 1
            binary_classifier = RandomForestClassifierModel(**binary_config)
            
            # Train binary model
            success = binary_classifier.train(
                X_train_bin, y_train_bin,
                class_names=['Low Risk', 'High Risk'],
                verbose=False
            )
            
            if success:
                print("✅ Binary training successful")
                
                # Evaluate binary model
                binary_metrics = binary_classifier.evaluate(X_test_bin, y_test_bin)
                print(f"📊 Binary Accuracy: {binary_metrics.get('accuracy', 0):.4f}")
                
                if 'roc_auc' in binary_metrics:
                    print(f"📊 Binary ROC AUC: {binary_metrics['roc_auc']:.4f}")
                
            else:
                print("❌ Binary training failed")
                
        except Exception as e:
            print(f"❌ Binary classification test failed: {e}")
        
        # Test cross-validation
        print(f"\n🌲 Testing Cross-Validation...")
        try:
            config = RandomForestConfig.get_config_by_scenario('fast')
            config['n_jobs'] = 1
            cv_model = RandomForestClassifierModel(**config)
            
            cv_results = cv_model.cross_validate(X_train, y_train, cv=3, scoring='accuracy')
            if cv_results:
                print(f"✅ Cross-validation completed")
                print(f"📊 Mean CV Score: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
        except Exception as e:
            print(f"❌ Cross-validation test failed: {e}")
        
        # Test hyperparameter tuning (with reduced grid for speed)
        print(f"\n🌲 Testing Hyperparameter Tuning...")
        try:
            config = RandomForestConfig.get_config_by_scenario('fast')
            tuning_model = RandomForestClassifierModel(**config)
            
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
                print(f"📊 Best score: {tuning_results['best_score']:.4f}")
        except Exception as e:
            print(f"❌ Hyperparameter tuning test failed: {e}")
        
        print("\n🎉 Random Forest Classifier Test Completed!")
    
    else:
        print("❌ Scikit-learn not available - Cannot test Random Forest models")
        print("Install scikit-learn with: pip install scikit-learn")