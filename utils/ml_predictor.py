"""
Complete ML Prediction Module for Financial Risk Assessment
==========================================================

This module provides comprehensive machine learning prediction capabilities
including confidence analysis, financial risk modeling, and ensemble methods.

Classes:
--------
- ConfidenceScoreCalculator: Calculate confidence scores for ML predictions
- FinancialFeatureExtractor: Extract and engineer financial features
- LiquidationPredictor: ML model for bankruptcy/liquidation prediction
- FinancialHealthPredictor: Overall financial health assessment
- ModelTrainer: Training and validation utilities
- EnsemblePredictor: Ensemble methods for improved accuracy
- MLPredictor: Main prediction orchestrator

Author: Prateek Dahiya
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Self, Union, Tuple
import warnings
import pickle
import json
from datetime import datetime
from pathlib import Path
import logging

from ml_algorithms.base_model import ModelType
from ml_algorithms.lightgbm.lgb_classifier import X_train
from ml_algorithms.lightgbm.lgb_regressor import X_test

# ML imports with fallbacks
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, mean_squared_error, r2_score
    from sklearn.impute import SimpleImputer
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ConfidenceScoreCalculator:
    """
    Calculate confidence scores for different ML algorithms
    """
    
    @staticmethod
    def calculate_classification_confidence(model, X: np.ndarray, 
                                          confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Calculate confidence scores for classification models
        
        Args:
            model: Trained classification model
            X: Input features
            confidence_threshold: Threshold for high confidence
            
        Returns:
            Dictionary with confidence analysis
        """
        try:
            # Get predictions and probabilities
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)
            
            # Calculate confidence scores (max probability)
            confidence_scores = np.max(probabilities, axis=1)
            predicted_classes = np.argmax(probabilities, axis=1)
            
            # Detailed confidence analysis
            confidence_details = []
            for i, (pred, conf) in enumerate(zip(predictions, confidence_scores)):
                # Determine confidence level
                if conf >= 0.9:
                    level = "Very High"
                elif conf >= 0.8:
                    level = "High"
                elif conf >= 0.7:
                    level = "Medium"
                elif conf >= 0.6:
                    level = "Low"
                else:
                    level = "Very Low"
                
                # Get class probabilities
                class_probs = {}
                if hasattr(model, 'classes_'):
                    for j, class_name in enumerate(model.classes_):
                        if j < probabilities.shape[1]:
                            class_probs[str(class_name)] = float(probabilities[i, j])
                else:
                    for j in range(probabilities.shape[1]):
                        class_probs[f'class_{j}'] = float(probabilities[i, j])
                
                confidence_details.append({
                    'sample_index': i,
                    'prediction': int(pred),
                    'confidence_score': float(conf),
                    'confidence_level': level,
                    'is_confident': conf >= confidence_threshold,
                    'class_probabilities': class_probs
                })
            
            # Summary statistics
            high_confidence_count = sum(1 for x in confidence_details if x['confidence_score'] >= confidence_threshold)
            
            confidence_summary = {
                'mean_confidence': float(np.mean(confidence_scores)),
                'min_confidence': float(np.min(confidence_scores)),
                'max_confidence': float(np.max(confidence_scores)),
                'std_confidence': float(np.std(confidence_scores)),
                'median_confidence': float(np.median(confidence_scores)),
                'high_confidence_count': high_confidence_count,
                'high_confidence_percentage': float(high_confidence_count / len(X) * 100),
                'low_confidence_count': len(X) - high_confidence_count,
                'confidence_distribution': {
                    'very_high (≥0.9)': int(sum(1 for x in confidence_scores if x >= 0.9)),
                    'high (0.8-0.9)': int(sum(1 for x in confidence_scores if 0.8 <= x < 0.9)),
                    'medium (0.7-0.8)': int(sum(1 for x in confidence_scores if 0.7 <= x < 0.8)),
                    'low (0.6-0.7)': int(sum(1 for x in confidence_scores if 0.6 <= x < 0.7)),
                    'very_low (<0.6)': int(sum(1 for x in confidence_scores if x < 0.6))
                }
            }
            
            return {
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist(),
                'confidence_scores': confidence_scores.tolist(),
                'confidence_details': confidence_details,
                'confidence_summary': confidence_summary
            }
            
        except Exception as e:
            raise Exception(f"Classification confidence calculation failed: {e}")
    
    @staticmethod
    def calculate_regression_confidence(model, X: np.ndarray,
                                      confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Calculate confidence scores for regression models
        """
        try:
            # Get predictions
            predictions = model.predict(X)
            
            # Use prediction magnitude as confidence proxy for basic models
            pred_magnitude = np.abs(predictions)
            if pred_magnitude.max() > 0:
                normalized_confidence = pred_magnitude / pred_magnitude.max()
            else:
                normalized_confidence = np.ones_like(predictions) * 0.5
            
            confidence_details = []
            for i, (pred, conf) in enumerate(zip(predictions, normalized_confidence)):
                if conf >= 0.8:
                    level = "High"
                elif conf >= 0.6:
                    level = "Medium"
                else:
                    level = "Low"
                    
                confidence_details.append({
                    'sample_index': i,
                    'prediction': float(pred),
                    'confidence_score': float(conf),
                    'confidence_level': level,
                    'is_confident': conf >= confidence_threshold
                })
            
            return {
                'predictions': predictions.tolist(),
                'confidence_scores': normalized_confidence.tolist(),
                'confidence_details': confidence_details,
                'confidence_summary': {
                    'mean_confidence': float(np.mean(normalized_confidence)),
                    'note': 'Using prediction magnitude as confidence proxy'
                }
            }
                
        except Exception as e:
            raise Exception(f"Regression confidence calculation failed: {e}")


class FinancialFeatureExtractor:
    """
    Extract and engineer features from financial data for ML models
    """
    
    def __init__(self):
        self.feature_names = []
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.imputer = SimpleImputer(strategy='median') if SKLEARN_AVAILABLE else None
        self.is_fitted = False
        
    def extract_features(self, financial_data: Dict[str, float]) -> np.ndarray:
        """
        Extract features from financial data
        
        Args:
            financial_data: Dictionary containing financial metrics
            
        Returns:
            Numpy array of features
        """
        try:
            features = []
            self.feature_names = []
            
            # Basic financial data
            total_assets = financial_data.get('total_assets', 0)
            current_assets = financial_data.get('current_assets', 0)
            current_liabilities = financial_data.get('current_liabilities', 0)
            total_liabilities = financial_data.get('total_liabilities', 0)
            total_equity = financial_data.get('total_equity', 0)
            revenue = financial_data.get('revenue', 0)
            net_income = financial_data.get('net_income', 0)
            operating_cash_flow = financial_data.get('net_cash_from_operating_activities', 0)
            cash = financial_data.get('cash_and_equivalents', 0)
            inventory = financial_data.get('inventory', 0)
            accounts_receivable = financial_data.get('accounts_receivable', 0)
            accounts_payable = financial_data.get('accounts_payable', 0)
            long_term_debt = financial_data.get('long_term_debt', 0)
            interest_expense = financial_data.get('interest_expense', 0)
            
            # Liquidity Ratios
            current_ratio = current_assets / current_liabilities if current_liabilities != 0 else 0
            features.append(current_ratio)
            self.feature_names.append('current_ratio')
            
            quick_ratio = (current_assets - inventory) / current_liabilities if current_liabilities != 0 else 0
            features.append(quick_ratio)
            self.feature_names.append('quick_ratio')
            
            cash_ratio = cash / current_liabilities if current_liabilities != 0 else 0
            features.append(cash_ratio)
            self.feature_names.append('cash_ratio')
            
            # Leverage Ratios
            debt_to_equity = total_liabilities / total_equity if total_equity != 0 else float('inf')
            debt_to_equity = min(debt_to_equity, 10)  # Cap at 10
            features.append(debt_to_equity)
            self.feature_names.append('debt_to_equity')
            
            debt_to_assets = total_liabilities / total_assets if total_assets != 0 else 0
            features.append(debt_to_assets)
            self.feature_names.append('debt_to_assets')
            
            # Profitability Ratios
            profit_margin = net_income / revenue if revenue != 0 else 0
            features.append(profit_margin)
            self.feature_names.append('profit_margin')
            
            roa = net_income / total_assets if total_assets != 0 else 0
            features.append(roa)
            self.feature_names.append('roa')
            
            roe = net_income / total_equity if total_equity != 0 else 0
            features.append(roe)
            self.feature_names.append('roe')
            
            # Efficiency Ratios
            asset_turnover = revenue / total_assets if total_assets != 0 else 0
            features.append(asset_turnover)
            self.feature_names.append('asset_turnover')
            
            # Cash Flow Ratios
            operating_cf_ratio = operating_cash_flow / current_liabilities if current_liabilities != 0 else 0
            features.append(operating_cf_ratio)
            self.feature_names.append('operating_cf_ratio')
            
            cash_flow_to_debt = operating_cash_flow / total_liabilities if total_liabilities != 0 else 0
            features.append(cash_flow_to_debt)
            self.feature_names.append('cash_flow_to_debt')
            
            # Interest Coverage
            interest_coverage = (net_income + interest_expense) / interest_expense if interest_expense != 0 else 10
            interest_coverage = min(interest_coverage, 50)  # Cap at 50
            features.append(interest_coverage)
            self.feature_names.append('interest_coverage')
            
            # Working Capital
            working_capital = current_assets - current_liabilities
            working_capital_ratio = working_capital / total_assets if total_assets != 0 else 0
            features.append(working_capital_ratio)
            self.feature_names.append('working_capital_ratio')
            
            # Z-Score Components
            if total_assets != 0:
                z1 = working_capital / total_assets
                z2 = financial_data.get('retained_earnings', 0) / total_assets
                z3 = financial_data.get('operating_income', net_income) / total_assets
                z4 = total_equity / total_liabilities if total_liabilities != 0 else 0
                z5 = revenue / total_assets
                
                features.extend([z1, z2, z3, z4, z5])
                self.feature_names.extend(['z_score_wc', 'z_score_re', 'z_score_ebit', 'z_score_equity', 'z_score_sales'])
            else:
                features.extend([0, 0, 0, 0, 0])
                self.feature_names.extend(['z_score_wc', 'z_score_re', 'z_score_ebit', 'z_score_equity', 'z_score_sales'])
            
            # Additional Risk Indicators
            revenue_growth = financial_data.get('revenue_growth', 0)
            features.append(revenue_growth)
            self.feature_names.append('revenue_growth')
            
            # Cash flow quality
            ocf_to_ni = operating_cash_flow / net_income if net_income != 0 else 0
            ocf_to_ni = max(-5, min(5, ocf_to_ni))  # Cap between -5 and 5
            features.append(ocf_to_ni)
            self.feature_names.append('ocf_to_ni_ratio')
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return np.zeros((1, 20))  # Return default features
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform features"""
        if not SKLEARN_AVAILABLE:
            return X
            
        try:
            # Handle missing values
            X_imputed = self.imputer.fit_transform(X)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X_imputed)
            
            self.is_fitted = True
            return X_scaled
            
        except Exception as e:
            logger.error(f"Error fitting features: {e}")
            return X
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler"""
        if not SKLEARN_AVAILABLE or not self.is_fitted:
            return X
            
        try:
            X_imputed = self.imputer.transform(X)
            X_scaled = self.scaler.transform(X_imputed)
            return X_scaled
        except Exception as e:
            logger.error(f"Error transforming features: {e}")
            return X


class LiquidationPredictor:
    """
    ML model for predicting liquidation/bankruptcy risk
    """
    
    def __init__(self):
        self.model = None
        self.feature_extractor = FinancialFeatureExtractor()
        self.is_trained = False
        self.model_type = 'random_forest'
        self.confidence_calculator = ConfidenceScoreCalculator()
        
    def train(self, training_data: List[Dict[str, Any]], model_type: str = 'random_forest') -> Dict[str, Any]:
        """
        Train liquidation prediction model
        
        Args:
            training_data: List of training examples with features and labels
            model_type: Type of model to train
            
        Returns:
            Training results
        """
        try:
            if not SKLEARN_AVAILABLE:
                return self._rule_based_fallback()
            
            # Extract features and labels
            X_data = []
            y_data = []
            
            for example in training_data:
                features = self.feature_extractor.extract_features(example['features']).flatten()
                label = example['label']  # 1 for liquidation risk, 0 for stable
                
                X_data.append(features)
                y_data.append(label)
            
            X = np.array(X_data)
            y = np.array(y_data)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            models = {}
            results = {}
            
            # Random Forest
            rf_model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
            rf_model.fit(X_train_scaled, y_train)
            rf_score = rf_model.score(X_test_scaled, y_test)
            models['random_forest'] = rf_model
            results['random_forest'] = {'accuracy': rf_score}
            
            # Logistic Regression
            lr_model = LogisticRegression(
                random_state=42,
                class_weight='balanced'
            )
            lr_model.fit(X_train_scaled, y_train)
            lr_score = lr_model.score(X_test_scaled, y_test)
            models['logistic_regression'] = lr_model
            results['logistic_regression'] = {'accuracy': lr_score}
            
            # XGBoost if available
            if XGBOOST_AVAILABLE:
                xgb_model = xgb.XGBClassifier(
                    random_state=42,
                    eval_metric='logloss'
                )
                xgb_model.fit(X_train_scaled, y_train)
                xgb_score = xgb_model.score(X_test_scaled, y_test)
                models['xgboost'] = xgb_model
                results['xgboost'] = {'accuracy': xgb_score}
            
            # Voting Classifier (Ensemble)
            voting_models = [
                ('rf', models['random_forest']),
                ('lr', models['logistic_regression'])
            ]
            
            if XGBOOST_AVAILABLE:
                voting_models.append(('xgb', models['xgboost']))
            
            ensemble_model = VotingClassifier(
                estimators=voting_models,
                voting='soft'
            )
            ensemble_model.fit(X_train_scaled, y_train)
            ensemble_score = ensemble_model.score(X_test_scaled, y_test)
            models['ensemble'] = ensemble_model
            results['ensemble'] = {'accuracy': ensemble_score}
            
            # Select best model
            best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
            best_model = models[best_model_name]
            best_score = results[best_model_name]['accuracy']
            
            self.trained_models = models
            self.trained_models['scaler'] = scaler
            
            return {
                'success': True,
                'best_model': best_model_name,
                'best_accuracy': best_score,
                'all_results': results,
                'models_trained': list(models.keys()),
                'scaler': scaler
            }
            
        except Exception as e:
            logger.error(f"Error training ensemble model: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def cross_validate_model(self, model, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation on model
        """
        try:
            if not SKLEARN_AVAILABLE:
                return {'success': False, 'error': 'Scikit-learn not available'}
            
            # Stratified K-Fold for classification
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
            
            # Additional metrics if classification
            if hasattr(model, 'predict_proba'):
                cv_auc = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
                cv_precision = cross_val_score(model, X, y, cv=skf, scoring='precision')
                cv_recall = cross_val_score(model, X, y, cv=skf, scoring='recall')
                cv_f1 = cross_val_score(model, X, y, cv=skf, scoring='f1')
                
                return {
                    'success': True,
                    'cv_accuracy': {
                        'mean': float(cv_scores.mean()),
                        'std': float(cv_scores.std()),
                        'scores': cv_scores.tolist()
                    },
                    'cv_auc': {
                        'mean': float(cv_auc.mean()),
                        'std': float(cv_auc.std()),
                        'scores': cv_auc.tolist()
                    },
                    'cv_precision': {
                        'mean': float(cv_precision.mean()),
                        'std': float(cv_precision.std()),
                        'scores': cv_precision.tolist()
                    },
                    'cv_recall': {
                        'mean': float(cv_recall.mean()),
                        'std': float(cv_recall.std()),
                        'scores': cv_recall.tolist()
                    },
                    'cv_f1': {
                        'mean': float(cv_f1.mean()),
                        'std': float(cv_f1.std()),
                        'scores': cv_f1.tolist()
                    }
                }
            else:
                return {
                    'success': True,
                    'cv_accuracy': {
                        'mean': float(cv_scores.mean()),
                        'std': float(cv_scores.std()),
                        'scores': cv_scores.tolist()
                    }
                }
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def hyperparameter_tuning(self, model, param_grid: Dict, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV
        """
        try:
            if not SKLEARN_AVAILABLE:
                return {'success': False, 'error': 'Scikit-learn not available'}
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                model, 
                param_grid, 
                cv=5, 
                scoring='accuracy', 
                n_jobs=-1
            )
            
            grid_search.fit(X, y)
            
            return {
                'success': True,
                'best_params': grid_search.best_params_,
                'best_score': float(grid_search.best_score_),
                'best_model': grid_search.best_estimator_,
                'cv_results': grid_search.cv_results_
            }
            
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {e}")
            return {
                'success': False,
                'error': str(e)
            }


class EnsemblePredictor:
    """
    Ensemble methods for improved prediction accuracy
    """
    
    def __init__(self):
        self.models = []
        self.weights = []
        self.is_fitted = False
    
    def add_model(self, model, weight: float = 1.0):
        """Add model to ensemble with weight"""
        self.models.append(model)
        self.weights.append(weight)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit all models in ensemble"""
        try:
            for model in self.models:
                model.fit(X, y)
            self.is_fitted = True
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def predict_weighted_voting(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Make predictions using weighted voting
        """
        try:
            if not self.is_fitted:
                return {'success': False, 'error': 'Ensemble not fitted'}
            
            predictions = []
            probabilities = []
            
            # Get predictions from each model
            for i, model in enumerate(self.models):
                pred = model.predict(X)
                predictions.append(pred)
                
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X)
                    probabilities.append(prob)
            
            # Weighted voting for final prediction
            if probabilities:
                # Average probabilities with weights
                weighted_probs = np.zeros_like(probabilities[0])
                total_weight = sum(self.weights)
                
                for i, prob in enumerate(probabilities):
                    weighted_probs += prob * (self.weights[i] / total_weight)
                
                final_predictions = np.argmax(weighted_probs, axis=1)
                confidence_scores = np.max(weighted_probs, axis=1)
                
                return {
                    'success': True,
                    'predictions': final_predictions.tolist(),
                    'probabilities': weighted_probs.tolist(),
                    'confidence_scores': confidence_scores.tolist(),
                    'individual_predictions': [pred.tolist() for pred in predictions]
                }
            else:
                # Simple majority voting for models without probabilities
                prediction_array = np.array(predictions)
                final_predictions = []
                
                for i in range(X.shape[0]):
                    sample_predictions = prediction_array[:, i]
                    # Weighted voting
                    weighted_votes = {}
                    for j, pred in enumerate(sample_predictions):
                        if pred not in weighted_votes:
                            weighted_votes[pred] = 0
                        weighted_votes[pred] += self.weights[j]
                    
                    final_pred = max(weighted_votes.keys(), key=lambda k: weighted_votes[k])
                    final_predictions.append(final_pred)
                
                return {
                    'success': True,
                    'predictions': final_predictions,
                    'individual_predictions': [pred.tolist() for pred in predictions]
                }
                
        except Exception as e:
            logger.error(f"Error in weighted voting: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def predict_stacking(self, X: np.ndarray, meta_learner=None) -> Dict[str, Any]:
        """
        Make predictions using stacking ensemble
        """
        try:
            if not self.is_fitted:
                return {'success': False, 'error': 'Ensemble not fitted'}
            
            # Get predictions from base models as meta-features
            meta_features = []
            
            for model in self.models:
                if hasattr(model, 'predict_proba'):
                    meta_feat = model.predict_proba(X)
                else:
                    meta_feat = model.predict(X).reshape(-1, 1)
                meta_features.append(meta_feat)
            
            # Combine meta-features
            if len(meta_features) > 0:
                if meta_features[0].ndim == 2:
                    combined_meta = np.hstack(meta_features)
                else:
                    combined_meta = np.column_stack(meta_features)
                
                # Use meta-learner if provided, otherwise use simple average
                if meta_learner and hasattr(meta_learner, 'predict'):
                    final_predictions = meta_learner.predict(combined_meta)
                    
                    if hasattr(meta_learner, 'predict_proba'):
                        final_probabilities = meta_learner.predict_proba(combined_meta)
                        confidence_scores = np.max(final_probabilities, axis=1)
                    else:
                        final_probabilities = None
                        confidence_scores = None
                else:
                    # Simple averaging
                    if combined_meta.shape[1] > 1:
                        final_probabilities = np.mean(combined_meta.reshape(X.shape[0], -1, 2), axis=1)
                        final_predictions = np.argmax(final_probabilities, axis=1)
                        confidence_scores = np.max(final_probabilities, axis=1)
                    else:
                        final_predictions = np.mean(combined_meta, axis=1)
                        final_probabilities = None
                        confidence_scores = None
                
                result = {
                    'success': True,
                    'predictions': final_predictions.tolist() if hasattr(final_predictions, 'tolist') else final_predictions,
                    'meta_features_shape': combined_meta.shape
                }
                
                if final_probabilities is not None:
                    result['probabilities'] = final_probabilities.tolist()
                
                if confidence_scores is not None:
                    result['confidence_scores'] = confidence_scores.tolist()
                
                return result
            else:
                return {'success': False, 'error': 'No valid meta-features generated'}
                
        except Exception as e:
            logger.error(f"Error in stacking: {e}")
            return {
                'success': False,
                'error': str(e)
            }


class MLPredictor:
    """
    Main ML Prediction orchestrator with all capabilities
    """
    
    def __init__(self):
        self.confidence_calculator = ConfidenceScoreCalculator()
        self.feature_extractor = FinancialFeatureExtractor()
        self.liquidation_predictor = LiquidationPredictor()
        self.health_predictor = None
        self.model_trainer = None
        self.ensemble_predictor = EnsemblePredictor()
        
    def predict_with_confidence(self, model, model_info: Dict, X: np.ndarray,
                              confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Make predictions with comprehensive confidence analysis
        """
        try:
            result = {
                'model_name': model_info.get('name', 'unknown'),
                'model_type': model_info.get('type', 'unknown'),
                'algorithm': model_info.get('algorithm', 'unknown'),
                'n_samples': len(X),
                'confidence_threshold': confidence_threshold
            }
            
            if model_info['algorithm'] == 'classifier':
                confidence_data = self.confidence_calculator.calculate_classification_confidence(
                    model, X, confidence_threshold
                )
                result.update(confidence_data)
                
            elif model_info['algorithm'] == 'regressor':
                confidence_data = self.confidence_calculator.calculate_regression_confidence(
                    model, X, confidence_threshold
                )
                result.update(confidence_data)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in prediction with confidence: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def comprehensive_financial_prediction(self, financial_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Comprehensive financial prediction combining all models
        """
        try:
            # Liquidation risk prediction
            liquidation_result = self.liquidation_predictor.predict(financial_data)
            
            # Financial health assessment
            health_result = self.health_predictor.assess_health(financial_data)
            
            # Extract features for additional analysis
            features = self.feature_extractor.extract_features(financial_data)
            
            # Generate recommendations based on results
            recommendations = self._generate_recommendations(liquidation_result, health_result)
            
            return {
                'success': True,
                'liquidation_prediction': liquidation_result,
                'health_assessment': health_result,
                'features_extracted': features.shape[1],
                'feature_names': self.feature_extractor.feature_names,
                'recommendations': recommendations,
                'prediction_date': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive prediction: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _generate_recommendations(self, liquidation_result: Dict, health_result: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Liquidation risk recommendations
        risk_level = liquidation_result.get('risk_level', 'unknown')
        if risk_level == 'critical':
            recommendations.extend([
                "🚨 URGENT: Implement immediate financial restructuring",
                "💼 Engage financial advisors and legal counsel",
                "⚡ Reduce all non-essential expenses immediately"
            ])
        elif risk_level == 'high':
            recommendations.extend([
                "⚠️ HIGH PRIORITY: Develop comprehensive turnaround plan",
                "💰 Improve cash flow through working capital optimization",
                "🤝 Negotiate with creditors for better terms"
            ])
        elif risk_level == 'medium':
            recommendations.extend([
                "📊 Monitor financial metrics closely",
                "🔄 Implement operational efficiency improvements",
                "💡 Consider strategic cost reduction initiatives"
            ])
        else:
            recommendations.append("✅ Maintain current financial management practices")
        
        # Health-based recommendations
        if health_result.get('success'):
            component_scores = health_result.get('component_scores', {})
            
            if component_scores.get('liquidity', 0) < 60:
                recommendations.append("💧 Improve liquidity through better cash management")
            
            if component_scores.get('profitability', 0) < 60:
                recommendations.append("📈 Focus on margin improvement and revenue growth")
            
            if component_scores.get('leverage', 0) < 60:
                recommendations.append("⚖️ Consider debt reduction strategies")
            
            if component_scores.get('efficiency', 0) < 60:
                recommendations.append("⚙️ Enhance operational efficiency and asset utilization")
            
            if component_scores.get('cash_flow', 0) < 60:
                recommendations.append("💵 Strengthen cash flow generation capabilities")
        
        return recommendations


def save_model(model, filepath: str) -> bool:
    """Save trained model to file"""
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        return True
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return False


def load_model(filepath: str):
    """Load trained model from file"""
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


def create_synthetic_training_data(n_samples: int = 1000) -> List[Dict[str, Any]]:
    """
    Create synthetic training data for model development
    """
    try:
        np.random.seed(42)
        data = []
        
        for i in range(n_samples):
            # Generate base financial metrics
            total_assets = np.random.uniform(1000000, 100000000)
            
            # Healthy companies (70% of data)
            if i < n_samples * 0.7:
                current_ratio = np.random.uniform(1.2, 3.0)
                debt_to_equity = np.random.uniform(0.1, 1.0)
                profit_margin = np.random.uniform(0.05, 0.25)
                liquidation_label = 0
            else:
                # Distressed companies (30% of data)
                current_ratio = np.random.uniform(0.3, 1.1)
                debt_to_equity = np.random.uniform(1.5, 5.0)
                profit_margin = np.random.uniform(-0.15, 0.05)
                liquidation_label = 1
            
            # Calculate dependent variables
            current_assets = total_assets * np.random.uniform(0.3, 0.6)
            current_liabilities = current_assets / current_ratio
            total_liabilities = total_assets * debt_to_equity / (1 + debt_to_equity)
            total_equity = total_assets - total_liabilities
            
            revenue = total_assets * np.random.uniform(0.5, 2.0)
            net_income = revenue * profit_margin
            operating_cash_flow = net_income * np.random.uniform(0.8, 1.5)
            
            record = {
                'features': {
                    'total_assets': total_assets,
                    'current_assets': current_assets,
                    'current_liabilities': current_liabilities,
                    'total_liabilities': total_liabilities,
                    'total_equity': total_equity,
                    'revenue': revenue,
                    'net_income': net_income,
                    'net_cash_from_operating_activities': operating_cash_flow,
                    'cash_and_equivalents': current_assets * np.random.uniform(0.1, 0.4),
                    'accounts_receivable': current_assets * np.random.uniform(0.2, 0.5),
                    'inventory': current_assets * np.random.uniform(0.1, 0.3),
                    'accounts_payable': current_liabilities * np.random.uniform(0.3, 0.7),
                    'long_term_debt': total_liabilities * np.random.uniform(0.4, 0.8),
                    'operating_income': net_income * np.random.uniform(1.1, 1.5),
                    'interest_expense': total_liabilities * np.random.uniform(0.03, 0.08),
                    'retained_earnings': total_equity * np.random.uniform(0.2, 0.8)
                },
                'label': liquidation_label
            }
            
            data.append(record)
        
        return data
        
    except Exception as e:
        logger.error(f"Error creating synthetic data: {e}")
        return []


# Utility functions
def format_confidence_response(prediction_data: Dict, include_details: bool = True) -> Dict[str, Any]:
    """Format confidence prediction response for API"""
    formatted = {
        'model_info': {
            'name': prediction_data.get('model_name'),
            'type': prediction_data.get('model_type'),
            'algorithm': prediction_data.get('algorithm')
        },
        'predictions': prediction_data.get('predictions', []),
        'confidence_summary': prediction_data.get('confidence_summary', {}),
        'n_samples': prediction_data.get('n_samples', 0)
    }
    
    if prediction_data.get('algorithm') == 'classifier':
        formatted['probabilities'] = prediction_data.get('probabilities', [])
        formatted['confidence_scores'] = prediction_data.get('confidence_scores', [])
    
    if include_details:
        formatted['confidence_details'] = prediction_data.get('confidence_details', [])
    
    return formatted


def get_confidence_recommendations(confidence_summary: Dict) -> List[str]:
    """Get recommendations based on confidence analysis"""
    recommendations = []
    
    mean_confidence = confidence_summary.get('mean_confidence', 0)
    high_confidence_percentage = confidence_summary.get('high_confidence_percentage', 0)
    
    if mean_confidence < 0.6:
        recommendations.append("⚠️ Low overall confidence detected. Consider retraining with more data.")
    elif mean_confidence < 0.7:
        recommendations.append("📊 Moderate confidence. Monitor predictions carefully.")
    else:
        recommendations.append("✅ Good overall confidence in predictions.")
    
    if high_confidence_percentage < 50:
        recommendations.append("🔄 Less than 50% high-confidence predictions. Consider feature engineering.")
    elif high_confidence_percentage > 90:
        recommendations.append("🎯 Excellent prediction confidence. Model performs well.")
    
    return recommendations


def get_model_requirements() -> Dict[str, bool]:
    """Get availability of ML libraries"""
    return {
        'sklearn': SKLEARN_AVAILABLE,
        'xgboost': XGBOOST_AVAILABLE,
        'lightgbm': LIGHTGBM_AVAILABLE
    }


        # Configuration constants   
DEFAULT_MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'random_state': 42,
        'class_weight': 'balanced'
    },
    'logistic_regression': {
        'random_state': 42,
        'class_weight': 'balanced',
        'max_iter': 1000
    },
    'xgboost': {
        'random_state': 42,
        'eval_metric': 'logloss'
    }
}

CONFIDENCE_THRESHOLDS = {
    'very_high': 0.9,
    'high': 0.8,
    'medium': 0.7,
    'low': 0.6
}

RISK_LEVELS = {
    'low': (0.0, 0.3),
    'medium': (0.3, 0.6),
    'high': (0.6, 0.8),
    'critical': (0.8, 1.0)
}
            
            
# Scale features
X_train_scaled = Self.feature_extractor.fit_transform(X_train)
X_test_scaled = Self.feature_extractor.transform(X_test)
            
    # Train model
if ModelType == 'random_forest':
    Self.model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    class_weight='balanced'
                )
elif ModelType == 'xgboost' and XGBOOST_AVAILABLE:
                Self.model = xgb.XGBClassifier(
                    random_state=42,
                    eval_metric='logloss'
                )
elif ModelType == 'logistic':
                Self.model = LogisticRegression(
                    random_state=42,
                    class_weight='balanced'
                )
else:
                Self.model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    class_weight='balanced'
                )
            
            # Train model
Self.model.fit(X_train_scaled, X_train)
            
            # Evaluate
y_pred = Self.model.predict(X_test_scaled)
y_pred_proba = Self.model.predict_proba(X_test_scaled)[:, 1]
            
accuracy = Self.model.score(X_test_scaled, X_test)
auc_score = roc_auc_score(X_test, y_pred_proba)
            
            # Cross-validation
cv_scores = cross_val_score(Self.model, X_train_scaled, X_train, cv=5)
            
Self.is_trained = True
Self.model_type = ModelType
 
            
def train(self, training_data, model_type='random_forest'):
    try:
        if not SKLEARN_AVAILABLE:
            return self._rule_based_fallback()
        
        # Extract features and labels
        X_data = []
        y_data = []
        
        for example in training_data:
            features = self.feature_extractor.extract_features(example['features']).flatten()
            label = example['label']  # 1 for liquidation risk, 0 for stable
            
            X_data.append(features)
            y_data.append(label)
        
        X = np.array(X_data)
        y = np.array(y_data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.feature_extractor.fit_transform(X_train)
        X_test_scaled = self.feature_extractor.transform(X_test)
        
        # Train model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
            self.model = xgb.XGBClassifier(
                random_state=42,
                eval_metric='logloss'
            )
        elif model_type == 'logistic':
            self.model = LogisticRegression(
                random_state=42,
                class_weight='balanced'
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = self.model.score(X_test_scaled, y_test)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        
        self.is_trained = True
        self.model_type = model_type
        
        return {
            'success': True,
            'accuracy': accuracy,
            'auc_score': auc_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'training_samples': len(training_data),
            'model_type': model_type
        }
        
    except Exception as e:
        logger.error(f"Error training liquidation model: {e}")
        return {
            'success': False,
            'error': str(e)
        }
    
def predict(self, financial_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict liquidation risk
        
        Args:
            financial_data: Financial metrics
            
        Returns:
            Prediction with confidence
        """
        try:
            if not self.is_trained:
                return self._rule_based_prediction(financial_data)
            
            # Extract and transform features
            features = self.feature_extractor.extract_features(financial_data)
            features_scaled = self.feature_extractor.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Calculate confidence
            confidence_data = self.confidence_calculator.calculate_classification_confidence(
                self.model, features_scaled, 0.7
            )
            
            risk_probability = probabilities[1]
            
            # Determine risk level
            if risk_probability < 0.3:
                risk_level = 'low'
                risk_description = 'Low liquidation risk'
            elif risk_probability < 0.6:
                risk_level = 'medium'
                risk_description = 'Moderate liquidation risk'
            elif risk_probability < 0.8:
                risk_level = 'high'
                risk_description = 'High liquidation risk'
            else:
                risk_level = 'critical'
                risk_description = 'Critical liquidation risk'
            
            return {
                'success': True,
                'prediction': int(prediction),
                'risk_probability': round(risk_probability, 3),
                'risk_level': risk_level,
                'risk_description': risk_description,
                'confidence_score': confidence_data['confidence_scores'][0],
                'model_type': self.model_type,
                'prediction_date': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error predicting liquidation risk: {e}")
            return {
                'success': False,
                'error': str(e),
                'risk_probability': 0.5,
                'risk_level': 'unknown'
            }
    
def _rule_based_prediction(self, financial_data: Dict[str, float]) -> Dict[str, Any]:
        """Rule-based prediction fallback"""
        try:
            risk_score = 0.0
            risk_factors = []
            
            # Extract key metrics
            current_ratio = financial_data.get('current_assets', 0) / financial_data.get('current_liabilities', 1)
            debt_to_equity = financial_data.get('total_liabilities', 0) / financial_data.get('total_equity', 1)
            net_income = financial_data.get('net_income', 0)
            operating_cash_flow = financial_data.get('net_cash_from_operating_activities', 0)
            
            # Apply rules
            if current_ratio < 1.0:
                risk_score += 0.15
                risk_factors.append('Poor liquidity')
            
            if debt_to_equity > 2.0:
                risk_score += 0.20
                risk_factors.append('High leverage')
            
            if net_income < 0:
                risk_score += 0.15
                risk_factors.append('Negative income')
            
            if operating_cash_flow < 0:
                risk_score += 0.20
                risk_factors.append('Negative cash flow')
            
            # Combined risk
            if net_income < 0 and operating_cash_flow < 0:
                risk_score += 0.20
                risk_factors.append('Combined losses')
            
            # Determine risk level
            if risk_score < 0.3:
                risk_level = 'low'
            elif risk_score < 0.6:
                risk_level = 'medium'
            elif risk_score < 0.8:
                risk_level = 'high'
            else:
                risk_level = 'critical'
            
            return {
                'success': True,
                'prediction': 1 if risk_score > 0.5 else 0,
                'risk_probability': min(1.0, risk_score),
                'risk_level': risk_level,
                'risk_factors': risk_factors,
                'method': 'rule_based',
                'confidence_score': 0.7,
                'prediction_date': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error in rule-based prediction: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
def _rule_based_fallback(self) -> Dict[str, Any]:
        """Fallback when sklearn not available"""
        self.is_trained = True  
        return {
            'success': True,
            'method': 'rule_based',
            'note': 'Using rule-based system (sklearn not available)'
        }


class FinancialHealthPredictor:
    """
    Comprehensive financial health assessment using ML
    """
    
    def __init__(self):
        self.liquidation_predictor = LiquidationPredictor()
        self.health_weights = {
            'liquidity': 0.25,
            'profitability': 0.25,
            'leverage': 0.20,
            'efficiency': 0.15,
            'cash_flow': 0.15
        }
    
    def assess_health(self, financial_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Assess comprehensive financial health
        
        Args:
            financial_data: Financial statement data
            
        Returns:
            Complete health assessment
        """
        try:
            # Get liquidation risk
            liquidation_result = self.liquidation_predictor.predict(financial_data)
            
            # Calculate component scores
            component_scores = self._calculate_component_scores(financial_data)
            
            # Calculate weighted overall score
            overall_score = sum(component_scores[component] * self.health_weights[component] 
                               for component in component_scores)
            
            # Determine health grade
            if overall_score >= 80:
                health_grade = 'A'
                health_status = 'Excellent'
            elif overall_score >= 70:
                health_grade = 'B'
                health_status = 'Good'
            elif overall_score >= 60:
                health_grade = 'C'
                health_status = 'Fair'
            elif overall_score >= 50:
                health_grade = 'D'
                health_status = 'Poor'
            else:
                health_grade = 'F'
                health_status = 'Critical'
            
            return {
                'success': True,
                'overall_score': round(overall_score, 1),
                'health_grade': health_grade,
                'health_status': health_status,
                'component_scores': {k: round(v, 1) for k, v in component_scores.items()},
                'liquidation_risk': liquidation_result,
                'assessment_date': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error assessing financial health: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _calculate_component_scores(self, financial_data: Dict[str, float]) -> Dict[str, float]:
        """Calculate individual component scores"""
        scores = {}
        
        # Liquidity Score
        current_ratio = financial_data.get('current_assets', 0) / financial_data.get('current_liabilities', 1)
        quick_ratio = (financial_data.get('current_assets', 0) - financial_data.get('inventory', 0)) / financial_data.get('current_liabilities', 1)
        
        liquidity_score = 0
        if current_ratio >= 2.0:
            liquidity_score += 50
        else:
            liquidity_score += current_ratio / 2.0 * 50
        
        if quick_ratio >= 1.0:
            liquidity_score += 50
        else:
            liquidity_score += quick_ratio * 50
        
        scores['liquidity'] = min(100, liquidity_score)
        
        # Profitability Score
        revenue = financial_data.get('revenue', 1)
        net_income = financial_data.get('net_income', 0)
        total_assets = financial_data.get('total_assets', 1)
        total_equity = financial_data.get('total_equity', 1)
        
        profit_margin = net_income / revenue
        roa = net_income / total_assets
        roe = net_income / total_equity
        
        profitability_score = 0
        if profit_margin >= 0.1:
            profitability_score += 40
        else:
            profitability_score += max(0, profit_margin * 400)
        
        if roa >= 0.05:
            profitability_score += 30
        else:
            profitability_score += max(0, roa * 600)
        
        if roe >= 0.1:
            profitability_score += 30
        else:
            profitability_score += max(0, roe * 300)
        
        scores['profitability'] = min(100, profitability_score)
        
        # Leverage Score (inverse - lower is better)
        debt_to_equity = financial_data.get('total_liabilities', 0) / financial_data.get('total_equity', 1)
        debt_to_assets = financial_data.get('total_liabilities', 0) / financial_data.get('total_assets', 1)
        
        if debt_to_equity <= 0.5:
            leverage_score = 100
        elif debt_to_equity <= 1.0:
            leverage_score = 80
        elif debt_to_equity <= 2.0:
            leverage_score = 60
        else:
            leverage_score = max(0, 60 - (debt_to_equity - 2.0) * 20)
        
        scores['leverage'] = min(100, leverage_score)
        
        # Efficiency Score
        asset_turnover = revenue / total_assets
        
        if asset_turnover >= 1.5:
            efficiency_score = 100
        else:
            efficiency_score = asset_turnover / 1.5 * 100
        
        scores['efficiency'] = min(100, efficiency_score)
        
        # Cash Flow Score
        operating_cash_flow = financial_data.get('net_cash_from_operating_activities', 0)
        
        ocf_to_ni = operating_cash_flow / net_income if net_income != 0 else 0
        ocf_to_assets = operating_cash_flow / total_assets
        
        cash_flow_score = 0
        if ocf_to_ni >= 1.0:
            cash_flow_score += 50
        else:
            cash_flow_score += max(0, ocf_to_ni * 50)
        
        if ocf_to_assets >= 0.1:
            cash_flow_score += 50
        else:
            cash_flow_score += max(0, ocf_to_assets * 500)
        
        scores['cash_flow'] = min(100, cash_flow_score)
        
        return scores


class ModelTrainer:
    """
    Comprehensive model training and validation utilities
    """
    
    def __init__(self):
        self.trained_models = {}
        self.validation_results = {}
    
    def train_ensemble_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train ensemble model with multiple algorithms
        """
        try:
            if not SKLEARN_AVAILABLE:
                return {'success': False, 'error': 'Scikit-learn not available'}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            models = {}
            results = {}
            
            # Random Forest
            rf_model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
            rf_model.fit(X_train_scaled, y_train)
            rf_score = rf_model.score(X_test_scaled, y_test)
            models['random_forest'] = rf_model
            results['random_forest'] = {'accuracy': rf_score}
            
            # Logistic Regression
            lr_model = LogisticRegression(
                random_state=42,
                class_weight='balanced'
            )
            lr_model.fit(X_train_scaled, y_train)
            lr_score = lr_model.score(X_test_scaled, y_test)
            models['logistic_regression'] = lr_model
            results['logistic_regression'] = {'accuracy': lr_score}
            
            # XGBoost if available
            if XGBOOST_AVAILABLE:
                xgb_model = xgb.XGBClassifier(
                    random_state=42,
                    eval_metric='logloss'
                )
                xgb_model.fit(X_train_scaled, y_train)
                xgb_score = xgb_model.score(X_test_scaled, y_test)
                models['xgboost'] = xgb_model
                results['xgboost'] = {'accuracy': xgb_score}
            
            # Voting Classifier (Ensemble)
            voting_models = [
                ('rf', models['random_forest']),
                ('lr', models['logistic_regression'])
            ]
            
            if XGBOOST_AVAILABLE:
                voting_models.append(('xgb', models['xgboost']))
            
            ensemble_model = VotingClassifier(
                estimators=voting_models,
                voting='soft'
            )
            ensemble_model.fit(X_train_scaled, y_train)
            ensemble_score = ensemble_model.score(X_test_scaled, y_test)
            models['ensemble'] = ensemble_model
            results['ensemble'] = {'accuracy': ensemble_score}
            
            # Select best model
            best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
            best_model = models[best_model_name]
            best_score = results[best_model_name]['accuracy']
            
            self.trained_models = models
            self.trained_models['scaler'] = scaler
            
            return {
                'success': True,
                'best_model': best_model_name,
                'best_accuracy': best_score,
                'all_results': results,
                'models_trained': list(models.keys()),
                'scaler': scaler
            }
            
        except Exception as e:
            logger.error(f"Error training ensemble model: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def train_single_model(self, X: np.ndarray, y: np.ndarray, model_type: str = 'random_forest',
                          test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Train a single model with comprehensive evaluation
        
        Args:
            X: Feature matrix
            y: Target labels
            model_type: Type of model to train
            test_size: Proportion of test data
            random_state: Random seed
            
        Returns:
            Training results with detailed metrics
        """
        try:
            if not SKLEARN_AVAILABLE:
                return {'success': False, 'error': 'Scikit-learn not available'}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Initialize model based on type
            if model_type == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=random_state,
                    class_weight='balanced'
                )
            elif model_type == 'logistic_regression':
                model = LogisticRegression(
                    random_state=random_state,
                    class_weight='balanced',
                    max_iter=1000
                )
            elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
                model = xgb.XGBClassifier(
                    random_state=random_state,
                    eval_metric='logloss'
                )
            elif model_type == 'knn':
                model = KNeighborsClassifier(n_neighbors=5)
            else:
                # Default to Random Forest
                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=random_state,
                    class_weight='balanced'
                )
                model_type = 'random_forest'
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            y_test_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            train_accuracy = model.score(X_train_scaled, y_train)
            test_accuracy = model.score(X_test_scaled, y_test)
            
            # Classification report
            train_report = classification_report(y_train, y_train_pred, output_dict=True)
            test_report = classification_report(y_test, y_test_pred, output_dict=True)
            
            # Confusion matrix
            train_cm = confusion_matrix(y_train, y_train_pred)
            test_cm = confusion_matrix(y_test, y_test_pred)
            
            # AUC score if probabilities available
            auc_score = None
            if y_test_proba is not None:
                try:
                    auc_score = roc_auc_score(y_test, y_test_proba)
                except ValueError:
                    auc_score = None
            
            # Feature importance if available
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_.tolist()
            elif hasattr(model, 'coef_'):
                feature_importance = np.abs(model.coef_[0]).tolist()
            
            # Store trained model
            self.trained_models[model_type] = {
                'model': model,
                'scaler': scaler,
                'training_date': datetime.now(),
                'model_type': model_type
            }
            
            return {
                'success': True,
                'model_type': model_type,
                'training_accuracy': round(train_accuracy, 4),
                'test_accuracy': round(test_accuracy, 4),
                'auc_score': round(auc_score, 4) if auc_score else None,
                'train_classification_report': train_report,
                'test_classification_report': test_report,
                'train_confusion_matrix': train_cm.tolist(),
                'test_confusion_matrix': test_cm.tolist(),
                'feature_importance': feature_importance,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'model_params': model.get_params() if hasattr(model, 'get_params') else {},
                'training_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error training single model: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def cross_validate_model(self, model, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform comprehensive cross-validation on model
        """
        try:
            if not SKLEARN_AVAILABLE:
                return {'success': False, 'error': 'Scikit-learn not available'}
            
            # Stratified K-Fold for classification
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            # Cross-validation scores
            cv_accuracy = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
            cv_precision = cross_val_score(model, X, y, cv=skf, scoring='precision')
            cv_recall = cross_val_score(model, X, y, cv=skf, scoring='recall')
            cv_f1 = cross_val_score(model, X, y, cv=skf, scoring='f1')
            
            # AUC if model supports probability prediction
            cv_auc = None
            if hasattr(model, 'predict_proba'):
                try:
                    cv_auc = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
                except ValueError:
                    cv_auc = None
            
            # Detailed fold-by-fold results
            fold_results = []
            for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_fold_train, X_fold_val = X[train_idx], X[val_idx]
                y_fold_train, y_fold_val = y[train_idx], y[val_idx]
                
                # Train on fold
                model.fit(X_fold_train, y_fold_train)
                fold_pred = model.predict(X_fold_val)
                fold_accuracy = (fold_pred == y_fold_val).mean()
                
                fold_results.append({
                    'fold': i + 1,
                    'accuracy': round(fold_accuracy, 4),
                    'samples': len(y_fold_val)
                })
            
            result = {
                'success': True,
                'cv_folds': cv_folds,
                'cv_accuracy': {
                    'mean': round(cv_accuracy.mean(), 4),
                    'std': round(cv_accuracy.std(), 4),
                    'scores': [round(score, 4) for score in cv_accuracy]
                },
                'cv_precision': {
                    'mean': round(cv_precision.mean(), 4),
                    'std': round(cv_precision.std(), 4),
                    'scores': [round(score, 4) for score in cv_precision]
                },
                'cv_recall': {
                    'mean': round(cv_recall.mean(), 4),
                    'std': round(cv_recall.std(), 4),
                    'scores': [round(score, 4) for score in cv_recall]
                },
                'cv_f1': {
                    'mean': round(cv_f1.mean(), 4),
                    'std': round(cv_f1.std(), 4),
                    'scores': [round(score, 4) for score in cv_f1]
                },
                'fold_results': fold_results
            }
            
            if cv_auc is not None:
                result['cv_auc'] = {
                    'mean': round(cv_auc.mean(), 4),
                    'std': round(cv_auc.std(), 4),
                    'scores': [round(score, 4) for score in cv_auc]
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def hyperparameter_tuning(self, model, param_grid: Dict, X: np.ndarray, y: np.ndarray,
                             cv_folds: int = 5, scoring: str = 'accuracy') -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Args:
            model: Base model to tune
            param_grid: Parameter grid for tuning
            X: Feature matrix
            y: Target labels
            cv_folds: Number of CV folds
            scoring: Scoring metric
            
        Returns:
            Hyperparameter tuning results
        """
        try:
            if not SKLEARN_AVAILABLE:
                return {'success': False, 'error': 'Scikit-learn not available'}
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                model, 
                param_grid, 
                cv=cv_folds, 
                scoring=scoring, 
                n_jobs=-1
            )
            
            # Fit grid search
            grid_search.fit(X, y)
            
            # Extract results
            cv_results = pd.DataFrame(grid_search.cv_results_)
            
            # Top 5 parameter combinations
            top_results = cv_results.nlargest(5, 'mean_test_score')[
                ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
            ].to_dict('records')
            
            return {
                'success': True,
                'best_params': grid_search.best_params_,
                'best_score': round(grid_search.best_score_, 4),
                'best_model': grid_search.best_estimator_,
                'best_index': int(grid_search.best_index_),
                'top_5_results': top_results,
                'total_combinations': len(cv_results),
                'scoring_metric': scoring,
                'cv_folds': cv_folds
            }
            
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def validate_model_performance(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Validate performance of a trained model on new test data
        
        Args:
            model_name: Name of the trained model
            X_test: Test feature matrix
            y_test: Test target labels
            
        Returns:
            Validation performance metrics
        """
        try:
            if model_name not in self.trained_models:
                return {
                    'success': False,
                    'error': f'Model {model_name} not found in trained models'
                }
            
            model_data = self.trained_models[model_name]
            model = model_data['model']
            scaler = model_data['scaler']
            
            # Scale test data
            X_test_scaled = scaler.transform(X_test)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = (y_pred == y_test).mean()
            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            confusion_mat = confusion_matrix(y_test, y_pred)
            
            # AUC score if probabilities available
            auc_score = None
            if y_proba is not None:
                try:
                    auc_score = roc_auc_score(y_test, y_proba)
                except ValueError:
                    auc_score = None
            
            # Prediction confidence analysis
            confidence_analysis = None
            if y_proba is not None:
                confidence_scores = np.max(model.predict_proba(X_test_scaled), axis=1)
                confidence_analysis = {
                    'mean_confidence': round(confidence_scores.mean(), 4),
                    'min_confidence': round(confidence_scores.min(), 4),
                    'max_confidence': round(confidence_scores.max(), 4),
                    'std_confidence': round(confidence_scores.std(), 4),
                    'high_confidence_count': int(sum(confidence_scores >= 0.8)),
                    'low_confidence_count': int(sum(confidence_scores < 0.6))
                }
            
            return {
                'success': True,
                'model_name': model_name,
                'model_type': model_data['model_type'],
                'test_accuracy': round(accuracy, 4),
                'auc_score': round(auc_score, 4) if auc_score else None,
                'classification_report': classification_rep,
                'confusion_matrix': confusion_mat.tolist(),
                'confidence_analysis': confidence_analysis,
                'test_samples': len(y_test),
                'validation_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error validating model performance: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def compare_models(self, X: np.ndarray, y: np.ndarray, model_types: List[str] = None) -> Dict[str, Any]:
        """
        Compare performance of multiple models
        
        Args:
            X: Feature matrix
            y: Target labels
            model_types: List of model types to compare
            
        Returns:
            Model comparison results
        """
        try:
            if not SKLEARN_AVAILABLE:
                return {'success': False, 'error': 'Scikit-learn not available'}
            
            if model_types is None:
                model_types = ['random_forest', 'logistic_regression', 'knn']
                if XGBOOST_AVAILABLE:
                    model_types.append('xgboost')
            
            comparison_results = {}
            
            # Train and evaluate each model
            for model_type in model_types:
                result = self.train_single_model(X, y, model_type)
                if result['success']:
                    comparison_results[model_type] = {
                        'test_accuracy': result['test_accuracy'],
                        'training_accuracy': result['training_accuracy'],
                        'auc_score': result['auc_score'],
                        'test_samples': result['test_samples']
                    }
                    
                    # Add cross-validation results
                    if model_type in self.trained_models:
                        model = self.trained_models[model_type]['model']
                        cv_result = self.cross_validate_model(model, X, y)
                        if cv_result['success']:
                            comparison_results[model_type]['cv_accuracy'] = cv_result['cv_accuracy']
            
            # Rank models by test accuracy
            sorted_models = sorted(
                comparison_results.items(),
                key=lambda x: x[1]['test_accuracy'],
                reverse=True
            )
            
            # Create summary
            summary = {
                'best_model': sorted_models[0][0] if sorted_models else None,
                'best_accuracy': sorted_models[0][1]['test_accuracy'] if sorted_models else None,
                'models_compared': len(comparison_results),
                'ranking': [{'model': name, 'accuracy': data['test_accuracy']} 
                           for name, data in sorted_models]
            }
            
            return {
                'success': True,
                'comparison_results': comparison_results,
                'summary': summary,
                'comparison_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def save_trained_model(self, model_name: str, filepath: str) -> bool:
        """
        Save a trained model to file
        
        Args:
            model_name: Name of the model to save
            filepath: Path to save the model
            
        Returns:
            Success status
        """
        try:
            if model_name not in self.trained_models:
                logger.error(f"Model {model_name} not found")
                return False
            
            model_data = self.trained_models[model_name]
            
            # Save model with metadata
            save_data = {
                'model': model_data['model'],
                'scaler': model_data['scaler'],
                'model_type': model_data['model_type'],
                'training_date': model_data['training_date'].isoformat(),
                'save_date': datetime.now().isoformat()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
            
            logger.info(f"Model {model_name} saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_trained_model(self, filepath: str, model_name: str = None) -> bool:
        """
        Load a trained model from file
        
        Args:
            filepath: Path to the saved model
            model_name: Name to assign to the loaded model
            
        Returns:
            Success status
        """
        try:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            # Extract model data
            if model_name is None:
                model_name = save_data.get('model_type', 'loaded_model')
            
            self.trained_models[model_name] = {
                'model': save_data['model'],
                'scaler': save_data['scaler'],
                'model_type': save_data['model_type'],
                'training_date': datetime.fromisoformat(save_data['training_date']),
                'loaded_date': datetime.now()
            }
            
            logger.info(f"Model loaded as {model_name} from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def get_model_info(self, model_name: str = None) -> Dict[str, Any]:
        """
        Get information about trained models
        
        Args:
            model_name: Specific model name, or None for all models
            
        Returns:
            Model information
        """
        try:
            if model_name:
                if model_name not in self.trained_models:
                    return {
                        'success': False,
                        'error': f'Model {model_name} not found'
                    }
                
                model_data = self.trained_models[model_name]
                return {
                    'success': True,
                    'model_name': model_name,
                    'model_type': model_data['model_type'],
                    'training_date': model_data['training_date'].isoformat(),
                    'model_params': model_data['model'].get_params() if hasattr(model_data['model'], 'get_params') else {}
                }
            else:
                # Return info for all models
                models_info = {}
                for name, data in self.trained_models.items():
                    if name != 'scaler':  # Skip scaler entry
                        models_info[name] = {
                            'model_type': data['model_type'],
                            'training_date': data['training_date'].isoformat() if hasattr(data['training_date'], 'isoformat') else str(data['training_date'])
                        }
                
                return {
                    'success': True,
                    'total_models': len(models_info),
                    'models': models_info
                }
                
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {
                'success': False,
                'error': str(e)
            }