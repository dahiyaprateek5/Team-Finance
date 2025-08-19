"""
Ensemble Manager for Financial ML Models
Manages XGBoost, Random Forest, Neural Networks, and LightGBM
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import numpy as np # type: ignore
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import json

# Import ML models with error handling
print("🎯 Loading Ensemble Manager...")

# Import model classes with availability checks
model_classes = {}
model_availability = {}

# XGBoost Models
try:
    from .xgboost.xgb_regressor import XGBRegressorModel  # type: ignore
    from .xgboost.xgb_classifier import XGBClassifierModel  # type: ignore
    model_classes['xgb_regressor'] = XGBRegressorModel
    model_classes['xgb_classifier'] = XGBClassifierModel
    model_availability['xgboost'] = True
    print("✅ XGBoost models imported")
except ImportError as e:
    print(f"⚠️ XGBoost models not available: {e}")
    model_availability['xgboost'] = False

# Random Forest Models
try:
    from .random_forest.rf_regressor import RFRegressorModel  # type: ignore
    from .random_forest.rf_classifier import RFClassifierModel  # type: ignore
    model_classes['rf_regressor'] = RFRegressorModel
    model_classes['rf_classifier'] = RFClassifierModel
    model_availability['random_forest'] = True
    print("✅ Random Forest models imported")
except ImportError as e:
    print(f"⚠️ Random Forest models not available: {e}")
    model_availability['random_forest'] = False

# Neural Network Models
try:
    from .neural_networks.nn_regressor import NNRegressorModel   # type: ignore
    from .neural_networks.nn_classifier import NNClassifierModel  # type: ignore
    model_classes['nn_regressor'] = NNRegressorModel
    model_classes['nn_classifier'] = NNClassifierModel
    model_availability['neural_networks'] = True
    print("✅ Neural Network models imported")
except ImportError as e:
    print(f"⚠️ Neural Network models not available: {e}")
    model_availability['neural_networks'] = False

# LightGBM Models
try:
    from .lightgbm.lgb_regressor import LGBRegressorModel  # type: ignore
    from .lightgbm.lgb_classifier import LGBClassifierModel  # type: ignore
    model_classes['lgb_regressor'] = LGBRegressorModel  
    model_classes['lgb_classifier'] = LGBClassifierModel
    model_availability['lightgbm'] = True
    print("✅ LightGBM models imported")
except ImportError as e:
    print(f"⚠️ LightGBM models not available: {e}")
    model_availability['lightgbm'] = False

# Sklearn for data preprocessing
try:
    from sklearn.model_selection import train_test_split  # type: ignore
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder  # type: ignore
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score  # type: ignore
    SKLEARN_AVAILABLE = True
    print("✅ Scikit-learn utilities imported")
except ImportError as e:
    print(f"❌ Scikit-learn not available: {e}")
    SKLEARN_AVAILABLE = False

class EnsembleManager:
    """
    Advanced Ensemble Manager for Financial ML Models
    Coordinates XGBoost, Random Forest, Neural Networks, and LightGBM
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.model_performance = {}
        self.feature_importance = {}
        self.is_trained = False
        self.training_history = []
        self.available_models = model_classes
        self.model_availability = model_availability
        
        print(f"🎯 Ensemble Manager initialized")
        print(f"📊 Available model types: {len(self.available_models)}")
        print(f"🎯 Model availability: {self.model_availability}")
    
    def initialize_models(self, model_configs: Optional[Dict] = None) -> bool:
        """
        Initialize all available ML models with configurations
        """
        try:
            if not SKLEARN_AVAILABLE:
                print("❌ Scikit-learn required for ensemble operations")
                return False
            
            print("🚀 Initializing ensemble models...")
            
            # Default configurations for each model type
            default_configs = {
                'xgb_regressor': {
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42
                },
                'xgb_classifier': {
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42
                },
                'rf_regressor': {
                    'n_estimators': 200,
                    'max_depth': 12,
                    'random_state': 42
                },
                'rf_classifier': {
                    'n_estimators': 200,
                    'max_depth': 12,
                    'random_state': 42
                },
                'nn_regressor': {
                    'hidden_layer_sizes': (128, 64, 32),
                    'max_iter': 1000,
                    'random_state': 42
                },
                'nn_classifier': {
                    'hidden_layer_sizes': (128, 64, 32),
                    'max_iter': 1000,
                    'random_state': 42
                },
                'lgb_regressor': {
                    'n_estimators': 200,
                    'max_depth': 8,
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'verbose': -1
                },
                'lgb_classifier': {
                    'n_estimators': 200,
                    'max_depth': 8,
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'verbose': -1
                }
            }
            
            # Update with user configurations
            if model_configs:
                for model_name, config in model_configs.items():
                    if model_name in default_configs:
                        default_configs[model_name].update(config)
            
            # Initialize models
            initialized_models = 0
            for model_name, model_class in self.available_models.items():
                try:
                    config = default_configs.get(model_name, {})
                    self.models[model_name] = model_class(**config)
                    initialized_models += 1
                    print(f"✅ {model_name} initialized")
                except Exception as e:
                    print(f"❌ {model_name} initialization failed: {e}")
            
            # Initialize scalers and encoders
            self.scalers['standard'] = StandardScaler()
            self.scalers['minmax'] = MinMaxScaler()
            self.label_encoders['health_category'] = LabelEncoder()
            
            print(f"🎉 Ensemble initialization complete!")
            print(f"   Models initialized: {initialized_models}/{len(self.available_models)}")
            
            return initialized_models > 0
            
        except Exception as e:
            print(f"❌ Ensemble initialization failed: {e}")
            return False
    
    def prepare_data(self, X: np.ndarray, y_reg: np.ndarray, y_clf: np.ndarray, 
                    test_size: float = 0.2) -> Tuple:
        """
        Prepare and split data for training
        """
        try:
            # Split data
            X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
                X, y_reg, y_clf, test_size=test_size, random_state=42, stratify=y_clf
            )
            
            # Scale features
            X_train_std = self.scalers['standard'].fit_transform(X_train)
            X_test_std = self.scalers['standard'].transform(X_test)
            
            X_train_mm = self.scalers['minmax'].fit_transform(X_train)
            X_test_mm = self.scalers['minmax'].transform(X_test)
            
            # Encode classification labels
            y_clf_train_encoded = self.label_encoders['health_category'].fit_transform(y_clf_train)
            y_clf_test_encoded = self.label_encoders['health_category'].transform(y_clf_test)
            
            return (X_train_std, X_test_std, X_train_mm, X_test_mm, 
                   y_reg_train, y_reg_test, y_clf_train_encoded, y_clf_test_encoded)
            
        except Exception as e:
            print(f"❌ Data preparation failed: {e}")
            return None
    
    def train_ensemble(self, X: np.ndarray, y_reg: np.ndarray, y_clf: np.ndarray, 
                      X_val: Optional[np.ndarray] = None, 
                      y_reg_val: Optional[np.ndarray] = None, 
                      y_clf_val: Optional[np.ndarray] = None) -> Optional[Dict]:
        """
        Train all models in the ensemble
        """
        try:
            if not self.models:
                print("❌ No models initialized. Call initialize_models() first.")
                return None
            
            print(f"🚀 Training ensemble with {len(self.models)} models...")
            
            # Prepare data if validation not provided
            if X_val is None:
                data_prep = self.prepare_data(X, y_reg, y_clf)
                if data_prep is None:
                    return None
                
                (X_train_std, X_test_std, X_train_mm, X_test_mm, 
                 y_reg_train, y_reg_test, y_clf_train, y_clf_test) = data_prep
            else:
                # Use provided validation data
                X_train_std = self.scalers['standard'].fit_transform(X)
                X_test_std = self.scalers['standard'].transform(X_val)
                
                X_train_mm = self.scalers['minmax'].fit_transform(X)
                X_test_mm = self.scalers['minmax'].transform(X_val)
                
                y_reg_train, y_reg_test = y_reg, y_reg_val
                y_clf_train = self.label_encoders['health_category'].fit_transform(y_clf)
                y_clf_test = self.label_encoders['health_category'].transform(y_clf_val)
            
            # Train each model
            training_results = {}
            training_start_time = datetime.now()
            
            for model_name, model in self.models.items():
                try:
                    print(f"🔄 Training {model_name}...")
                    
                    # Choose appropriate data scaling
                    if 'nn_' in model_name:  # Neural networks work better with MinMax scaling
                        X_train_scaled = X_train_mm
                        X_test_scaled = X_test_mm
                    else:
                        X_train_scaled = X_train_std
                        X_test_scaled = X_test_std
                    
                    # Choose target variable
                    if 'regressor' in model_name:
                        y_train = y_reg_train
                        y_test = y_reg_test
                    else:  # classifier
                        y_train = y_clf_train
                        y_test = y_clf_test
                    
                    # Train model
                    success = model.train(X_train_scaled, y_train)
                    
                    if success:
                        # Evaluate model
                        performance = model.evaluate(X_test_scaled, y_test)
                        self.model_performance[model_name] = performance
                        
                        # Store feature importance if available
                        if hasattr(model, 'feature_importance') and model.feature_importance is not None:
                            self.feature_importance[model_name] = model.feature_importance.tolist()
                        
                        training_results[model_name] = {
                            'success': True,
                            'performance': performance,
                            'training_time': model.training_time
                        }
                        
                        print(f"✅ {model_name} trained successfully")
                        
                        # Print key performance metric
                        if 'regressor' in model_name:
                            r2 = performance.get('r2_score', 0)
                            print(f"   R² Score: {r2:.3f}")
                        else:
                            acc = performance.get('accuracy', 0)
                            print(f"   Accuracy: {acc:.3f}")
                    else:
                        training_results[model_name] = {
                            'success': False,
                            'error': 'Training failed'
                        }
                        print(f"❌ {model_name} training failed")
                        
                except Exception as e:
                    print(f"❌ {model_name} training error: {e}")
                    training_results[model_name] = {
                        'success': False,
                        'error': str(e)
                    }
            
            training_end_time = datetime.now()
            total_training_time = (training_end_time - training_start_time).total_seconds()
            
            # Mark as trained if at least one model succeeded
            successful_models = sum(1 for result in training_results.values() if result.get('success', False))
            self.is_trained = successful_models > 0
            
            # Create training session record
            training_session = {
                'timestamp': training_start_time.isoformat(),
                'total_training_time': total_training_time,
                'models_attempted': len(self.models),
                'models_successful': successful_models,
                'training_results': training_results,
                'model_performance': self.model_performance,
                'feature_importance': self.feature_importance
            }
            
            # Add to training history
            self.training_history.append(training_session)
            
            print(f"\n🎉 Ensemble training completed!")
            print(f"   Total time: {total_training_time:.2f} seconds")
            print(f"   Successful models: {successful_models}/{len(self.models)}")
            
            return training_session
            
        except Exception as e:
            print(f"❌ Ensemble training failed: {e}")
            return None
    
    def predict_ensemble(self, X: np.ndarray) -> Optional[Dict]:
        """
        Make ensemble predictions using all trained models
        """
        try:
            if not self.is_trained:
                print("❌ Ensemble not trained yet")
                return None
            
            predictions = {}
            regression_predictions = []
            classification_predictions = []
            
            for model_name, model in self.models.items():
                if model.is_trained:
                    try:
                        # Choose appropriate scaling
                        if 'nn_' in model_name:
                            X_scaled = self.scalers['minmax'].transform(X)
                        else:
                            X_scaled = self.scalers['standard'].transform(X)
                        
                        # Make prediction
                        pred = model.predict(X_scaled)
                        predictions[model_name] = pred.tolist() if hasattr(pred, 'tolist') else pred
                        
                        # Collect for ensemble
                        if 'regressor' in model_name:
                            weight = self.model_performance.get(model_name, {}).get('r2_score', 0.5)
                            regression_predictions.append((pred[0] if len(pred) > 0 else 0, weight))
                        else:  # classifier
                            classification_predictions.append(pred[0] if len(pred) > 0 else 0)
                            
                    except Exception as e:
                        print(f"⚠️ {model_name} prediction failed: {e}")
            
            # Calculate ensemble predictions
            ensemble_regression = None
            if regression_predictions:
                weighted_sum = sum(pred * weight for pred, weight in regression_predictions)
                total_weight = sum(weight for _, weight in regression_predictions)
                ensemble_regression = weighted_sum / total_weight if total_weight > 0 else None
            
            ensemble_classification = None
            if classification_predictions:
                # Majority voting
                ensemble_classification = max(set(classification_predictions), 
                                            key=classification_predictions.count)
                
                # Decode if needed
                try:
                    ensemble_classification = self.label_encoders['health_category'].inverse_transform([ensemble_classification])[0]
                except:
                    pass
            
            # Calculate confidence
            confidence = 75.0
            if regression_predictions and len(regression_predictions) > 1:
                pred_values = [pred for pred, _ in regression_predictions]
                std_dev = np.std(pred_values)
                confidence = max(60.0, 95.0 - (std_dev * 2))
            
            return {
                'individual_predictions': predictions,
                'ensemble_regression': ensemble_regression,
                'ensemble_classification': ensemble_classification,
                'prediction_confidence': round(confidence, 2),
                'models_used': list(predictions.keys()),
                'total_models': len(self.models),
                'prediction_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Ensemble prediction failed: {e}")
            return None
    
    # NEW METHOD: Comprehensive financial analysis prediction
    def predict_comprehensive(self, financial_data: Dict, algorithms: List[str] = None, analysis_type: str = 'risk_assessment') -> Dict:
        """
        Comprehensive financial analysis prediction - MISSING METHOD
        This is what your frontend is calling but doesn't exist!
        """
        try:
            print(f"🎯 Starting comprehensive analysis for: {analysis_type}")
            print(f"📊 Requested algorithms: {algorithms}")
            
            # Convert financial data to numpy array format
            feature_array = self._prepare_financial_features(financial_data)
            
            if feature_array is None:
                return {
                    'success': False,
                    'error': 'Invalid financial data format',
                    'health_score': 50,
                    'liquidation_risk': 50,
                    'risk_level': 'Unknown',
                    'confidence': 0
                }
            
            # Use ensemble prediction if trained
            if self.is_trained:
                ensemble_result = self.predict_ensemble(feature_array)
                
                if ensemble_result:
                    # Convert ensemble prediction to comprehensive format
                    health_score = self._calculate_health_score(ensemble_result, financial_data)
                    liquidation_risk = 100 - health_score
                    risk_level = self._determine_risk_level(liquidation_risk)
                    recommendations = self._generate_recommendations(financial_data, health_score)
                    
                    return {
                        'success': True,
                        'health_score': health_score,
                        'liquidation_risk': liquidation_risk,
                        'risk_level': risk_level,
                        'confidence': ensemble_result.get('prediction_confidence', 75),
                        'recommendations': recommendations,
                        'analysis_type': analysis_type,
                        'model_used': 'ensemble',
                        'models_used': ensemble_result.get('models_used', []),
                        'ensemble_details': ensemble_result
                    }
            
            # Fallback to statistical analysis
            return self._statistical_fallback_analysis(financial_data, analysis_type)
            
        except Exception as e:
            print(f"❌ Comprehensive analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'health_score': 50,
                'liquidation_risk': 50,
                'risk_level': 'Unknown',
                'confidence': 0
            }

    # NEW METHOD: Prepare financial features
    def _prepare_financial_features(self, financial_data: Dict) -> Optional[np.ndarray]:
        """
        Convert financial dictionary to numpy array for ML models
        """
        try:
            # Expected feature order (adjust based on your training data)
            feature_keys = [
                'net_income',
                'operating_cash_flow', 
                'free_cash_flow',
                'current_ratio',
                'debt_equity_ratio',
                'interest_coverage_ratio',
                'working_capital_change',
                'accounts_receivable_change',
                'inventory_change',
                'accounts_payable_change',
                'capital_expenditure',
                'depreciation',
                'ocf_to_net_income_ratio'
            ]
            
            # Extract and convert features
            features = []
            for key in feature_keys:
                value = financial_data.get(key, 0)
                # Handle string values and convert to float
                if isinstance(value, str):
                    # Remove commas, dollar signs, parentheses
                    value = value.replace(',', '').replace('$', '').replace('(', '-').replace(')', '')
                    try:
                        value = float(value)
                    except:
                        value = 0.0
                elif value is None:
                    value = 0.0
                features.append(float(value))
            
            # Return as 2D array (required for sklearn)
            return np.array([features])
            
        except Exception as e:
            print(f"❌ Feature preparation failed: {e}")
            return None

    # NEW METHOD: Calculate health score
    def _calculate_health_score(self, ensemble_result: Dict, financial_data: Dict) -> int:
        """
        Calculate health score from ensemble prediction and financial data
        """
        try:
            # Start with ensemble regression prediction if available
            base_score = 50
            
            if ensemble_result.get('ensemble_regression') is not None:
                # Assume regression predicts health score (0-100)
                base_score = max(0, min(100, float(ensemble_result['ensemble_regression'])))
            
            # Adjust based on key financial indicators
            adjustments = 0
            
            # Operating cash flow check
            ocf = float(financial_data.get('operating_cash_flow', 0))
            if ocf > 0:
                adjustments += 10
            elif ocf < 0:
                adjustments -= 15
            
            # Profitability check
            net_income = float(financial_data.get('net_income', 0))
            if net_income > 0:
                adjustments += 10
            elif net_income < 0:
                adjustments -= 10
            
            # Liquidity check
            current_ratio = float(financial_data.get('current_ratio', 1))
            if current_ratio >= 2:
                adjustments += 5
            elif current_ratio < 1:
                adjustments -= 10
            
            # Debt check
            debt_equity = float(financial_data.get('debt_equity_ratio', 0))
            if debt_equity < 0.3:
                adjustments += 5
            elif debt_equity > 1:
                adjustments -= 10
            
            final_score = base_score + adjustments
            return max(0, min(100, int(final_score)))
            
        except Exception as e:
            print(f"❌ Health score calculation failed: {e}")
            return 50

    # NEW METHOD: Determine risk level
    def _determine_risk_level(self, liquidation_risk: float) -> str:
        """
        Determine risk level based on liquidation risk percentage
        """
        if liquidation_risk < 20:
            return 'Low'
        elif liquidation_risk < 40:
            return 'Medium-Low'
        elif liquidation_risk < 60:
            return 'Medium'
        elif liquidation_risk < 80:
            return 'High'
        else:
            return 'Critical'

    # NEW METHOD: Generate recommendations
    def _generate_recommendations(self, financial_data: Dict, health_score: int) -> List[str]:
        """
        Generate actionable recommendations based on financial analysis
        """
        recommendations = []
        
        try:
            # Operating cash flow recommendations
            ocf = float(financial_data.get('operating_cash_flow', 0))
            if ocf <= 0:
                recommendations.append("Focus on improving operational efficiency to generate positive cash flow")
            
            # Liquidity recommendations
            current_ratio = float(financial_data.get('current_ratio', 1))
            if current_ratio < 1.5:
                recommendations.append("Improve liquidity by increasing current assets or reducing short-term liabilities")
            
            # Debt management recommendations
            debt_equity = float(financial_data.get('debt_equity_ratio', 0))
            if debt_equity > 0.6:
                recommendations.append("Consider reducing debt burden or increasing equity capital")
            
            # Profitability recommendations
            net_income = float(financial_data.get('net_income', 0))
            if net_income <= 0:
                recommendations.append("Review cost structure and pricing strategy to improve profitability")
            
            # Free cash flow recommendations
            fcf = float(financial_data.get('free_cash_flow', 0))
            if fcf <= 0:
                recommendations.append("Focus on generating positive free cash flow for long-term sustainability")
            
            # Working capital recommendations
            wc_change = float(financial_data.get('working_capital_change', 0))
            if wc_change < 0:
                recommendations.append("Optimize working capital management to free up cash")
            
            # General health-based recommendations
            if health_score < 30:
                recommendations.append("Consider comprehensive financial restructuring with professional guidance")
            elif health_score < 50:
                recommendations.append("Implement immediate cost reduction measures and cash preservation strategies")
            elif health_score < 70:
                recommendations.append("Monitor key financial metrics closely and maintain conservative approach")
            
            return recommendations[:5]  # Limit to top 5 recommendations
            
        except Exception as e:
            print(f"❌ Recommendation generation failed: {e}")
            return ["Conduct detailed financial review with professional guidance"]

    # NEW METHOD: Statistical fallback analysis
    def _statistical_fallback_analysis(self, financial_data: Dict, analysis_type: str) -> Dict:
        """
        Fallback analysis when ML models are not available or trained
        """
        try:
            print("📊 Using statistical fallback analysis")
            
            # Extract key metrics
            net_income = float(financial_data.get('net_income', 0))
            operating_cf = float(financial_data.get('operating_cash_flow', 0))
            current_ratio = float(financial_data.get('current_ratio', 1))
            debt_equity_ratio = float(financial_data.get('debt_equity_ratio', 0))
            free_cash_flow = float(financial_data.get('free_cash_flow', 0))
            
            # Calculate health score using weighted scoring
            health_score = 50  # Start neutral
            
            # Operating Cash Flow (30% weight)
            if operating_cf > 0:
                health_score += 20
            elif operating_cf < 0:
                health_score -= 25
            
            # Profitability (25% weight)
            if net_income > 0:
                health_score += 15
            elif net_income < 0:
                health_score -= 20
            
            # Liquidity (20% weight)
            if current_ratio >= 2:
                health_score += 10
            elif current_ratio >= 1.2:
                health_score += 5
            elif current_ratio < 1:
                health_score -= 15
            
            # Leverage (15% weight)
            if debt_equity_ratio < 0.3:
                health_score += 8
            elif debt_equity_ratio < 0.6:
                health_score += 3
            elif debt_equity_ratio > 1:
                health_score -= 12
            
            # Free Cash Flow (10% weight)
            if free_cash_flow > 0:
                health_score += 7
            elif free_cash_flow < 0:
                health_score -= 7
            
            # Ensure score is within bounds
            health_score = max(0, min(100, int(health_score)))
            
            # Calculate liquidation risk and determine level
            liquidation_risk = 100 - health_score
            risk_level = self._determine_risk_level(liquidation_risk)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(financial_data, health_score)
            
            return {
                'success': True,
                'health_score': health_score,
                'liquidation_risk': liquidation_risk,
                'risk_level': risk_level,
                'confidence': 70,  # Lower confidence for statistical method
                'recommendations': recommendations,
                'analysis_type': f'{analysis_type} (Statistical)',
                'model_used': 'statistical_fallback',
                'method': 'weighted_scoring'
            }
            
        except Exception as e:
            print(f"❌ Statistical fallback failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'health_score': 50,
                'liquidation_risk': 50,
                'risk_level': 'Unknown',
                'confidence': 0
            }

    # NEW METHOD: Get available methods
    def get_available_methods(self) -> Dict:
        """
        Return available methods for API endpoint
        """
        return {
            'available_methods': [
                'predict',
                'predict_comprehensive',  # Now available!
                'predict_ensemble',
                'train_ensemble',
                'get_ensemble_summary',
                'get_feature_importance',
                'get_model_performance'
            ],
            'model_availability': self.model_availability,
            'is_trained': self.is_trained,
            'total_models': len(self.models),
            'status': 'active'
        }
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance from all models"""
        return self.feature_importance.copy()
    
    def get_model_performance(self) -> Dict:
        """Get performance metrics for all models"""
        return self.model_performance.copy()
    
    def get_ensemble_summary(self) -> Dict:
        """Get comprehensive ensemble summary"""
        try:
            trained_models = sum(1 for model in self.models.values() if model.is_trained)
            
            summary = {
                'ensemble_status': {
                    'is_trained': self.is_trained,
                    'total_models': len(self.models),
                    'trained_models': trained_models,
                    'model_availability': self.model_availability
                },
                'model_performance': self.model_performance,
                'feature_importance': self.feature_importance,
                'training_history': self.training_history,
                'model_details': {
                    name: {
                        'is_trained': model.is_trained,
                        'model_name': model.model_name,
                        'training_time': getattr(model, 'training_time', 0)
                    } for name, model in self.models.items()
                },
                'generated_at': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            print(f"❌ Ensemble summary failed: {e}")
            return {}
    
    def save_ensemble(self, filepath: str) -> bool:
        """Save ensemble configuration and performance"""
        try:
            ensemble_data = {
                'ensemble_summary': self.get_ensemble_summary(),
                'model_configs': {
                    name: getattr(model, 'model_params', {}) 
                    for name, model in self.models.items()
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(ensemble_data, f, indent=2, default=str)
            
            print(f"✅ Ensemble saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Ensemble save failed: {e}")
            return False
    
    def get_best_models(self, top_n: int = 3) -> Dict:
        """Get top performing models"""
        try:
            # Separate regressors and classifiers
            regressors = {name: perf for name, perf in self.model_performance.items() 
                         if 'regressor' in name}
            classifiers = {name: perf for name, perf in self.model_performance.items() 
                          if 'classifier' in name}
            
            # Sort by performance
            best_regressors = sorted(regressors.items(), 
                                   key=lambda x: x[1].get('r2_score', 0), 
                                   reverse=True)[:top_n]
            
            best_classifiers = sorted(classifiers.items(), 
                                    key=lambda x: x[1].get('accuracy', 0), 
                                    reverse=True)[:top_n]
            
            return {
                'best_regressors': [{'model': name, 'r2_score': perf.get('r2_score', 0)} 
                                   for name, perf in best_regressors],
                'best_classifiers': [{'model': name, 'accuracy': perf.get('accuracy', 0)} 
                                    for name, perf in best_classifiers]
            }
            
        except Exception as e:
            print(f"❌ Best models retrieval failed: {e}")
            return {}

# Global ensemble manager instance
ensemble_manager = EnsembleManager()

# Print initialization status
print("=" * 60)
print("🎯 ENSEMBLE MANAGER READY")
print(f"📊 Available Models: {len(model_classes)}")
print(f"🎯 Sklearn Available: {'✅' if SKLEARN_AVAILABLE else '❌'}")
print("=" * 60)