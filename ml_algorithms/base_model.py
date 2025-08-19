"""
Base Financial Model Module
============================

This module provides base classes and utilities for financial risk assessment models.
It includes abstract base classes that all specific models should inherit from,
ensuring consistent interfaces and functionality across different model types.

Author: Prateek Dahiya
Project: Financial Risk Assessment Model for Small Companies and Startups
"""

import abc
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Enumeration of supported model types."""
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"
    TIME_SERIES = "time_series"

class RiskLevel(Enum):
    """Risk level classifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class PredictionType(Enum):
    """Types of predictions the model can make."""
    LIQUIDATION_RISK = "liquidation_risk"
    FINANCIAL_HEALTH = "financial_health"
    CREDIT_SCORE = "credit_score"
    CASH_FLOW_FORECAST = "cash_flow_forecast"

@dataclass
class ModelConfig:
    """Configuration class for financial models."""
    model_type: ModelType
    prediction_type: PredictionType
    features: List[str]
    target_column: str
    test_size: float = 0.2
    random_state: int = 42
    cross_validation_folds: int = 5
    hyperparameter_tuning: bool = True
    feature_selection: bool = True
    scale_features: bool = True
    handle_imbalance: bool = True
    save_model: bool = True
    model_save_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        config_dict = asdict(self)
        # Convert enums to strings
        config_dict['model_type'] = self.model_type.value
        config_dict['prediction_type'] = self.prediction_type.value
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary."""
        config_dict = config_dict.copy()
        config_dict['model_type'] = ModelType(config_dict['model_type'])
        config_dict['prediction_type'] = PredictionType(config_dict['prediction_type'])
        return cls(**config_dict)

@dataclass
class PredictionResult:
    """Result of model prediction."""
    prediction: Union[float, int, str]
    probability: Optional[float] = None
    risk_level: Optional[RiskLevel] = None
    confidence: Optional[float] = None
    explanation: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, float]] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        result_dict = asdict(self)
        if self.risk_level:
            result_dict['risk_level'] = self.risk_level.value
        if self.timestamp:
            result_dict['timestamp'] = self.timestamp.isoformat()
        return result_dict

@dataclass
class ModelPerformance:
    """Model performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None
    classification_report: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert performance to dictionary."""
        return asdict(self)

class BaseFinancialModel(abc.ABC):
    """
    Abstract base class for all financial risk assessment models.
    
    This class defines the interface that all financial models must implement,
    ensuring consistency across different model types and enabling 
    interchangeable usage in the application.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the base financial model.
        
        Args:
            config (ModelConfig): Model configuration
        """
        self.config = config
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.target_name = None
        self.scaler = None
        self.feature_selector = None
        self.performance_metrics = None
        self.training_history = []
        
        # Setup logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
    @abc.abstractmethod
    def train(self, 
              X_train: pd.DataFrame, 
              y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> ModelPerformance:
        """
        Train the model with provided data.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training targets
            X_val (pd.DataFrame, optional): Validation features
            y_val (pd.Series, optional): Validation targets
            
        Returns:
            ModelPerformance: Training performance metrics
        """
        pass
    
    @abc.abstractmethod
    def predict(self, X: pd.DataFrame) -> List[PredictionResult]:
        """
        Make predictions on new data.
        
        Args:
            X (pd.DataFrame): Features for prediction
            
        Returns:
            List[PredictionResult]: Prediction results
        """
        pass
    
    @abc.abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X (pd.DataFrame): Features for prediction
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        pass
    
    def preprocess_data(self, 
                       X: pd.DataFrame, 
                       y: Optional[pd.Series] = None,
                       is_training: bool = False) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Preprocess input data (scaling, feature selection, etc.).
        
        Args:
            X (pd.DataFrame): Input features
            y (pd.Series, optional): Target variable
            is_training (bool): Whether this is training data
            
        Returns:
            Tuple[pd.DataFrame, Optional[pd.Series]]: Preprocessed data
        """
        try:
            X_processed = X.copy()
            
            # Handle missing values
            X_processed = self._handle_missing_values(X_processed)
            
            # Feature engineering
            X_processed = self._engineer_features(X_processed)
            
            # Feature scaling
            if self.config.scale_features:
                X_processed = self._scale_features(X_processed, is_training)
            
            # Feature selection
            if self.config.feature_selection and is_training:
                X_processed = self._select_features(X_processed, y)
            elif self.config.feature_selection and not is_training and self.feature_selector:
                X_processed = pd.DataFrame(
                    self.feature_selector.transform(X_processed),
                    columns=self.feature_names,
                    index=X_processed.index
                )
            
            return X_processed, y
            
        except Exception as e:
            self.logger.error(f"Error in data preprocessing: {e}")
            raise
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Simple imputation - can be enhanced
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        
        categorical_cols = X.select_dtypes(include=['object']).columns
        X[categorical_cols] = X[categorical_cols].fillna(X[categorical_cols].mode().iloc[0])
        
        return X
    
    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Engineer new features from existing ones."""
        X_engineered = X.copy()
        
        # Financial ratios and derived features
        try:
            # Liquidity ratios
            if 'current_assets' in X.columns and 'current_liabilities' in X.columns:
                X_engineered['current_ratio'] = X['current_assets'] / X['current_liabilities']
            
            # Profitability ratios
            if 'net_income' in X.columns and 'total_assets' in X.columns:
                X_engineered['roa'] = X['net_income'] / X['total_assets']
            
            if 'net_income' in X.columns and 'total_equity' in X.columns:
                X_engineered['roe'] = X['net_income'] / X['total_equity']
            
            # Leverage ratios
            if 'total_debt' in X.columns and 'total_equity' in X.columns:
                X_engineered['debt_to_equity'] = X['total_debt'] / X['total_equity']
            
            # Efficiency ratios
            if 'revenue' in X.columns and 'total_assets' in X.columns:
                X_engineered['asset_turnover'] = X['revenue'] / X['total_assets']
            
            # Growth ratios (if historical data available)
            for col in ['revenue', 'net_income', 'total_assets']:
                if f'{col}_prev_year' in X.columns and col in X.columns:
                    X_engineered[f'{col}_growth'] = (X[col] - X[f'{col}_prev_year']) / X[f'{col}_prev_year']
        
        except Exception as e:
            self.logger.warning(f"Feature engineering warning: {e}")
        
        return X_engineered
    
    def _scale_features(self, X: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Scale features using StandardScaler."""
        from sklearn.preprocessing import StandardScaler
        
        if is_training:
            self.scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Please train the model first.")
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        
        return X_scaled
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Select most important features."""
        from sklearn.feature_selection import SelectKBest, f_classif
        
        # Select top features
        k = min(20, len(X.columns))  # Select top 20 features or all if less
        self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        
        X_selected = self.feature_selector.fit_transform(X, y)
        selected_features = X.columns[self.feature_selector.get_support()]
        self.feature_names = list(selected_features)
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> ModelPerformance:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test targets
            
        Returns:
            ModelPerformance: Performance metrics
        """
        try:
            # Preprocess test data
            X_processed, _ = self.preprocess_data(X_test, is_training=False)
            
            # Make predictions
            predictions = [result.prediction for result in self.predict(X_processed)]
            probabilities = self.predict_proba(X_processed)
            
            # Calculate metrics
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
                roc_auc_score, confusion_matrix, classification_report
            )
            
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, average='weighted')
            recall = recall_score(y_test, predictions, average='weighted')
            f1 = f1_score(y_test, predictions, average='weighted')
            
            # AUC-ROC for binary classification
            auc_roc = None
            if len(np.unique(y_test)) == 2 and probabilities.shape[1] == 2:
                auc_roc = roc_auc_score(y_test, probabilities[:, 1])
            
            conf_matrix = confusion_matrix(y_test, predictions).tolist()
            class_report = classification_report(y_test, predictions, output_dict=True)
            
            # Feature importance
            feature_importance = self.get_feature_importance()
            
            self.performance_metrics = ModelPerformance(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                auc_roc=auc_roc,
                confusion_matrix=conf_matrix,
                classification_report=class_report,
                feature_importance=feature_importance
            )
            
            return self.performance_metrics
            
        except Exception as e:
            self.logger.error(f"Error during model evaluation: {e}")
            raise
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores."""
        if not self.is_trained:
            return None
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance_scores = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                importance_scores = np.abs(self.model.coef_[0])
            else:
                return None
            
            if self.feature_names:
                return dict(zip(self.feature_names, importance_scores))
            else:
                return dict(zip(self.config.features, importance_scores))
                
        except Exception as e:
            self.logger.warning(f"Could not extract feature importance: {e}")
            return None
    
    def save_model(self, filepath: Optional[str] = None) -> str:
        """
        Save the trained model to disk.
        
        Args:
            filepath (str, optional): Path to save the model
            
        Returns:
            str: Path where model was saved
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        if filepath is None:
            filepath = self.config.model_save_path or f"model_{self.config.model_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and associated objects
        model_data = {
            'model': self.model,
            'config': self.config.to_dict(),
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names,
            'performance_metrics': self.performance_metrics.to_dict() if self.performance_metrics else None,
            'training_history': self.training_history,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {filepath}")
        return filepath
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.config = ModelConfig.from_dict(model_data['config'])
            self.scaler = model_data.get('scaler')
            self.feature_selector = model_data.get('feature_selector')
            self.feature_names = model_data.get('feature_names')
            self.training_history = model_data.get('training_history', [])
            
            if model_data.get('performance_metrics'):
                self.performance_metrics = ModelPerformance(**model_data['performance_metrics'])
            
            self.is_trained = True
            self.logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Perform cross-validation on the model.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Targets
            
        Returns:
            Dict[str, float]: Cross-validation scores
        """
        from sklearn.model_selection import cross_val_score
        
        try:
            # Preprocess data
            X_processed, _ = self.preprocess_data(X, y, is_training=True)
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                self.model, X_processed, y, 
                cv=self.config.cross_validation_folds,
                scoring='accuracy'
            )
            
            return {
                'mean_cv_score': float(cv_scores.mean()),
                'std_cv_score': float(cv_scores.std()),
                'cv_scores': cv_scores.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Error during cross-validation: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'model_type': self.config.model_type.value,
            'prediction_type': self.config.prediction_type.value,
            'is_trained': self.is_trained,
            'features_count': len(self.feature_names) if self.feature_names else len(self.config.features),
            'config': self.config.to_dict(),
            'performance': self.performance_metrics.to_dict() if self.performance_metrics else None,
            'feature_names': self.feature_names or self.config.features
        }
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return (f"{self.__class__.__name__}("
                f"model_type={self.config.model_type.value}, "
                f"trained={self.is_trained})")

# Utility functions
def create_model_config(
    model_type: str,
    prediction_type: str,
    features: List[str],
    target_column: str,
    **kwargs
) -> ModelConfig:
    """
    Create a model configuration with validation.
    
    Args:
        model_type (str): Type of model
        prediction_type (str): Type of prediction
        features (List[str]): Feature columns
        target_column (str): Target column
        **kwargs: Additional configuration parameters
        
    Returns:
        ModelConfig: Validated model configuration
    """
    return ModelConfig(
        model_type=ModelType(model_type),
        prediction_type=PredictionType(prediction_type),
        features=features,
        target_column=target_column,
        **kwargs
    )

def validate_data(X: pd.DataFrame, y: Optional[pd.Series] = None) -> bool:
    """
    Validate input data for model training/prediction.
    
    Args:
        X (pd.DataFrame): Feature data
        y (pd.Series, optional): Target data
        
    Returns:
        bool: True if data is valid
        
    Raises:
        ValueError: If data validation fails
    """
    if X.empty:
        raise ValueError("Feature data cannot be empty")
    
    if X.isnull().all().any():
        raise ValueError("Some columns contain only null values")
    
    if y is not None:
        if len(X) != len(y):
            raise ValueError("Feature and target data must have same length")
        
        if y.isnull().all():
            raise ValueError("Target data cannot be all null")
    
    return True