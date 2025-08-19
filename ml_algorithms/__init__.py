# Import main classes for easy access
from json5 import __version__


try:
    from .base_model import BaseFinancialModel, ModelConfig, PredictionResult
    from .time_series_analyzer import (
        FinancialTimeSeriesAnalyzer,
        TimeSeriesBatchAnalyzer,
        TimeSeriesVisualizer,
        TimeSeriesComponents,
        ForecastResult,
        AnomalyDetectionResult,
        TrendType,
        SeasonalityType
    )
    from .essemble_manager import EnsembleManager, ModelWeights
    from .industry_benchmarks import IndustryBenchmarkAnalyzer
    from .knn_imputer import KNNFinancialImputer
    from .peer_analysis import PeerCompanyAnalyzer
    
except ImportError as e:
    print(f"Warning: Some modules could not be imported: {e}")
    print("Please ensure all required dependencies are installed.")

# Package-level constants
SUPPORTED_MODELS = [
    'random_forest',
    'xgboost',
    'lightgbm',
    'neural_network',
    'ensemble'
]

FINANCIAL_METRICS = [
    'revenue',
    'net_income',
    'total_assets',
    'current_ratio',
    'debt_to_equity',
    'roa',
    'roe',
    'profit_margin',
    'operating_cash_flow',
    'free_cash_flow'
]

RISK_CATEGORIES = {
    'low': 0,
    'medium': 1,
    'high': 2
}

# Model performance thresholds
PERFORMANCE_THRESHOLDS = {
    'accuracy': 0.75,
    'precision': 0.70,
    'recall': 0.70,
    'f1_score': 0.70
}

# Default configuration
DEFAULT_CONFIG = {
    'model_type': 'ensemble',
    'test_size': 0.2,
    'random_state': 42,
    'cross_validation_folds': 5,
    'feature_selection': True,
    'hyperparameter_tuning': True,
    'save_models': True
}

def get_version():
    """Get package version."""
    return __version__

def get_supported_models():
    """Get list of supported ML models."""
    return SUPPORTED_MODELS.copy()

def get_financial_metrics():
    """Get list of supported financial metrics."""
    return FINANCIAL_METRICS.copy()

def validate_model_type(model_type):
    """Validate if model type is supported."""
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model type: {model_type}. "
                        f"Supported models: {SUPPORTED_MODELS}")
    return True

def get_default_config():
    """Get default configuration for models."""
    return DEFAULT_CONFIG.copy()

# Package information
__all__ = [
    # Classes
    'BaseFinancialModel',
    'ModelConfig', 
    'PredictionResult',
    'FinancialTimeSeriesAnalyzer',
    'TimeSeriesBatchAnalyzer',
    'TimeSeriesVisualizer',
    'TimeSeriesComponents',
    'ForecastResult',
    'AnomalyDetectionResult',
    'TrendType',
    'SeasonalityType',
    'EnsembleManager',
    'ModelWeights',
    'IndustryBenchmarkAnalyzer',
    'KNNFinancialImputer',
    'PeerCompanyAnalyzer',
    
    # Functions
    'get_version',
    'get_supported_models',
    'get_financial_metrics',
    'validate_model_type',
    'get_default_config',
    
    # Constants
    'SUPPORTED_MODELS',
    'FINANCIAL_METRICS',
    'RISK_CATEGORIES',
    'PERFORMANCE_THRESHOLDS',
    'DEFAULT_CONFIG'
]