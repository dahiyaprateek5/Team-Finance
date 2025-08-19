from .financial_health_model import FinancialHealthModel
from .balance_sheet_generator import BalanceSheetGenerator
from .balance_sheet_models import BalanceSheetPredictionModels
from .cash_flow_generator import CashFlowGenerator
from .cash_flow_models import CashFlowPredictionModels

__version__ = "1.0.0"
__author__ = "Prateek Dahiya"
__email__ = "prateek.dahiya@liverpool.ac.uk"

# Export main classes
__all__ = [
    'FinancialHealthModel',
    'BalanceSheetGenerator', 
    'BalanceSheetPredictionModels',
    'CashFlowGenerator',
    'CashFlowPredictionModels'
]

# Configuration constants
DATABASE_TABLES = {
    'balance_sheet': 'balance_sheet_1',
    'cash_flow': 'cash_flow_statement'
}

SUPPORTED_FILE_FORMATS = [
    '.csv', '.xlsx', '.xls', '.pdf'
]

INDUSTRY_CATEGORIES = [
    'technology',
    'manufacturing', 
    'retail',
    'healthcare',
    'financial',
    'industrial_goods',
    'aerospace_defence',
    'construction_machinery',
    'biotechnology',
    'medical_equipment',
    'asset_management'
]

# Model accuracy thresholds
ACCURACY_THRESHOLDS = {
    'balance_sheet_min_accuracy': 70.0,
    'cash_flow_min_accuracy': 65.0,
    'liquidation_prediction_threshold': 0.6
}

def get_model_info():
    """Get information about available models"""
    return {
        'version': __version__,
        'author': __author__,
        'supported_formats': SUPPORTED_FILE_FORMATS,
        'database_tables': DATABASE_TABLES,
        'industry_categories': INDUSTRY_CATEGORIES,
        'accuracy_thresholds': ACCURACY_THRESHOLDS
    }

def validate_database_config(db_config):
    """
    Validate database configuration
    
    Args:
        db_config (dict): Database configuration
        
    Returns:
        bool: True if valid, False otherwise
    """
    required_keys = ['host', 'database', 'user', 'password', 'port']
    
    if not isinstance(db_config, dict):
        return False
    
    for key in required_keys:
        if key not in db_config:
            print(f"Missing required database config key: {key}")
            return False
    
    return True

def create_model_instance(model_type, db_config):
    """
    Factory function to create model instances
    
    Args:
        model_type (str): Type of model to create
        db_config (dict): Database configuration
        
    Returns:
        Model instance or None if invalid
    """
    if not validate_database_config(db_config):
        print("Invalid database configuration")
        return None
    
    model_classes = {
        'financial_health': FinancialHealthModel,
        'balance_sheet_generator': BalanceSheetGenerator,
        'balance_sheet_models': BalanceSheetPredictionModels,
        'cash_flow_generator': CashFlowGenerator,
        'cash_flow_models': CashFlowPredictionModels
    }
    
    if model_type not in model_classes:
        print(f"Unknown model type: {model_type}")
        print(f"Available types: {list(model_classes.keys())}")
        return None
    
    try:
        return model_classes[model_type](db_config)
    except Exception as e:
        print(f"Error creating {model_type} model: {e}")
        return None