# Import all services for easy access
try:
    from .ai_service import AIAnalysisService, AnalysisRequest, AnalysisResponse
    from .balance_sheet_service import BalanceSheetService, BalanceSheetAnalysis
    from .cash_flow_service import CashFlowService, CashFlowAnalysis
    from .data_imputation_service import DataImputationService, ImputationResult
    from .financial_processing_service import FinancialProcessingService, ProcessingResult
    
    # Service status
    SERVICES_LOADED = True
    
except ImportError as e:
    print(f"Warning: Some services could not be imported: {e}")
    print("Please ensure all required dependencies are installed.")
    SERVICES_LOADED = False

# Service configuration
SERVICE_CONFIG = {
    'ai_service': {
        'enabled': True,
        'model_path': './models/',
        'confidence_threshold': 0.75
    },
    'balance_sheet_service': {
        'enabled': True,
        'validation_enabled': True,
        'auto_correction': True
    },
    'cash_flow_service': {
        'enabled': True,
        'forecasting_enabled': True,
        'historical_periods': 3
    },
    'data_imputation_service': {
        'enabled': True,
        'imputation_methods': ['knn', 'random_forest', 'industry_benchmark'],
        'validation_split': 0.2
    },
    'financial_processing_service': {
        'enabled': True,
        'batch_processing': True,
        'parallel_processing': True,
        'max_workers': 4
    }
}

# Supported file formats
SUPPORTED_FORMATS = {
    'excel': ['.xlsx', '.xls'],
    'csv': ['.csv'],
    'pdf': ['.pdf'],
    'json': ['.json'],
    'xml': ['.xml']
}

# Financial statement types
STATEMENT_TYPES = {
    'balance_sheet': 'Balance Sheet',
    'income_statement': 'Income Statement',
    'cash_flow_statement': 'Cash Flow Statement',
    'statement_of_equity': 'Statement of Equity'
}

# Analysis types
ANALYSIS_TYPES = {
    'liquidation_risk': 'Liquidation Risk Analysis',
    'financial_health': 'Financial Health Assessment',
    'cash_flow_forecast': 'Cash Flow Forecasting',
    'peer_comparison': 'Peer Company Comparison',
    'industry_benchmark': 'Industry Benchmark Analysis',
    'time_series_analysis': 'Time Series Analysis'
}

def get_service_status() -> dict:
    """Get status of all services."""
    return {
        'services_loaded': SERVICES_LOADED,
        'available_services': list(SERVICE_CONFIG.keys()),
        'supported_formats': SUPPORTED_FORMATS,
        'analysis_types': list(ANALYSIS_TYPES.keys())
    }

def get_service_config(service_name: str) -> dict:
    """Get configuration for a specific service."""
    return SERVICE_CONFIG.get(service_name, {})

def validate_file_format(filename: str) -> bool:
    """Validate if file format is supported."""
    import os
    file_ext = os.path.splitext(filename)[1].lower()
    
    for format_type, extensions in SUPPORTED_FORMATS.items():
        if file_ext in extensions:
            return True
    return False

def get_supported_formats() -> list:
    """Get list of all supported file formats."""
    formats = []
    for ext_list in SUPPORTED_FORMATS.values():
        formats.extend(ext_list)
    return formats

# Service factory functions
def create_ai_service(**kwargs):
    """Create AI Analysis Service instance."""
    if not SERVICES_LOADED:
        raise ImportError("Services not properly loaded")
    return AIAnalysisService(**kwargs)

def create_balance_sheet_service(**kwargs):
    """Create Balance Sheet Service instance."""
    if not SERVICES_LOADED:
        raise ImportError("Services not properly loaded")
    return BalanceSheetService(**kwargs)

def create_cash_flow_service(**kwargs):
    """Create Cash Flow Service instance."""
    if not SERVICES_LOADED:
        raise ImportError("Services not properly loaded")
    return CashFlowService(**kwargs)

def create_data_imputation_service(**kwargs):
    """Create Data Imputation Service instance."""
    if not SERVICES_LOADED:
        raise ImportError("Services not properly loaded")
    return DataImputationService(**kwargs)

def create_financial_processing_service(**kwargs):
    """Create Financial Processing Service instance."""
    if not SERVICES_LOADED:
        raise ImportError("Services not properly loaded")
    return FinancialProcessingService(**kwargs)

# Error codes for services
SERVICE_ERROR_CODES = {
    'INVALID_FILE_FORMAT': 1001,
    'DATA_VALIDATION_FAILED': 1002,
    'INSUFFICIENT_DATA': 1003,
    'MODEL_PREDICTION_FAILED': 1004,
    'SERVICE_UNAVAILABLE': 1005,
    'PROCESSING_TIMEOUT': 1006,
    'MEMORY_LIMIT_EXCEEDED': 1007,
    'INVALID_CONFIGURATION': 1008
}

class ServiceException(Exception):
    """Base exception for all service-related errors."""
    
    def __init__(self, message: str, error_code: int = None, details: dict = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}

# Export all public classes and functions
__all__ = [
    # Service classes
    'AIAnalysisService',
    'AnalysisRequest', 
    'AnalysisResponse',
    'BalanceSheetService',
    'BalanceSheetAnalysis',
    'CashFlowService',
    'CashFlowAnalysis',
    'DataImputationService',
    'ImputationResult',
    'FinancialProcessingService',
    'ProcessingResult',
    
    # Factory functions
    'create_ai_service',
    'create_balance_sheet_service',
    'create_cash_flow_service',
    'create_data_imputation_service',
    'create_financial_processing_service',
    
    # Utility functions
    'get_service_status',
    'get_service_config',
    'validate_file_format',
    'get_supported_formats',
    
    # Constants
    'SERVICE_CONFIG',
    'SUPPORTED_FORMATS',
    'STATEMENT_TYPES',
    'ANALYSIS_TYPES',
    'SERVICE_ERROR_CODES',
    
    # Exception
    'ServiceException'
]