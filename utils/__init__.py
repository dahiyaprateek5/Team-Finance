import sys
import logging
from datetime import datetime
from pathlib import Path

# Configure logging for the entire package
def setup_logging():
    """Setup logging configuration for the utils package"""
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_filename = log_dir / f"financial_utils_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create logger for this package
    logger = logging.getLogger('financial_utils')
    logger.info("Financial Utils Package Initialized")
    
    return logger

# Initialize logging
logger = setup_logging()

# Package version
__version__ = "1.0.0"
__author__ = "Prateek Dahiya"
__email__ = "prateek.dahiya@student.liverpool.ac.uk"
__description__ = "Financial Risk Assessment Platform - Utilities Package"

# Import all modules and make them available at package level
try:
    from .balance_sheet_utils import (
        BalanceSheetProcessor,
        BalanceSheetAnalyzer,
        BalanceSheetValidator,
        calculate_balance_sheet_ratios,
        validate_balance_sheet_equation,
        generate_balance_sheet_insights
    )
    logger.info("✅ Balance Sheet Utils imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Balance Sheet Utils import failed: {e}")

try:
    from .cash_flow_utils import (
        CashFlowProcessor,
        CashFlowAnalyzer,
        CashFlowGenerator,
        calculate_cash_flow_ratios,
        generate_cash_flow_from_balance_sheet,
        validate_cash_flow_data
    )
    logger.info("✅ Cash Flow Utils imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Cash Flow Utils import failed: {e}")

try:
    from .data_processing import (
        DocumentProcessor,
        DataExtractor,
        FileManager,
        process_uploaded_files,
        extract_financial_data,
        clean_and_validate_data
    )
    logger.info("✅ Data Processing imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Data Processing import failed: {e}")

try:
    from .financial_calculations import (
        FinancialCalculator,
        RatioCalculator,
        RiskAnalyzer,
        calculate_all_ratios,
        assess_financial_health,
        calculate_liquidation_risk
    )
    logger.info("✅ Financial Calculations imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Financial Calculations import failed: {e}")

try:
    from .helpers import (
        format_currency,
        get_industry_benchmarks,
        log_processing_step,
        process_uploaded_document,
        generate_processing_report,
        detect_financial_anomalies
    )
    logger.info("✅ Helpers imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Helpers import failed: {e}")

try:
    from .ml_predictor import (
        LiquidationPredictor,
        CashFlowGenerator,
        FinancialRatioCalculator,
        generate_financial_insights,
        validate_cash_flow_data
    )
    logger.info("✅ ML Predictor imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ ML Predictor import failed: {e}")

# Global configuration dictionary
CONFIG = {
    'package_name': 'financial_utils',
    'version': __version__,
    'max_file_size': 50 * 1024 * 1024,  # 50MB
    'allowed_extensions': {'.pdf', '.xlsx', '.xls', '.csv', '.docx'},
    'temp_folder': 'temp',
    'upload_folder': 'uploads',
    'logs_folder': 'logs',
    'models_folder': 'models',
    'reports_folder': 'reports',
    'default_industry': 'Technology',
    'validation_tolerance': 0.05,  # 5% tolerance for validation
    'min_accuracy_threshold': 70.0,  # Minimum acceptable accuracy percentage
    'supported_currencies': ['USD', 'GBP', 'EUR'],
    'date_formats': ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y'],
    'decimal_places': 2,
    'enable_debug': True,
    'enable_caching': True
}

# Database table schemas for reference
BALANCE_SHEET_SCHEMA = {
    'table_name': 'balance_sheet_1',
    'columns': [
        'id', 'company_id', 'year', 'generated_at', 'current_assets',
        'cash_and_equivalents', 'accounts_receivable', 'inventory',
        'prepaid_expenses', 'other_current_assets', 'non_current_assets',
        'property_plant_equipment', 'accumulated_depreciation', 'net_ppe',
        'intangible_assets', 'goodwill', 'investments', 'other_non_current_assets',
        'total_assets', 'current_liabilities', 'accounts_payable', 'short_term_debt',
        'accrued_liabilities', 'deferred_revenue', 'other_current_liabilities',
        'non_current_liabilities', 'long_term_debt', 'deferred_tax_liabilities',
        'pension_obligations', 'other_non_current_liabilities', 'total_liabilities',
        'share_capital', 'retained_earnings', 'additional_paid_in_capital',
        'treasury_stock', 'accumulated_other_comprehensive_income', 'total_equity',
        'balance_check', 'accuracy_percentage', 'data_source', 'validation_errors'
    ]
}

CASH_FLOW_SCHEMA = {
    'table_name': 'cash_flow_statement',
    'columns': [
        'id', 'company_id', 'year', 'generated_at', 'company_name', 'industry',
        'net_income', 'depreciation_and_amortization', 'stock_based_compensation',
        'changes_in_working_capital', 'accounts_receivable', 'inventory',
        'accounts_payable', 'net_cash_from_operating_activities', 'capital_expenditures',
        'acquisitions', 'net_cash_from_investing_activities', 'dividends_paid',
        'share_repurchases', 'net_cash_from_financing_activities', 'free_cash_flow',
        'ocf_to_net_income_ratio', 'liquidation_label', 'debt_to_equity_ratio',
        'interest_coverage_ratio'
    ]
}

# Supported industries with their characteristics
INDUSTRY_CONFIG = {
    'Technology': {
        'typical_ratios': {
            'current_ratio': 2.5,
            'debt_to_equity': 0.3,
            'cash_ratio': 0.15
        },
        'risk_factors': ['market_volatility', 'rapid_obsolescence', 'competition'],
        'key_metrics': ['cash_burn_rate', 'revenue_growth', 'r_and_d_spending']
    },
    'Manufacturing': {
        'typical_ratios': {
            'current_ratio': 1.8,
            'debt_to_equity': 0.7,
            'inventory_turnover': 6.0
        },
        'risk_factors': ['supply_chain', 'commodity_prices', 'regulatory_changes'],
        'key_metrics': ['capacity_utilization', 'inventory_levels', 'capex_ratio']
    },
    'Healthcare': {
        'typical_ratios': {
            'current_ratio': 2.2,
            'debt_to_equity': 0.4,
            'receivables_turnover': 8.0
        },
        'risk_factors': ['regulatory_approval', 'reimbursement_changes', 'litigation'],
        'key_metrics': ['patient_volume', 'reimbursement_rates', 'clinical_outcomes']
    },
    'Retail': {
        'typical_ratios': {
            'current_ratio': 1.5,
            'debt_to_equity': 0.8,
            'inventory_turnover': 12.0
        },
        'risk_factors': ['consumer_spending', 'seasonality', 'e_commerce_disruption'],
        'key_metrics': ['same_store_sales', 'inventory_levels', 'customer_acquisition']
    }
}

# API response templates
API_RESPONSE_TEMPLATES = {
    'success': {
        'status': 'success',
        'message': 'Operation completed successfully',
        'data': None,
        'timestamp': None
    },
    'error': {
        'status': 'error',
        'message': 'An error occurred',
        'error_code': None,
        'details': None,
        'timestamp': None
    },
    'validation_error': {
        'status': 'validation_error',
        'message': 'Data validation failed',
        'errors': [],
        'timestamp': None
    }
}

def get_package_info():
    """Get package information"""
    return {
        'name': CONFIG['package_name'],
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'modules': [
            'balance_sheet_utils',
            'cash_flow_utils', 
            'data_processing',
            'financial_calculations',
            'helpers',
            'ml_predictor'
        ],
        'config': CONFIG
    }

def create_response(status='success', message='', data=None, **kwargs):
    """Create standardized API response"""
    template = API_RESPONSE_TEMPLATES.get(status, API_RESPONSE_TEMPLATES['success'])
    response = template.copy()
    
    response['message'] = message or template['message']
    response['data'] = data
    response['timestamp'] = datetime.now().isoformat()
    
    # Add any additional fields
    for key, value in kwargs.items():
        response[key] = value
    
    return response

def setup_directories():
    """Create necessary directories for the application"""
    directories = [
        CONFIG['temp_folder'],
        CONFIG['upload_folder'],
        CONFIG['logs_folder'],
        CONFIG['models_folder'],
        CONFIG['reports_folder']
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.debug(f"Directory ensured: {directory}")

def cleanup_temp_files(max_age_hours=24):
    """Clean up temporary files older than specified hours"""
    try:
        temp_dir = Path(CONFIG['temp_folder'])
        if temp_dir.exists():
            current_time = datetime.now()
            
            for file_path in temp_dir.iterdir():
                if file_path.is_file():
                    file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_age.total_seconds() > max_age_hours * 3600:
                        file_path.unlink()
                        logger.debug(f"Cleaned up temp file: {file_path}")
                        
    except Exception as e:
        logger.error(f"Error cleaning temp files: {e}")

def validate_environment():
    """Validate that all required dependencies are available"""
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'openpyxl', 
        'PyPDF2', 'pdfplumber', 'werkzeug'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        return False, missing_packages
    
    logger.info("✅ All required packages are available")
    return True, []

# Initialize the package
def initialize_package():
    """Initialize the financial utils package"""
    logger.info(f"Initializing {CONFIG['package_name']} v{__version__}")
    
    # Setup directories
    setup_directories()
    
    # Validate environment
    env_valid, missing_packages = validate_environment()
    
    if not env_valid:
        logger.warning(f"Some packages are missing: {missing_packages}")
        logger.warning("Install them using: pip install " + " ".join(missing_packages))
    
    # Clean up old temp files
    cleanup_temp_files()
    
    logger.info("🚀 Financial Utils Package initialization complete")
    
    return {
        'status': 'initialized',
        'version': __version__,
        'config': CONFIG,
        'environment_valid': env_valid,
        'missing_packages': missing_packages
    }

# Auto-initialize when package is imported
_initialization_result = initialize_package()

# Export commonly used functions and classes at package level
__all__ = [
    # Core classes
    'BalanceSheetProcessor',
    'CashFlowProcessor', 
    'DocumentProcessor',
    'FinancialCalculator',
    'LiquidationPredictor',
    
    # Utility functions
    'format_currency',
    'get_industry_benchmarks',
    'process_uploaded_document',
    'calculate_all_ratios',
    'generate_financial_insights',
    
    # Configuration and schemas
    'CONFIG',
    'BALANCE_SHEET_SCHEMA',
    'CASH_FLOW_SCHEMA',
    'INDUSTRY_CONFIG',
    
    # Package utilities
    'get_package_info',
    'create_response',
    'setup_directories',
    'cleanup_temp_files',
    'validate_environment',
    'initialize_package'
]

# Print initialization status
if CONFIG.get('enable_debug', False):
    print(f"📊 {CONFIG['package_name']} v{__version__} - Ready for Financial Analysis!")
    if not _initialization_result['environment_valid']:
        print(f"⚠️  Warning: Missing packages - {_initialization_result['missing_packages']}")