from flask import Blueprint
# Import all route blueprints
from .ai_routes import ai_bp
from .analytics_routes import analytics_bp
from .api_routes import api_bp
from .balance_sheet_routes import balance_sheet_bp
from .cash_flow_routes import cash_flow_bp
from .main_routes import main_bp
from .upload_routes import upload_bp
__version__ = "1.0.0"
__author__ = "Prateek Dahiya"

# Export all blueprints
__all__ = [
    'ai_bp',
    'analytics_bp', 
    'api_bp',
    'balance_sheet_bp',
    'cash_flow_bp',
    'main_bp',
    'upload_bp',
    'register_blueprints'
]

def register_blueprints(app):
    """
    Register all blueprints with the Flask application
    
    Args:
        app: Flask application instance
    """
    
    # Main routes (dashboard, home page)
    app.register_blueprint(main_bp, url_prefix='/')
    
    # API utilities and health checks
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Balance sheet operations
    app.register_blueprint(balance_sheet_bp, url_prefix='/api/balance-sheet')
    
    # Cash flow operations  
    app.register_blueprint(cash_flow_bp, url_prefix='/api/cash-flow')
    
    # File upload operations
    app.register_blueprint(upload_bp, url_prefix='/api/upload')
    
    # Analytics and reporting
    app.register_blueprint(analytics_bp, url_prefix='/api/analytics')
    
    # AI-powered features
    app.register_blueprint(ai_bp, url_prefix='/api/ai')
    
    print("✅ All blueprints registered successfully")

def get_route_info():
    """
    Get information about all available routes
    
    Returns:
        dict: Route information
    """
    return {
        'version': __version__,
        'author': __author__,
        'routes': {
            'main': {
                'prefix': '/',
                'description': 'Main dashboard and web pages',
                'endpoints': [
                    'GET /',
                    'GET /dashboard',
                    'GET /companies',
                    'GET /about'
                ]
            },
            'api': {
                'prefix': '/api',
                'description': 'General API utilities',
                'endpoints': [
                    'GET /api/health',
                    'GET /api/status',
                    'GET /api/info'
                ]
            },
            'balance_sheet': {
                'prefix': '/api/balance-sheet',
                'description': 'Balance sheet processing and management',
                'endpoints': [
                    'POST /api/balance-sheet/generate',
                    'POST /api/balance-sheet/process',
                    'GET /api/balance-sheet/get/<company_id>/<year>',
                    'GET /api/balance-sheet/list/<company_id>',
                    'POST /api/balance-sheet/validate',
                    'GET /api/balance-sheet/export/<company_id>/<year>',
                    'GET /api/balance-sheet/benchmark/<company_id>/<year>'
                ]
            },
            'cash_flow': {
                'prefix': '/api/cash-flow',
                'description': 'Cash flow statement processing and analysis',
                'endpoints': [
                    'POST /api/cash-flow/generate',
                    'POST /api/cash-flow/process',
                    'GET /api/cash-flow/get/<company_id>/<year>',
                    'GET /api/cash-flow/list/<company_id>',
                    'POST /api/cash-flow/validate',
                    'POST /api/cash-flow/bulk-generate',
                    'POST /api/cash-flow/predict',
                    'GET /api/cash-flow/trends/<company_id>',
                    'GET /api/cash-flow/benchmark/<company_id>/<year>',
                    'GET /api/cash-flow/health-score/<company_id>/<year>',
                    'POST /api/cash-flow/compare',
                    'GET /api/cash-flow/export/<company_id>/<year>',
                    'GET /api/cash-flow/search',
                    'GET /api/cash-flow/analytics/summary'
                ]
            },
            'upload': {
                'prefix': '/api/upload',
                'description': 'File upload and processing',
                'endpoints': [
                    'POST /api/upload/balance-sheet',
                    'POST /api/upload/cash-flow',
                    'POST /api/upload/combined',
                    'POST /api/upload/validate-file',
                    'POST /api/upload/preview',
                    'POST /api/upload/batch',
                    'GET /api/upload/supported-formats'
                ]
            },
            'analytics': {
                'prefix': '/api/analytics',
                'description': 'Analytics and reporting',
                'endpoints': [
                    'GET /api/analytics/dashboard',
                    'GET /api/analytics/companies',
                    'GET /api/analytics/industry/<industry>',
                    'GET /api/analytics/trends',
                    'POST /api/analytics/custom-report'
                ]
            },
            'ai': {
                'prefix': '/api/ai',
                'description': 'AI-powered analysis and predictions',
                'endpoints': [
                    'POST /api/ai/predict-risk',
                    'POST /api/ai/financial-health',
                    'POST /api/ai/recommendations',
                    'POST /api/ai/forecast'
                ]
            }
        }
    }

# Route validation and error handling utilities
def validate_company_id(company_id):
    """
    Validate company ID format
    
    Args:
        company_id (str): Company identifier
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not company_id or not isinstance(company_id, str):
        return False
    
    # Basic validation - alphanumeric and underscores, 3-20 characters
    import re
    pattern = r'^[A-Za-z0-9_]{3,20}$'
    return bool(re.match(pattern, company_id))

def validate_year(year):
    """
    Validate year parameter
    
    Args:
        year: Year value
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        year_int = int(year)
        # Valid range: 2000 to current year + 10
        current_year = datetime.now().year
        return 2000 <= year_int <= current_year + 10
    except (ValueError, TypeError):
        return False

def standardize_response(success=True, data=None, message=None, error=None, status_code=200):
    """
    Standardize API response format
    
    Args:
        success (bool): Whether the operation was successful
        data: Response data
        message (str): Success message
        error (str): Error message
        status_code (int): HTTP status code
        
    Returns:
        tuple: (response_dict, status_code)
    """
    from datetime import datetime
    
    response = {
        'success': success,
        'timestamp': datetime.now().isoformat()
    }
    
    if success:
        if message:
            response['message'] = message
        if data is not None:
            response['data'] = data
    else:
        if error:
            response['error'] = error
    
    return response, status_code

# Common error responses
def error_response(message, status_code=400):
    """Generate standardized error response"""
    return standardize_response(
        success=False, 
        error=message, 
        status_code=status_code
    )

def success_response(data=None, message=None, status_code=200):
    """Generate standardized success response"""
    return standardize_response(
        success=True,
        data=data,
        message=message,
        status_code=status_code
    )

# Database configuration helper
def get_db_config_from_app(app):
    """
    Extract database configuration from Flask app config
    
    Args:
        app: Flask application instance
        
    Returns:
        dict: Database configuration
    """
    return {
        'host': app.config.get('DB_HOST', 'localhost'),
        'database': app.config.get('DB_NAME', 'financial_db'),
        'user': app.config.get('DB_USER', 'postgres'),
        'password': app.config.get('DB_PASSWORD', 'password'),
        'port': app.config.get('DB_PORT', 5432)
    }

# Logging configuration
def setup_route_logging(app):
    """
    Setup logging for all routes
    
    Args:
        app: Flask application instance
    """
    import logging
    from flask import request, g
    import time
    
    # Create logger
    logger = logging.getLogger('routes')
    logger.setLevel(logging.INFO)
    
    # Create handler if not exists
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    @app.before_request
    def before_request():
        g.start_time = time.time()
    
    @app.after_request
    def after_request(response):
        if hasattr(g, 'start_time'):
            duration = time.time() - g.start_time
            logger.info(f"{request.method} {request.path} - {response.status_code} - {duration:.3f}s")
        return response

# Route documentation generator
def generate_route_docs():
    """
    Generate documentation for all routes
    
    Returns:
        dict: Route documentation
    """
    route_info = get_route_info()
    
    docs = {
        'title': 'Financial Risk Assessment Platform API',
        'version': route_info['version'],
        'author': route_info['author'],
        'base_url': '/api',
        'authentication': 'None required for demo',
        'content_type': 'application/json',
        'routes': route_info['routes']
    }
    
    return docs

# Import datetime for validation functions
from datetime import datetime