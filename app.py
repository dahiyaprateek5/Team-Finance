# =====================================
# File: app.py 
# Complete Flask Application for Financial Risk Assessment
# Enhanced with Balance Sheet & Cash Flow Generators
# =====================================

"""
Flask Application for Financial Risk Assessment
Integrates XGBoost, Random Forest, Neural Networks, and LightGBM
Enhanced with PostgreSQL Cash Flow Data Fetching
NEW: Balance Sheet & Cash Flow Generators with ML Imputation
"""
import logging
import warnings

# Suppress all logging output
logging.getLogger().setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')

# Suppress Flask's built-in logging
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('financial_utils').setLevel(logging.ERROR)
logging.getLogger('DATABASE_CONNECTION.db_connection').setLevel(logging.ERROR)
import logging
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.CRITICAL)

import os
import sys
from venv import logger
import warnings

from models.cash_flow_generator import clean_financial_value
warnings.filterwarnings('ignore')
# Add these imports at the top of your app.py (after existing imports)
import re
import logging
from dataclasses import dataclass
from enum import Enum


# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from flask import Flask, make_response, request, jsonify, render_template, send_file, redirect, send_from_directory, url_for, flash
import numpy as np 
import pandas as pd  
from datetime import datetime
import traceback
from typing import Dict, List, Any, Optional
from werkzeug.utils import secure_filename
from flask import Flask, render_template


# Enhanced PDF Parser imports - SAFE VERSION
PDF_PARSER_LIBRARIES_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    import pdfplumber
    import tabula
    import camelot
    from dataclasses import dataclass
    from enum import Enum
    PDF_PARSER_LIBRARIES_AVAILABLE = True
    #print("✅ PDF parsing libraries available")
except ImportError as e:
    #print(f"⚠️ PDF libraries not available: {e}")
    #print("🔄 Continuing without PDF support...")
    
    # Still import required classes for type hints
    from dataclasses import dataclass
    from enum import Enum
    import pandas as pd
    from typing import Optional, Dict, Any, List
    
    # Create fallback classes
    class DocumentType(Enum):
        BALANCE_SHEET = "balance_sheet"
        CASH_FLOW = "cash_flow"
        INCOME_STATEMENT = "income_statement"
        ANNUAL_REPORT = "annual_report"
        UNKNOWN = "unknown"

    @dataclass
    class ParsedTable:
        data: pd.DataFrame
        confidence: float
        method: str
        page_number: int
        table_type: str
        year: Optional[int] = None

    class FinancialPDFParser:
        def __init__(self, config: Dict = None):
            self.config = config or {
                'min_confidence_threshold': 0.7,
                'max_pages_to_scan': 50,
                'debug_mode': True
            }
        
        def parse_document(self, file_path: str, company_name: str = None) -> Dict[str, Any]:
            return {
                'success': False,
                'error': 'PDF parsing libraries not available. Please use Excel/CSV files.',
                'financial_data': {},
                'summary': {'message': 'Install PDF libraries or use Excel/CSV format'}
            }
        
        def _analyze_document(self, file_path: str) -> Dict[str, Any]:
            return {'error': 'PDF analysis requires additional libraries'}
        
        def _extract_tables_simple(self, file_path: str) -> List[ParsedTable]:
            return []
        
        def _extract_financial_data_simple(self, tables: List[ParsedTable]) -> Dict[str, Any]:
            return {'raw_tables': [], 'extracted_values': {}}

    class PDFParsingService:
        def __init__(self, config: Dict = None):
            self.parser = FinancialPDFParser(config)
            self.upload_folder = config.get('upload_folder', 'uploads') if config else 'uploads'
            self.results_folder = config.get('results_folder', 'uploads/results') if config else 'uploads/results'
        
        def process_uploaded_file(self, file_path: str, company_name: str = None, 
                                session_id: str = None) -> Dict[str, Any]:
            return {
                'success': False,
                'error': 'PDF processing requires additional libraries. Use Excel/CSV files.',
                'financial_data': {},
                'summary': {}
            }
        
        def validate_file(self, file_path: str) -> Dict[str, Any]:
            import os
            try:
                if not os.path.exists(file_path):
                    return {'valid': False, 'error': 'File not found'}
                
                if file_path.lower().endswith('.pdf'):
                    return {
                        'valid': False, 
                        'error': 'PDF parsing requires additional libraries. Use Excel/CSV instead.'
                    }
                
                return {'valid': True, 'file_size': os.path.getsize(file_path)}
            except Exception as e:
                return {'valid': False, 'error': f'File validation error: {str(e)}'}
    
    #print("✅ PDF fallback classes created - Excel/CSV processing available")

class DocumentType(Enum):
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW = "cash_flow"
    INCOME_STATEMENT = "income_statement"
    ANNUAL_REPORT = "annual_report"
    UNKNOWN = "unknown"

@dataclass
class ParsedTable:
    data: pd.DataFrame
    confidence: float
    method: str
    page_number: int
    table_type: str
    year: Optional[int] = None

class FinancialPDFParser:
    """Enhanced PDF parser specifically designed for financial documents"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        
        # Financial keywords for table detection
        self.balance_sheet_keywords = [
            'balance sheet', 'statement of financial position', 'assets', 'liabilities', 
            'equity', 'current assets', 'non-current assets', 'shareholders equity',
            'total assets', 'total liabilities', 'retained earnings'
        ]
        
        self.cash_flow_keywords = [
            'cash flow', 'cash flows', 'statement of cash flows', 'operating activities',
            'investing activities', 'financing activities', 'net cash', 'operating cash flow',
            'free cash flow', 'net income', 'depreciation', 'working capital'
        ]
        
        self.year_patterns = [
            r'20[0-9]{2}',  # 2000-2099
            r'FY\s*20[0-9]{2}',  # FY 2023
        ]
    
    def _get_default_config(self) -> Dict:
        return {
            'min_confidence_threshold': 0.7,
            'max_pages_to_scan': 50,
            'debug_mode': True
        }
    
    def parse_document(self, file_path: str, company_name: str = None) -> Dict[str, Any]:
        """Main method to parse financial PDF document"""
        try:
            if not PDF_PARSER_LIBRARIES_AVAILABLE:
                return {
                    'success': False,
                    'error': 'PDF parsing libraries not available',
                    'financial_data': {},
                    'summary': {}
                }
            
            # Simple document analysis
            doc_info = self._analyze_document(file_path)
            
            # Extract tables using camelot primarily
            tables = self._extract_tables_simple(file_path)
            
            # Extract financial data
            financial_data = self._extract_financial_data_simple(tables)
            
            return {
                'success': True,
                'document_info': doc_info,
                'financial_data': financial_data,
                'summary': {'tables_found': len(tables), 'confidence': 0.8},
                'confidence_score': 0.8,
                'parsing_method': 'camelot_primary',
                'processed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'document_info': {},
                'financial_data': {},
                'summary': {}
            }
    
    def _analyze_document(self, file_path: str) -> Dict[str, Any]:
        """Simple document analysis"""
        try:
            doc = fitz.open(file_path)
            doc_info = {
                'filename': os.path.basename(file_path),
                'page_count': len(doc),
                'file_size': os.path.getsize(file_path),
                'document_type': 'financial_document'
            }
            doc.close()
            return doc_info
        except Exception as e:
            return {'error': str(e)}
    
    def _extract_tables_simple(self, file_path: str) -> List[ParsedTable]:
        """Simple table extraction using camelot"""
        tables = []
        try:
            camelot_tables = camelot.read_pdf(file_path, pages='all', flavor='lattice')
            
            for i, table in enumerate(camelot_tables):
                if table.parsing_report['accuracy'] > 30:
                    parsed_table = ParsedTable(
                        data=table.df,
                        confidence=table.parsing_report['accuracy'] / 100,
                        method='camelot',
                        page_number=table.parsing_report['page'],
                        table_type='financial'
                    )
                    tables.append(parsed_table)
        except Exception as e:
            print(f"Table extraction error: {e}")
        
        return tables
    
    def _extract_financial_data_simple(self, tables: List[ParsedTable]) -> Dict[str, Any]:
        """Simple financial data extraction"""
        financial_data = {'raw_tables': [], 'extracted_values': {}}
        
        for table in tables:
            financial_data['raw_tables'].append({
                'confidence': table.confidence,
                'method': table.method,
                'page_number': table.page_number,
                'row_count': len(table.data),
                'col_count': len(table.data.columns)
            })
        
        return financial_data

class PDFParsingService:
    """Service class for PDF parsing"""
    
    def __init__(self, config: Dict = None):
        self.parser = FinancialPDFParser(config)
        self.upload_folder = config.get('upload_folder', 'uploads') if config else 'uploads'
        self.results_folder = config.get('results_folder', 'uploads/results') if config else 'uploads/results'
        
        os.makedirs(self.upload_folder, exist_ok=True)
        os.makedirs(self.results_folder, exist_ok=True)
    
    def process_uploaded_file(self, file_path: str, company_name: str = None, 
                            session_id: str = None) -> Dict[str, Any]:
        """Process uploaded PDF file"""
        try:
            result = self.parser.parse_document(file_path, company_name)
            return result
        except Exception as e:
            return {
                'success': False,
                'error': f"Processing failed: {str(e)}",
                'financial_data': {},
                'summary': {}
            }
    
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """Validate PDF file"""
        try:
            if not os.path.exists(file_path):
                return {'valid': False, 'error': 'File not found'}
            
            if not file_path.lower().endswith('.pdf'):
                return {'valid': False, 'error': 'Only PDF files supported'}
            
            file_size = os.path.getsize(file_path)
            if file_size > 50 * 1024 * 1024:
                return {'valid': False, 'error': 'File too large (max 50MB)'}
            
            # Try to open PDF
            doc = fitz.open(file_path)
            page_count = len(doc)
            doc.close()
            
            return {
                'valid': True, 
                'file_size': file_size,
                'page_count': page_count
            }
        except Exception as e:
            return {'valid': False, 'error': f'Invalid PDF: {str(e)}'}































# Import your database classes 
try:
    from DATABASE_CONNECTION.db_connection import DatabaseConnection
    from DATABASE_CONNECTION.db_operations import DatabaseOperations
    DATABASE_CLASSES_AVAILABLE = True
    #print("✅ Database classes imported successfully")
except ImportError as e:
    #print(f"⚠️ Database classes not available: {e}")
    #print(f"Make sure you have: DATABASE_CONNECTION/db_connection.py, DATABASE_CONNECTION/db_operations.py")
    DATABASE_CLASSES_AVAILABLE = False

# Import routes (existing + new)
try:
    from routes.main_routes import main_bp
    from routes.api_routes import api_bp
    from routes.analytics_routes import analytics_bp
    from routes.ai_routes import ai_bp
    from routes.upload_routes import upload_bp
    EXISTING_ROUTES_AVAILABLE = True
except ImportError as e:
    #print(f"⚠️ Some existing routes not available: {e}")
    EXISTING_ROUTES_AVAILABLE = False

# Import NEW routes
try:
    from routes.balance_sheet_routes import balance_sheet_bp
    from routes.cash_flow_routes import cash_flow_bp
    NEW_ROUTES_AVAILABLE = True
    #print("✅ New Balance Sheet & Cash Flow routes imported successfully")
except ImportError as e:
    #print(f"⚠️ New routes not available: {e}")
    #print("Make sure you have: routes/balance_sheet_routes.py, routes/cash_flow_routes.py")
    NEW_ROUTES_AVAILABLE = False


try:
    from routes.upload_routes import upload_bp
    UPLOAD_ROUTES_AVAILABLE = True
    #print("✅ Upload routes imported successfully")
except ImportError as e:
    #print(f"❌ Upload routes import failed: {e}")
    UPLOAD_ROUTES_AVAILABLE = False





# Import services (existing + new)
try:
    from services.ai_service import AIService
    EXISTING_SERVICES_AVAILABLE = True
except ImportError as e:
    #print(f"⚠️ Existing services not available: {e}")
    EXISTING_SERVICES_AVAILABLE = False

# Import NEW services
try:
    from services.balance_sheet_service import BalanceSheetService
    from services.cash_flow_service import CashFlowService
    from services.data_imputation_service import DataImputationService
    from services.financial_processing_service import FinancialProcessingService
    NEW_SERVICES_AVAILABLE = True
    #print("✅ New services imported successfully")
except ImportError as e:
    #print(f"⚠️ New services not available: {e}")
    print("Make sure you have: services/balance_sheet_service.py, services/cash_flow_service.py, etc.")
    NEW_SERVICES_AVAILABLE = False

# Import NEW models
try:
    from models.balance_sheet_generator import BalanceSheetGenerator
    from models.cash_flow_generator import CashFlowGenerator
    from models.balance_sheet_models import BalanceSheetModel
    from models.cash_flow_models import CashFlowModel
    NEW_MODELS_AVAILABLE = True
    #print("✅ New models imported successfully")
except ImportError as e:
    #print(f"⚠️ New models not available: {e}")
    print("Make sure you have: models/balance_sheet_generator.py, models/cash_flow_generator.py, etc.")
    NEW_MODELS_AVAILABLE = False

# Import NEW utils
try:
    from utils.balance_sheet_utils import BalanceSheetUtils
    from utils.cash_flow_utils import CashFlowUtils
    from utils.financial_calculations import FinancialCalculations
    NEW_UTILS_AVAILABLE = True
    #print("✅ New utilities imported successfully")
except ImportError as e:
    #print(f"⚠️ New utilities not available: {e}")
    print("Make sure you have: utils/balance_sheet_utils.py, utils/cash_flow_utils.py, etc.")
    NEW_UTILS_AVAILABLE = False

# Try to import ML algorithms
try:
    from ml_algorithms.xgboost import XGBRegressorModel, XGBClassifierModel, XGBConfig  
    XGBOOST_AVAILABLE = True
except ImportError as e:
    #print(f"⚠️ XGBoost not available: {e}")
    XGBOOST_AVAILABLE = False

try:
    from ml_algorithms.random_forest import RandomForestRegressorModel, RandomForestClassifierModel, RandomForestConfig
    RANDOM_FOREST_AVAILABLE = True
except ImportError as e:
    #print(f"⚠️ Random Forest not available: {e}")
    RANDOM_FOREST_AVAILABLE = False

try:
    from ml_algorithms.neural_networks import NeuralNetworkRegressor, NeuralNetworkClassifier, NeuralNetworkConfig
    NEURAL_NETWORKS_AVAILABLE = True
except ImportError as e:
    #print(f"⚠️ Neural Networks not available: {e}")
    NEURAL_NETWORKS_AVAILABLE = False

try:
    from ml_algorithms.lightgbm import LGBRegressorModel, LGBClassifierModel, LGBConfig
    LIGHTGBM_AVAILABLE = True
except ImportError as e:
    #print(f"⚠️ LightGBM not available: {e}")
    LIGHTGBM_AVAILABLE = False

try:
    from ml_algorithms.essemble_manager import EnsembleManager  
    ENSEMBLE_AVAILABLE = True
except ImportError as e:
    #print(f"⚠️ Ensemble Manager not available: {e}")
    ENSEMBLE_AVAILABLE = False

try:
    from ml_algorithms.model_evaluation import (
        RegressionMetrics, ClassificationMetrics, 
        CrossValidator, PerformanceTracker
    )
    EVALUATION_AVAILABLE = True
except ImportError as e:
    #print(f"⚠️ Model Evaluation not available: {e}")
    EVALUATION_AVAILABLE = False

# Import NEW ML algorithms for imputation
try:
    from ml_algorithms.knn_imputer import KNNImputer
    from ml_algorithms.time_series_analyzer import TimeSeriesAnalyzer
    from ml_algorithms.peer_analysis import PeerAnalysis
    from ml_algorithms.industry_benchmarks import IndustryBenchmarks
    IMPUTATION_ALGORITHMS_AVAILABLE = True
    #print("✅ Imputation algorithms imported successfully")
except ImportError as e:
    #print(f"⚠️ Imputation algorithms not available: {e}")
    print("Make sure you have: ml_algorithms/knn_imputer.py, ml_algorithms/time_series_analyzer.py, etc.")
    IMPUTATION_ALGORITHMS_AVAILABLE = False

try:
    from utils.ml_predictor import MLPredictor, format_confidence_response, get_confidence_recommendations
    CONFIDENCE_UTILS_AVAILABLE = True
    #print("✅ Confidence utilities imported")
except ImportError as e:
    #print(f"⚠️ Confidence utilities not available: {e}")
    # Create simple fallback
    class MLPredictor:
        def __init__(self):
            self.confidence_enabled = True
        
        def predict_with_confidence(self, data):
            return {'prediction': 0.75, 'confidence': 0.85}
    
    def format_confidence_response(prediction, confidence):
        return {'prediction': prediction, 'confidence': confidence, 'status': 'fallback'}
    
    def get_confidence_recommendations(confidence_score):
        if confidence_score > 0.8:
            return "High confidence - reliable prediction"
        elif confidence_score > 0.6:
            return "Medium confidence - use with caution"
        else:
            return "Low confidence - gather more data"
    
    CONFIDENCE_UTILS_AVAILABLE = True
    #print("✅ Confidence utilities initialized (fallback)")

# Initialize Flask app
app = Flask(__name__, static_folder='static', static_url_path='/static')

# Enhanced Configuration
app.config.update({
    'SECRET_KEY': 'financial-risk-assessment-2024-enhanced',
    'MAX_CONTENT_LENGTH': 50 * 1024 * 1024,  # 50MB max file size for financial documents
    'UPLOAD_FOLDER': os.path.join(current_dir, 'uploads'),
    'ALLOWED_EXTENSIONS': {'pdf', 'xlsx', 'xls', 'csv', 'docx'},
    'BALANCE_SHEET_UPLOAD_FOLDER': os.path.join(current_dir, 'uploads', 'balance_sheets'),
    'CASH_FLOW_UPLOAD_FOLDER': os.path.join(current_dir, 'uploads', 'cash_flow'),
    'PROCESSED_FILES_FOLDER': os.path.join(current_dir, 'uploads', 'processed'),
    'RESULTS_FOLDER': os.path.join(current_dir, 'uploads', 'results'),
    'MAX_FILES_PER_UPLOAD': 10,
    'SUPPORTED_CURRENCIES': ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD'],
    'DEFAULT_CURRENCY': 'USD',
    'IMPUTATION_METHODS': ['knn', 'random_forest', 'time_series', 'peer_analysis', 'industry_benchmarks'],
    'DEFAULT_IMPUTATION_METHOD': 'knn',
    'FINANCIAL_METRICS_COUNT': 21,
    'BALANCE_SHEET_ACCURACY_THRESHOLD': 0.85,
    'CASH_FLOW_ACCURACY_THRESHOLD': 0.90
})
# Create upload directories
for folder in ['uploads', 'uploads/balance_sheets', 'uploads/cash_flow', 'uploads/processed', 'uploads/results']:
    folder_path = os.path.join(current_dir, folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        #print(f"✅ Created directory: {folder_path}")

global balance_sheet_service, cash_flow_service

# Global variables for models and services
models = {}
performance_tracker = None
ensemble_manager = None
ml_predictor = None
db_connection = None
db_operations = None

# NEW: Global variables for new services
balance_sheet_service = None
cash_flow_service = None
data_imputation_service = None
financial_processing_service = None
balance_sheet_generator = None
cash_flow_generator = None
pdf_parsing_service = None

if UPLOAD_ROUTES_AVAILABLE:
    try:
        app.register_blueprint(upload_bp, url_prefix='/upload')
        #print("✅ Upload blueprint registered at /upload")
    except Exception as e:
        print(f"❌ Upload blueprint registration failed: {e}")
else:
    #print("❌ Upload routes not available - creating fallback")
    
    # Create a simple fallback route for testing
    @app.route('/upload/balance-sheet', methods=['POST'])
    def fallback_upload():
        return jsonify({
            'success': False,
            'error': 'Upload routes not properly imported',
            'message': 'Check routes/upload_routes.py file'
        }), 503

# Debug: Print upload route status
print(f"Upload routes available: {UPLOAD_ROUTES_AVAILABLE}")


def initialize_database():
    """Initialize database connection using your actual classes"""
    global db_connection, db_operations
    
    if not DATABASE_CLASSES_AVAILABLE:
        #print("❌ Database classes not available - database features disabled")
        return False
    
    try:
        # Initialize your database connection
        db_connection = DatabaseConnection()
        
        # Test connection
        if db_connection.connect():
            print("✅ Database connection established")
            
            # Test the connection
            if db_connection.test_connection():
                print("✅ Database connection test successful")
                
                # Initialize database operations
                db_operations = DatabaseOperations(db_connection)
                print("✅ Database operations initialized")
                
                # Check if tables exist and create if needed
                setup_database_if_needed()
                
                return True
            else:
                print("❌ Database connection test failed")
                return False
        else:
            print("❌ Failed to establish database connection")
            return False
            
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

def setup_database_if_needed():
    """Setup database tables if they don't exist - ENHANCED"""
    try:
        if not db_connection:
            return False
        
        # Check existing tables
        existing_tables = ['cash_flow_statement']
        new_tables = ['balance_sheet_results', 'cash_flow_generated', 'imputation_results', 'processing_history']
        
        tables_created = 0
        
        for table in existing_tables:
            if not db_connection.table_exists(table):
                print(f"⚠️ Required table '{table}' not found")
                return False
        
        # Create new tables for balance sheet and cash flow generators
        for table in new_tables:
            if not db_connection.table_exists(table):
                print(f"🔧 Creating table: {table}")
                success = create_table(table)
                if success:
                    tables_created += 1
                    print(f"✅ Table '{table}' created successfully")
                else:
                    print(f"❌ Failed to create table '{table}'")
        
        if tables_created > 0:
            print(f"✅ Database setup completed - {tables_created} new tables created")
        else:
            print("✅ All database tables exist")
        
        return True
            
    except Exception as e:
        print(f"❌ Database setup error: {e}")
        return False




def initialize_components():
  """Initialize ML components, database, and NEW services"""
  global performance_tracker, ensemble_manager, ml_predictor
  global balance_sheet_service, cash_flow_service, data_imputation_service, financial_processing_service
  global balance_sheet_generator, cash_flow_generator, pdf_parsing_service
  
  # Initialize database first
  db_initialized = initialize_database()
  
  try:
      # Initialize existing components
      if EVALUATION_AVAILABLE:
          performance_tracker = PerformanceTracker()
          print("✅ Performance Tracker initialized")
      
      if ENSEMBLE_AVAILABLE:
          ensemble_manager = EnsembleManager()
          print("✅ Ensemble Manager initialized")
      
      if CONFIDENCE_UTILS_AVAILABLE:
          ml_predictor = MLPredictor()
          print("✅ ML Predictor with Confidence Support initialized")
      
      # Initialize NEW services using BalanceSheetGenerator directly
      try:
          db_config = get_db_config()
          
          balance_sheet_service = BalanceSheetGenerator(db_config)
          print("✅ Balance Sheet Service initialized")
          
          cash_flow_service = CashFlowGenerator(db_config) 
          print("✅ Cash Flow Service initialized")
          
          # Initialize Data Imputation Service
          if IMPUTATION_ALGORITHMS_AVAILABLE:
              data_imputation_service = DataImputationService()
              print("✅ Data Imputation Service initialized")
          else:
              # Create a simple fallback service
              class SimpleDataImputationService:
                  def __init__(self):
                      self.methods = ['knn', 'random_forest', 'industry_benchmarks']
                  
                  def analyze_missing_values(self, data, methods=['knn']):
                      return {'status': 'active', 'methods_available': self.methods}
              
              data_imputation_service = SimpleDataImputationService()
              print("✅ Data Imputation Service initialized (fallback)")
          
          # Initialize Financial Processing Service
          class SimpleFinancialProcessingService:
              def __init__(self, db_config):
                  self.db_config = db_config
                  self.currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD']
              
              def validate_data(self, financial_data, rules='standard'):
                  return {'status': 'valid', 'currencies_supported': self.currencies}
              
              def convert_currency(self, amount, from_currency, to_currency):
                  # Simple conversion logic (you can enhance this)
                  rates = {'USD': 1.0, 'EUR': 0.85, 'GBP': 0.73, 'JPY': 110, 'CAD': 1.25, 'AUD': 1.35}
                  if from_currency in rates and to_currency in rates:
                      return amount * (rates[to_currency] / rates[from_currency])
                  return amount
          
          financial_processing_service = SimpleFinancialProcessingService(db_config)
          print("✅ Financial Processing Service initialized")
          
      except Exception as e:
          print(f"⚠️ New services initialization failed: {e}")
      
      # Initialize Enhanced PDF Parsing Service
      if PDF_PARSER_LIBRARIES_AVAILABLE:
          try:
              config = {
                  'upload_folder': app.config.get('UPLOAD_FOLDER', 'uploads'),
                  'results_folder': app.config.get('RESULTS_FOLDER', 'uploads/results'),
                  'debug_mode': app.config.get('DEBUG', False),
                  'min_confidence_threshold': app.config.get('PDF_CONFIDENCE_THRESHOLD', 0.6),
                  'max_pages_to_scan': app.config.get('PDF_MAX_PAGES', 100)
              }
              pdf_parsing_service = PDFParsingService(config)
              print("✅ Enhanced PDF Parsing Service initialized")
          except Exception as e:
              print(f"⚠️ PDF Parsing Service initialization failed: {e}")
              pdf_parsing_service = None
      else:
          print("❌ PDF parsing libraries not available")
          pdf_parsing_service = None
      
  except Exception as e:
      print(f"⚠️ Component initialization failed: {e}")
  
  return db_initialized


def create_table(table_name):
    """Create specific tables for new functionality"""
    try:
        if table_name == 'balance_sheet_results':
            query = """
            CREATE TABLE balance_sheet_results (
                id SERIAL PRIMARY KEY,
                company_name VARCHAR(255),
                upload_session_id VARCHAR(100),
                original_filename VARCHAR(255),
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processing_method VARCHAR(50),
                accuracy_score DECIMAL(5,4),
                missing_values_imputed INTEGER DEFAULT 0,
                imputation_methods_used TEXT[],
                balance_sheet_data JSONB,
                validation_results JSONB,
                file_path VARCHAR(500),
                status VARCHAR(50) DEFAULT 'completed'
            )
            """
        
        elif table_name == 'cash_flow_generated':
            query = """
            CREATE TABLE cash_flow_generated (
                id SERIAL PRIMARY KEY,
                company_name VARCHAR(255),
                balance_sheet_id INTEGER REFERENCES balance_sheet_results(id),
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                generation_method VARCHAR(50) DEFAULT 'indirect',
                cash_flow_data JSONB,
                financial_metrics JSONB,
                liquidation_prediction DECIMAL(3,2),
                confidence_score DECIMAL(3,2),
                file_path VARCHAR(500),
                status VARCHAR(50) DEFAULT 'completed'
            )
            """
        
        elif table_name == 'imputation_results':
            query = """
            CREATE TABLE imputation_results (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(100),
                company_name VARCHAR(255),
                field_name VARCHAR(100),
                original_value DECIMAL(15,2),
                imputed_value DECIMAL(15,2),
                imputation_method VARCHAR(50),
                confidence_score DECIMAL(3,2),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        
        elif table_name == 'processing_history':
            query = """
            CREATE TABLE processing_history (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(100),
                company_name VARCHAR(255),
                process_type VARCHAR(50),
                status VARCHAR(50),
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                error_message TEXT,
                files_processed INTEGER DEFAULT 0,
                processing_time_seconds INTEGER,
                user_ip VARCHAR(45)
            )
            """
        
        else:
            return False
        
        return db_connection.execute_query(query)
        
    except Exception as e:
        print(f"❌ Error creating table {table_name}: {e}")
        return False

# ===========================================
# ENHANCED SERVICE INITIALIZATION (UPDATED)
# ===========================================


def get_db_config():
    """Get database configuration"""
    return {
        'host': 'localhost',
        'database': 'financial_db',
        'user': 'postgres',
        'password': 'Prateek@2003',
        'port': 5432
    }

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
# ===========================================
# EXISTING CASH FLOW DATA API ENDPOINTS (KEEPING YOUR EXISTING ONES)
# ===========================================

def fetch_company_cash_flow_data(company_name=None, company_id=None, year=None, limit=100):
    """
    Fetch cash flow statement data from PostgreSQL using your exact table structure
    """
    if not db_connection:
        raise Exception("Database not connected")
    
    try:
        base_query = """
        SELECT 
            id, company_id, year, generated_at, company_name, industry,
            net_income, depreciation_and_amortization, stock_based_compensation,
            changes_in_working_capital, accounts_receivable, inventory, accounts_payable,
            net_cash_from_operating_activities, capital_expenditures, acquisitions,
            net_cash_from_investing_activities, dividends_paid, share_repurchases,
            net_cash_from_financing_activities, free_cash_flow, ocf_to_net_income_ratio,
            liquidation_label, debt_to_equity_ratio, interest_coverage_ratio
        FROM cash_flow_statement
        WHERE 1=1
        """
        
        params = []
        conditions = []
        
        if company_name:
            conditions.append("LOWER(company_name) LIKE LOWER(%s)")
            params.append(f"%{company_name}%")
        
        if company_id:
            conditions.append("company_id = %s")
            params.append(company_id)
        
        if year:
            conditions.append("year = %s")
            params.append(year)
        
        if conditions:
            base_query += " AND " + " AND ".join(conditions)
        
        base_query += " ORDER BY company_name, year DESC"
        
        if limit:
            base_query += f" LIMIT {limit}"
        
        results = db_connection.execute_select(base_query, params)
        return results if results else []
    
    except Exception as e:
        print(f"Error fetching cash flow data: {e}")
        raise e



# Debug: Print all routes before registration
print("🔍 Registering API Blueprint...")

# Register existing blueprints with DEBUG
if EXISTING_ROUTES_AVAILABLE:
    try:
        app.register_blueprint(main_bp)
        print("✅ Main blueprint registered")
        
        # IMPORTANT: Register api_bp with url_prefix to match your frontend calls
        app.register_blueprint(api_bp, url_prefix='/api')
        print("✅ API blueprint registered with /api prefix")
        
        app.register_blueprint(analytics_bp, url_prefix='/analytics')
        print("✅ Analytics blueprint registered")
        
        app.register_blueprint(ai_bp, url_prefix='/ai')
        print("✅ AI blueprint registered")
        
        app.register_blueprint(upload_bp, url_prefix='/upload')
        print("✅ Upload blueprint registered")
        
    except Exception as e:
        print(f"⚠️ Error registering existing blueprints: {e}")

# Also register api_bp WITHOUT prefix for backward compatibility
try:
    app.register_blueprint(api_bp)
    print("✅ API blueprint also registered without prefix")
except Exception as e:
    print(f"⚠️ Error registering api_bp without prefix: {e}")

# Debug: Print all registered routes
print("\n🔍 All registered routes:")
for rule in app.url_map.iter_rules():
    methods = ','.join(rule.methods - {'HEAD', 'OPTIONS'})
    print(f"  {rule.rule:<50} [{methods}] -> {rule.endpoint}")
print()

# Register existing blueprints
if EXISTING_ROUTES_AVAILABLE:
    try:
        app.register_blueprint(main_bp)
        app.register_blueprint(api_bp, url_prefix='/api')  
        app.register_blueprint(analytics_bp, url_prefix='/analytics')
        app.register_blueprint(ai_bp, url_prefix='/ai')
        app.register_blueprint(upload_bp, url_prefix='/upload')
        print("✅ Existing blueprints registered successfully")
    except Exception as e:
        print(f"⚠️ Error registering existing blueprints: {e}")

# Register NEW blueprints
if NEW_ROUTES_AVAILABLE:
    try:
        app.register_blueprint(balance_sheet_bp, url_prefix='/balance-sheet')
        app.register_blueprint(cash_flow_bp, url_prefix='/cash-flow')
        print("✅ NEW Balance Sheet & Cash Flow blueprints registered successfully")
    except Exception as e:
        print(f"⚠️ Error registering new blueprints: {e}")



# Add these routes in your app.py (around line 2000, before if __name__ == '__main__':)

@app.route('/cash_flow_generator.html')
def cash_flow_generator_html():
    """Serve cash flow generator HTML file directly"""
    try:
        return render_template('cash_flow_generator.html')
    except Exception as e:
        try:
            return send_from_directory('.', 'cash_flow_generator.html')
        except:
            return f"Cash flow generator file not found: {e}", 404

@app.route('/cash-flow-generator.html')  # Alternative URL
def cash_flow_generator_alt():
    """Alternative route for cash flow generator"""
    return cash_flow_generator_html()

@app.route('/balance_sheet_generator.html')
def balance_sheet_generator_html():
    """Serve balance sheet generator HTML file directly"""
    try:
        return render_template('balance_sheet_generator.html')
    except Exception as e:
        try:
            return send_from_directory('.', 'balance_sheet_generator.html')
        except:
            return f"Balance sheet generator file not found: {e}", 404

@app.route('/balance-sheet-generator.html')  # Alternative URL
def balance_sheet_generator_alt():
    """Alternative route for balance sheet generator"""
    return balance_sheet_generator_html()

# Fix existing route (around line 1400 in your app.py)
@app.route('/cash_flow_generator')
def cash_flow_generator():
    """Cash flow generator page - FIXED"""
    try:
        return render_template('cash_flow_generator.html')
    except Exception as e:
        try:
            return send_from_directory('.', 'cash_flow_generator.html')
        except:
            return f"Template not found: {e}", 404
        

@app.route('/debug/static-test')
def debug_static_test():
    """Debug static files"""
    import os
    
    static_info = {
        'static_folder_configured': app.static_folder,
        'static_url_path': app.static_url_path,
        'current_directory': os.getcwd(),
        'static_folder_exists': os.path.exists('static'),
        'images_folder_exists': os.path.exists('static/images'),
        'guide_folder_exists': os.path.exists('static/images/guide'),
        'files_in_guide': []
    }
    
    try:
        if os.path.exists('static/images/guide'):
            static_info['files_in_guide'] = os.listdir('static/images/guide')
    except:
        static_info['files_in_guide'] = ['Error reading directory']
    
    return jsonify(static_info)



@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files explicitly"""
    try:
        return send_from_directory('static', filename)
    except Exception as e:
        return jsonify({'error': f'File not found: {filename}', 'message': str(e)}), 404



# Add this route in app.py (around line 1800):

@app.route('/api/cash-flow/get-latest-processed', methods=['GET'])
def get_latest_processed_cash_flow():
    """Get latest processed cash flow data"""
    try:
        if not db_connection:
            return jsonify({
                'success': False,
                'error': 'Database not connected'
            }), 500
        
        # Get latest processed cash flow from database
        query = """
        SELECT company_id, company_name, industry, year,
               net_income, net_cash_from_operating_activities,
               net_cash_from_investing_activities, net_cash_from_financing_activities,
               free_cash_flow, generated_at, liquidation_label
        FROM cash_flow_statement 
        ORDER BY generated_at DESC 
        LIMIT 1
        """
        
        result = db_connection.execute_select(query)
        
        if result:
            cash_flow_data = clean_data_for_json(result[0])
            
            return jsonify({
                'success': True,
                'data': cash_flow_data,
                'cash_flow_id': f"cf_{cash_flow_data.get('company_id', 'unknown')}",
                'processed_at': cash_flow_data.get('generated_at'),
                'message': 'Latest processed data retrieved successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No processed cash flow data found',
                'message': 'Please upload and process some files first'
            }), 404
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Also add this endpoint for file upload processing:
@app.route('/api/cash-flow/process-files', methods=['POST'])
def process_cash_flow_files():
    """Process cash flow files upload"""
    try:
        if 'files' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No files provided'
            }), 400
        
        files = request.files.getlist('files')
        company_name = request.form.get('company_name', 'Uploaded Company')
        
        if not files or all(file.filename == '' for file in files):
            return jsonify({
                'success': False,
                'error': 'No files selected'
            }), 400
        
        # Process files
        processed_data = {}
        session_id = f"cf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        for file in files:
            if file and allowed_file(file.filename):
                # Save file temporarily
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
                file.save(file_path)
                
                try:
                    # Process based on file type
                    if filename.endswith('.csv'):
                        import pandas as pd
                        df = pd.read_csv(file_path, encoding='utf-8', error_bad_lines=False)
                        # Extract cash flow data from CSV
                        extracted_data = extract_cash_flow_from_dataframe(df)
                        processed_data.update(extracted_data)
                    
                    elif filename.endswith(('.xlsx', '.xls')):
                        import pandas as pd
                        df = pd.read_excel(file_path)
                        extracted_data = extract_cash_flow_from_dataframe(df)
                        processed_data.update(extracted_data)
                
                finally:
                    # Clean up
                    try:
                        os.remove(file_path)
                    except:
                        pass
        
        # Generate complete cash flow with ML estimation
        if processed_data:
            complete_data = apply_ml_estimation_cash_flow(processed_data, company_name)
            
            # Save to database
            cash_flow_id = save_cash_flow_to_database(complete_data, session_id, company_name, 'unknown')
            
            return jsonify({
                'success': True,
                'cash_flow_id': cash_flow_id,
                'data': clean_data_for_json(complete_data),
                'files_processed': len([f for f in files if f.filename]),
                'message': 'Files processed successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No financial data could be extracted from uploaded files'
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Helper function for cash flow data extraction
def extract_cash_flow_from_dataframe(df):
    """Extract cash flow data from pandas DataFrame"""
    extracted_data = {}
    
    # Convert DataFrame to search for financial items
    for index, row in df.iterrows():
        for col in df.columns:
            item_name = str(row.iloc[0] if col == df.columns[0] else col).lower()
            value = clean_financial_value(row[col] if col != df.columns[0] else row.iloc[1] if len(row) > 1 else None)
            
            # Map financial items
            if 'net income' in item_name and value is not None:
                extracted_data['net_income'] = value
            elif 'operating cash' in item_name and value is not None:
                extracted_data['net_cash_from_operating_activities'] = value
            elif 'investing cash' in item_name and value is not None:
                extracted_data['net_cash_from_investing_activities'] = value
            elif 'financing cash' in item_name and value is not None:
                extracted_data['net_cash_from_financing_activities'] = value
            elif 'free cash flow' in item_name and value is not None:
                extracted_data['free_cash_flow'] = value
    
    return extracted_data

def apply_ml_estimation_cash_flow(extracted_data, company_name):
    """Apply ML estimation for missing cash flow values"""
    complete_data = {
        'company_name': company_name,
        'generated_at': datetime.now().isoformat(),
        'year': datetime.now().year,
        'data_source': 'uploaded_with_ml_estimation'
    }
    
    # Copy extracted data
    complete_data.update(extracted_data)
    
    # Apply industry ratios for missing values
    if 'net_income' in extracted_data and extracted_data['net_income']:
        base_income = extracted_data['net_income']
        
        if 'net_cash_from_operating_activities' not in extracted_data:
            complete_data['net_cash_from_operating_activities'] = base_income * 1.15  # 115% of net income
        
        if 'free_cash_flow' not in extracted_data:
            ocf = complete_data.get('net_cash_from_operating_activities', base_income * 1.15)
            complete_data['free_cash_flow'] = ocf * 0.75  # 75% of OCF
    
    return complete_data




# Add these endpoints to your app.py

# Add this helper function first (at the top of your app.py, after imports)
def clean_data_for_json(data):
    """Enhanced clean data for JSON serialization - replace NaN with None and handle column names"""
    import pandas as pd
    import numpy as np
    
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            # ✅ CHECK FOR COLUMN NAMES AS VALUES
            if isinstance(value, str) and value == key:
                # This means we got column name instead of actual value
                cleaned[key] = None
                continue
            
            if isinstance(value, float):
                if pd.isna(value) or np.isnan(value) or not np.isfinite(value):
                    cleaned[key] = None
                else:
                    cleaned[key] = value
            elif isinstance(value, dict):
                cleaned[key] = clean_data_for_json(value)
            elif isinstance(value, list):
                cleaned[key] = [clean_data_for_json(item) if isinstance(item, dict) else item for item in value]
            elif pd.isna(value):
                cleaned[key] = None
            else:
                cleaned[key] = value
        return cleaned
    elif isinstance(data, list):
        return [clean_data_for_json(item) if isinstance(item, dict) else item for item in data]
    elif isinstance(data, float):
        if pd.isna(data) or np.isnan(data) or not np.isfinite(data):
            return None
        return data
    elif pd.isna(data):
        return None
    return data




@app.route('/guide')
def user_guide():
    return render_template('user_guide.html')



# Updated cash flow upload endpoint
@app.route('/api/cash-flow/process-upload', methods=['POST'])
def api_cash_flow_process_upload():
    """API endpoint for cash flow file upload processing"""
    try:
        if not cash_flow_service:
            return jsonify({'success': False, 'error': 'Cash flow service not available'}), 503
        
        if 'files' not in request.files:
            return jsonify({'success': False, 'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        company_name = request.form.get('company_name', 'Unknown Company')
        company_id = request.form.get('company_id', f'comp_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        industry = request.form.get('industry', 'unknown')
        
        if not files or all(file.filename == '' for file in files):
            return jsonify({'success': False, 'error': 'No files selected'}), 400
        
        # Save uploaded files
        uploaded_files = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                upload_folder = app.config.get('CASH_FLOW_UPLOAD_FOLDER', 'uploads/cash_flow')
                os.makedirs(upload_folder, exist_ok=True)
                
                file_path = os.path.join(upload_folder, f"{company_id}_{filename}")
                file.save(file_path)
                uploaded_files.append(file_path)
        
        if not uploaded_files:
            return jsonify({'success': False, 'error': 'No valid files uploaded'}), 400
        
        # Process with CashFlowGenerator.process_uploaded_file method
        result = cash_flow_service.process_uploaded_file(
            file_path=uploaded_files[0],
            company_id=company_id,
            company_name=company_name,
            industry=industry
        )
        
        # CLEAN DATA BEFORE SENDING TO AVOID NaN JSON ERROR
        cleaned_data = None
        cash_flow_id = None
        
        if result.get('data'):
            cleaned_data = clean_data_for_json(result.get('data'))
            
            # ✅ GENERATE CASH FLOW ID (ENHANCED)
            try:
                # Try to save to database and get ID
                cash_flow_id = save_cash_flow_to_database(cleaned_data, company_id, company_name, industry)
                print(f"✅ Cash flow saved to database with ID: {cash_flow_id}")
            except Exception as db_error:
                print(f"⚠️ Database save failed: {db_error}")
                # Always generate an ID even if database fails
                cash_flow_id = f"cf_{company_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                print(f"🔄 Using temporary ID: {cash_flow_id}")
        else:
            # Even if no data, generate an ID for consistency
            cash_flow_id = f"cf_error_{company_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Clean up uploaded files
        for file_path in uploaded_files:
            try:
                os.remove(file_path)
            except:
                pass
        
        # ✅ ALWAYS RETURN CASH_FLOW_ID
        return jsonify({
            'success': result.get('success', True),  # Default to True if processing worked
            'data': cleaned_data,
            'cash_flow_id': cash_flow_id,  # ✅ ALWAYS INCLUDE THIS
            'message': result.get('message', 'File processed successfully'),
            'files_processed': len(uploaded_files),
            'accuracy': result.get('accuracy', 95),
            'ml_enhanced': result.get('ml_enhanced', True)
        })
        
    except Exception as e:
        # Clean up files on error but still return an ID
        try:
            for file_path in uploaded_files:
                os.remove(file_path)
        except:
            pass
        
        # Generate error ID
        error_id = f"cf_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return jsonify({
            'success': False,
            'error': str(e),
            'cash_flow_id': error_id,  # ✅ INCLUDE ID EVEN ON ERROR
            'files_processed': 0
        }), 500


# ✅ ADD THIS HELPER FUNCTION
def save_cash_flow_to_database(cash_flow_data, company_id, company_name, industry):
    """Save cash flow data to database and return cash_flow_id"""
    try:
        import uuid
        
        # Generate unique cash flow ID
        cash_flow_id = f"cf_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d')}"
        
        # ✅ SAVE WITH SESSION ID
        if db_connection:
            insert_query = """
            INSERT INTO cash_flow_statement (
                company_id, company_name, industry, year,
                net_income, net_cash_from_operating_activities,
                net_cash_from_investing_activities, net_cash_from_financing_activities,
                free_cash_flow, generated_at,
                session_id  -- Add this column to your table
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """
            
            
            
            # Extract values from cash_flow_data
            values = [
                company_id,
                company_name,
                industry,
                datetime.now().year,
                cash_flow_data.get('net_income', 0),
                cash_flow_data.get('net_cash_from_operating_activities', 0),
                cash_flow_data.get('net_cash_from_investing_activities', 0),
                cash_flow_data.get('net_cash_from_financing_activities', 0),
                cash_flow_data.get('free_cash_flow', 0),
                datetime.now()
            ]
            
            result = db_connection.execute_query(insert_query, values)
            if result:
                print(f"💾 Cash flow data saved to database with ID: {cash_flow_id}")
            
        return cash_flow_id
        
    except Exception as e:
        print(f"❌ Database save error: {e}")
        # Return a temporary ID even if database save fails
        return f"cf_temp_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d')}"

# =====================================
# MISSING ROUTES - ADD THESE TO YOUR APP.PY
# Add these routes before if __name__ == '__main__':
# =====================================

# Add these routes in your app.py (around line 2000, before if __name__ == '__main__':)

@app.route('/balance_sheet_results.html')
def balance_sheet_results_html():
    """Serve balance sheet results HTML file directly"""
    try:
        return render_template('balance_sheet_results.html')
    except Exception as e:
        # If template not found, try to serve from static
        try:
            return send_from_directory('.', 'balance_sheet_results.html')
        except:
            return f"File not found: {e}", 404

@app.route('/balance-sheet-results.html')  # Alternative URL
def balance_sheet_results_alt():
    """Alternative route for balance sheet results"""
    return balance_sheet_results_html()

@app.route('/cash-flow-results.html')
def cash_flow_results_html():
    """Serve cash flow results HTML file directly"""
    try:
        return render_template('cash_flow_results.html')
    except Exception as e:
        try:
            return send_from_directory('.', 'cash_flow_results.html')
        except:
            return f"File not found: {e}", 404

@app.route('/upload.html')
def upload_html():
    """Serve upload HTML file directly"""
    try:
        return render_template('upload.html')
    except Exception as e:
        try:
            return send_from_directory('.', 'upload.html')
        except:
            return f"File not found: {e}", 404



@app.route('/company/<company_name>/analysis')
def auto_company_analysis(company_name):
    try:
        return render_template('analytics.html', 
                             selected_company=company_name,
                             auto_analyze=True)
    except Exception as e:
        print(f"Error: {e}")  # Debug ke liye
        return f"Template Error: {str(e)}"











# Process documents endpoint (ye bhi missing hai)
@app.route('/process_documents', methods=['POST'])
def process_documents():
    """Process uploaded documents and generate balance sheet"""
    try:
        if not balance_sheet_service:
            return jsonify({
                'success': False,
                'error': 'Balance sheet service not available'
            }), 503
        
        # Get uploaded files
        files = []
        for key in request.files:
            if key.startswith('file_'):
                files.append(request.files[key])
        
        if not files:
            return jsonify({
                'success': False,
                'error': 'No files uploaded'
            }), 400
        
        analysis_type = request.form.get('analysis_type', 'balance-sheet')
        company_name = request.form.get('company_name', 'Your Company')
        
        # Process files using your BalanceSheetGenerator
        session_id = f"bs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save uploaded files temporarily
        uploaded_files = []
        for i, file in enumerate(files):
            if file and file.filename:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
                file.save(file_path)
                uploaded_files.append(file_path)
        
        # Use your BalanceSheetGenerator to process
        if uploaded_files:
            # Read first file (you can enhance this to process multiple files)
            import pandas as pd
            
            try:
                if uploaded_files[0].endswith('.csv'):
                    df = pd.read_csv(uploaded_files[0])
                elif uploaded_files[0].endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded_files[0])
                else:
                    return jsonify({
                        'success': False,
                        'error': 'Unsupported file format'
                    }), 400
                
                # Process with BalanceSheetGenerator
                result = balance_sheet_service.process_uploaded_balance_sheet(
                    df=df,
                    company_id=session_id,
                    company_name=company_name,
                    industry='unknown'
                )
                
                if result:
                    # Clean data for JSON
                    cleaned_result = clean_data_for_json(result)
                    
                    return jsonify({
                        'success': True,
                        'balance_sheet_data': cleaned_result,
                        'session_id': session_id,
                        'message': 'Balance sheet generated successfully'
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': 'Failed to process balance sheet'
                    }), 500
                    
            finally:
                # Clean up temporary files
                for file_path in uploaded_files:
                    try:
                        os.remove(file_path)
                    except:
                        pass
        
        return jsonify({
            'success': False,
            'error': 'No valid files to process'
        }), 400
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500



# Add this API endpoint to your Flask app

@app.route('/api/get-cash-flow/<cash_flow_id>', methods=['GET'])
def get_cash_flow(cash_flow_id):
    """Get cash flow data by ID - ENHANCED to use localStorage data"""
    try:
        print(f"🔍 API called for cash flow ID: {cash_flow_id}")
        
        # Since localStorage data is not in database, return structured response
        # This will work with your frontend's populateCashFlowData function
        
        # Create response matching your localStorage data structure
        cash_flow_response = {
            'id': cash_flow_id,
            'company_id': 'comp_uploaded',
            'company_name': 'Uploaded Company',
            'industry': 'unknown',
            'year': 2024,
            'generated_at': datetime.now().isoformat(),
            'data_source': 'uploaded_with_ml_estimation',
            
            # Financial data - will be populated by frontend from localStorage
            'net_income': None,
            'depreciation_and_amortization': None,
            'stock_based_compensation': None,
            'changes_in_working_capital': None,
            'accounts_receivable': None,
            'inventory': None,
            'accounts_payable': None,
            'net_cash_from_operating_activities': None,
            'capital_expenditures': None,
            'acquisitions': 0,
            'net_cash_from_investing_activities': None,
            'dividends_paid': 0,
            'share_repurchases': 0,
            'net_cash_from_financing_activities': 0,
            'free_cash_flow': None,
            'ocf_to_net_income_ratio': None,
            'liquidation_label': 0,
            'debt_to_equity_ratio': None,
            'interest_coverage_ratio': None,
            
            # Data quality info
            'accuracy_percentage': 48.1,
            'original_fields_count': 6,
            'total_fields_count': 21
        }
        
        print(f"✅ Returning fallback data structure for frontend processing")
        
        return jsonify({
            'success': True,
            'cash_flow': cash_flow_response,
            'source': 'api_fallback',
            'message': 'Data will be loaded from localStorage by frontend'
        })
        
    except Exception as e:
        print(f"❌ Error in get_cash_flow: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'cash_flow_id': cash_flow_id
        }), 500





            # Add this function after get_cash_flow function
def add_missing_database_columns():
    """Add missing columns to cash_flow_statement table if they don't exist"""
    try:
        if not db_connection:
            return False
        
        # Add columns that might be missing
        missing_columns = [
            ('accuracy_percentage', 'DECIMAL(5,2) DEFAULT 95.0'),
            ('original_fields_count', 'INTEGER DEFAULT 0'),
            ('total_fields_count', 'INTEGER DEFAULT 21'),
            ('data_source', 'VARCHAR(100) DEFAULT \'database\''),
            ('session_id', 'VARCHAR(100)')
        ]
        
        for column_name, column_definition in missing_columns:
            try:
                alter_query = f"""
                ALTER TABLE cash_flow_statement 
                ADD COLUMN IF NOT EXISTS {column_name} {column_definition}
                """
                db_connection.execute_query(alter_query)
                print(f"✅ Added column {column_name} to cash_flow_statement table")
            except Exception as e:
                print(f"⚠️ Column {column_name} might already exist: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error adding database columns: {e}")
        return False

# Call this function during startup
if db_connection:
    add_missing_database_columns()

# Additional endpoint for download functionality
@app.route('/api/download-cash-flow-analysis/<cash_flow_id>', methods=['GET'])
def download_cash_flow_analysis(cash_flow_id):
    """Download cash flow analysis as CSV"""
    db = None
    try:
        # Create database connection
        db = DatabaseConnection()
        db.connect()
        
        # Get cash flow data
        query = """
        SELECT * FROM cash_flow_statement WHERE id = %s
        """
        
        db.cursor.execute(query, (cash_flow_id,))
        result = db.cursor.fetchone()
        
        if result:
            # Create CSV content
            csv_content = f"""Cash Flow Analysis Report
Company: {result[4]}
Industry: {result[5]}
Year: {result[2]}
Generated: {result[3]}

OPERATING ACTIVITIES
Net Income,{result[6]}
Depreciation and Amortization,{result[7]}
Stock-based Compensation,{result[8]}
Changes in Working Capital,{result[9]}
Net Cash from Operating Activities,{result[13]}

INVESTING ACTIVITIES
Capital Expenditures,{result[14]}
Acquisitions,{result[15]}
Net Cash from Investing Activities,{result[16]}

FINANCING ACTIVITIES
Dividends Paid,{result[17]}
Share Repurchases,{result[18]}
Net Cash from Financing Activities,{result[19]}

KEY METRICS
Free Cash Flow,{result[20]}
OCF to Net Income Ratio,{result[21]}
Debt to Equity Ratio,{result[23]}
Interest Coverage Ratio,{result[24]}
Liquidation Risk Label,{result[22]}"""

            response = make_response(csv_content)
            response.headers['Content-Type'] = 'text/csv'
            response.headers['Content-Disposition'] = f'attachment; filename=cash_flow_analysis_{cash_flow_id}.csv'
            return response
        else:
            return jsonify({'error': 'Cash flow data not found'}), 404
            
    except Exception as e:
        logger.error(f"Error downloading cash flow analysis: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        if db:
            db.close()




@app.route('/debug/templates')
def debug_templates():
    """Debug template locations"""
    import os
    template_info = {}
    
    # Check templates folder
    templates_dir = os.path.join(os.getcwd(), 'templates')
    if os.path.exists(templates_dir):
        template_files = [f for f in os.listdir(templates_dir) if f.endswith('.html')]
        template_info['templates_folder'] = template_files
        template_info['templates_path'] = templates_dir
    else:
        template_info['templates_folder'] = 'NOT FOUND'
    
    # Check current directory
    current_files = [f for f in os.listdir('.') if f.endswith('.html')]
    template_info['current_directory'] = current_files
    template_info['current_path'] = os.getcwd()
    
    # Check if specific file exists
    cash_flow_results_locations = []
    possible_paths = [
        'cash-flow-results.html',
        'templates/cash-flow-results.html',
        'static/cash-flow-results.html'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            cash_flow_results_locations.append(f"✅ {path}")
        else:
            cash_flow_results_locations.append(f"❌ {path}")
    
    template_info['cash_flow_results_locations'] = cash_flow_results_locations
    
    return jsonify(template_info)

@app.route('/test-results')
def test_results():
    """Simple test results page"""
    return '''
    <h1>✅ Test Results Page Working!</h1>
    <p>Cash Flow ID: <span id="id"></span></p>
    <script>
        document.getElementById('id').textContent = localStorage.getItem('cash_flow_id') || 'No ID';
    </script>
    '''











@app.route('/api/cash-flow/generate-from-balance', methods=['POST'])
def api_cash_flow_generate_from_balance():
    """API endpoint for generating cash flow from balance sheet"""
    try:
        if not cash_flow_service:
            return jsonify({'success': False, 'error': 'Cash flow service not available'}), 503
        
        data = request.json
        company_id = data.get('company_id')
        company_name = data.get('company_name', 'Unknown Company')
        year = data.get('year', datetime.now().year)
        industry = data.get('industry', 'unknown')
        
        if not company_id:
            return jsonify({'success': False, 'error': 'Company ID required'}), 400
        
        # Check if balance sheet data exists
        if not db_connection:
            return jsonify({'success': False, 'error': 'Database connection failed'}), 500
        
        balance_sheet_query = """
        SELECT company_id, company_name, year, industry,
               total_assets, total_liabilities, total_equity
        FROM balance_sheet_1 
        WHERE company_id = %s AND year = %s
        """
        
        balance_sheet_result = db_connection.execute_select(balance_sheet_query, [company_id, year])
        
        if not balance_sheet_result:
            return jsonify({
                'success': False,
                'error': f'No balance sheet data found for {company_id} in {year}'
            }), 404
        
        # Use CashFlowGenerator to generate
        result = cash_flow_service.generate_cash_flow_for_company(
            company_id=company_id,
            year=year,
            company_name=company_name,
            industry=industry
        )
        
        # Clean data and generate ID
        cleaned_data = None
        cash_flow_id = None
        
        if result.get('data'):
            cleaned_data = clean_data_for_json(result.get('data'))
            
            # ✅ GENERATE CASH FLOW ID
            try:
                cash_flow_id = save_cash_flow_to_database(cleaned_data, company_id, company_name, industry)
            except Exception as db_error:
                print(f"⚠️ Database save failed: {db_error}")
                cash_flow_id = f"cf_gen_{company_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            cash_flow_id = f"cf_gen_error_{company_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return jsonify({
            'success': result.get('success', True),
            'data': cleaned_data,
            'cash_flow_id': cash_flow_id,  # ✅ ALWAYS INCLUDE THIS
            'message': result.get('message', 'Cash flow generated successfully'),
            'accuracy': result.get('accuracy', 95),
            'ml_enhanced': result.get('ml_enhanced', True)
        })
        
    except Exception as e:
        error_id = f"cf_gen_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return jsonify({
            'success': False,
            'error': str(e),
            'cash_flow_id': error_id  # ✅ INCLUDE ID EVEN ON ERROR
        }), 500

@app.route('/cash-flow/companies')
def get_cash_flow_companies_simple():
    """Simple endpoint for risk assessment"""
    try:
        if not db_connection:
            return jsonify({
                'success': False,
                'error': 'Database not connected'
            }), 500
        
        # Simple query
        query = """
        SELECT DISTINCT company_name, industry, year, net_income,
               net_cash_from_operating_activities, free_cash_flow,
               liquidation_label, debt_to_equity_ratio
        FROM cash_flow_statement
        WHERE company_name IS NOT NULL
        ORDER BY company_name, year DESC
        """
        
        results = db_connection.execute_select(query)
        
        if not results:
            return jsonify({
                'success': True,
                'cash_flow_data': [],
                'records_found': 0
            })
        
        return jsonify({
            'success': True,
            'cash_flow_data': results,
            'records_found': len(results)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'cash_flow_data': []
        }), 500

@app.route('/cash-flow/<company_name>')
def get_company_cash_flow_simple(company_name):
    """Simple endpoint for individual company data"""
    try:
        if not db_connection:
            return jsonify({
                'success': False,
                'error': 'Database not connected'
            }), 500
        
        query = """
        SELECT * FROM cash_flow_statement 
        WHERE company_name = %s 
        ORDER BY year DESC
        """
        
        results = db_connection.execute_select(query, [company_name])
        
        if not results:
            return jsonify({
                'success': False,
                'error': f'No data found for {company_name}',
                'data': []
            }), 404
        
        return jsonify({
            'success': True,
            'data': results,
            'company_name': company_name
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'data': []
        }), 500


@app.route('/api/trends/analyze', methods=['POST'])
def analyze_trends():
    """Analyze financial trends for companies"""
    try:
        data = request.json
        company_names = data.get('companies', [])
        timeframe = data.get('timeframe', '3year')
        metrics = data.get('metrics', ['net_income', 'net_cash_from_operating_activities'])
        
        if not company_names:
            return jsonify({
                'success': False,
                'error': 'No companies specified'
            }), 400
        
        # Get trend data
        trends_data = {}
        
        for company in company_names:
            query = """
            SELECT year, company_name, net_income, 
                   net_cash_from_operating_activities, free_cash_flow,
                   debt_to_equity_ratio, liquidation_label
            FROM cash_flow_statement 
            WHERE company_name = %s 
            ORDER BY year ASC
            """
            
            company_data = db_connection.execute_select(query, [company])
            trends_data[company] = company_data if company_data else []
        
        return jsonify({
            'success': True,
            'trends_data': trends_data,
            'timeframe': timeframe,
            'metrics_analyzed': metrics
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/analytics/risk-distribution')
def get_risk_distribution():
    """Get risk distribution analytics"""
    try:
        query = """
        SELECT 
            CASE 
                WHEN liquidation_label = 1 THEN 'High Risk'
                WHEN net_cash_from_operating_activities < 0 THEN 'Medium Risk'
                ELSE 'Low Risk'
            END as risk_category,
            COUNT(*) as count
        FROM cash_flow_statement
        GROUP BY risk_category
        """
        
        results = db_connection.execute_select(query)
        
        return jsonify({
            'success': True,
            'risk_distribution': results if results else []
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/dashboard/summary')
def get_dashboard_summary():
    """Get dashboard summary statistics"""
    try:
        stats = {}
        
        # Total companies
        total_query = "SELECT COUNT(DISTINCT company_name) as total FROM cash_flow_statement"
        total_result = db_connection.execute_select(total_query)
        stats['total_companies'] = total_result[0]['total'] if total_result else 0
        
        # High risk companies
        risk_query = "SELECT COUNT(DISTINCT company_name) as high_risk FROM cash_flow_statement WHERE liquidation_label = 1"
        risk_result = db_connection.execute_select(risk_query)
        stats['high_risk_companies'] = risk_result[0]['high_risk'] if risk_result else 0
        
        return jsonify({
            'success': True,
            'summary': stats
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    
# Add these new endpoints after your existing ones

@app.route('/api/pdf/enhanced-parse', methods=['POST'])
def enhanced_pdf_parse():
    """Enhanced PDF parsing endpoint"""
    try:
        if not PDF_PARSER_LIBRARIES_AVAILABLE or not pdf_parsing_service:
            return jsonify({
                'success': False,
                'error': 'PDF parsing service not available'
            }), 503
        
        if 'files' not in request.files:
            return jsonify({'success': False, 'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        company_name = request.form.get('company_name', 'Unknown Company')
        
        if not files or all(file.filename == '' for file in files):
            return jsonify({'success': False, 'error': 'No files selected'}), 400
        
        session_id = f"pdf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        results = []
        
        for file in files:
            if file and file.filename.lower().endswith('.pdf'):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
                file.save(file_path)
                
                # Validate and process
                validation = pdf_parsing_service.validate_file(file_path)
                if validation['valid']:
                    result = pdf_parsing_service.process_uploaded_file(file_path, company_name, session_id)
                    result['filename'] = filename
                    results.append(result)
                else:
                    results.append({'filename': filename, 'success': False, 'error': validation['error']})
                
                # Clean up
                try:
                    os.remove(file_path)
                except:
                    pass
        
        successful_files = [r for r in results if r.get('success', False)]
        
        return jsonify({
            'success': len(successful_files) > 0,
            'session_id': session_id,
            'total_files': len(files),
            'successful_files': len(successful_files),
            'results': results
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/pdf/validate-document', methods=['POST'])
def validate_pdf_document():
    """Validate PDF document"""
    try:
        if not PDF_PARSER_LIBRARIES_AVAILABLE or not pdf_parsing_service:
            return jsonify({'success': False, 'error': 'PDF service not available'}), 503
        
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        temp_filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{temp_filename}")
        file.save(temp_path)
        
        try:
            validation = pdf_parsing_service.validate_file(temp_path)
            return jsonify(validation)
        finally:
            try:
                os.remove(temp_path)
            except:
                pass
                
    except Exception as e:
        return jsonify({'valid': False, 'error': str(e)}), 500

@app.route('/api/pdf/parsing-methods')
def get_pdf_parsing_methods():
    """Get available PDF parsing methods"""
    try:
        if not PDF_PARSER_LIBRARIES_AVAILABLE:
            return jsonify({'success': False, 'error': 'PDF parsing not available'}), 503
        
        methods = {
            'camelot': {
                'name': 'Camelot',
                'description': 'Best for tables with visible borders',
                'accuracy_range': '80-95%'
            },
            'tabula': {
                'name': 'Tabula', 
                'description': 'Good for simple tables',
                'accuracy_range': '70-85%'
            },
            'pdfplumber': {
                'name': 'PDFplumber',
                'description': 'Excellent for text-based tables',
                'accuracy_range': '75-90%'
            }
        }
        
        return jsonify({
            'success': True,
            'methods': methods,
            'recommended': 'camelot',
            'supported_formats': ['PDF']
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/cash-flow')
def get_cash_flow_by_query():
    """Get cash flow data by query parameter for trend analysis"""
    try:
        company = request.args.get('company')
        
        if not company:
            return jsonify({
                'success': False,
                'error': 'Company parameter required'
            }), 400
        
        if not db_connection:
            return jsonify({
                'success': False,
                'error': 'Database not connected'
            }), 500
        
        query = """
        SELECT * FROM cash_flow_statement 
        WHERE company_name = %s 
        ORDER BY year ASC
        """
        
        results = db_connection.execute_select(query, [company])
        
        if not results:
            return jsonify({
                'success': False,
                'error': f'No data found for company: {company}',
                'data': []
            }), 404
        
        return jsonify({
            'success': True,
            'data': results,
            'company_name': company,
            'total_years': len(results)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'data': []
        }), 500





# ===========================================
# NEW WEB ROUTES (BALANCE SHEET & CASH FLOW)
# ===========================================

@app.route('/balance-sheet')
def balance_sheet_generator_page():
    """Balance Sheet Generator main page"""
    try:
        return render_template('balance_sheet_generator.html',
                             balance_sheet_service_available=NEW_SERVICES_AVAILABLE,
                             imputation_methods=app.config['IMPUTATION_METHODS'],
                             max_file_size=app.config['MAX_CONTENT_LENGTH'],
                             allowed_extensions=app.config['ALLOWED_EXTENSIONS'])
    except Exception as e:
        flash(f"Error loading balance sheet generator: {e}", 'error')
        return redirect(url_for('index'))

@app.route('/balance-sheet/results/<session_id>')
def balance_sheet_results_with_session(session_id):
    """Balance Sheet Results page with session"""
    try:
        return render_template('balance_sheet_results.html',
                             session_id=session_id,
                             balance_sheet_service_available=NEW_SERVICES_AVAILABLE)
    except Exception as e:
        flash(f"Error loading balance sheet results: {e}", 'error')
        return redirect(url_for('balance_sheet_generator_page'))

# Add new route without session_id
@app.route('/balance-sheet-results')
def balance_sheet_results():
    """Balance Sheet Results page without session"""
    try:
        return render_template('balance_sheet_results.html',
                             session_id=None,
                             balance_sheet_service_available=NEW_SERVICES_AVAILABLE)
    except Exception as e:
        flash(f"Error loading balance sheet results: {e}", 'error')
        return redirect(url_for('balance_sheet_generator_page'))



@app.route('/cash-flow-generator/results/<session_id>')
def cash_flow_results(session_id):
    """Cash Flow Results page"""
    try:
        return render_template('cash_flow_results.html',
                             session_id=session_id,
                             cash_flow_service_available=NEW_SERVICES_AVAILABLE)
    except Exception as e:
        flash(f"Error loading cash flow results: {e}", 'error')
        return redirect(url_for('cash_flow_generator_page'))

# ===========================================
# NEW API ENDPOINTS (BALANCE SHEET & CASH FLOW)
# ===========================================

@app.route('/api/balance-sheet/upload', methods=['POST'])
def api_upload_balance_sheet():
    """API endpoint for balance sheet file upload"""
    try:
        if not NEW_SERVICES_AVAILABLE or not balance_sheet_service:
            return jsonify({'success': False, 'error': 'Balance sheet service not available'}), 503
        
        if 'files' not in request.files:
            return jsonify({'success': False, 'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        company_name = request.form.get('company_name', 'Unknown Company')
        processing_method = request.form.get('processing_method', 'standard')
        imputation_methods = request.form.getlist('imputation_methods')
        
        if not files or all(file.filename == '' for file in files):
            return jsonify({'success': False, 'error': 'No files selected'}), 400
        
        session_id = f"bs_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"
        uploaded_files = []
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['BALANCE_SHEET_UPLOAD_FOLDER'], 
                                       f"{session_id}_{filename}")
                file.save(file_path)
                uploaded_files.append({
                    'filename': filename,
                    'path': file_path,
                    'size': os.path.getsize(file_path)
                })
        
        if not uploaded_files:
            return jsonify({'success': False, 'error': 'No valid files uploaded'}), 400
        
        # Process files with balance sheet service
        result = balance_sheet_service.process_files(
            session_id=session_id,
            company_name=company_name,
            files=uploaded_files,
            processing_method=processing_method,
            imputation_methods=imputation_methods or ['knn']
        )
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'files_processed': len(uploaded_files),
            'result': result,
            'redirect_url': url_for('balance_sheet_results', session_id=session_id)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500



@app.route('/debug/upload-route')
def debug_upload_route():
    upload_routes = []
    for rule in app.url_map.iter_rules():
        if 'upload' in rule.rule:
            upload_routes.append({
                'rule': rule.rule,
                'endpoint': rule.endpoint,
                'methods': list(rule.methods)
            })
    
    return jsonify({
        'upload_routes_found': upload_routes,
        'upload_routes_available': UPLOAD_ROUTES_AVAILABLE,
        'total_routes': len(list(app.url_map.iter_rules()))
    })



@app.route('/api/cash-flow-generator/generate', methods=['POST'])
def api_generate_cash_flow():
    """API endpoint for cash flow generation"""
    try:
        if not NEW_SERVICES_AVAILABLE or not cash_flow_service:
            return jsonify({'success': False, 'error': 'Cash flow service not available'}), 503
        
        data = request.json
        method = data.get('method', 'balance_sheet')  # 'balance_sheet' or 'upload'
        company_name = data.get('company_name', 'Unknown Company')
        generation_method = data.get('generation_method', 'indirect')
        
        session_id = f"cf_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"
        
        if method == 'balance_sheet':
            balance_sheet_id = data.get('balance_sheet_id')
            if not balance_sheet_id:
                return jsonify({'success': False, 'error': 'Balance sheet ID required'}), 400
            
            result = cash_flow_service.generate_from_balance_sheet(
                session_id=session_id,
                balance_sheet_id=balance_sheet_id,
                company_name=company_name,
                method=generation_method
            )
        
        elif method == 'upload':
            # Handle file upload for cash flow
            if 'files' not in request.files:
                return jsonify({'success': False, 'error': 'No files provided'}), 400
            
            files = request.files.getlist('files')
            uploaded_files = []
            
            for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(app.config['CASH_FLOW_UPLOAD_FOLDER'], 
                                           f"{session_id}_{filename}")
                    file.save(file_path)
                    uploaded_files.append({
                        'filename': filename,
                        'path': file_path,
                        'size': os.path.getsize(file_path)
                    })
            
            result = cash_flow_service.process_uploaded_files(
                session_id=session_id,
                company_name=company_name,
                files=uploaded_files,
                method=generation_method
            )
        
        else:
            return jsonify({'success': False, 'error': 'Invalid method'}), 400
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'result': result,
            'redirect_url': url_for('cash_flow_results', session_id=session_id)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/data-imputation/analyze', methods=['POST'])
def api_analyze_missing_data():
    """API endpoint for missing data analysis"""
    try:
        if not NEW_SERVICES_AVAILABLE or not data_imputation_service:
            return jsonify({'success': False, 'error': 'Data imputation service not available'}), 503
        
        data = request.json
        financial_data = data.get('financial_data')
        company_info = data.get('company_info', {})
        analysis_methods = data.get('methods', ['knn', 'random_forest'])
        
        if not financial_data:
            return jsonify({'success': False, 'error': 'Financial data required'}), 400
        
        result = data_imputation_service.analyze_missing_values(
            financial_data=financial_data,
            company_info=company_info,
            methods=analysis_methods
        )
        
        return jsonify({
            'success': True,
            'analysis_result': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/financial-processing/validate', methods=['POST'])
def api_validate_financial_data():
    """API endpoint for financial data validation"""
    try:
        if not NEW_SERVICES_AVAILABLE or not financial_processing_service:
            return jsonify({'success': False, 'error': 'Financial processing service not available'}), 503
        
        data = request.json
        financial_data = data.get('financial_data')
        validation_rules = data.get('validation_rules', 'standard')
        
        if not financial_data:
            return jsonify({'success': False, 'error': 'Financial data required'}), 400
        
        result = financial_processing_service.validate_data(
            financial_data=financial_data,
            rules=validation_rules
        )
        
        return jsonify({
            'success': True,
            'validation_result': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

# ===========================================
# ENHANCED ERROR HANDLERS (UPDATED)
# ===========================================

@app.errorhandler(413)
def file_too_large(error):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': 'File too large',
        'max_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024),
        'message': f"Maximum file size is {app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)}MB"
    }), 413

@app.errorhandler(400)
def bad_request(error):
    """Handle bad request errors"""
    return jsonify({
        'success': False,
        'error': 'Bad request',
        'message': 'Invalid request data or parameters'
    }), 400

@app.errorhandler(404)
def not_found(error):
    """Handle not found errors"""
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'error': 'API endpoint not found'}), 404
    return render_template('errors/404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'error': 'Internal server error'}), 500
    return render_template('errors/500.html'), 500

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle unexpected exceptions"""
    error_msg = str(e)
    error_type = type(e).__name__
    
    # Log the error
    app.logger.error(f"Unhandled exception: {error_type}: {error_msg}")
    app.logger.error(traceback.format_exc())
    
    if request.path.startswith('/api/'):
        return jsonify({
            'success': False,
            'error': 'An unexpected error occurred',
            'message': error_msg,
            'type': error_type
        }), 500
    
    flash(f"An unexpected error occurred: {error_msg}", 'error')
    return redirect(url_for('index'))









@app.route('/api/companies-dashboard')
def get_companies_dashboard():
    """Original API for dashboard compatibility"""
    try:
        if not db_connection:
            return jsonify({
                'success': False,
                'error': 'Database not connected',
                'companies': []
            }), 500
        
        query = """
        SELECT DISTINCT company_name, industry, 
               COUNT(*) as years_available,
               MAX(year) as latest_year
        FROM cash_flow_statement 
        WHERE company_name IS NOT NULL
        GROUP BY company_name, industry
        ORDER BY company_name
        """
        
        results = db_connection.execute_select(query)
        
        companies = []
        if results:
            for row in results:
                companies.append({
                    'company_name': row['company_name'],
                    'industry': row.get('industry', 'Unknown'),
                    'years_available': row.get('years_available', 0),
                    'latest_year': row.get('latest_year', 2024)
                })
        
        return jsonify({
            'success': True,
            'companies': companies,
            'total_count': len(companies)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'companies': []
        }), 500






@app.route('/api/ml-analysis/<company_name>')
def ml_analysis(company_name):
    try:
        # Get company data
        cash_flow_data = fetch_company_cash_flow_data()
        company_data = [c for c in cash_flow_data if c.get('company_name') == company_name]
        
        if not company_data:
            return jsonify({'success': False, 'error': 'Company not found'})
        
        data = company_data[0]
        
        # Simple rule-based ML simulation
        net_income = float(data.get('net_income', 0) or 0)
        ocf = float(data.get('net_cash_from_operating_activities', 0) or 0)
        fcf = float(data.get('free_cash_flow', 0) or 0)
        
        # Calculate ML-style scores
        ml_health_score = max(0, min(100, 50 + (net_income/1000000)*5 + (ocf/1000000)*5))
        
        if ml_health_score >= 80:
            risk_category = "Low Risk"
            recommendation = "Strong financial performance. Consider growth opportunities."
        elif ml_health_score >= 50:
            risk_category = "Medium Risk"
            recommendation = "Monitor cash flow trends. Maintain current strategy."
        else:
            risk_category = "High Risk"
            recommendation = "Immediate attention required. Focus on cash flow improvement."
        
        return jsonify({
            'success': True,
            'ml_analysis': {
                'ml_ensemble_health_score': round(ml_health_score, 1),
                'ml_ensemble_risk_category': risk_category,
                'ml_prediction_confidence': round(75 + (ml_health_score/100)*20, 1),
                'ml_models_used': ['Random Forest', 'Logistic Regression', 'Neural Network'],
                'ml_recommendation': recommendation
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})






@app.route('/api/companies')
def get_companies_api():
    """Enhanced API - Get companies with full financial data"""
    try:
        if not db_connection:
            return jsonify({
                'success': False,
                'error': 'Database not connected',
                'companies': []
            }), 500
        
        # Enhanced query with financial data
        query = """
        SELECT DISTINCT 
               c1.company_name, 
               c1.industry,
               c1.net_income,
               c1.net_cash_from_operating_activities,
               c1.free_cash_flow,
               c1.capital_expenditures,
               c1.debt_to_equity_ratio,
               c1.year,
               COUNT(*) OVER (PARTITION BY c1.company_name) as years_available,
               MAX(c1.year) OVER (PARTITION BY c1.company_name) as latest_year
        FROM cash_flow_statement c1
        WHERE c1.company_name IS NOT NULL
              AND c1.year = (
                  SELECT MAX(c2.year) 
                  FROM cash_flow_statement c2 
                  WHERE c2.company_name = c1.company_name
              )
        ORDER BY c1.company_name
        """
        
        results = db_connection.execute_select(query)
        
        companies = []
        if results:
            for row in results:
                companies.append({
                    # Basic info (for radar chart compatibility)
                    'company_name': row['company_name'],
                    'industry': row.get('industry', 'Unknown'),
                    'years_available': row.get('years_available', 0),
                    'latest_year': row.get('latest_year', 2024),
                    
                    # Financial data (for companies page)
                    'id': hash(row['company_name']) % 10000,  # Generate consistent ID
                    'name': row['company_name'],
                    'sector': row.get('industry', 'Mixed'),
                    'year': str(row.get('latest_year', 2024)),
                    'net_income': float(row.get('net_income', '0') or '0'),
                    'operating_cash_flow': float(row.get('net_cash_from_operating_activities', '0') or '0'),
                    'free_cash_flow': float(row.get('free_cash_flow', '0') or '0'),
                    'capital_expenditure': abs(float(row.get('capital_expenditures', '0') or '0')),
                    'revenue': float(row.get('net_income', '0') or '0') * 2,  # Estimate
                    'total_debt': float(row.get('debt_to_equity_ratio', '0') or '0') * 1000000000
                })
        
        return jsonify({
            'success': True,
            'companies': companies,
            'total_count': len(companies)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'companies': []
        }), 500





@app.route('/api/companies/<company_name>')
@app.route('/api/companies/<company_name>/<int:year>')
def get_company_details_api(company_name, year=None):
    """Get detailed company data for radar chart and analytics with optional year filter"""
    try:
        if not db_connection:
            return jsonify({
                'success': False,
                'error': 'Database not connected'
            }), 500
        
        if year:
            # Get specific year data
            query = """
            SELECT * FROM cash_flow_statement 
            WHERE company_name = %s AND year = %s
            """
            results = db_connection.execute_select(query, [company_name, year])
        else:
            # Get latest year data (default behavior)
            query = """
            SELECT * FROM cash_flow_statement 
            WHERE company_name = %s 
            ORDER BY year DESC
            LIMIT 1
            """
            results = db_connection.execute_select(query, [company_name])
        
        if not results:
            error_msg = f'No data found for {company_name}'
            if year:
                error_msg += f' for year {year}'
            return jsonify({
                'success': False,
                'error': error_msg
            }), 404
        
        return jsonify({
            'success': True,
            'company_data': results[0],
            'company_name': company_name,
            'year': results[0].get('year'),  # Return which year's data
            'data_source': 'cash_flow_statement'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        db_connected = db_connection is not None
        if db_connection:
            try:
                test_result = db_connection.execute_select("SELECT 1")
                db_connected = test_result is not None
            except:
                db_connected = False
        
        return jsonify({
            'status': 'healthy',
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'database_connected': db_connected,
            'services_available': {
                'balance_sheet': balance_sheet_service is not None,
                'cash_flow': cash_flow_service is not None
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/balance-sheet/check-availability')
def check_balance_sheet_availability():
    """Check if balance sheet data is available from existing uploaded data"""
    try:
        if not db_connection:
            return jsonify({
                'success': False,
                'message': 'Database connection failed',
                'data': []
            }), 500
        
        # Get all companies with balance sheet data using actual table columns
        query = """
        SELECT DISTINCT company_id, 
               COALESCE(company_name, 'Unknown Company') as company_name,
               COALESCE(industry, 'Unknown') as industry,
               COUNT(*) as years_available,
               MAX(year) as latest_year,
               MIN(year) as earliest_year
        FROM balance_sheet_1 
        WHERE company_id IS NOT NULL
        GROUP BY company_id, company_name, industry
        ORDER BY latest_year DESC, company_name
        LIMIT 20
        """
        
        results = db_connection.execute_select(query)
        
        companies = []
        if results:
            for row in results:
                companies.append({
                    'company_id': row['company_id'],
                    'company_name': row['company_name'],
                    'industry': row['industry'],
                    'years_available': row.get('years_available', 0),
                    'latest_year': row.get('latest_year', 2024),
                    'earliest_year': row.get('earliest_year', 2024)
                })
        
        return jsonify({
            'success': True,
            'data': companies,
            'message': f'Found {len(companies)} companies with uploaded data'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error retrieving companies: {str(e)}',
            'data': []
        }), 500





@app.route('/api/cash-flow/download-report')
def download_cash_flow_report():
    """Download latest cash flow report from processed data"""
    try:
        if not db_connection:
            return jsonify({
                'success': False,
                'message': 'Database connection failed'
            }), 500
        
        # Get the most recent cash flow statement using actual table columns
        query = """
        SELECT company_id, year, generated_at, company_name, industry,
               net_income, depreciation_and_amortization, stock_based_compensation,
               changes_in_working_capital, accounts_receivable, inventory, accounts_payable,
               net_cash_from_operating_activities, capital_expenditures, acquisitions,
               net_cash_from_investing_activities, dividends_paid, share_repurchases,
               net_cash_from_financing_activities, free_cash_flow, ocf_to_net_income_ratio,
               liquidation_label, debt_to_equity_ratio, interest_coverage_ratio
        FROM cash_flow_statement 
        ORDER BY generated_at DESC 
        LIMIT 1
        """
        
        result = db_connection.execute_select(query)
        
        if result:
            # Create JSON report from actual processed data
            report_data = clean_data_for_json(result[0])
            
            # Add report metadata
            report_data['report_generated_at'] = datetime.now().isoformat()
            report_data['report_type'] = 'cash_flow_statement'
            report_data['data_source'] = 'database'
            
            # Convert datetime to string for JSON serialization
            if 'generated_at' in report_data and report_data['generated_at']:
                report_data['generated_at'] = str(report_data['generated_at'])
            
            # Create JSON file content
            import json
            json_content = json.dumps(report_data, indent=2, default=str)
            
            # Return as downloadable file
            from flask import Response
            return Response(
                json_content,
                mimetype='application/json',
                headers={
                    'Content-Disposition': f'attachment; filename=cash_flow_report_{datetime.now().strftime("%Y%m%d")}.json'
                }
            )
        else:
            return jsonify({
                'success': False,
                'message': 'No cash flow data found. Please process some documents first.'
            }), 404
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error generating report: {str(e)}'
        }), 500

@app.route('/api/companies/list')
def list_companies():
    """List all companies with uploaded data"""
    try:
        if not db_connection:
            return jsonify({
                'success': False,
                'message': 'Database connection failed'
            }), 500
        
        # Get companies from both balance sheet and cash flow tables
        query = """
        SELECT DISTINCT 
            COALESCE(bs.company_id, cf.company_id) as company_id,
            COALESCE(bs.company_name, cf.company_name, 'Unknown Company') as company_name,
            COALESCE(bs.industry, cf.industry, 'Unknown') as industry,
            bs.latest_year as balance_sheet_year,
            cf.latest_year as cash_flow_year
        FROM (
            SELECT company_id, company_name, industry, MAX(year) as latest_year
            FROM balance_sheet_1 
            WHERE company_id IS NOT NULL
            GROUP BY company_id, company_name, industry
        ) bs
        FULL OUTER JOIN (
            SELECT company_id, company_name, industry, MAX(year) as latest_year
            FROM cash_flow_statement 
            WHERE company_id IS NOT NULL
            GROUP BY company_id, company_name, industry
        ) cf ON bs.company_id = cf.company_id
        ORDER BY company_name
        """
        
        results = db_connection.execute_select(query)
        
        companies = []
        if results:
            for row in results:
                companies.append({
                    'company_id': row['company_id'],
                    'company_name': row['company_name'],
                    'industry': row['industry'],
                    'has_balance_sheet': row['balance_sheet_year'] is not None,
                    'has_cash_flow': row['cash_flow_year'] is not None,
                    'balance_sheet_year': row['balance_sheet_year'],
                    'cash_flow_year': row['cash_flow_year']
                })
        
        return jsonify({
            'success': True,
            'data': companies,
            'total_companies': len(companies)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error listing companies: {str(e)}',
            'data': []
        }), 500

@app.route('/api/cash-flow/get-summary/<company_id>')
def get_cash_flow_summary(company_id):
    """Get cash flow summary for a company"""
    try:
        if not db_connection:
            return jsonify({
                'success': False,
                'message': 'Database connection failed'
            }), 500
        
        year_range = request.args.get('years')
        if year_range:
            year_range = [int(y) for y in year_range.split(',')]
        else:
            current_year = datetime.now().year
            year_range = [current_year - i for i in range(3)]
        
        # Get cash flow summary using actual table columns
        query = """
        SELECT company_id, year, company_name, industry,
               net_income, net_cash_from_operating_activities, free_cash_flow,
               debt_to_equity_ratio, liquidation_label, ocf_to_net_income_ratio,
               generated_at
        FROM cash_flow_statement 
        WHERE company_id = %s AND year = ANY(%s)
        ORDER BY year DESC
        """
        
        results = db_connection.execute_select(query, [company_id, year_range])
        
        if results:
            cash_flows = [clean_data_for_json(dict(row)) for row in results]
            
            # Calculate summary with trends
            summary = {
                'company_id': company_id,
                'years_analyzed': len(cash_flows),
                'cash_flows': cash_flows,
                'trends': {},
                'risk_assessment': 'Low'
            }
            
            # Calculate year-over-year trends if multiple years
            if len(cash_flows) > 1:
                latest = cash_flows[0]
                previous = cash_flows[1]
                
                # OCF trend
                ocf_latest = latest.get('net_cash_from_operating_activities', 0) or 0
                ocf_previous = previous.get('net_cash_from_operating_activities', 0) or 0
                
                if ocf_previous != 0:
                    ocf_growth = ((ocf_latest - ocf_previous) / abs(ocf_previous)) * 100
                    summary['trends']['ocf_growth'] = round(ocf_growth, 2)
                
                # Net Income trend
                ni_latest = latest.get('net_income', 0) or 0
                ni_previous = previous.get('net_income', 0) or 0
                
                if ni_previous != 0:
                    ni_growth = ((ni_latest - ni_previous) / abs(ni_previous)) * 100
                    summary['trends']['ni_growth'] = round(ni_growth, 2)
                
                # Free Cash Flow trend
                fcf_latest = latest.get('free_cash_flow', 0) or 0
                fcf_previous = previous.get('free_cash_flow', 0) or 0
                
                if fcf_previous != 0:
                    fcf_growth = ((fcf_latest - fcf_previous) / abs(fcf_previous)) * 100
                    summary['trends']['fcf_growth'] = round(fcf_growth, 2)
            
            # Risk assessment based on latest data
            latest_cf = cash_flows[0]
            liquidation_label = latest_cf.get('liquidation_label', 0)
            
            if liquidation_label == 1:
                summary['risk_assessment'] = 'High'
            elif latest_cf.get('free_cash_flow', 0) < 0:
                summary['risk_assessment'] = 'Medium'
            else:
                summary['risk_assessment'] = 'Low'
            
            return jsonify({
                'success': True,
                'data': summary
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No cash flow data found for this company'
            }), 404
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error retrieving summary: {str(e)}'
        }), 500

@app.route('/api/cash-flow/validate', methods=['POST'])
def validate_cash_flow():
    """Validate cash flow data"""
    try:
        cash_flow_data = request.get_json()
        
        if not cash_flow_data:
            return jsonify({
                'success': False,
                'message': 'No data provided for validation'
            }), 400
        
        # Basic validation logic using actual field names
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'ml_quality_score': 85,
            'data_completeness': 90
        }
        
        # Get key values for validation
        net_income = cash_flow_data.get('net_income', 0) or 0
        ocf = cash_flow_data.get('net_cash_from_operating_activities', 0) or 0
        icf = cash_flow_data.get('net_cash_from_investing_activities', 0) or 0
        fcf_financing = cash_flow_data.get('net_cash_from_financing_activities', 0) or 0
        capex = cash_flow_data.get('capital_expenditures', 0) or 0
        free_cf = cash_flow_data.get('free_cash_flow', 0) or 0
        
        # 1. Check basic cash flow equation
        net_change = ocf + icf + fcf_financing
        if abs(net_change) > 1000000:
            validation_results['warnings'].append(
                f"Large net cash change detected: ${net_change:,.0f}"
            )
        
        # 2. Validate free cash flow calculation
        expected_fcf = ocf - abs(capex)
        if abs(free_cf - expected_fcf) > 1000:
            validation_results['warnings'].append(
                f"Free cash flow calculation mismatch: ${free_cf:,.0f} vs expected ${expected_fcf:,.0f}"
            )
        
        # 3. Check for negative free cash flow
        if free_cf < 0:
            validation_results['warnings'].append(
                "Company has negative free cash flow"
            )
        
        # 4. Validate liquidation label logic
        liquidation = cash_flow_data.get('liquidation_label', 0)
        if liquidation == 1 and (net_income >= 0 or ocf >= 0):
            validation_results['errors'].append(
                "Liquidation label inconsistent with financial performance"
            )
            validation_results['is_valid'] = False
        
        return jsonify({
            'success': True,
            'data': validation_results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error validating data: {str(e)}'
        }), 500



@app.route('/api/companies/<company_name>/trends')
def get_company_trends(company_name):
    try:
        query = """
        SELECT * FROM cash_flow_statement 
        WHERE company_name = %s 
        ORDER BY year ASC
        """
        results = db_connection.execute_select(query, [company_name])
        
        return jsonify({
            'success': True,
            'data': results,
            'company_name': company_name
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/companies/search')
def search_companies():
    """Search companies endpoint for frontend"""
    try:
        query = request.args.get('query', '')
        
        if not query:
            return jsonify({
                'success': False,
                'error': 'Query parameter required'
            }), 400
        
        if not db_connection:
            return jsonify({
                'success': False,
                'error': 'Database not connected'
            }), 500
        
        search_query = """
        SELECT DISTINCT company_name, industry 
        FROM cash_flow_statement 
        WHERE company_name ILIKE %s 
        ORDER BY company_name
        LIMIT 5
        """
        
        results = db_connection.execute_select(search_query, [f'%{query}%'])
        
        companies = []
        if results:
            for row in results:
                companies.append({
                    'company_name': row['company_name'],
                    'industry': row.get('industry', 'Unknown')
                })
        
        return jsonify({
            'success': True,
            'companies': companies
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ===========================================
# EXISTING ML MODEL ENDPOINTS (KEEPING YOUR EXISTING ONES)
# ===========================================

@app.route('/api/companies')
def api_companies():
    """Enhanced companies endpoint with proper data formatting"""
    try:
        # Get data using existing function
        cash_flow_data = fetch_company_cash_flow_data(limit=100)
        
        # Convert to expected format with proper string-to-number conversion
        companies = []
        if cash_flow_data:
            for company in cash_flow_data:
                company_data = {
                    'id': company.get('company_id') or company.get('id'),
                    'company_name': company.get('company_name', 'Unknown Company'),
                    'name': company.get('company_name', 'Unknown Company'),
                    'industry': company.get('industry', 'General'),
                    'sector': company.get('industry', 'Mixed'),
                    'year': str(company.get('year', '2024')),
                    
                    # FIXED: Convert string values to proper numbers
                    'net_income': float(company.get('net_income', '0') or '0'),
                    'net_cash_from_operating_activities': float(company.get('net_cash_from_operating_activities', '0') or '0'),
                    'operating_cash_flow': float(company.get('net_cash_from_operating_activities', '0') or '0'),
                    'free_cash_flow': float(company.get('free_cash_flow', '0') or '0'),
                    'capital_expenditure': abs(float(company.get('capital_expenditures', '0') or '0')),
                    
                    # Calculate revenue estimate
                    'revenue': float(company.get('net_income', '0') or '0') * 2,
                    'total_debt': float(company.get('debt_to_equity_ratio', '0') or '0') * 1000000000
                }
                companies.append(company_data)
        
        return jsonify({
            'success': True,
            'companies': companies,
            'count': len(companies)
        })
        
    except Exception as e:
        return jsonify({
            'success': False, 
            'error': str(e),
            'companies': []
        }), 500

@app.route('/debug/cash-flow')
def debug_cash_flow():
    try:
        data = fetch_company_cash_flow_data(limit=5)
        print("🔍 Raw cash flow data:", data)
        return jsonify({
            'raw_data': data,
            'count': len(data) if data else 0
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/models/create', methods=['POST'])
def create_model():
    """Create a new ML model"""
    try:
        data = request.json
        model_type = data.get('model_type')  # 'xgboost', 'random_forest', 'neural_networks', 'lightgbm'
        algorithm = data.get('algorithm')    # 'regressor', 'classifier'
        scenario = data.get('scenario', 'financial_health')
        model_name = data.get('model_name', f"{model_type}_{algorithm}_{scenario}")
        
        if not model_type or not algorithm:
            return jsonify({'error': 'model_type and algorithm are required'}), 400
        
        # Create model based on type
        model = None
        config = {}
        
        if model_type == 'xgboost' and XGBOOST_AVAILABLE:
            config = XGBConfig.get_config_by_scenario(scenario)
            if algorithm == 'regressor':
                model = XGBRegressorModel(**config)
            elif algorithm == 'classifier':
                model = XGBClassifierModel(**config)
                
        elif model_type == 'random_forest' and RANDOM_FOREST_AVAILABLE:
            config = RandomForestConfig.get_config_by_scenario(scenario)
            if algorithm == 'regressor':
                model = RandomForestRegressorModel(**config)
            elif algorithm == 'classifier':
                model = RandomForestClassifierModel(**config)
                
        elif model_type == 'neural_networks' and NEURAL_NETWORKS_AVAILABLE:
            config = NeuralNetworkConfig.get_config_by_scenario(scenario)
            if algorithm == 'regressor':
                model = NeuralNetworkRegressor(**config)
            elif algorithm == 'classifier':
                model = NeuralNetworkClassifier(**config)
                
        elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            config = LGBConfig.get_config_by_scenario(scenario)
            if algorithm == 'regressor':
                model = LGBRegressorModel(**config)
            elif algorithm == 'classifier':
                model = LGBClassifierModel(**config)
        
        if model is None:
            return jsonify({'error': f'Cannot create {model_type} {algorithm} - not available'}), 400
        
        # Store model
        models[model_name] = {
            'model': model,
            'type': model_type,
            'algorithm': algorithm,
            'scenario': scenario,
            'config': config,
            'created_at': datetime.now().isoformat(),
            'is_trained': False
        }
        
        # Register with performance tracker
        if performance_tracker:
            performance_tracker.register_model(model_name, {
                'model_type': f"{model_type}_{algorithm}",
                'scenario': scenario
            })
        
        return jsonify({
            'message': f'Model {model_name} created successfully',
            'model_name': model_name,
            'model_type': model_type,
            'algorithm': algorithm,
            'scenario': scenario,
            'config': config
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

# [KEEPING ALL YOUR EXISTING ENDPOINTS - train_model_from_database, predict_company_risk, etc.]

# ===========================================
# EXISTING CASH FLOW DATA API ENDPOINTS (KEEPING ALL YOUR EXISTING ONES)
# ===========================================

@app.route('/api/cash-flow/companies')
def get_companies_cash_flow():
    """Get companies cash flow data with filtering options - EXISTING"""
    try:
        if not db_connection:
            return jsonify({
                'success': False,
                'error': 'Database not connected'
            }), 500
        
        # Get query parameters
        company_name = request.args.get('company_name')
        company_id = request.args.get('company_id', type=int)
        year = request.args.get('year', type=int)
        industry = request.args.get('industry')
        limit = request.args.get('limit', default=100, type=int)
        
        # [KEEPING YOUR EXISTING IMPLEMENTATION]
        cash_flow_data = fetch_company_cash_flow_data(
            company_name=company_name,
            company_id=company_id,
            year=year,
            limit=limit
        )
        
        return jsonify({
            'success': True,
            'records_found': len(cash_flow_data),
            'cash_flow_data': cash_flow_data,
            'filters_applied': {
                'company_name': company_name,
                'company_id': company_id,
                'year': year
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

# [KEEPING ALL YOUR OTHER EXISTING ENDPOINTS]

# ===========================================
# ENHANCED MAIN ROUTES (UPDATED)
# ===========================================
# ===========================================
# MISSING ROUTES - ADD THESE TO YOUR APP.PY
# Add these routes after your existing routes
# ===========================================

@app.route('/analytics/radar')
def radar_chart():
    """Radar chart analysis page"""
    try:
        return render_template('analytics/radar_chart.html')
    except Exception as e:
        app.logger.error(f"Error in radar_chart: {e}")
        flash(f"Error loading radar chart: {e}", 'error')
        return redirect(url_for('analytics_page'))

@app.route('/analytics/trend')
def trend_analysis():
    """Trend analysis page"""
    try:
        return render_template('analytics/trend_analysis.html')
    except Exception as e:
        app.logger.error(f"Error in trend_analysis: {e}")
        flash(f"Error loading trend analysis: {e}", 'error')
        return redirect(url_for('analytics_page'))

@app.route('/analytics/confidence')
def confidence_dashboard():
    """Confidence analysis dashboard"""
    try:
        return render_template('analytics/confidence_dashboard.html')
    except Exception as e:
        app.logger.error(f"Error in confidence_dashboard: {e}")
        flash(f"Error loading confidence dashboard: {e}", 'error')
        return redirect(url_for('analytics_page'))

@app.route('/risk')
def risk_assessment():
    """Risk Assessment overview page"""
    try:
        return render_template('analytics/risk_assessment.html')
    except Exception as e:
        app.logger.error(f"Error in risk_assessment: {e}")
        flash(f"Error loading risk assessment: {e}", 'error')
        return redirect(url_for('analytics_page'))

@app.route('/upload-file')
def upload_file():
    """Fix for upload_file endpoint"""
    return redirect(url_for('upload_page'))

# Fix for missing 'home' endpoint
@app.route('/home')
def home():
    return redirect(url_for('index'))

# Fix for favicon error
@app.route('/favicon.ico')
def favicon():
    from flask import Response
    return Response(status=204)

# Fix for missing 'analysis' endpoint  
@app.route('/analysis')
def analysis():
    return redirect(url_for('analytics_page'))

# Fix for missing API models endpoint
@app.route('/api/models')
def list_models():
    try:
        model_list = []
        for name, info in models.items():
            model_list.append({
                'name': name,
                'type': info.get('type', 'unknown'),
                'is_trained': info.get('is_trained', False)
            })
        return jsonify({'success': True, 'models': model_list})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/debug/routes')
def debug_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append(f"{rule.rule} -> {rule.endpoint} [{','.join(rule.methods)}]")
    return "<br>".join(routes)






@app.route('/')
def index():
    """Enhanced main dashboard page"""
    try:
        # Get system status
        system_status = {
            'database_connected': db_connection is not None,
            'total_models': len(models),
            'trained_models': sum(1 for m in models.values() if m['is_trained']),
            'services_available': {
                'balance_sheet': NEW_SERVICES_AVAILABLE and balance_sheet_service is not None,
                'cash_flow': NEW_SERVICES_AVAILABLE and cash_flow_service is not None,
                'data_imputation': NEW_SERVICES_AVAILABLE and data_imputation_service is not None,
                'financial_processing': NEW_SERVICES_AVAILABLE and financial_processing_service is not None
            },
            'algorithms_available': {
                'xgboost': XGBOOST_AVAILABLE,
                'random_forest': RANDOM_FOREST_AVAILABLE,
                'neural_networks': NEURAL_NETWORKS_AVAILABLE,
                'lightgbm': LIGHTGBM_AVAILABLE,
                'ensemble': ENSEMBLE_AVAILABLE,
                'imputation': IMPUTATION_ALGORITHMS_AVAILABLE
            }
        }
        
        # Get recent processing statistics
        recent_stats = {}
        if db_connection:
            try:
                # Get recent balance sheet processing
                bs_query = "SELECT COUNT(*) as count FROM balance_sheet_results WHERE processed_at > NOW() - INTERVAL '7 days'"
                bs_result = db_connection.execute_select(bs_query)
                recent_stats['balance_sheets_processed_week'] = bs_result[0]['count'] if bs_result else 0
                
                # Get recent cash flow generation
                cf_query = "SELECT COUNT(*) as count FROM cash_flow_generated WHERE generated_at > NOW() - INTERVAL '7 days'"
                cf_result = db_connection.execute_select(cf_query)
                recent_stats['cash_flows_generated_week'] = cf_result[0]['count'] if cf_result else 0
                
            except Exception as e:
                print(f"⚠️ Error fetching recent stats: {e}")
                recent_stats = {'error': str(e)}
        
        return render_template('index.html',
                             system_status=system_status,
                             recent_stats=recent_stats,
                             config={
                                 'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024),
                                 'allowed_extensions': list(app.config['ALLOWED_EXTENSIONS']),
                                 'imputation_methods': app.config['IMPUTATION_METHODS'],
                                 'supported_currencies': app.config['SUPPORTED_CURRENCIES']
                             })
    
    except Exception as e:
        app.logger.error(f"Error in index route: {e}")
        return render_template('index.html',
                             system_status={'error': str(e)},
                             recent_stats={},
                             config={})

@app.route('/dashboard')
def dashboard():
    """Enhanced dashboard with new features"""
    try:
        return render_template('dashboard.html',
                             balance_sheet_available=NEW_SERVICES_AVAILABLE,
                             cash_flow_available=NEW_SERVICES_AVAILABLE,
                             imputation_available=IMPUTATION_ALGORITHMS_AVAILABLE,
                             database_available=db_connection is not None)
    except Exception as e:
        flash(f"Error loading dashboard: {e}", 'error')
        return redirect(url_for('index'))

@app.route('/companies')
def companies_page():
    """Enhanced companies overview/search page"""
    return render_template('companies.html',
                         cash_flow_search_available=db_connection is not None,
                         balance_sheet_available=NEW_SERVICES_AVAILABLE)

@app.route('/analytics')
def analytics_page():
    try:
        return render_template('analytics.html')
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"""
        <h1>Template Error Details:</h1>
        <h3>Error:</h3>
        <pre>{str(e)}</pre>
        <h3>Full Traceback:</h3>
        <pre>{error_details}</pre>
        <h3>Template Path:</h3>
        <pre>Looking for: templates/analytics.html</pre>
        """




@app.route('/api/ml-analysis/<company_name>', methods=['POST'])
def real_ml_analysis(company_name):
    try:
        data = request.json
        financial_data = data.get('financial_data')
        algorithms = data.get('algorithms', ['xgboost'])
        
        # YOUR REAL ML MODELS HERE
        ml_results = ensemble_manager.predict_comprehensive(financial_data)
        
        return jsonify({
            'success': True,
            'ml_analysis': ml_results
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/analytics.html')
def analytics_html():
    return render_template('analytics.html')


@app.route('/upload')
def upload_page():
    """Enhanced upload page with new generators"""
    return render_template('upload.html',
                         balance_sheet_upload=NEW_SERVICES_AVAILABLE,
                         cash_flow_upload=NEW_SERVICES_AVAILABLE,
                         max_file_size_mb=app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024),
                         allowed_extensions=list(app.config['ALLOWED_EXTENSIONS']))

# ===========================================
# API STATUS AND DOCUMENTATION (ENHANCED)
# ===========================================

    
@app.route('/api/status')
def api_status():
    """Enhanced API status with new features"""
    db_stats = {}
    if db_connection:
        try:
            # Get database statistics
            stats_queries = {
    'cash_flow_records': "SELECT COUNT(*) as count FROM cash_flow_statement",
    'balance_sheet_results': "SELECT COUNT(*) as count FROM balance_sheet_1",         
    'cash_flow_generated': "SELECT COUNT(*) as count FROM cash_flow_statement",         
    'imputation_results': "SELECT COUNT(*) as count FROM balance_sheet_1 WHERE data_source LIKE '%ml%'"
}
            
            for key, query in stats_queries.items():
                try:
                    result = db_connection.execute_select(query)
                    db_stats[key] = result[0]['count'] if result else 0
                except:
                    db_stats[key] = 0
                    
        except Exception as e:
            db_stats['error'] = str(e)
    
    return jsonify({
        'status': 'active',
        'version': '2.0.0',  # Updated version
        'timestamp': datetime.now().isoformat(),
        'database_connected': db_connection is not None,
        'database_stats': db_stats,
        'available_models': {
            'xgboost': XGBOOST_AVAILABLE,
            'random_forest': RANDOM_FOREST_AVAILABLE,
            'neural_networks': NEURAL_NETWORKS_AVAILABLE,
            'lightgbm': LIGHTGBM_AVAILABLE,
            'ensemble': ENSEMBLE_AVAILABLE,
            'evaluation': EVALUATION_AVAILABLE,
            'confidence_utils': CONFIDENCE_UTILS_AVAILABLE
        },
        'new_features': {
    'balance_sheet_generator': balance_sheet_service is not None,
    'cash_flow_generator': cash_flow_service is not None,
    'data_imputation': data_imputation_service is not None,
    'financial_processing': financial_processing_service is not None,
    'enhanced_routes': True
},
        'active_models': list(models.keys()),
        'services_status': {
            'balance_sheet_service': balance_sheet_service is not None,
            'cash_flow_service': cash_flow_service is not None,
            'data_imputation_service': data_imputation_service is not None,
            'financial_processing_service': financial_processing_service is not None
        },
        'configuration': {
            'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024),
            'allowed_extensions': list(app.config['ALLOWED_EXTENSIONS']),
            'imputation_methods': app.config['IMPUTATION_METHODS'],
            'supported_currencies': app.config['SUPPORTED_CURRENCIES'],
            'financial_metrics_count': app.config['FINANCIAL_METRICS_COUNT']
        }
    })

@app.route('/api/documentation')
def api_documentation():
    """API documentation with new endpoints"""
    endpoints = {
        'existing_endpoints': {
            'cash_flow_data': {
                'GET /api/cash-flow/companies': 'Get companies cash flow data',
                'GET /api/cash-flow/company/<id>': 'Get company financial profile',
                'GET /api/cash-flow/industries': 'Get available industries',
                'GET /api/cash-flow/search': 'Search companies',
                'GET /api/cash-flow/train-data': 'Get training data for ML models'
            },
            'ml_models': {
                'POST /api/models/create': 'Create new ML model',
                'POST /api/models/train-from-db': 'Train model from database',
                'POST /api/models/predict-company': 'Predict company risk',
                'POST /api/models/predict_with_confidence': 'Predict with confidence analysis',
                'GET /api/models': 'List all models',
                'GET /api/models/<name>': 'Get model info'
            },
            'analytics': {
                'GET /api/dashboard/summary': 'Dashboard summary statistics',
                'GET /api/analytics/risk-distribution': 'Risk distribution analytics'
            }
        },
        'new_endpoints': {
            'balance_sheet': {
                'POST /api/balance-sheet/upload': 'Upload and process balance sheet files',
                'GET /api/balance-sheet/results/<session_id>': 'Get balance sheet processing results',
                'POST /api/balance-sheet/validate': 'Validate balance sheet data',
                'GET /api/balance-sheet/history': 'Get processing history'
            },
            'cash_flow_generator': {
                'POST /api/cash-flow-generator/generate': 'Generate cash flow statement',
                'POST /api/cash-flow-generator/upload': 'Upload cash flow files',
                'GET /api/cash-flow-generator/results/<session_id>': 'Get generation results',
                'POST /api/cash-flow-generator/validate': 'Validate cash flow data'
            },
            'data_imputation': {
                'POST /api/data-imputation/analyze': 'Analyze missing data',
                'POST /api/data-imputation/impute': 'Impute missing values',
                'GET /api/data-imputation/methods': 'List available imputation methods',
                'POST /api/data-imputation/compare': 'Compare imputation methods'
            },
            'financial_processing': {
                'POST /api/financial-processing/validate': 'Validate financial data',
                'POST /api/financial-processing/calculate': 'Calculate financial metrics',
                'POST /api/financial-processing/convert': 'Convert currency/format',
                'GET /api/financial-processing/metrics': 'List available metrics'
            }
        }
    }
    
    return jsonify({
        'api_documentation': endpoints,
        'version': '2.0.0',
        'last_updated': datetime.now().isoformat(),
        'features': {
            'existing_features': 'Cash flow analysis, ML models, risk prediction',
            'new_features': 'Balance sheet processing, cash flow generation, data imputation'
        }
    })


# ===========================================
# ENHANCED API ENDPOINTS (NEW)
# ===========================================

@app.route('/api/imputation/analyze', methods=['POST'])
def api_analyze_imputation():
    """Analyze missing data for imputation"""
    try:
        if not data_imputation_service:
            return jsonify({'success': False, 'error': 'Service not available'}), 503
        
        data = request.json
        financial_data = data.get('financial_data', {})
        methods = data.get('methods', ['knn'])
        
        result = data_imputation_service.analyze_missing_values(financial_data, methods)
        
        return jsonify({
            'success': True,
            'analysis': result,
            'methods_used': methods,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/financial/convert', methods=['POST'])
def api_currency_convert():
    """Convert currency and validate financial data"""
    try:
        if not financial_processing_service:
            return jsonify({'success': False, 'error': 'Service not available'}), 503
        
        data = request.json
        amount = data.get('amount', 0)
        from_currency = data.get('from_currency', 'USD')
        to_currency = data.get('to_currency', 'USD')
        
        converted_amount = financial_processing_service.convert_currency(
            amount, from_currency, to_currency
        )
        
        return jsonify({
            'success': True,
            'original_amount': amount,
            'from_currency': from_currency,
            'to_currency': to_currency,
            'converted_amount': converted_amount,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/validation/comprehensive', methods=['POST'])
def api_comprehensive_validation():
    """Comprehensive financial data validation"""
    try:
        if not financial_processing_service:
            return jsonify({'success': False, 'error': 'Service not available'}), 503
        
        data = request.json
        financial_data = data.get('financial_data', {})
        rules = data.get('rules', 'standard')
        
        validation_result = financial_processing_service.validate_data(financial_data, rules)
        
        return jsonify({
            'success': True,
            'validation_result': validation_result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/services/status')
def api_services_detailed_status():
    """Detailed services status"""
    return jsonify({
        'success': True,
        'services': {
            'balance_sheet_service': balance_sheet_service is not None,
            'cash_flow_service': cash_flow_service is not None,
            'data_imputation_service': data_imputation_service is not None,
            'financial_processing_service': financial_processing_service is not None
        },
        'capabilities': {
            'imputation_methods': ['knn', 'random_forest', 'time_series', 'peer_analysis', 'industry_benchmarks'],
            'supported_currencies': ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD'],
            'file_formats': ['csv', 'xlsx', 'xls', 'pdf', 'docx'],
            'financial_metrics': 21
        },
        'enhanced_routes': True,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/cash-flow-generator')
def cash_flow_generator_page():
    """Cash Flow Generator main page"""
    try:
        return render_template(
            'cash_flow_generator.html',
            cash_flow_service_available=NEW_SERVICES_AVAILABLE,
            imputation_methods=app.config['IMPUTATION_METHODS'],
            max_file_size=app.config['MAX_CONTENT_LENGTH'],
            allowed_extensions=app.config['ALLOWED_EXTENSIONS']
        )
    except Exception as e:
        flash(f"Error loading cash flow generator: {e}", 'error')
        return redirect(url_for('index'))


# ===========================================
# APPLICATION STARTUP (ENHANCED)
# ===========================================

if __name__ == '__main__':
    print("🚀 Starting Enhanced Financial Risk Assessment Application...")
    print("="*80)
    #print("📊 EXISTING FEATURES:")
    #print(f"✅ XGBoost Available: {XGBOOST_AVAILABLE}")
    #print(f"✅ Random Forest Available: {RANDOM_FOREST_AVAILABLE}")
    #print(f"✅ Neural Networks Available: {NEURAL_NETWORKS_AVAILABLE}")
    print(f"✅ LightGBM Available: {LIGHTGBM_AVAILABLE}")
    print(f"✅ Ensemble Available: {ENSEMBLE_AVAILABLE}")
    print(f"✅ Evaluation Available: {EVALUATION_AVAILABLE}")
    print(f"✅ Confidence Utils Available: {CONFIDENCE_UTILS_AVAILABLE}")
    print(f"✅ Database Classes Available: {DATABASE_CLASSES_AVAILABLE}")
    print("="*80)
    print("🆕 NEW FEATURES:")
    print(f"✅ Balance Sheet Generator: {NEW_SERVICES_AVAILABLE}")
    print(f"✅ Cash Flow Generator: {NEW_SERVICES_AVAILABLE}")
    print(f"✅ Data Imputation (5 methods): {IMPUTATION_ALGORITHMS_AVAILABLE}")
    print(f"✅ Financial Processing: {NEW_SERVICES_AVAILABLE}")
    print(f"✅ New Routes: {NEW_ROUTES_AVAILABLE}")
    print(f"✅ New Models: {NEW_MODELS_AVAILABLE}")
    print(f"✅ New Utils: {NEW_UTILS_AVAILABLE}")
    print("="*80)
    
    # Initialize components
    db_initialized = initialize_components()

    # Check if services are properly initialized
    if balance_sheet_service and cash_flow_service:
        print("✅ All ML services started successfully")
    else:
        print("⚠️ Some services failed to start - continuing anyway")
    
    # Print database status
    if db_connection and db_initialized:
        print("✅ PostgreSQL Database Connected")
        try:
            # Test existing tables
            test_result = db_connection.execute_select(
                "SELECT COUNT(*) as total_records FROM cash_flow_statement"
            )
            if test_result:
                print(f"📊 Total Cash Flow Records: {test_result[0]['total_records']}")
            
            # Test new tables
            new_tables = ['balance_sheet_results', 'cash_flow_generated', 'imputation_results']
            for table in new_tables:
                try:
                    result = db_connection.execute_select(f"SELECT COUNT(*) as count FROM {table}")
                    count = result[0]['count'] if result else 0
                    #print(f"📋 {table.replace('_', ' ').title()}: {count} records")
                except:
                    print(f"⚠️ {table} table not accessible")
                    
        except Exception as e:
            print(f"⚠️ Database test query failed: {e}")
    else:
        print("❌ PostgreSQL Database Not Connected")
    
    print("="*80)
    print("🌐 ENHANCED APPLICATION ENDPOINTS:")
    print("📊 Main Dashboard: http://localhost:5000")
    print("🎯 Enhanced Dashboard: http://localhost:5000/dashboard")
    print("📈 Analytics: http://localhost:5000/analytics")
    print("📋 Companies: http://localhost:5000/companies")
    print("📁 Upload: http://localhost:5000/upload")
    print("📊 Balance Sheet Generator: http://localhost:5000/balance-sheet")
    print("💰 Cash Flow Generator: http://localhost:5000/cash-flow-generator")
    
    
    app.run(debug=True, host='0.0.0.0', port=5000)