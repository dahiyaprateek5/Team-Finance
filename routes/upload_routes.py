import re
from flask import Blueprint, request, jsonify, current_app
import pandas as pd
import io
import os
import psycopg2
from datetime import datetime
from werkzeug.utils import secure_filename
from models.financial_health_model import FinancialHealthModel
from models.balance_sheet_generator import BalanceSheetGenerator
from models.cash_flow_generator import CashFlowGenerator

def validate_balance_sheet_columns(df):
    """Validate if uploaded file has required balance sheet columns"""
    required_keywords = ['asset', 'liability', 'equity', 'cash']
    column_text = ' '.join(df.columns.astype(str)).lower()
    
    found_keywords = [kw for kw in required_keywords if kw in column_text]
    
    if len(found_keywords) < 2:
        return False, f"Missing financial keywords. Found: {found_keywords}"
    
    return True, "Columns validated successfully"

def validate_cash_flow_columns(df):
    """Validate if uploaded file has required cash flow columns"""
    required_keywords = ['cash', 'flow', 'operating', 'income']
    column_text = ' '.join(df.columns.astype(str)).lower()
    
    found_keywords = [kw for kw in required_keywords if kw in column_text]
    
    if len(found_keywords) < 2:
        return False, f"Missing cash flow keywords. Found: {found_keywords}"
    
    return True, "Cash flow columns validated successfully"

# Create Blueprint
upload_bp = Blueprint('upload', __name__)

# Configuration
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'pdf'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

def get_db_config():
    """Get database configuration from app config"""
    return {
        'host': current_app.config.get('DB_HOST', 'localhost'),
        'database': current_app.config.get('DB_NAME', 'financial_db'),
        'user': current_app.config.get('DB_USER', 'postgres'),
        'password': current_app.config.get('DB_PASSWORD', 'password'),
        'port': current_app.config.get('DB_PORT', 5432)
    }

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file_size(file):
    """Check if file size is within limits"""
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)  # Reset file pointer
    return file_size <= MAX_FILE_SIZE

def read_uploaded_file(file):
    """Read uploaded file into pandas DataFrame"""
    try:
        filename = secure_filename(file.filename)
        file_extension = os.path.splitext(filename)[1].lower()
        
        if file_extension == '.csv':
            # Try different encodings for CSV
            try:
                content = file.read().decode('utf-8')
            except UnicodeDecodeError:
                file.seek(0)
                content = file.read().decode('latin-1')
            
            file.seek(0)
            df = pd.read_csv(io.StringIO(content))
            
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(io.BytesIO(file.read()))
            
        elif file_extension == '.pdf':
            return None, "PDF file processing not implemented yet. Please use CSV or Excel files."
            
        else:
            return None, f"Unsupported file format: {file_extension}"
        
        return df, None
        
    except Exception as e:
        return None, f"Error reading file: {str(e)}"

# ========== BALANCE SHEET UPLOAD ROUTE WITH ML INTEGRATION ==========
@upload_bp.route('/balance-sheet', methods=['POST'])
def upload_balance_sheet():
    """
    Upload and process balance sheet file with ML algorithms
    
    Form Data:
    - file: Balance sheet file (CSV, Excel)
    - company_id: Company identifier (required)
    - company_name: Company name (required)  
    - industry: Company industry (optional, default: manufacturing)
    - year: Year (optional, auto-detected from file)
    """
    try:
        # Validate file upload
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'File type not allowed. Supported: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        if not validate_file_size(file):
            return jsonify({
                'success': False,
                'error': f'File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.1f}MB'
            }), 400
        
        # Get form data with proper data type conversion
        try:
            # Fix 1: Convert company_id to integer (STRING to INTEGER)
            company_id = int(request.form.get('company_id'))
        except (ValueError, TypeError):
            return jsonify({
                'success': False,
                'error': 'company_id is required and must be a valid number'
            }), 400
            
        company_name = request.form.get('company_name')
        industry = request.form.get('industry', 'manufacturing')
        year = request.form.get('year')
        
        # Validate required fields
        if not company_id:
            return jsonify({
                'success': False,
                'error': 'company_id is required and must be a number'
            }), 400
        
        if not company_name:
            return jsonify({
                'success': False,
                'error': 'company_name is required'
            }), 400
        
        # Fix 2: Ensure company exists before processing
        db_config = get_db_config()
        
        # Create company if it doesn't exist
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        
        try:
            # Check if company exists
            cursor.execute("SELECT id FROM companies WHERE id = %s", (company_id,))
            if not cursor.fetchone():
                # Create company
                cursor.execute("""
                    INSERT INTO companies (id, company_name, industry, created_at, updated_at)
                    VALUES (%s, %s, %s, NOW(), NOW())
                    ON CONFLICT (id) DO UPDATE SET
                        company_name = EXCLUDED.company_name,
                        industry = EXCLUDED.industry,
                        updated_at = NOW()
                """, (company_id, company_name, industry))
                conn.commit()
                current_app.logger.info(f"Company {company_id} created/updated in database")
        except Exception as e:
            current_app.logger.error(f"Error managing company record: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()
        
        # Read file
        df, error = read_uploaded_file(file)
        if error:
            return jsonify({
                'success': False,
                'error': error
            }), 400
        
        # Validate DataFrame
        if df.empty:
            return jsonify({
                'success': False,
                'error': 'Uploaded file is empty'
            }), 400
        
        if len(df.columns) < 2:
            return jsonify({
                'success': False,
                'error': 'File must have at least 2 columns'
            }), 400
        
        # Column validation
        is_valid, validation_msg = validate_balance_sheet_columns(df)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': f'Column validation failed: {validation_msg}'
            }), 400
        
        # ========== ML PROCESSING WITH BALANCE SHEET GENERATOR ==========
        bs_generator = BalanceSheetGenerator(db_config)
        
        # Process the uploaded file with ML algorithms and industry benchmarks
        complete_balance_sheet = bs_generator.process_uploaded_balance_sheet(
            df=df,
            company_id=company_id,  # Now properly converted to int
            company_name=company_name,
            industry=industry
        )
        
        if not complete_balance_sheet:
            return jsonify({
                'success': False,
                'error': 'Failed to process balance sheet data'
            }), 400
        
        # Override year if provided
        if year:
            try:
                complete_balance_sheet['year'] = int(year)
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': 'Invalid year format'
                }), 400
        
        # Fix 3: Ensure proper data types for database insertion
        # Convert company_id to integer
        complete_balance_sheet['company_id'] = int(complete_balance_sheet.get('company_id', company_id))
        
        # Handle NaN values for PostgreSQL compatibility
        import pandas as pd
        for key, value in complete_balance_sheet.items():
            if pd.isna(value):
                complete_balance_sheet[key] = None
            elif isinstance(value, (int, float)) and str(value).lower() in ['nan', 'inf', '-inf']:
                complete_balance_sheet[key] = None
        
        # Validate generated balance sheet
        validation_results = bs_generator.validate_balance_sheet(complete_balance_sheet)
        if not validation_results['is_valid']:
            complete_balance_sheet['validation_errors'] = '; '.join(validation_results['errors'])
        elif validation_results['warnings']:
            complete_balance_sheet['validation_errors'] = '; '.join(validation_results['warnings'])
        
        # Save complete balance sheet to database (all 40 columns)
        save_success = bs_generator.save_balance_sheet(complete_balance_sheet)
        
        if not save_success:
            # Add detailed logging
            current_app.logger.error(f"Database insertion failed for company: {company_id}")
            current_app.logger.error(f"Balance sheet data keys: {list(complete_balance_sheet.keys())}")
            current_app.logger.error(f"Data types: {[(k, type(v).__name__) for k, v in complete_balance_sheet.items()]}")
            
            return jsonify({
                'success': False,
                'error': 'Failed to save balance sheet to database',
                'debug_info': {
                    'company_id': company_id,
                    'columns_processed': list(complete_balance_sheet.keys()),
                    'data_source': complete_balance_sheet.get('data_source', 'unknown'),
                    'accuracy_percentage': complete_balance_sheet.get('accuracy_percentage', 0),
                    'data_types': {k: type(v).__name__ for k, v in complete_balance_sheet.items() if k in ['company_id', 'year', 'total_assets', 'total_liabilities']}
                }
            }), 500
        
        # Get industry benchmark comparison
        try:
            industry_comparison = bs_generator.get_industry_benchmark_comparison(
                complete_balance_sheet, industry
            )
        except Exception as e:
            current_app.logger.warning(f"Industry comparison failed: {e}")
            industry_comparison = {'error': 'Industry comparison unavailable'}
        
        # Return comprehensive success response with all data
        return jsonify({
            'success': True,
            'message': 'Balance sheet uploaded and processed successfully with ML algorithms',
            'data': {
                'company_id': complete_balance_sheet.get('company_id', company_id),
                'company_name': company_name,
                'year': complete_balance_sheet.get('year', 'Unknown'),
                'industry': industry,
                
                # Financial Summary
                'financial_summary': {
                    'total_assets': float(complete_balance_sheet.get('total_assets', 0) or 0),
                    'total_liabilities': float(complete_balance_sheet.get('total_liabilities', 0) or 0),
                    'total_equity': float(complete_balance_sheet.get('total_equity', 0) or 0),
                    'current_assets': float(complete_balance_sheet.get('current_assets', 0) or 0),
                    'current_liabilities': float(complete_balance_sheet.get('current_liabilities', 0) or 0),
                    'cash_and_equivalents': float(complete_balance_sheet.get('cash_and_equivalents', 0) or 0)
                },
                
                # Asset Breakdown
                'assets': {
                    'current_assets': {
                        'cash_and_equivalents': float(complete_balance_sheet.get('cash_and_equivalents', 0) or 0),
                        'accounts_receivable': float(complete_balance_sheet.get('accounts_receivable', 0) or 0),
                        'inventory': float(complete_balance_sheet.get('inventory', 0) or 0),
                        'prepaid_expenses': float(complete_balance_sheet.get('prepaid_expenses', 0) or 0),
                        'other_current_assets': float(complete_balance_sheet.get('other_current_assets', 0) or 0)
                    },
                    'non_current_assets': {
                        'property_plant_equipment': float(complete_balance_sheet.get('property_plant_equipment', 0) or 0),
                        'accumulated_depreciation': float(complete_balance_sheet.get('accumulated_depreciation', 0) or 0),
                        'net_ppe': float(complete_balance_sheet.get('net_ppe', 0) or 0),
                        'intangible_assets': float(complete_balance_sheet.get('intangible_assets', 0) or 0),
                        'goodwill': float(complete_balance_sheet.get('goodwill', 0) or 0),
                        'investments': float(complete_balance_sheet.get('investments', 0) or 0),
                        'other_non_current_assets': float(complete_balance_sheet.get('other_non_current_assets', 0) or 0)
                    }
                },
                
                # Liability Breakdown
                'liabilities': {
                    'current_liabilities': {
                        'accounts_payable': float(complete_balance_sheet.get('accounts_payable', 0) or 0),
                        'short_term_debt': float(complete_balance_sheet.get('short_term_debt', 0) or 0),
                        'accrued_liabilities': float(complete_balance_sheet.get('accrued_liabilities', 0) or 0),
                        'deferred_revenue': float(complete_balance_sheet.get('deferred_revenue', 0) or 0),
                        'other_current_liabilities': float(complete_balance_sheet.get('other_current_liabilities', 0) or 0)
                    },
                    'non_current_liabilities': {
                        'long_term_debt': float(complete_balance_sheet.get('long_term_debt', 0) or 0),
                        'deferred_tax_liabilities': float(complete_balance_sheet.get('deferred_tax_liabilities', 0) or 0),
                        'pension_obligations': float(complete_balance_sheet.get('pension_obligations', 0) or 0),
                        'other_non_current_liabilities': float(complete_balance_sheet.get('other_non_current_liabilities', 0) or 0)
                    }
                },
                
                # Equity Breakdown
                'equity': {
                    'share_capital': float(complete_balance_sheet.get('share_capital', 0) or 0),
                    'retained_earnings': float(complete_balance_sheet.get('retained_earnings', 0) or 0),
                    'additional_paid_in_capital': float(complete_balance_sheet.get('additional_paid_in_capital', 0) or 0),
                    'treasury_stock': float(complete_balance_sheet.get('treasury_stock', 0) or 0),
                    'accumulated_other_comprehensive_income': float(complete_balance_sheet.get('accumulated_other_comprehensive_income', 0) or 0)
                },
                
                # Quality Metrics
                'quality_metrics': {
                    'accuracy_percentage': float(complete_balance_sheet.get('accuracy_percentage', 0) or 0),
                    'balance_check': float(complete_balance_sheet.get('balance_check', 0) or 0),
                    'data_source': complete_balance_sheet.get('data_source', 'unknown'),
                    'validation_errors': complete_balance_sheet.get('validation_errors', None),
                    'generated_at': complete_balance_sheet.get('generated_at', datetime.now()).isoformat()
                },
                
                # Industry Comparison
                'industry_benchmarks': industry_comparison if 'error' not in industry_comparison else None,
                
                # Processing Info
                'processing_info': {
                    'ml_algorithms_used': True,
                    'industry_benchmarks_applied': True,
                    'missing_data_estimated': True,
                    'total_columns_processed': len(complete_balance_sheet),
                    'columns_with_data': len([v for v in complete_balance_sheet.values() if v is not None and str(v) != 'nan']),
                    'data_types_validated': True,
                    'company_record_created': True
                }
            }
        }), 200
        
    except ValueError as ve:
        current_app.logger.error(f"Data validation error: {str(ve)}")
        return jsonify({
            'success': False,
            'error': f'Data validation error: {str(ve)}'
        }), 400
        
    except psycopg2.Error as pe:
        current_app.logger.error(f"Database error: {str(pe)}")
        return jsonify({
            'success': False,
            'error': 'Database operation failed. Please check your data and try again.'
        }), 500
        
    except Exception as e:
        current_app.logger.error(f"Error uploading balance sheet: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

# ========== CASH FLOW UPLOAD ROUTE ==========
@upload_bp.route('/cash-flow', methods=['POST'])
def upload_cash_flow():
    """
    Upload and process cash flow statement file
    
    Form Data:
    - file: Cash flow statement file (CSV, Excel)
    - company_id: Company identifier (required)
    - company_name: Company name (required)
    - industry: Company industry (optional, default: manufacturing)
    - year: Year (optional, auto-detected from file)
    """
    try:
        # Validate file upload
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'File type not allowed. Supported: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        if not validate_file_size(file):
            return jsonify({
                'success': False,
                'error': f'File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.1f}MB'
            }), 400
        
        # Get form data
        company_id = request.form.get('company_id')
        company_name = request.form.get('company_name')
        industry = request.form.get('industry', 'manufacturing')
        year = request.form.get('year')
        
        # Validate required fields
        if not company_id:
            return jsonify({
                'success': False,
                'error': 'company_id is required'
            }), 400
        
        if not company_name:
            return jsonify({
                'success': False,
                'error': 'company_name is required'
            }), 400
        
        # Read file
        df, error = read_uploaded_file(file)
        if error:
            return jsonify({
                'success': False,
                'error': error
            }), 400
        
        # Validate DataFrame
        if df.empty:
            return jsonify({
                'success': False,
                'error': 'Uploaded file is empty'
            }), 400
        
        # Column validation
        is_valid, validation_msg = validate_cash_flow_columns(df)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': f'Column validation failed: {validation_msg}'
            }), 400
        
        # Process cash flow statement
        db_config = get_db_config()
        cf_generator = CashFlowGenerator(db_config)
        
        # Process the uploaded file
        cash_flow_data = cf_generator.process_uploaded_cash_flow(
            df=df,
            company_id=company_id,
            company_name=company_name,
            industry=industry
        )
        
        if not cash_flow_data:
            return jsonify({
                'success': False,
                'error': 'Failed to process cash flow statement'
            }), 400
        
        # Override year if provided
        if year:
            try:
                cash_flow_data['year'] = int(year)
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': 'Invalid year format'
                }), 400
        
        # Save to cash_flow_statement table
        save_success = cf_generator.save_cash_flow_to_db(cash_flow_data)
        
        if not save_success:
            # Add detailed logging
            current_app.logger.error(f"Database insertion failed for company: {company_id}")
            current_app.logger.error(f"Cash flow data keys: {list(cash_flow_data.keys())}")
            
            return jsonify({
                'success': False,
                'error': 'Failed to save cash flow to database',
                'debug_info': {
                    'company_id': company_id,
                    'columns_processed': list(cash_flow_data.keys()),
                    'data_source': cash_flow_data.get('data_source', 'unknown')
                }
            }), 500
        
        # Return success response
        return jsonify({
            'success': True,
            'message': 'Cash flow statement uploaded and processed successfully',
            'data': {
                'company_id': cash_flow_data.get('company_id', company_id),
                'company_name': company_name,
                'year': cash_flow_data.get('year', 'Unknown'),
                'industry': industry,
                
                # Core Cash Flow Metrics
                'cash_flow_summary': {
                    'net_income': cash_flow_data.get('net_income', 0),
                    'net_cash_from_operating_activities': cash_flow_data.get('net_cash_from_operating_activities', 0),
                    'net_cash_from_investing_activities': cash_flow_data.get('net_cash_from_investing_activities', 0),
                    'net_cash_from_financing_activities': cash_flow_data.get('net_cash_from_financing_activities', 0),
                    'free_cash_flow': cash_flow_data.get('free_cash_flow', 0)
                },
                
                # Operating Activities Detail
                'operating_activities': {
                    'net_income': cash_flow_data.get('net_income', 0),
                    'depreciation_and_amortization': cash_flow_data.get('depreciation_and_amortization', 0),
                    'stock_based_compensation': cash_flow_data.get('stock_based_compensation', 0),
                    'changes_in_working_capital': cash_flow_data.get('changes_in_working_capital', 0),
                    'accounts_receivable': cash_flow_data.get('accounts_receivable', 0),
                    'inventory': cash_flow_data.get('inventory', 0),
                    'accounts_payable': cash_flow_data.get('accounts_payable', 0)
                },
                
                # Investing Activities Detail
                'investing_activities': {
                    'capital_expenditures': cash_flow_data.get('capital_expenditures', 0),
                    'acquisitions': cash_flow_data.get('acquisitions', 0)
                },
                
                # Financing Activities Detail
                'financing_activities': {
                    'dividends_paid': cash_flow_data.get('dividends_paid', 0),
                    'share_repurchases': cash_flow_data.get('share_repurchases', 0)
                },
                
                # Financial Ratios
                'financial_ratios': {
                    'ocf_to_net_income_ratio': cash_flow_data.get('ocf_to_net_income_ratio', 0),
                    'debt_to_equity_ratio': cash_flow_data.get('debt_to_equity_ratio', 0),
                    'interest_coverage_ratio': cash_flow_data.get('interest_coverage_ratio', 0),
                    'liquidation_label': cash_flow_data.get('liquidation_label', 0)
                },
                
                # Processing Info
                'processing_info': {
                    'generated_at': cash_flow_data.get('generated_at', datetime.now()).isoformat(),
                    'data_source': cash_flow_data.get('data_source', 'uploaded'),
                    'total_columns_processed': len(cash_flow_data)
                }
            }
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error uploading cash flow: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

# ========== COMBINED UPLOAD ROUTE WITH ML INTEGRATION ==========
@upload_bp.route('/combined', methods=['POST'])
def upload_combined_financial_statements():
    """
    Upload multiple financial statement files at once with ML processing
    
    Form Data:
    - balance_sheet_file: Balance sheet file (optional)
    - cash_flow_file: Cash flow statement file (optional)
    - company_id: Company identifier (required)
    - company_name: Company name (required)
    - industry: Company industry (optional)
    - year: Year (optional)
    """
    try:
        # Get form data
        company_id = request.form.get('company_id')
        company_name = request.form.get('company_name')
        industry = request.form.get('industry', 'manufacturing')
        year = request.form.get('year')
        
        # Validate required fields
        if not company_id or not company_name:
            return jsonify({
                'success': False,
                'error': 'company_id and company_name are required'
            }), 400
        
        results = {
            'balance_sheet': None,
            'cash_flow': None,
            'errors': []
        }
        
        db_config = get_db_config()
        
        # Process balance sheet if provided with ML
        if 'balance_sheet_file' in request.files:
            bs_file = request.files['balance_sheet_file']
            if bs_file.filename != '':
                try:
                    if not allowed_file(bs_file.filename):
                        results['errors'].append('Balance sheet file type not allowed')
                    elif not validate_file_size(bs_file):
                        results['errors'].append('Balance sheet file too large')
                    else:
                        # Read and process balance sheet with ML
                        df, error = read_uploaded_file(bs_file)
                        if error:
                            results['errors'].append(f'Balance sheet error: {error}')
                        else:
                            bs_generator = BalanceSheetGenerator(db_config)
                            balance_sheet_data = bs_generator.process_uploaded_balance_sheet(
                                df, company_id, company_name, industry
                            )
                            
                            if balance_sheet_data:
                                if year:
                                    balance_sheet_data['year'] = int(year)
                                
                                save_success = bs_generator.save_balance_sheet(balance_sheet_data)
                                if save_success:
                                    results['balance_sheet'] = {
                                        'success': True,
                                        'year': balance_sheet_data['year'],
                                        'accuracy_percentage': balance_sheet_data['accuracy_percentage'],
                                        'total_assets': balance_sheet_data.get('total_assets', 0),
                                        'total_liabilities': balance_sheet_data.get('total_liabilities', 0),
                                        'total_equity': balance_sheet_data.get('total_equity', 0),
                                        'ml_processing': True
                                    }
                                else:
                                    results['errors'].append('Failed to save balance sheet')
                            else:
                                results['errors'].append('Failed to process balance sheet with ML')
                                
                except Exception as e:
                    results['errors'].append(f'Balance sheet processing error: {str(e)}')
        
        # Process cash flow if provided
        if 'cash_flow_file' in request.files:
            cf_file = request.files['cash_flow_file']
            if cf_file.filename != '':
                try:
                    if not allowed_file(cf_file.filename):
                        results['errors'].append('Cash flow file type not allowed')
                    elif not validate_file_size(cf_file):
                        results['errors'].append('Cash flow file too large')
                    else:
                        # Read and process cash flow
                        df, error = read_uploaded_file(cf_file)
                        if error:
                            results['errors'].append(f'Cash flow error: {error}')
                        else:
                            cf_generator = CashFlowGenerator(db_config)
                            cash_flow_data = cf_generator.process_uploaded_cash_flow(
                                df, company_id, company_name, industry
                            )
                            
                            if cash_flow_data:
                                if year:
                                    cash_flow_data['year'] = int(year)
                                
                                save_success = cf_generator.save_cash_flow_to_db(cash_flow_data)
                                if save_success:
                                    results['cash_flow'] = {
                                        'success': True,
                                        'year': cash_flow_data['year'],
                                        'net_income': cash_flow_data['net_income'],
                                        'free_cash_flow': cash_flow_data.get('free_cash_flow', 0),
                                        'liquidation_label': cash_flow_data['liquidation_label']
                                    }
                                else:
                                    results['errors'].append('Failed to save cash flow')
                            else:
                                results['errors'].append('Failed to process cash flow statement')
                                
                except Exception as e:
                    results['errors'].append(f'Cash flow processing error: {str(e)}')
        
        # Check if at least one file was processed successfully
        success_count = sum(1 for x in [results['balance_sheet'], results['cash_flow']] if x)
        
        if success_count == 0:
            return jsonify({
                'success': False,
                'error': 'No files were processed successfully',
                'details': results['errors']
            }), 400
        
        return jsonify({
            'success': True,
            'message': f'Processed {success_count} file(s) successfully with ML algorithms',
            'data': {
                'company_id': company_id,
                'company_name': company_name,
                'industry': industry,
                'results': results,
                'processing_summary': {
                    'balance_sheet_ml_processed': results['balance_sheet'] is not None,
                    'cash_flow_processed': results['cash_flow'] is not None,
                    'total_files_processed': success_count,
                    'errors_count': len(results['errors'])
                }
            }
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error in combined upload: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

# ========== BATCH UPLOAD WITH ML INTEGRATION ==========
@upload_bp.route('/batch', methods=['POST'])
def batch_upload():
    """
    Upload multiple files for multiple companies with ML processing
    
    Form Data:
    - files: Multiple files
    - companies_data: JSON string with company information
    """
    try:
        if 'files' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No files uploaded'
            }), 400
        
        files = request.files.getlist('files')
        companies_data_str = request.form.get('companies_data', '[]')
        
        try:
            import json
            companies_data = json.loads(companies_data_str)
        except json.JSONDecodeError:
            return jsonify({
                'success': False,
                'error': 'Invalid companies_data JSON format'
            }), 400
        
        if len(files) != len(companies_data):
            return jsonify({
                'success': False,
                'error': 'Number of files must match number of company data entries'
            }), 400
        
        results = []
        db_config = get_db_config()
        
        for i, (file, company_info) in enumerate(zip(files, companies_data)):
            result = {
                'file_index': i,
                'filename': file.filename,
                'company_id': company_info.get('company_id'),
                'success': False,
                'error': None,
                'data': None,
                'ml_processed': False
            }
            
            try:
                # Validate company info
                if not company_info.get('company_id') or not company_info.get('company_name'):
                    result['error'] = 'Missing company_id or company_name'
                    results.append(result)
                    continue
                
                # Validate file
                if not allowed_file(file.filename):
                    result['error'] = 'File type not allowed'
                    results.append(result)
                    continue
                
                if not validate_file_size(file):
                    result['error'] = 'File too large'
                    results.append(result)
                    continue
                
                # Read file
                df, error = read_uploaded_file(file)
                if error:
                    result['error'] = error
                    results.append(result)
                    continue
                
                # Determine file type based on content or company_info
                file_type = company_info.get('file_type', 'balance_sheet')
                
                if file_type == 'balance_sheet':
                    # Process as balance sheet with ML
                    bs_generator = BalanceSheetGenerator(db_config)
                    processed_data = bs_generator.process_uploaded_balance_sheet(
                        df,
                        company_info['company_id'],
                        company_info['company_name'],
                        company_info.get('industry', 'manufacturing')
                    )
                    
                    if processed_data:
                        save_success = bs_generator.save_balance_sheet(processed_data)
                        result['ml_processed'] = True
                    else:
                        save_success = False
                        result['error'] = 'Failed to process balance sheet with ML'
                    
                elif file_type == 'cash_flow':
                    # Process as cash flow
                    cf_generator = CashFlowGenerator(db_config)
                    processed_data = cf_generator.process_uploaded_cash_flow(
                        df,
                        company_info['company_id'],
                        company_info['company_name'],
                        company_info.get('industry', 'manufacturing')
                    )
                    
                    if processed_data:
                        save_success = cf_generator.save_cash_flow_to_db(processed_data)
                    else:
                        save_success = False
                        result['error'] = 'Failed to process cash flow data'
                
                else:
                    result['error'] = f'Unknown file_type: {file_type}'
                    results.append(result)
                    continue
                
                if save_success:
                    result['success'] = True
                    result['data'] = {
                        'company_id': processed_data['company_id'],
                        'year': processed_data['year'],
                        'file_type': file_type,
                        'accuracy_percentage': processed_data.get('accuracy_percentage', 0) if file_type == 'balance_sheet' else None
                    }
                else:
                    result['error'] = 'Failed to save to database'
                
            except Exception as e:
                result['error'] = f'Processing error: {str(e)}'
            
            results.append(result)
        
        # Calculate summary
        successful_uploads = sum(1 for r in results if r['success'])
        failed_uploads = len(results) - successful_uploads
        ml_processed_count = sum(1 for r in results if r.get('ml_processed', False))
        
        return jsonify({
            'success': successful_uploads > 0,
            'message': f'Batch upload completed: {successful_uploads} successful, {failed_uploads} failed',
            'summary': {
                'total_files': len(results),
                'successful': successful_uploads,
                'failed': failed_uploads,
                'ml_processed': ml_processed_count
            },
            'results': results
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error in batch upload: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

# ========== FILE VALIDATION ROUTE ==========
@upload_bp.route('/validate-file', methods=['POST'])
def validate_file():
    """
    Validate uploaded file without processing
    
    Form Data:
    - file: File to validate
    - file_type: Expected file type (balance_sheet or cash_flow)
    """
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        file_type = request.form.get('file_type', 'unknown')
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        validation_results = {
            'filename': secure_filename(file.filename),
            'file_type': file_type,
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check file extension
        if not allowed_file(file.filename):
            validation_results['is_valid'] = False
            validation_results['errors'].append(f'File type not allowed. Supported: {", ".join(ALLOWED_EXTENSIONS)}')
        
        # Check file size
        if not validate_file_size(file):
            validation_results['is_valid'] = False
            validation_results['errors'].append(f'File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.1f}MB')
        
        # Try to read file structure
        try:
            df, error = read_uploaded_file(file)
            if error:
                validation_results['is_valid'] = False
                validation_results['errors'].append(error)
            else:
                # Analyze file structure
                validation_results['file_info'] = {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'column_names': df.columns.tolist()[:10],  # First 10 columns
                    'has_data': not df.empty
                }
                
                # Check minimum requirements
                if df.empty:
                    validation_results['is_valid'] = False
                    validation_results['errors'].append('File is empty')
                
                if len(df.columns) < 2:
                    validation_results['is_valid'] = False
                    validation_results['errors'].append('File must have at least 2 columns')
                
                # Check for financial keywords
                column_text = ' '.join(df.columns.astype(str)).lower()
                
                if file_type == 'balance_sheet':
                    bs_keywords = ['asset', 'liability', 'equity', 'cash', 'receivable', 'inventory']
                    found_keywords = [kw for kw in bs_keywords if kw in column_text]
                    
                    if len(found_keywords) < 2:
                        validation_results['warnings'].append('Few balance sheet keywords found in columns')
                    else:
                        validation_results['detected_keywords'] = found_keywords
                
                elif file_type == 'cash_flow':
                    cf_keywords = ['cash', 'operating', 'investing', 'financing', 'income', 'flow']
                    found_keywords = [kw for kw in cf_keywords if kw in column_text]
                    
                    if len(found_keywords) < 2:
                        validation_results['warnings'].append('Few cash flow keywords found in columns')
                    else:
                        validation_results['detected_keywords'] = found_keywords
                
                # Check for year information
                year_pattern = r'20\d{2}'
                year_found = False
                for col in df.columns:
                    if re.search(year_pattern, str(col)):
                        year_found = True
                        break
                
                if not year_found:
                    validation_results['warnings'].append('No year information detected in columns')
                
                # ML Processing Readiness Check
                validation_results['ml_readiness'] = {
                    'can_process_with_ml': validation_results['is_valid'],
                    'estimated_accuracy': 'High' if len(found_keywords) >= 3 else 'Medium' if len(found_keywords) >= 2 else 'Low',
                    'industry_benchmarks_applicable': True,
                    'missing_data_estimation_available': True
                }
                
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f'File structure analysis failed: {str(e)}')
        
        return jsonify({
            'success': True,
            'validation': validation_results
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error validating file: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

# ========== FILE PREVIEW ROUTE ==========
@upload_bp.route('/preview', methods=['POST'])
def preview_file():
    """
    Preview uploaded file content without saving
    
    Form Data:
    - file: File to preview
    - rows: Number of rows to preview (default: 10)
    """
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        rows_to_preview = int(request.form.get('rows', 10))
        rows_to_preview = min(rows_to_preview, 50)  # Limit to 50 rows
        
        # Validate file
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'File type not allowed. Supported: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Read file
        df, error = read_uploaded_file(file)
        if error:
            return jsonify({
                'success': False,
                'error': error
            }), 400
        
        # Create preview
        preview_df = df.head(rows_to_preview)
        
        preview_data = {
            'filename': secure_filename(file.filename),
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'preview_rows': len(preview_df),
            'columns': df.columns.tolist(),
            'data': preview_df.fillna('').to_dict('records'),
            'data_types': df.dtypes.astype(str).to_dict(),
            'ml_processing_info': {
                'ready_for_ml': True,
                'supported_algorithms': ['Random Forest', 'KNN', 'Industry Benchmarks', 'Time Series'],
                'estimated_completion_columns': {
                    'balance_sheet': 40,
                    'cash_flow': 25
                }
            }
        }
        
        return jsonify({
            'success': True,
            'preview': preview_data
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error previewing file: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

# ========== SUPPORTED FORMATS ROUTE ==========
@upload_bp.route('/supported-formats', methods=['GET'])
def get_supported_formats():
    """Get information about supported file formats and ML capabilities"""
    return jsonify({
        'success': True,
        'data': {
            'supported_extensions': list(ALLOWED_EXTENSIONS),
            'max_file_size_mb': MAX_FILE_SIZE / (1024 * 1024),
            'ml_capabilities': {
                'balance_sheet_generator': {
                    'total_columns_generated': 40,
                    'algorithms_used': ['Random Forest', 'KNN', 'Industry Benchmarks'],
                    'supported_industries': ['technology', 'manufacturing', 'retail', 'healthcare', 'financial'],
                    'accuracy_targets': {
                        'complete_data': '90%+',
                        'partial_data': '85%+',
                        'minimal_data': '75%+'
                    }
                },
                'cash_flow_generator': {
                    'total_columns_generated': 25,
                    'calculations_included': ['Free Cash Flow', 'OCF Ratios', 'Liquidation Prediction'],
                    'industry_benchmarks': True
                }
            },
            'requirements': {
                'balance_sheet': {
                    'min_columns': 2,
                    'required_keywords': ['asset', 'liability', 'equity'],
                    'optional_keywords': ['cash', 'receivable', 'inventory', 'debt'],
                    'ml_processing': True
                },
                'cash_flow': {
                    'min_columns': 2,
                    'required_keywords': ['cash', 'flow'],
                    'optional_keywords': ['operating', 'investing', 'financing', 'income'],
                    'ml_processing': True
                }
            },
            'column_mapping': {
                'balance_sheet': {
                    'total_assets': ['total assets', 'assets total', 'sum of assets'],
                    'cash_and_equivalents': ['cash', 'cash and cash equivalents', 'liquid assets'],
                    'accounts_receivable': ['accounts receivable', 'receivables', 'trade receivables'],
                    'total_equity': ['total equity', 'shareholders equity', 'equity total'],
                    'property_plant_equipment': ['ppe', 'fixed assets', 'plant and equipment'],
                    'accounts_payable': ['payables', 'trade payables', 'creditors'],
                    'long_term_debt': ['long term debt', 'term loans', 'non-current debt']
                },
                'cash_flow': {
                    'net_income': ['net income', 'profit after tax', 'net profit'],
                    'operating_cash_flow': ['operating cash flow', 'cash from operations'],
                    'free_cash_flow': ['free cash flow', 'fcf'],
                    'capital_expenditures': ['capex', 'capital expenditure', 'investments in ppe'],
                    'depreciation': ['depreciation', 'amortization', 'non-cash charges']
                }
            }
        }
    }), 200

# ========== ML PROCESSING STATUS ROUTE ==========
@upload_bp.route('/ml-status/<company_id>', methods=['GET'])
def get_ml_processing_status(company_id):
    """
    Get ML processing status for a company
    
    Args:
        company_id: Company identifier
    """
    try:
        db_config = get_db_config()
        bs_generator = BalanceSheetGenerator(db_config)
        
        # Check if company data exists and get processing status
        conn = bs_generator.connect_db()
        if not conn:
            return jsonify({
                'success': False,
                'error': 'Database connection failed'
            }), 500
        
        cursor = conn.cursor()
        
        # Check balance sheet data
        cursor.execute("""
            SELECT year, accuracy_percentage, data_source, generated_at, validation_errors
            FROM balance_sheet_1 
            WHERE company_id = %s 
            ORDER BY year DESC
        """, (company_id,))
        
        balance_sheet_records = cursor.fetchall()
        
        # Check cash flow data
        cursor.execute("""
            SELECT year, liquidation_label, generated_at
            FROM cash_flow_statement 
            WHERE company_id = %s 
            ORDER BY year DESC
        """, (company_id,))
        
        cash_flow_records = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        status_info = {
            'company_id': company_id,
            'balance_sheet_status': {
                'records_found': len(balance_sheet_records),
                'latest_year': balance_sheet_records[0][0] if balance_sheet_records else None,
                'latest_accuracy': balance_sheet_records[0][1] if balance_sheet_records else None,
                'ml_generated': balance_sheet_records[0][2] == 'generated' if balance_sheet_records else False,
                'last_updated': balance_sheet_records[0][3].isoformat() if balance_sheet_records else None,
                'has_validation_errors': balance_sheet_records[0][4] is not None if balance_sheet_records else False
            },
            'cash_flow_status': {
                'records_found': len(cash_flow_records),
                'latest_year': cash_flow_records[0][0] if cash_flow_records else None,
                'liquidation_risk': cash_flow_records[0][1] if cash_flow_records else None,
                'last_updated': cash_flow_records[0][2].isoformat() if cash_flow_records else None
            },
            'overall_status': {
                'has_balance_sheet': len(balance_sheet_records) > 0,
                'has_cash_flow': len(cash_flow_records) > 0,
                'ml_processing_complete': len(balance_sheet_records) > 0 or len(cash_flow_records) > 0,
                'data_quality': 'High' if balance_sheet_records and balance_sheet_records[0][1] > 80 else 'Medium' if balance_sheet_records and balance_sheet_records[0][1] > 60 else 'Low'
            }
        }
        
        return jsonify({
            'success': True,
            'data': status_info
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error getting ML status for {company_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

# ========== ERROR HANDLERS ==========
@upload_bp.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'success': False,
        'error': f'File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.1f}MB'
    }), 413

@upload_bp.errorhandler(400)
def bad_request(error):
    return jsonify({
        'success': False,
        'error': 'Bad request'
    }), 400

@upload_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500