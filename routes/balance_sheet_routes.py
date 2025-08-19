from flask import Blueprint, request, jsonify, current_app
import pandas as pd
import io
import os
from datetime import datetime
from models.financial_health_model import FinancialHealthModel
from models.balance_sheet_generator import BalanceSheetGenerator
from models.balance_sheet_models import BalanceSheetPredictionModels

# Create Blueprint
balance_sheet_bp = Blueprint('balance_sheet', __name__)

def get_db_config():
    """Get database configuration from app config"""
    return {
        'host': current_app.config.get('DB_HOST', 'localhost'),
        'database': current_app.config.get('DB_NAME', 'financial_db'),
        'user': current_app.config.get('DB_USER', 'postgres'),
        'password': current_app.config.get('DB_PASSWORD', 'Prateek@2003'),
        'port': current_app.config.get('DB_PORT', 5432)
    }

@balance_sheet_bp.route('/generate', methods=['POST'])
def generate_balance_sheet():
    """
    Generate complete balance sheet from minimal data
    
    Request JSON:
    {
        "company_id": "COMP001",
        "company_name": "ABC Company",
        "year": 2024,
        "industry": "technology",
        "known_data": {
            "total_assets": 1000000,
            "cash_and_equivalents": 150000,
            "accounts_receivable": 200000
        }
    }
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['company_id', 'company_name']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        company_id = data['company_id']
        company_name = data['company_name']
        year = data.get('year', datetime.now().year)
        industry = data.get('industry', 'manufacturing')
        known_data = data.get('known_data', {})
        
        # Initialize balance sheet generator
        db_config = get_db_config()
        bs_generator = BalanceSheetGenerator(db_config)
        
        # Generate complete balance sheet
        balance_sheet = bs_generator.generate_complete_balance_sheet(
            company_id=company_id,
            year=year,
            industry=industry,
            known_data=known_data
        )
        
        # Save to database (balance_sheet_1 table)
        save_success = bs_generator.save_balance_sheet(balance_sheet)
        
        if not save_success:
            return jsonify({
                'success': False,
                'error': 'Failed to save balance sheet to database'
            }), 500
        
        # Return generated balance sheet
        return jsonify({
            'success': True,
            'message': 'Balance sheet generated successfully',
            'data': {
                'company_id': balance_sheet['company_id'],
                'year': balance_sheet['year'],
                'accuracy_percentage': balance_sheet['accuracy_percentage'],
                'balance_check': balance_sheet['balance_check'],
                'balance_sheet': balance_sheet
            }
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error generating balance sheet: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@balance_sheet_bp.route('/process', methods=['POST'])
def process_balance_sheet_file():
    """
    Process uploaded balance sheet file and save to balance_sheet_1 table
    
    Form Data:
    - file: Balance sheet file (CSV, Excel)
    - company_id: Company identifier
    - company_name: Company name
    - industry: Company industry (optional)
    """
    try:
        # Check if file is present
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
        
        # Get form data
        company_id = request.form.get('company_id')
        company_name = request.form.get('company_name')
        industry = request.form.get('industry', 'manufacturing')
        
        if not company_id or not company_name:
            return jsonify({
                'success': False,
                'error': 'Missing company_id or company_name'
            }), 400
        
        # Read file into DataFrame
        try:
            file_extension = os.path.splitext(file.filename)[1].lower()
            
            if file_extension == '.csv':
                df = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(io.BytesIO(file.read()))
            else:
                return jsonify({
                    'success': False,
                    'error': 'Unsupported file format. Use CSV or Excel files.'
                }), 400
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Error reading file: {str(e)}'
            }), 400
        
        # Initialize financial health model
        db_config = get_db_config()
        health_model = FinancialHealthModel(db_config)
        
        # Process balance sheet
        balance_sheet_data = health_model.process_balance_sheet(
            df=df,
            company_id=company_id,
            company_name=company_name,
            industry=industry
        )
        
        # Save to balance_sheet_1 table
        save_success = health_model.save_balance_sheet_to_db(balance_sheet_data)
        
        if not save_success:
            return jsonify({
                'success': False,
                'error': 'Failed to save to database'
            }), 500
        
        return jsonify({
            'success': True,
            'message': 'Balance sheet processed and saved successfully',
            'data': {
                'company_id': balance_sheet_data['company_id'],
                'year': balance_sheet_data['year'],
                'accuracy_percentage': balance_sheet_data['accuracy_percentage'],
                'balance_check': balance_sheet_data['balance_check'],
                'total_assets': balance_sheet_data['total_assets'],
                'total_liabilities': balance_sheet_data['total_liabilities'],
                'total_equity': balance_sheet_data['total_equity']
            }
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error processing balance sheet: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@balance_sheet_bp.route('/get/<company_id>/<int:year>', methods=['GET'])
def get_balance_sheet(company_id, year):
    """
    Get balance sheet for specific company and year from balance_sheet_1 table
    """
    try:
        db_config = get_db_config()
        health_model = FinancialHealthModel(db_config)
        
        conn = health_model.connect_db()
        if not conn:
            return jsonify({
                'success': False,
                'error': 'Database connection failed'
            }), 500
        
        cursor = conn.cursor()
        
        # Query balance sheet from balance_sheet_1 table
        query = """
        SELECT * FROM balance_sheet_1 
        WHERE company_id = %s AND year = %s
        """
        
        cursor.execute(query, (company_id, year))
        result = cursor.fetchone()
        
        if not result:
            cursor.close()
            conn.close()
            return jsonify({
                'success': False,
                'error': 'Balance sheet not found'
            }), 404
        
        # Get column names
        column_names = [desc[0] for desc in cursor.description]
        
        # Convert to dictionary
        balance_sheet = dict(zip(column_names, result))
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'data': balance_sheet
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error getting balance sheet: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@balance_sheet_bp.route('/list/<company_id>', methods=['GET'])
def list_balance_sheets(company_id):
    """
    List all balance sheets for a company from balance_sheet_1 table
    """
    try:
        db_config = get_db_config()
        health_model = FinancialHealthModel(db_config)
        
        conn = health_model.connect_db()
        if not conn:
            return jsonify({
                'success': False,
                'error': 'Database connection failed'
            }), 500
        
        cursor = conn.cursor()
        
        # Query all balance sheets for company
        query = """
        SELECT year, generated_at, accuracy_percentage, total_assets, 
               total_liabilities, total_equity, data_source
        FROM balance_sheet_1 
        WHERE company_id = %s 
        ORDER BY year DESC
        """
        
        cursor.execute(query, (company_id,))
        results = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        # Convert to list of dictionaries
        balance_sheets = []
        for result in results:
            balance_sheet = {
                'year': result[0],
                'generated_at': result[1].isoformat() if result[1] else None,
                'accuracy_percentage': result[2],
                'total_assets': result[3],
                'total_liabilities': result[4],
                'total_equity': result[5],
                'data_source': result[6]
            }
            balance_sheets.append(balance_sheet)
        
        return jsonify({
            'success': True,
            'data': balance_sheets,
            'count': len(balance_sheets)
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error listing balance sheets: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@balance_sheet_bp.route('/validate', methods=['POST'])
def validate_balance_sheet():
    """
    Validate balance sheet data for consistency
    
    Request JSON:
    {
        "company_id": "COMP001",
        "year": 2024
    }
    """
    try:
        data = request.get_json()
        company_id = data.get('company_id')
        year = data.get('year')
        
        if not company_id or not year:
            return jsonify({
                'success': False,
                'error': 'Missing company_id or year'
            }), 400
        
        db_config = get_db_config()
        bs_generator = BalanceSheetGenerator(db_config)
        
        # Get balance sheet from database
        conn = bs_generator.connect_db()
        cursor = conn.cursor()
        
        query = "SELECT * FROM balance_sheet_1 WHERE company_id = %s AND year = %s"
        cursor.execute(query, (company_id, year))
        result = cursor.fetchone()
        
        if not result:
            cursor.close()
            conn.close()
            return jsonify({
                'success': False,
                'error': 'Balance sheet not found'
            }), 404
        
        # Convert to dictionary
        column_names = [desc[0] for desc in cursor.description]
        balance_sheet = dict(zip(column_names, result))
        
        cursor.close()
        conn.close()
        
        # Validate balance sheet
        validation_results = bs_generator.validate_balance_sheet(balance_sheet)
        
        return jsonify({
            'success': True,
            'validation': validation_results
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error validating balance sheet: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@balance_sheet_bp.route('/export/<company_id>/<int:year>', methods=['GET'])
def export_balance_sheet(company_id, year):
    """
    Export balance sheet to Excel format
    """
    try:
        # Get balance sheet data
        db_config = get_db_config()
        health_model = FinancialHealthModel(db_config)
        
        conn = health_model.connect_db()
        cursor = conn.cursor()
        
        query = "SELECT * FROM balance_sheet_1 WHERE company_id = %s AND year = %s"
        cursor.execute(query, (company_id, year))
        result = cursor.fetchone()
        
        if not result:
            cursor.close()
            conn.close()
            return jsonify({
                'success': False,
                'error': 'Balance sheet not found'
            }), 404
        
        column_names = [desc[0] for desc in cursor.description]
        balance_sheet = dict(zip(column_names, result))
        
        cursor.close()
        conn.close()
        
        # Create export file
        bs_generator = BalanceSheetGenerator(db_config)
        export_filename = f"balance_sheet_{company_id}_{year}.xlsx"
        
        # Export to Excel (you can customize the export path)
        export_success = bs_generator.export_balance_sheet_to_excel(
            balance_sheet, 
            export_filename
        )
        
        if export_success:
            return jsonify({
                'success': True,
                'message': 'Balance sheet exported successfully',
                'filename': export_filename
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Export failed'
            }), 500
        
    except Exception as e:
        current_app.logger.error(f"Error exporting balance sheet: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@balance_sheet_bp.route('/benchmark/<company_id>/<int:year>', methods=['GET'])
def benchmark_balance_sheet(company_id, year):
    """
    Compare balance sheet against industry benchmarks
    """
    try:
        # Get industry from query params
        industry = request.args.get('industry', 'manufacturing')
        
        # Get balance sheet data
        db_config = get_db_config()
        health_model = FinancialHealthModel(db_config)
        
        conn = health_model.connect_db()
        cursor = conn.cursor()
        
        query = "SELECT * FROM balance_sheet_1 WHERE company_id = %s AND year = %s"
        cursor.execute(query, (company_id, year))
        result = cursor.fetchone()
        
        if not result:
            cursor.close()
            conn.close()
            return jsonify({
                'success': False,
                'error': 'Balance sheet not found'
            }), 404
        
        column_names = [desc[0] for desc in cursor.description]
        balance_sheet = dict(zip(column_names, result))
        
        cursor.close()
        conn.close()
        
        # Get benchmark comparison
        bs_generator = BalanceSheetGenerator(db_config)
        benchmark_comparison = bs_generator.get_industry_benchmark_comparison(
            balance_sheet, industry
        )
        
        return jsonify({
            'success': True,
            'data': {
                'company_id': company_id,
                'year': year,
                'industry': industry,
                'benchmark_comparison': benchmark_comparison
            }
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error benchmarking balance sheet: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

# Error handlers
@balance_sheet_bp.errorhandler(400)
def bad_request(error):
    return jsonify({
        'success': False,
        'error': 'Bad request'
    }), 400

@balance_sheet_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Resource not found'
    }), 404

@balance_sheet_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500