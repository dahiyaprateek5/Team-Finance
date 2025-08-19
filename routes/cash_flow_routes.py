from flask import Blueprint, request, jsonify, current_app
import pandas as pd
import io
import os
import re
from datetime import datetime
from models.financial_health_model import FinancialHealthModel
from models.cash_flow_generator import CashFlowGenerator
from models.cash_flow_models import CashFlowPredictionModels

# Create Blueprint
cash_flow_bp = Blueprint('cash_flow', __name__)

def get_db_config():
    """Get database configuration from app config"""
    return {
        'host': current_app.config.get('DB_HOST', 'localhost'),
        'database': current_app.config.get('DB_NAME', 'financial_db'),
        'user': current_app.config.get('DB_USER', 'postgres'),
        'password': current_app.config.get('DB_PASSWORD', 'password'),
        'port': current_app.config.get('DB_PORT', 5432)
    }

@cash_flow_bp.route('/generate', methods=['POST'])
def generate_cash_flow():
    """
    Generate cash flow statement from balance sheet data
    
    Request JSON:
    {
        "company_id": "COMP001",
        "year": 2024,
        "company_name": "ABC Company",
        "industry": "technology"
    }
    
    Response:
    {
        "success": true,
        "message": "Cash flow statement generated successfully",
        "data": {
            "company_id": "COMP001",
            "year": 2024,
            "net_income": 50000,
            "net_cash_from_operating_activities": 75000,
            "free_cash_flow": 45000,
            "liquidation_label": 0,
            "cash_flow_statement": {...}
        }
    }
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['company_id', 'year']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        company_id = data['company_id']
        year = data['year']
        company_name = data.get('company_name', '')
        industry = data.get('industry', 'manufacturing')
        
        # Initialize cash flow generator
        db_config = get_db_config()
        cf_generator = CashFlowGenerator(db_config)
        
        # Generate cash flow from balance sheets
        cash_flow_data = cf_generator.calculate_cash_flow_from_balance_sheets(
            company_id=company_id,
            current_year=year,
            company_name=company_name,
            industry=industry
        )
        
        if not cash_flow_data:
            return jsonify({
                'success': False,
                'error': 'Could not generate cash flow. Balance sheet data may be missing.'
            }), 400
        
        # Save to cash_flow_statement table
        save_success = cf_generator.save_cash_flow_to_db(cash_flow_data)
        
        if not save_success:
            return jsonify({
                'success': False,
                'error': 'Failed to save cash flow to database'
            }), 500
        
        return jsonify({
            'success': True,
            'message': 'Cash flow statement generated successfully',
            'data': {
                'company_id': cash_flow_data['company_id'],
                'year': cash_flow_data['year'],
                'net_income': cash_flow_data['net_income'],
                'net_cash_from_operating_activities': cash_flow_data['net_cash_from_operating_activities'],
                'free_cash_flow': cash_flow_data['free_cash_flow'],
                'liquidation_label': cash_flow_data['liquidation_label'],
                'cash_flow_statement': cash_flow_data
            }
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error generating cash flow: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@cash_flow_bp.route('/process', methods=['POST'])
def process_cash_flow_file():
    """
    Process uploaded cash flow statement file and save to cash_flow_statement table
    
    Form Data:
    - file: Cash flow statement file (CSV, Excel)
    - company_id: Company identifier
    - company_name: Company name
    - industry: Company industry (optional)
    
    Response:
    {
        "success": true,
        "message": "Cash flow statement processed and saved successfully",
        "data": {
            "company_id": "COMP001",
            "year": 2024,
            "net_income": 50000,
            "net_cash_from_operating_activities": 75000,
            "free_cash_flow": 45000,
            "liquidation_label": 0
        }
    }
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
        
        # Initialize cash flow generator
        db_config = get_db_config()
        cf_generator = CashFlowGenerator(db_config)
        
        # Process uploaded cash flow statement
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
        
        # Save to cash_flow_statement table
        save_success = cf_generator.save_cash_flow_to_db(cash_flow_data)
        
        if not save_success:
            return jsonify({
                'success': False,
                'error': 'Failed to save to database'
            }), 500
        
        return jsonify({
            'success': True,
            'message': 'Cash flow statement processed and saved successfully',
            'data': {
                'company_id': cash_flow_data['company_id'],
                'year': cash_flow_data['year'],
                'net_income': cash_flow_data['net_income'],
                'net_cash_from_operating_activities': cash_flow_data['net_cash_from_operating_activities'],
                'free_cash_flow': cash_flow_data['free_cash_flow'],
                'liquidation_label': cash_flow_data['liquidation_label']
            }
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error processing cash flow: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@cash_flow_bp.route('/get/<company_id>/<int:year>', methods=['GET'])
def get_cash_flow(company_id, year):
    """
    Get cash flow statement for specific company and year from cash_flow_statement table
    
    URL Parameters:
    - company_id: Company identifier
    - year: Year (integer)
    
    Response:
    {
        "success": true,
        "data": {
            "company_id": "COMP001",
            "year": 2024,
            "net_income": 50000,
            "net_cash_from_operating_activities": 75000,
            ...
        }
    }
    """
    try:
        db_config = get_db_config()
        cf_generator = CashFlowGenerator(db_config)
        
        conn = cf_generator.connect_db()
        if not conn:
            return jsonify({
                'success': False,
                'error': 'Database connection failed'
            }), 500
        
        cursor = conn.cursor()
        
        # Query cash flow from cash_flow_statement table
        query = """
        SELECT * FROM cash_flow_statement 
        WHERE company_id = %s AND year = %s
        """
        
        cursor.execute(query, (company_id, year))
        result = cursor.fetchone()
        
        if not result:
            cursor.close()
            conn.close()
            return jsonify({
                'success': False,
                'error': 'Cash flow statement not found'
            }), 404
        
        # Get column names
        column_names = [desc[0] for desc in cursor.description]
        
        # Convert to dictionary
        cash_flow = dict(zip(column_names, result))
        
        # Convert datetime to ISO format
        if cash_flow.get('generated_at'):
            cash_flow['generated_at'] = cash_flow['generated_at'].isoformat()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'data': cash_flow
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error getting cash flow: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@cash_flow_bp.route('/list/<company_id>', methods=['GET'])
def list_cash_flows(company_id):
    """
    List all cash flow statements for a company from cash_flow_statement table
    
    URL Parameters:
    - company_id: Company identifier
    
    Query Parameters:
    - limit: Maximum number of records (default: 10)
    - offset: Number of records to skip (default: 0)
    
    Response:
    {
        "success": true,
        "data": [
            {
                "year": 2024,
                "generated_at": "2024-01-01T00:00:00",
                "company_name": "ABC Company",
                "industry": "technology",
                "net_income": 50000,
                ...
            }
        ],
        "count": 5
    }
    """
    try:
        # Get query parameters
        limit = int(request.args.get('limit', 10))
        offset = int(request.args.get('offset', 0))
        
        db_config = get_db_config()
        cf_generator = CashFlowGenerator(db_config)
        
        conn = cf_generator.connect_db()
        if not conn:
            return jsonify({
                'success': False,
                'error': 'Database connection failed'
            }), 500
        
        cursor = conn.cursor()
        
        # Query all cash flows for company
        query = """
        SELECT year, generated_at, company_name, industry, net_income,
               net_cash_from_operating_activities, free_cash_flow, liquidation_label
        FROM cash_flow_statement 
        WHERE company_id = %s 
        ORDER BY year DESC
        LIMIT %s OFFSET %s
        """
        
        cursor.execute(query, (company_id, limit, offset))
        results = cursor.fetchall()
        
        # Get total count
        count_query = "SELECT COUNT(*) FROM cash_flow_statement WHERE company_id = %s"
        cursor.execute(count_query, (company_id,))
        total_count = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        # Convert to list of dictionaries
        cash_flows = []
        for result in results:
            cash_flow = {
                'year': result[0],
                'generated_at': result[1].isoformat() if result[1] else None,
                'company_name': result[2],
                'industry': result[3],
                'net_income': result[4],
                'net_cash_from_operating_activities': result[5],
                'free_cash_flow': result[6],
                'liquidation_label': result[7]
            }
            cash_flows.append(cash_flow)
        
        return jsonify({
            'success': True,
            'data': cash_flows,
            'count': len(cash_flows),
            'total_count': total_count,
            'pagination': {
                'limit': limit,
                'offset': offset,
                'has_more': offset + len(cash_flows) < total_count
            }
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error listing cash flows: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@cash_flow_bp.route('/validate', methods=['POST'])
def validate_cash_flow():
    """
    Validate cash flow statement for consistency
    
    Request JSON:
    {
        "company_id": "COMP001",
        "year": 2024
    }
    
    Response:
    {
        "success": true,
        "validation": {
            "is_valid": true,
            "warnings": [],
            "errors": []
        }
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
        cf_generator = CashFlowGenerator(db_config)
        
        # Get cash flow from database
        conn = cf_generator.connect_db()
        cursor = conn.cursor()
        
        query = "SELECT * FROM cash_flow_statement WHERE company_id = %s AND year = %s"
        cursor.execute(query, (company_id, year))
        result = cursor.fetchone()
        
        if not result:
            cursor.close()
            conn.close()
            return jsonify({
                'success': False,
                'error': 'Cash flow statement not found'
            }), 404
        
        # Convert to dictionary
        column_names = [desc[0] for desc in cursor.description]
        cash_flow = dict(zip(column_names, result))
        
        cursor.close()
        conn.close()
        
        # Validate cash flow
        validation_results = cf_generator.validate_cash_flow_data(cash_flow)
        
        return jsonify({
            'success': True,
            'validation': validation_results
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error validating cash flow: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@cash_flow_bp.route('/bulk-generate', methods=['POST'])
def bulk_generate_cash_flows():
    """
    Generate cash flows for multiple companies
    
    Request JSON:
    {
        "year_range": [2022, 2023, 2024]
    }
    
    Response:
    {
        "success": true,
        "message": "Bulk cash flow generation completed",
        "results": {
            "total_companies": 10,
            "successful": 8,
            "failed": 2,
            "errors": []
        }
    }
    """
    try:
        data = request.get_json()
        year_range = data.get('year_range')
        
        if not year_range:
            # Default to last 3 years
            current_year = datetime.now().year
            year_range = [current_year - i for i in range(3)]
        
        db_config = get_db_config()
        cf_generator = CashFlowGenerator(db_config)
        
        # Bulk generate cash flows
        results = cf_generator.bulk_generate_cash_flows(year_range)
        
        return jsonify({
            'success': True,
            'message': 'Bulk cash flow generation completed',
            'results': results
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error in bulk generation: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@cash_flow_bp.route('/predict', methods=['POST'])
def predict_cash_flow():
    """
    Predict cash flow using ML models
    
    Request JSON:
    {
        "balance_sheet_data": {
            "total_assets": 1000000,
            "current_assets": 400000,
            "total_equity": 600000
        },
        "company_info": {
            "company_name": "ABC Company",
            "industry": "technology"
        }
    }
    
    Response:
    {
        "success": true,
        "message": "Cash flow predicted successfully",
        "data": {
            "company_id": "UNKNOWN",
            "year": 2024,
            "net_income": 50000,
            "net_cash_from_operating_activities": 75000,
            ...
        }
    }
    """
    try:
        data = request.get_json()
        balance_sheet_data = data.get('balance_sheet_data', {})
        company_info = data.get('company_info', {})
        
        if not balance_sheet_data:
            return jsonify({
                'success': False,
                'error': 'Missing balance_sheet_data'
            }), 400
        
        db_config = get_db_config()
        cf_models = CashFlowPredictionModels(db_config)
        
        # Predict complete cash flow
        predicted_cash_flow = cf_models.predict_complete_cash_flow(
            balance_sheet_data, company_info
        )
        
        return jsonify({
            'success': True,
            'message': 'Cash flow predicted successfully',
            'data': predicted_cash_flow
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error predicting cash flow: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@cash_flow_bp.route('/trends/<company_id>', methods=['GET'])
def get_cash_flow_trends(company_id):
    """
    Get cash flow trends for a company with optional future predictions
    
    URL Parameters:
    - company_id: Company identifier
    
    Query Parameters:
    - years_ahead: Number of years to predict (default: 0)
    
    Response:
    {
        "success": true,
        "data": {
            "company_id": "COMP001",
            "historical_trends": [...],
            "predictions": [...] (if years_ahead > 0)
        }
    }
    """
    try:
        # Get query parameters
        years_ahead = int(request.args.get('years_ahead', 0))
        
        db_config = get_db_config()
        
        if years_ahead > 0:
            # Use ML models for predictions
            cf_models = CashFlowPredictionModels(db_config)
            predictions = cf_models.predict_cash_flow_trends(company_id, years_ahead)
            
            if predictions:
                return jsonify({
                    'success': True,
                    'data': {
                        'company_id': company_id,
                        'predictions': predictions,
                        'years_ahead': years_ahead
                    }
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'error': 'Could not generate predictions'
                }), 400
        else:
            # Get historical data only
            cf_generator = CashFlowGenerator(db_config)
            conn = cf_generator.connect_db()
            cursor = conn.cursor()
            
            query = """
            SELECT year, net_income, net_cash_from_operating_activities, 
                   free_cash_flow, liquidation_label
            FROM cash_flow_statement 
            WHERE company_id = %s 
            ORDER BY year ASC
            """
            
            cursor.execute(query, (company_id,))
            results = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            trends = []
            for result in results:
                trend = {
                    'year': result[0],
                    'net_income': result[1],
                    'operating_cash_flow': result[2],
                    'free_cash_flow': result[3],
                    'liquidation_label': result[4]
                }
                trends.append(trend)
            
            return jsonify({
                'success': True,
                'data': {
                    'company_id': company_id,
                    'historical_trends': trends
                }
            }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error getting trends: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@cash_flow_bp.route('/benchmark/<company_id>/<int:year>', methods=['GET'])
def benchmark_cash_flow(company_id, year):
    """
    Benchmark company's cash flow against industry
    
    URL Parameters:
    - company_id: Company identifier
    - year: Year (integer)
    
    Query Parameters:
    - industry: Industry category (required)
    
    Response:
    {
        "success": true,
        "data": {
            "company_id": "COMP001",
            "industry": "technology",
            "year": 2024,
            "benchmarks": {...}
        }
    }
    """
    try:
        # Get industry from query params
        industry = request.args.get('industry')
        
        if not industry:
            return jsonify({
                'success': False,
                'error': 'Industry parameter required'
            }), 400
        
        db_config = get_db_config()
        cf_models = CashFlowPredictionModels(db_config)
        
        # Get benchmark comparison
        benchmark_results = cf_models.benchmark_against_industry(company_id, industry)
        
        if benchmark_results and 'error' not in benchmark_results:
            return jsonify({
                'success': True,
                'data': benchmark_results
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': benchmark_results.get('error', 'Benchmarking failed')
            }), 400
        
    except Exception as e:
        current_app.logger.error(f"Error benchmarking cash flow: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@cash_flow_bp.route('/health-score/<company_id>/<int:year>', methods=['GET'])
def get_cash_flow_health_score(company_id, year):
    """
    Calculate cash flow health score for a company
    
    URL Parameters:
    - company_id: Company identifier
    - year: Year (integer)
    
    Response:
    {
        "success": true,
        "data": {
            "company_id": "COMP001",
            "year": 2024,
            "health_score": {
                "total_score": 75,
                "components": {...},
                "risk_level": "low"
            },
            "cash_flow_summary": {...}
        }
    }
    """
    try:
        db_config = get_db_config()
        health_model = FinancialHealthModel(db_config)
        
        # Get cash flow data
        conn = health_model.connect_db()
        cursor = conn.cursor()
        
        query = "SELECT * FROM cash_flow_statement WHERE company_id = %s AND year = %s"
        cursor.execute(query, (company_id, year))
        result = cursor.fetchone()
        
        if not result:
            cursor.close()
            conn.close()
            return jsonify({
                'success': False,
                'error': 'Cash flow data not found'
            }), 404
        
        column_names = [desc[0] for desc in cursor.description]
        cash_flow = dict(zip(column_names, result))
        
        cursor.close()
        conn.close()
        
        # Calculate health score
        health_score = {
            'overall_score': 0,
            'components': {},
            'risk_level': 'unknown'
        }
        
        # Operating Cash Flow Score (40 points)
        ocf = cash_flow.get('net_cash_from_operating_activities', 0) or 0
        if ocf > 0:
            health_score['components']['operating_cash_flow'] = 40
        elif ocf > -0.05 * abs(cash_flow.get('net_income', 1)):
            health_score['components']['operating_cash_flow'] = 20
        else:
            health_score['components']['operating_cash_flow'] = 0
        
        # Free Cash Flow Score (30 points)
        fcf = cash_flow.get('free_cash_flow', 0) or 0
        if fcf > 0:
            health_score['components']['free_cash_flow'] = 30
        elif fcf > -0.1 * abs(cash_flow.get('net_income', 1)):
            health_score['components']['free_cash_flow'] = 15
        else:
            health_score['components']['free_cash_flow'] = 0
        
        # Profitability Score (30 points)
        net_income = cash_flow.get('net_income', 0) or 0
        if net_income > 0:
            health_score['components']['profitability'] = 30
        elif net_income > -0.05 * abs(ocf or 1):
            health_score['components']['profitability'] = 15
        else:
            health_score['components']['profitability'] = 0
        
        # Calculate overall score
        health_score['overall_score'] = sum(health_score['components'].values())
        
        # Determine risk level
        if health_score['overall_score'] >= 80:
            health_score['risk_level'] = 'low'
        elif health_score['overall_score'] >= 50:
            health_score['risk_level'] = 'medium'
        else:
            health_score['risk_level'] = 'high'
        
        # Add liquidation warning
        liquidation_label = cash_flow.get('liquidation_label', 0)
        if liquidation_label == 1:
            health_score['risk_level'] = 'critical'
            health_score['warning'] = 'Company shows signs of financial distress'
        
        return jsonify({
            'success': True,
            'data': {
                'company_id': company_id,
                'year': year,
                'health_score': health_score,
                'cash_flow_summary': {
                    'net_income': net_income,
                    'operating_cash_flow': ocf,
                    'free_cash_flow': fcf,
                    'liquidation_label': liquidation_label
                }
            }
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error calculating health score: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@cash_flow_bp.route('/compare', methods=['POST'])
def compare_cash_flows():
    """
    Compare cash flows between multiple companies or years
    
    Request JSON:
    {
        "comparisons": [
            {"company_id": "COMP001", "year": 2024},
            {"company_id": "COMP002", "year": 2024}
        ]
    }
    
    Response:
    {
        "success": true,
        "data": {
            "comparison_results": [...],
            "metrics": {...}
        }
    }
    """
    try:
        data = request.get_json()
        comparisons = data.get('comparisons', [])
        
        if len(comparisons) < 2:
            return jsonify({
                'success': False,
                'error': 'At least 2 companies/years required for comparison'
            }), 400
        
        db_config = get_db_config()
        cf_generator = CashFlowGenerator(db_config)
        
        conn = cf_generator.connect_db()
        cursor = conn.cursor()
        
        comparison_results = []
        
        for comp in comparisons:
            company_id = comp.get('company_id')
            year = comp.get('year')
            
            if not company_id or not year:
                continue
            
            query = """
            SELECT company_id, year, company_name, net_income,
                   net_cash_from_operating_activities, free_cash_flow,
                   liquidation_label
            FROM cash_flow_statement 
            WHERE company_id = %s AND year = %s
            """
            
            cursor.execute(query, (company_id, year))
            result = cursor.fetchone()
            
            if result:
                comparison_results.append({
                    'company_id': result[0],
                    'year': result[1],
                    'company_name': result[2],
                    'net_income': result[3],
                    'operating_cash_flow': result[4],
                    'free_cash_flow': result[5],
                    'liquidation_label': result[6]
                })
        
        cursor.close()
        conn.close()
        
        # Calculate comparison metrics
        if comparison_results:
            net_incomes = [r['net_income'] for r in comparison_results if r['net_income']]
            ocfs = [r['operating_cash_flow'] for r in comparison_results if r['operating_cash_flow']]
            fcfs = [r['free_cash_flow'] for r in comparison_results if r['free_cash_flow']]
            
            metrics = {
                'net_income': {
                    'average': sum(net_incomes) / len(net_incomes) if net_incomes else 0,
                    'highest': max(net_incomes) if net_incomes else 0,
                    'lowest': min(net_incomes) if net_incomes else 0
                },
                'operating_cash_flow': {
                    'average': sum(ocfs) / len(ocfs) if ocfs else 0,
                    'highest': max(ocfs) if ocfs else 0,
                    'lowest': min(ocfs) if ocfs else 0
                },
                'free_cash_flow': {
                    'average': sum(fcfs) / len(fcfs) if fcfs else 0,
                    'highest': max(fcfs) if fcfs else 0,
                    'lowest': min(fcfs) if fcfs else 0
                },
                'at_risk_count': sum(1 for r in comparison_results if r['liquidation_label'] == 1)
            }
        else:
            metrics = {}
        
        return jsonify({
            'success': True,
            'data': {
                'comparison_results': comparison_results,
                'metrics': metrics,
                'total_compared': len(comparison_results)
            }
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error comparing cash flows: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@cash_flow_bp.route('/export/<company_id>/<int:year>', methods=['GET'])
def export_cash_flow(company_id, year):
    """
    Export cash flow statement to various formats
    
    URL Parameters:
    - company_id: Company identifier
    - year: Year (integer)
    
    Query Parameters:
    - format: Export format (json, csv, excel) - default: json
    
    Response:
    {
        "success": true,
        "data": {...} or file download
    }
    """
    try:
        export_format = request.args.get('format', 'json').lower()
        
        if export_format not in ['json', 'csv', 'excel']:
            return jsonify({
                'success': False,
                'error': 'Invalid format. Supported: json, csv, excel'
            }), 400
        
        db_config = get_db_config()
        cf_generator = CashFlowGenerator(db_config)
        
        conn = cf_generator.connect_db()
        cursor = conn.cursor()
        
        query = "SELECT * FROM cash_flow_statement WHERE company_id = %s AND year = %s"
        cursor.execute(query, (company_id, year))
        result = cursor.fetchone()
        
        if not result:
            cursor.close()
            conn.close()
            return jsonify({
                'success': False,
                'error': 'Cash flow statement not found'
            }), 404
        
        column_names = [desc[0] for desc in cursor.description]
        cash_flow = dict(zip(column_names, result))
        
        cursor.close()
        conn.close()
        
        # Convert datetime to string for JSON serialization
        if cash_flow.get('generated_at'):
            cash_flow['generated_at'] = cash_flow['generated_at'].isoformat()
        
        if export_format == 'json':
            return jsonify({
                'success': True,
                'data': cash_flow
            }), 200
        
        elif export_format == 'csv':
            # Create CSV response
            from flask import make_response
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow(cash_flow.keys())
            # Write data
            writer.writerow(cash_flow.values())
            
            response = make_response(output.getvalue())
            response.headers['Content-Type'] = 'text/csv'
            response.headers['Content-Disposition'] = f'attachment; filename=cash_flow_{company_id}_{year}.csv'
            
            return response
        
        elif export_format == 'excel':
            # Create Excel response
            from flask import make_response
            import io
            
            output = io.BytesIO()
            df = pd.DataFrame([cash_flow])
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Cash Flow', index=False)
            
            output.seek(0)
            
            response = make_response(output.read())
            response.headers['Content-Type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            response.headers['Content-Disposition'] = f'attachment; filename=cash_flow_{company_id}_{year}.xlsx'
            
            return response
        
    except Exception as e:
        current_app.logger.error(f"Error exporting cash flow: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@cash_flow_bp.route('/search', methods=['GET'])
def search_cash_flows():
    """
    Search cash flow statements with filters
    
    Query Parameters:
    - company_name: Filter by company name (partial match)
    - industry: Filter by industry
    - year_from: Filter by year range (start)
    - year_to: Filter by year range (end)
    - min_income: Minimum net income
    - max_income: Maximum net income
    - liquidation_risk: Filter by liquidation label (0 or 1)
    - limit: Maximum number of results (default: 20)
    - offset: Number of records to skip (default: 0)
    
    Response:
    {
        "success": true,
        "data": [...],
        "count": 15,
        "total_count": 150,
        "filters_applied": {...}
    }
    """
    try:
        # Get query parameters
        company_name = request.args.get('company_name')
        industry = request.args.get('industry')
        year_from = request.args.get('year_from', type=int)
        year_to = request.args.get('year_to', type=int)
        min_income = request.args.get('min_income', type=float)
        max_income = request.args.get('max_income', type=float)
        liquidation_risk = request.args.get('liquidation_risk', type=int)
        limit = int(request.args.get('limit', 20))
        offset = int(request.args.get('offset', 0))
        
        db_config = get_db_config()
        cf_generator = CashFlowGenerator(db_config)
        
        conn = cf_generator.connect_db()
        cursor = conn.cursor()
        
        # Build dynamic query
        where_conditions = []
        params = []
        
        if company_name:
            where_conditions.append("LOWER(company_name) LIKE LOWER(%s)")
            params.append(f'%{company_name}%')
        
        if industry:
            where_conditions.append("industry = %s")
            params.append(industry)
        
        if year_from:
            where_conditions.append("year >= %s")
            params.append(year_from)
        
        if year_to:
            where_conditions.append("year <= %s")
            params.append(year_to)
        
        if min_income is not None:
            where_conditions.append("net_income >= %s")
            params.append(min_income)
        
        if max_income is not None:
            where_conditions.append("net_income <= %s")
            params.append(max_income)
        
        if liquidation_risk is not None:
            where_conditions.append("liquidation_label = %s")
            params.append(liquidation_risk)
        
        # Construct WHERE clause
        where_clause = ""
        if where_conditions:
            where_clause = "WHERE " + " AND ".join(where_conditions)
        
        # Main query
        query = f"""
        SELECT company_id, year, company_name, industry, net_income,
               net_cash_from_operating_activities, free_cash_flow, 
               liquidation_label, generated_at
        FROM cash_flow_statement 
        {where_clause}
        ORDER BY year DESC, company_name
        LIMIT %s OFFSET %s
        """
        
        cursor.execute(query, params + [limit, offset])
        results = cursor.fetchall()
        
        # Count query
        count_query = f"""
        SELECT COUNT(*) FROM cash_flow_statement {where_clause}
        """
        
        cursor.execute(count_query, params)
        total_count = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        # Convert results
        cash_flows = []
        for result in results:
            cash_flow = {
                'company_id': result[0],
                'year': result[1],
                'company_name': result[2],
                'industry': result[3],
                'net_income': result[4],
                'net_cash_from_operating_activities': result[5],
                'free_cash_flow': result[6],
                'liquidation_label': result[7],
                'generated_at': result[8].isoformat() if result[8] else None
            }
            cash_flows.append(cash_flow)
        
        # Track applied filters
        filters_applied = {}
        if company_name:
            filters_applied['company_name'] = company_name
        if industry:
            filters_applied['industry'] = industry
        if year_from:
            filters_applied['year_from'] = year_from
        if year_to:
            filters_applied['year_to'] = year_to
        if min_income is not None:
            filters_applied['min_income'] = min_income
        if max_income is not None:
            filters_applied['max_income'] = max_income
        if liquidation_risk is not None:
            filters_applied['liquidation_risk'] = liquidation_risk
        
        return jsonify({
            'success': True,
            'data': cash_flows,
            'count': len(cash_flows),
            'total_count': total_count,
            'filters_applied': filters_applied,
            'pagination': {
                'limit': limit,
                'offset': offset,
                'has_more': offset + len(cash_flows) < total_count
            }
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error searching cash flows: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@cash_flow_bp.route('/analytics/summary', methods=['GET'])
def get_analytics_summary():
    """
    Get analytics summary for all cash flow data
    
    Query Parameters:
    - industry: Filter by industry (optional)
    - year: Filter by specific year (optional)
    
    Response:
    {
        "success": true,
        "data": {
            "total_companies": 150,
            "total_records": 450,
            "industry_breakdown": {...},
            "year_breakdown": {...},
            "financial_metrics": {...}
        }
    }
    """
    try:
        industry_filter = request.args.get('industry')
        year_filter = request.args.get('year', type=int)
        
        db_config = get_db_config()
        cf_generator = CashFlowGenerator(db_config)
        
        conn = cf_generator.connect_db()
        cursor = conn.cursor()
        
        # Build WHERE clause for filters
        where_conditions = []
        params = []
        
        if industry_filter:
            where_conditions.append("industry = %s")
            params.append(industry_filter)
        
        if year_filter:
            where_conditions.append("year = %s")
            params.append(year_filter)
        
        where_clause = ""
        if where_conditions:
            where_clause = "WHERE " + " AND ".join(where_conditions)
        
        # Total companies and records
        cursor.execute(f"SELECT COUNT(DISTINCT company_id) FROM cash_flow_statement {where_clause}", params)
        total_companies = cursor.fetchone()[0]
        
        cursor.execute(f"SELECT COUNT(*) FROM cash_flow_statement {where_clause}", params)
        total_records = cursor.fetchone()[0]
        
        # Industry breakdown
        cursor.execute(f"""
            SELECT industry, COUNT(*) as count 
            FROM cash_flow_statement {where_clause}
            GROUP BY industry 
            ORDER BY count DESC
        """, params)
        industry_breakdown = dict(cursor.fetchall())
        
        # Year breakdown
        cursor.execute(f"""
            SELECT year, COUNT(*) as count 
            FROM cash_flow_statement {where_clause}
            GROUP BY year 
            ORDER BY year DESC
        """, params)
        year_breakdown = dict(cursor.fetchall())
        
        # Financial metrics
        cursor.execute(f"""
            SELECT 
                AVG(net_income) as avg_net_income,
                AVG(net_cash_from_operating_activities) as avg_ocf,
                AVG(free_cash_flow) as avg_fcf,
                COUNT(CASE WHEN liquidation_label = 1 THEN 1 END) as at_risk_count
            FROM cash_flow_statement {where_clause}
        """, params)
        
        metrics_result = cursor.fetchone()
        financial_metrics = {
            'average_net_income': metrics_result[0] or 0,
            'average_operating_cash_flow': metrics_result[1] or 0,
            'average_free_cash_flow': metrics_result[2] or 0,
            'companies_at_risk': metrics_result[3] or 0,
            'risk_percentage': (metrics_result[3] / total_companies * 100) if total_companies > 0 else 0
        }
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'data': {
                'total_companies': total_companies,
                'total_records': total_records,
                'industry_breakdown': industry_breakdown,
                'year_breakdown': year_breakdown,
                'financial_metrics': financial_metrics,
                'filters_applied': {
                    'industry': industry_filter,
                    'year': year_filter
                }
            }
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error getting analytics summary: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

# Error handlers
@cash_flow_bp.errorhandler(400)
def bad_request(error):
    return jsonify({
        'success': False,
        'error': 'Bad request'
    }), 400

@cash_flow_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Resource not found'
    }), 404

@cash_flow_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500