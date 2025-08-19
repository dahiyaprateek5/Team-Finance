from flask import Blueprint, jsonify, request 
from datetime import datetime
from utils.helpers import (
    calculate_cash_flow_health_score, 
    safe_float, 
    safe_int,
    format_currency,
    get_cash_flow_status_from_amount,
    get_profitability_status_from_amount,
    map_industry_to_sector
)
from config.database import db_conn
from utils.ml_predictor import ml_predictor

api_bp = Blueprint('api', __name__)

@api_bp.route('/cash-flow/companies', methods=['GET'])
def get_cash_flow_companies():
    """API endpoint to fetch companies from cash_flow_statement table"""
    try:
        print("🔍 Fetching companies from cash_flow_statement table...")  # Debug log
        
        companies_query = """
        SELECT DISTINCT company_name, industry, year, net_income,
               net_cash_from_operating_activities, free_cash_flow,
               liquidation_label, debt_to_equity_ratio, generated_at
        FROM cash_flow_statement
        WHERE company_name IS NOT NULL
        ORDER BY company_name, year DESC
        """
        
        cash_flow_data = db_conn.execute_query(companies_query)
        print(f"📊 Found {len(cash_flow_data) if cash_flow_data else 0} records")  # Debug log
        
        if not cash_flow_data:
            return jsonify({
                'success': True,
                'companies': [],
                'message': 'No companies found'
            })
        
        # Process and group by company
        companies_dict = {}
        for row in cash_flow_data:
            company_name = row['company_name']
            if company_name not in companies_dict:
                companies_dict[company_name] = row
            elif row.get('year') and companies_dict[company_name].get('year'):
                if row['year'] > companies_dict[company_name]['year']:
                    companies_dict[company_name] = row
        
        print(f"📈 Processed {len(companies_dict)} unique companies")  # Debug log
        
        # Train ML models if enough data
        if len(companies_dict) >= 10 and ml_predictor:
            companies_list = list(companies_dict.values())
            ml_predictor.train_models(companies_list)
        
        # Convert to frontend format
        companies = []
        for company_name, data in companies_dict.items():
            if ml_predictor and ml_predictor.is_trained:
                ml_analysis = ml_predictor.predict_financial_health(data)
                health_score = ml_analysis['ml_ensemble_health_score']
                risk_category = ml_analysis['ml_ensemble_risk_category']
                ml_models_used = ml_analysis['ml_models_used']
            else:
                health_score = calculate_cash_flow_health_score(data)
                risk_category = 'Medium Risk'
                ml_models_used = ['Traditional Analysis']
            
            ocf = data.get('net_cash_from_operating_activities') or 0
            cash_flow_status = get_cash_flow_status_from_amount(ocf)
            
            net_income = data.get('net_income') or 0
            profitability_status = get_profitability_status_from_amount(net_income)
            
            company = {
                'company_name': company_name,
                'industry': data.get('industry') or 'General',
                'sector': map_industry_to_sector(data.get('industry')),
                'year': data.get('year') or 2024,
                'net_income': safe_float(data.get('net_income')),
                'net_cash_from_operating_activities': safe_float(data.get('net_cash_from_operating_activities')),
                'free_cash_flow': safe_float(data.get('free_cash_flow')),
                'financial_health_score': health_score,
                'cash_flow_status': cash_flow_status,
                'profitability_status': profitability_status,
                'risk_category': risk_category,
                'ml_models_used': ml_models_used,
                'last_analysis_date': data.get('generated_at'),
                'data_source': 'cash_flow_statement'
            }
            companies.append(company)
        
        print(f"✅ Returning {len(companies)} companies to frontend")  # Debug log
        
        return jsonify({
            'success': True,
            'companies': companies,
            'total_count': len(companies),
            'ml_powered': ml_predictor.is_trained if ml_predictor else False,
            'last_updated': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Error in get_cash_flow_companies: {str(e)}")  # Debug log
        return jsonify({
            'success': False,
            'error': f'Database error: {str(e)}',
            'companies': [],
            'total_count': 0
        }), 500

@api_bp.route('/cash-flow/<company_name>', methods=['GET'])
def get_cash_flow_data(company_name):
    """Get detailed cash flow data for a specific company"""
    try:
        print(f"🔍 Fetching cash flow data for company: {company_name}")  # Debug log
        
        query = """
        SELECT * FROM cash_flow_statement 
        WHERE company_name = %s 
        ORDER BY year DESC
        """
        
        cash_flow_data = db_conn.execute_query(query, (company_name,))
        print(f"📊 Found {len(cash_flow_data) if cash_flow_data else 0} records for {company_name}")  # Debug log
        
        if not cash_flow_data:
            return jsonify({
                'success': False, 
                'error': f'No cash flow data found for company: {company_name}',
                'data': []
            }), 404
        
        # Format the data
        formatted_data = []
        for row in cash_flow_data:
            formatted_row = dict(row)
            for key, value in formatted_row.items():
                if hasattr(value, 'quantize'):  # Decimal type
                    formatted_row[key] = float(value)
            formatted_data.append(formatted_row)
        
        print(f"✅ Returning {len(formatted_data)} years of data for {company_name}")  # Debug log
        
        return jsonify({
            'success': True, 
            'data': formatted_data,
            'company_name': company_name,
            'total_years': len(formatted_data)
        })
        
    except Exception as e:
        print(f"❌ Error in get_cash_flow_data for {company_name}: {str(e)}")  # Debug log
        return jsonify({
            'success': False, 
            'error': f'Database error: {str(e)}',
            'data': []
        }), 500

# Alternative endpoints with /api prefix for backwards compatibility
@api_bp.route('/api/cash-flow/companies', methods=['GET'])
def get_cash_flow_companies_api():
    """Alternative endpoint with /api prefix"""
    return get_cash_flow_companies()

@api_bp.route('/api/cash-flow/<company_name>', methods=['GET'])
def get_cash_flow_data_api(company_name):
    """Alternative endpoint with /api prefix"""
    return get_cash_flow_data(company_name)