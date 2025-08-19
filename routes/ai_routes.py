from flask import Blueprint, render_template, jsonify, request # type: ignore
from datetime import datetime
from utils.helpers import calculate_cash_flow_health_score, safe_float, safe_int # type: ignore
from config.database import db_conn
from services.ai_service import get_chatgpt_recommendations_service # type: ignore
from utils.ml_predictor import ml_predictor # type: ignore

ai_bp = Blueprint('ai', __name__)

@ai_bp.route('/insights')
def ai_insights_panel():
    """Main AI Insights Panel Route"""
    return render_template('ai_insights_panel.html')

@ai_bp.route('/financial-data')
def get_financial_data():
    """Get current financial data for AI insights"""
    try:
        query = """
        SELECT company_name, industry, net_income, 
               net_cash_from_operating_activities, free_cash_flow,
               liquidation_label, year, debt_to_equity_ratio,
               ocf_to_net_income_ratio, capital_expenditures, generated_at
        FROM cash_flow_statement 
        ORDER BY generated_at DESC 
        LIMIT 1
        """
        
        result = db_conn.execute_query(query)
        
        if not result:
            return jsonify({
                'success': False,
                'error': 'No financial data available'
            })
        
        data = result[0]
        health_score = calculate_cash_flow_health_score(data)
        
        # Determine risk level and trends
        if health_score >= 75:
            risk_level = 'low'
            trend_direction = 'up'
            trend = 'improving'
            trend_class = 'improving'
        elif health_score >= 50:
            risk_level = 'medium'
            trend_direction = 'right'
            trend = 'stable'
            trend_class = 'stable'
        else:
            risk_level = 'high'
            trend_direction = 'down'
            trend = 'declining'
            trend_class = 'declining'
        
        analysis_time = 3.2  # Simulated analysis time
        
        return jsonify({
            'success': True,
            'company_name': data.get('company_name', 'Unknown Company'),
            'health_score': health_score,
            'risk_level': risk_level,
            'trend': trend,
            'trend_direction': trend_direction,
            'trend_class': trend_class,
            'data_points': 247,
            'analysis_time': analysis_time,
            'metrics': {
                'liquidity': min(100, max(0, safe_float(data.get('net_cash_from_operating_activities', 0)) / 1000000 * 20)),
                'profitability': min(100, max(0, safe_float(data.get('net_income', 0)) / 1000000 * 25)),
                'debt': max(0, min(100, 100 - safe_float(data.get('debt_to_equity_ratio', 1)) * 20)),
                'stability': 100 - (safe_int(data.get('liquidation_label', 0)) * 50)
            },
            'alerts': [] if health_score >= 60 else [
                {
                    'type': 'warning',
                    'icon': 'fa-exclamation-triangle',
                    'message': 'Financial health score below 60 - monitor closely'
                }
            ]
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Database error: {str(e)}'
        }), 500

@ai_bp.route('/chat-gpt-recommendations', methods=['POST'])
def get_chatgpt_recommendations():
    """Get AI recommendations from ChatGPT"""
    return get_chatgpt_recommendations_service(request)

@ai_bp.route('/ml-analysis/<company_name>')
def get_ml_company_analysis(company_name):
    """Get ML-powered financial analysis for a company"""
    try:
        query = """
        SELECT * FROM cash_flow_statement 
        WHERE company_name = %s 
        ORDER BY year DESC 
        LIMIT 1
        """
        result = db_conn.execute_query(query, (company_name,))
        
        if not result:
            return jsonify({'success': False, 'error': 'Company not found'}), 404
        
        cash_flow_data = result[0]
        
        # Get ML prediction
        if ml_predictor and ml_predictor.is_trained:
            ml_analysis = ml_predictor.predict_financial_health(cash_flow_data)
        else:
            ml_analysis = {
                'ml_ensemble_health_score': 50,
                'ml_ensemble_risk_category': 'Medium Risk',
                'ml_prediction_confidence': 0,
                'ml_models_used': ['None - Install ML libraries'],
                'ml_recommendation': 'ML libraries not available'
            }
        
        traditional_health_score = calculate_cash_flow_health_score(cash_flow_data)
        
        return jsonify({
            'success': True,
            'company_name': company_name,
            'ml_analysis': ml_analysis,
            'traditional_health_score': traditional_health_score,
            'financial_data': {
                'net_income': safe_float(cash_flow_data.get('net_income')),
                'operating_cash_flow': safe_float(cash_flow_data.get('net_cash_from_operating_activities')),
                'free_cash_flow': safe_float(cash_flow_data.get('free_cash_flow')),
                'liquidation_label': safe_int(cash_flow_data.get('liquidation_label')),
                'year': cash_flow_data.get('year')
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False, 
            'error': f'Analysis error: {str(e)}'
        }), 500