import json
import pandas as pd
import io
import base64
from datetime import datetime, timedelta
from flask import Blueprint, jsonify, request, send_file
from .financial_analyzer import FinancialAnalyzer
from .ai_insights import AIInsights
from .chart_generator import ChartGenerator
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint for dashboard routes
dashboard_bp = Blueprint('dashboard', __name__)

# Database configuration (would come from app config)
def get_db_config():
    return {
        'host': 'localhost',
        'database': 'financial_db',
        'user': 'postgres',
        'password': 'Prateek@2003',
        'port': 5432
    }

@dashboard_bp.route('/executive/<company_id>')
def executive_dashboard(company_id):
    """
    Generate executive dashboard for a company
    
    Args:
        company_id (str): Company identifier
        
    Returns:
        JSON: Dashboard data with charts and KPIs
    """
    try:
        db_config = get_db_config()
        analyzer = FinancialAnalyzer(db_config)
        chart_gen = ChartGenerator(db_config)
        
        # Get comprehensive analysis
        analysis = analyzer.analyze_company_performance(company_id)
        if 'error' in analysis:
            return jsonify({'error': 'Unable to generate dashboard data'}), 400
        
        # Create dashboard components
        dashboard_data = {
            'dashboard_id': f"exec_{company_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'company_id': company_id,
            'generated_at': datetime.now().isoformat(),
            'summary': create_executive_summary(analysis),
            'kpis': create_executive_kpis(analysis),
            'charts': create_executive_charts(company_id, analysis, chart_gen),
            'alerts': generate_executive_alerts(analysis),
            'recommendations': format_executive_recommendations(analysis)
        }
        
        return jsonify({
            'success': True,
            'dashboard': dashboard_data
        })
        
    except Exception as e:
        logger.error(f"Executive dashboard error: {e}")
        return jsonify({'error': f'Dashboard generation failed: {str(e)}'}), 500

@dashboard_bp.route('/operational/<company_id>')
def operational_dashboard(company_id):
    """
    Generate operational dashboard for detailed analysis
    
    Args:
        company_id (str): Company identifier
        
    Returns:
        JSON: Operational dashboard data
    """
    try:
        db_config = get_db_config()
        analyzer = FinancialAnalyzer(db_config)
        chart_gen = ChartGenerator(db_config)
        
        # Get analysis data
        analysis = analyzer.analyze_company_performance(company_id)
        if 'error' in analysis:
            return jsonify({'error': 'Unable to generate operational dashboard'}), 400
        
        # Create operational dashboard
        dashboard_data = {
            'dashboard_id': f"ops_{company_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'company_id': company_id,
            'type': 'operational',
            'generated_at': datetime.now().isoformat(),
            'metrics': create_operational_metrics(analysis),
            'efficiency_analysis': create_efficiency_analysis(analysis),
            'working_capital': create_working_capital_analysis(company_id, analyzer),
            'charts': create_operational_charts(company_id, analysis, chart_gen),
            'trends': create_operational_trends(company_id, analyzer),
            'actionable_insights': generate_operational_insights(analysis)
        }
        
        return jsonify({
            'success': True,
            'dashboard': dashboard_data
        })
        
    except Exception as e:
        logger.error(f"Operational dashboard error: {e}")
        return jsonify({'error': f'Operational dashboard generation failed: {str(e)}'}), 500

@dashboard_bp.route('/risk/<company_id>')
def risk_dashboard(company_id):
    """
    Generate risk-focused dashboard
    
    Args:
        company_id (str): Company identifier
        
    Returns:
        JSON: Risk dashboard data
    """
    try:
        db_config = get_db_config()
        analyzer = FinancialAnalyzer(db_config)
        ai_insights = AIInsights(db_config)
        
        # Get risk analysis
        analysis = analyzer.analyze_company_performance(company_id)
        if 'error' in analysis:
            return jsonify({'error': 'Unable to generate risk dashboard'}), 400
        
        # Get AI-powered risk insights
        risk_insights = ai_insights.generate_risk_insights(company_id)
        predictions = analyzer.predict_financial_health(company_id, 12)
        
        # Create risk dashboard
        dashboard_data = {
            'dashboard_id': f"risk_{company_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'company_id': company_id,
            'type': 'risk_assessment',
            'generated_at': datetime.now().isoformat(),
            'risk_summary': create_risk_summary(analysis),
            'risk_factors': analysis['risk_assessment']['risk_factors'],
            'risk_levels': analysis['risk_assessment']['risk_levels'],
            'predictions': predictions,
            'ai_insights': risk_insights,
            'scenarios': create_risk_scenarios(analysis, predictions),
            'charts': create_risk_charts(company_id, analysis),
            'mitigation_strategies': generate_risk_mitigation(analysis)
        }
        
        return jsonify({
            'success': True,
            'dashboard': dashboard_data
        })
        
    except Exception as e:
        logger.error(f"Risk dashboard error: {e}")
        return jsonify({'error': f'Risk dashboard generation failed: {str(e)}'}), 500

@dashboard_bp.route('/industry-benchmark/<industry>')
def industry_benchmark_dashboard(industry):
    """
    Generate industry benchmarking dashboard
    
    Args:
        industry (str): Industry name
        
    Returns:
        JSON: Industry benchmark dashboard
    """
    try:
        db_config = get_db_config()
        analyzer = FinancialAnalyzer(db_config)
        
        # Get industry benchmark
        benchmark = analyzer.generate_industry_benchmark(industry)
        if 'error' in benchmark:
            return jsonify({'error': f'Unable to generate benchmark for {industry}'}), 400
        
        # Create benchmark dashboard
        dashboard_data = {
            'dashboard_id': f"benchmark_{industry}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'industry': industry,
            'type': 'industry_benchmark',
            'generated_at': datetime.now().isoformat(),
            'industry_overview': create_industry_overview(benchmark),
            'performance_distribution': create_performance_distribution(benchmark),
            'top_performers': benchmark['top_performers'],
            'industry_averages': benchmark['averages'],
            'percentiles': benchmark['percentiles'],
            'charts': create_benchmark_charts(benchmark, industry),
            'insights': generate_industry_insights(benchmark)
        }
        
        return jsonify({
            'success': True,
            'dashboard': dashboard_data
        })
        
    except Exception as e:
        logger.error(f"Industry benchmark error: {e}")
        return jsonify({'error': f'Industry benchmark generation failed: {str(e)}'}), 500

@dashboard_bp.route('/comparison')
def comparison_dashboard():
    """
    Generate company comparison dashboard
    
    Query Parameters:
        companies: Comma-separated list of company IDs
        
    Returns:
        JSON: Comparison dashboard data
    """
    try:
        companies = request.args.get('companies', '').split(',')
        if len(companies) < 2:
            return jsonify({'error': 'At least 2 companies required for comparison'}), 400
        
        db_config = get_db_config()
        analyzer = FinancialAnalyzer(db_config)
        
        # Get comparison analysis
        comparison = analyzer.compare_companies(companies)
        if 'error' in comparison:
            return jsonify({'error': 'Unable to generate comparison'}), 400
        
        # Create comparison dashboard
        dashboard_data = {
            'dashboard_id': f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'companies': companies,
            'type': 'comparison',
            'generated_at': datetime.now().isoformat(),
            'comparison_summary': create_comparison_summary(comparison),
            'performance_rankings': comparison['rankings'],
            'metric_comparisons': comparison['comparison_metrics'],
            'charts': create_comparison_charts(comparison),
            'competitive_analysis': generate_competitive_analysis(comparison),
            'investment_recommendations': generate_investment_recommendations(comparison)
        }
        
        return jsonify({
            'success': True,
            'dashboard': dashboard_data
        })
        
    except Exception as e:
        logger.error(f"Comparison dashboard error: {e}")
        return jsonify({'error': f'Comparison dashboard generation failed: {str(e)}'}), 500

@dashboard_bp.route('/charts/<chart_type>/<company_id>')
def generate_chart(chart_type, company_id):
    """
    Generate specific chart for dashboard
    
    Args:
        chart_type (str): Type of chart to generate
        company_id (str): Company identifier
        
    Returns:
        JSON: Chart configuration
    """
    try:
        db_config = get_db_config()
        analyzer = FinancialAnalyzer(db_config)
        chart_gen = ChartGenerator(db_config)
        
        # Get company analysis
        analysis = analyzer.analyze_company_performance(company_id)
        if 'error' in analysis:
            return jsonify({'error': 'Unable to generate chart'}), 400
        
        # Generate specific chart
        chart_config = None
        
        if chart_type == 'financial_health_radar':
            chart_config = create_health_radar_chart(analysis)
        elif chart_type == 'trend_analysis':
            chart_config = chart_gen.create_trend_chart(company_id)
        elif chart_type == 'ratio_comparison':
            chart_config = create_ratio_comparison_chart(analysis)
        elif chart_type == 'cash_flow_waterfall':
            chart_config = create_cash_flow_waterfall(company_id, analyzer)
        elif chart_type == 'risk_assessment':
            chart_config = create_risk_assessment_chart(analysis)
        else:
            return jsonify({'error': f'Unknown chart type: {chart_type}'}), 400
        
        return jsonify({
            'success': True,
            'chart': chart_config
        })
        
    except Exception as e:
        logger.error(f"Chart generation error: {e}")
        return jsonify({'error': f'Chart generation failed: {str(e)}'}), 500

@dashboard_bp.route('/kpis/<company_id>')
def get_kpis(company_id):
    """
    Get key performance indicators for a company
    
    Args:
        company_id (str): Company identifier
        
    Returns:
        JSON: KPI data
    """
    try:
        db_config = get_db_config()
        analyzer = FinancialAnalyzer(db_config)
        
        # Get analysis
        analysis = analyzer.analyze_company_performance(company_id)
        if 'error' in analysis:
            return jsonify({'error': 'Unable to get KPIs'}), 400
        
        # Create KPIs
        kpis = {
            'financial_health_score': {
                'value': analysis['summary']['overall_score'],
                'unit': '/100',
                'trend': 'stable',
                'status': get_status_from_score(analysis['summary']['overall_score'])
            },
            'current_ratio': {
                'value': round(analysis['ratios']['liquidity'].get('current_ratio', 0), 2),
                'unit': ':1',
                'benchmark': 2.0,
                'status': get_ratio_status(analysis['ratios']['liquidity'].get('current_ratio', 0), 2.0, 'higher_better')
            },
            'debt_to_equity': {
                'value': round(analysis['ratios']['leverage'].get('debt_to_equity', 0), 2),
                'unit': ':1',
                'benchmark': 0.5,
                'status': get_ratio_status(analysis['ratios']['leverage'].get('debt_to_equity', 0), 0.5, 'lower_better')
            },
            'roa': {
                'value': round(analysis['ratios']['profitability'].get('roa', 0) * 100, 1),
                'unit': '%',
                'benchmark': 10.0,
                'status': get_ratio_status(analysis['ratios']['profitability'].get('roa', 0) * 100, 10.0, 'higher_better')
            },
            'asset_turnover': {
                'value': round(analysis['ratios']['efficiency'].get('asset_turnover', 0), 2),
                'unit': 'x',
                'benchmark': 1.0,
                'status': get_ratio_status(analysis['ratios']['efficiency'].get('asset_turnover', 0), 1.0, 'higher_better')
            }
        }
        
        return jsonify({
            'success': True,
            'kpis': kpis,
            'updated_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"KPI retrieval error: {e}")
        return jsonify({'error': f'KPI retrieval failed: {str(e)}'}), 500

@dashboard_bp.route('/alerts/<company_id>')
def get_alerts(company_id):
    """
    Get financial alerts for a company
    
    Args:
        company_id (str): Company identifier
        
    Returns:
        JSON: Alert data
    """
    try:
        db_config = get_db_config()
        analyzer = FinancialAnalyzer(db_config)
        
        # Get analysis
        analysis = analyzer.analyze_company_performance(company_id)
        if 'error' in analysis:
            return jsonify({'error': 'Unable to generate alerts'}), 400
        
        # Generate alerts
        alerts = generate_financial_alerts(analysis)
        
        return jsonify({
            'success': True,
            'alerts': alerts,
            'alert_count': len(alerts),
            'updated_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Alert generation error: {e}")
        return jsonify({'error': f'Alert generation failed: {str(e)}'}), 500

@dashboard_bp.route('/export/<dashboard_type>/<company_id>')
def export_dashboard(dashboard_type, company_id):
    """
    Export dashboard data
    
    Args:
        dashboard_type (str): Type of dashboard to export
        company_id (str): Company identifier
        
    Query Parameters:
        format: Export format (json, csv, excel)
        
    Returns:
        File: Exported dashboard data
    """
    try:
        export_format = request.args.get('format', 'json')
        
        # Get dashboard data based on type
        if dashboard_type == 'executive':
            response = executive_dashboard(company_id)
        elif dashboard_type == 'operational':
            response = operational_dashboard(company_id)
        elif dashboard_type == 'risk':
            response = risk_dashboard(company_id)
        else:
            return jsonify({'error': f'Unknown dashboard type: {dashboard_type}'}), 400
        
        dashboard_data = response.get_json()
        
        if export_format == 'json':
            return jsonify(dashboard_data)
        elif export_format == 'csv':
            # Convert to CSV format
            return export_to_csv(dashboard_data)
        elif export_format == 'excel':
            # Convert to Excel format
            return export_to_excel(dashboard_data)
        else:
            return jsonify({'error': f'Unsupported export format: {export_format}'}), 400
        
    except Exception as e:
        logger.error(f"Export error: {e}")
        return jsonify({'error': f'Export failed: {str(e)}'}), 500

# Helper functions for creating dashboard components
def create_executive_summary(analysis):
    """Create executive summary from analysis"""
    return {
        'overall_score': analysis['summary']['overall_score'],
        'overall_rating': analysis['summary']['overall_rating'],
        'category_scores': analysis['summary']['category_scores'],
        'key_strengths': identify_key_strengths(analysis),
        'key_concerns': analysis['risk_assessment']['risk_factors'][:3],
        'recommendation_count': len(analysis.get('recommendations', []))
    }

def create_executive_kpis(analysis):
    """Create KPIs for executive dashboard"""
    return {
        'health_score': {
            'value': analysis['summary']['overall_score'],
            'status': get_status_from_score(analysis['summary']['overall_score']),
            'trend': 'stable'
        },
        'liquidity': {
            'value': analysis['ratios']['liquidity'].get('current_ratio', 0),
            'status': get_ratio_status(analysis['ratios']['liquidity'].get('current_ratio', 0), 1.5, 'higher_better')
        },
        'profitability': {
            'value': analysis['ratios']['profitability'].get('roa', 0) * 100,
            'status': get_ratio_status(analysis['ratios']['profitability'].get('roa', 0) * 100, 5.0, 'higher_better')
        },
        'leverage': {
            'value': analysis['ratios']['leverage'].get('debt_to_equity', 0),
            'status': get_ratio_status(analysis['ratios']['leverage'].get('debt_to_equity', 0), 1.0, 'lower_better')
        }
    }

def create_executive_charts(company_id, analysis, chart_gen):
    """Create charts for executive dashboard"""
    charts = {}
    
    try:
        # Financial health radar
        charts['health_radar'] = create_health_radar_chart(analysis)
        
        # Key trends
        charts['trends'] = chart_gen.create_trend_chart(company_id)
        
        # Risk assessment
        charts['risk_assessment'] = create_risk_assessment_chart(analysis)
        
        # Performance overview
        charts['performance_overview'] = create_performance_overview_chart(analysis)
        
    except Exception as e:
        logger.error(f"Error creating charts: {e}")
        charts['error'] = str(e)
    
    return charts

def create_operational_metrics(analysis):
    """Create operational metrics for operational dashboard"""
    try:
        return {
            'efficiency_ratios': {
                'asset_turnover': analysis['ratios']['efficiency'].get('asset_turnover', 0),
                'inventory_turnover': analysis['ratios']['efficiency'].get('inventory_turnover', 0),
                'receivables_turnover': analysis['ratios']['efficiency'].get('receivables_turnover', 0)
            },
            'working_capital_metrics': {
                'current_ratio': analysis['ratios']['liquidity'].get('current_ratio', 0),
                'quick_ratio': analysis['ratios']['liquidity'].get('quick_ratio', 0),
                'cash_ratio': analysis['ratios']['liquidity'].get('cash_ratio', 0)
            },
            'cash_flow_metrics': {
                'operating_cf_ratio': analysis['ratios'].get('cash_flow', {}).get('operating_cf_ratio', 0),
                'free_cf_margin': analysis['ratios'].get('cash_flow', {}).get('free_cf_margin', 0)
            }
        }
    except Exception as e:
        logger.error(f"Error creating operational metrics: {e}")
        return {}

def create_efficiency_analysis(analysis):
    """Create efficiency analysis"""
    try:
        efficiency_ratios = analysis['ratios']['efficiency']
        
        return {
            'asset_utilization': {
                'asset_turnover': efficiency_ratios.get('asset_turnover', 0),
                'benchmark': 1.0,
                'status': get_ratio_status(efficiency_ratios.get('asset_turnover', 0), 1.0, 'higher_better')
            },
            'inventory_management': {
                'inventory_turnover': efficiency_ratios.get('inventory_turnover', 0),
                'days_inventory': 365 / max(efficiency_ratios.get('inventory_turnover', 1), 1),
                'benchmark': 60,
                'status': 'good' if (365 / max(efficiency_ratios.get('inventory_turnover', 1), 1)) <= 60 else 'poor'
            },
            'receivables_management': {
                'receivables_turnover': efficiency_ratios.get('receivables_turnover', 0),
                'days_receivables': 365 / max(efficiency_ratios.get('receivables_turnover', 1), 1),
                'benchmark': 45,
                'status': 'good' if (365 / max(efficiency_ratios.get('receivables_turnover', 1), 1)) <= 45 else 'poor'
            }
        }
    except Exception as e:
        logger.error(f"Error creating efficiency analysis: {e}")
        return {}

def create_working_capital_analysis(company_id, analyzer):
    """Create working capital analysis"""
    try:
        # Get company data for working capital analysis
        company_data = analyzer.get_company_data(company_id)
        if not company_data or not company_data.get('balance_sheets'):
            return {}
        
        latest_bs = company_data['balance_sheets'][0]
        
        current_assets = float(latest_bs.get('current_assets', 0) or 0)
        current_liabilities = float(latest_bs.get('current_liabilities', 0) or 0)
        working_capital = current_assets - current_liabilities
        
        return {
            'current_assets': current_assets,
            'current_liabilities': current_liabilities,
            'working_capital': working_capital,
            'working_capital_ratio': working_capital / current_assets if current_assets > 0 else 0,
            'status': 'positive' if working_capital > 0 else 'negative'
        }
    except Exception as e:
        logger.error(f"Error creating working capital analysis: {e}")
        return {}

def create_operational_charts(company_id, analysis, chart_gen):
    """Create charts for operational dashboard"""
    charts = {}
    
    try:
        # Efficiency trends
        charts['efficiency_trends'] = create_efficiency_trends_chart(analysis)
        
        # Working capital analysis
        charts['working_capital'] = create_working_capital_chart(company_id, analysis)
        
        # Cash flow analysis
        charts['cash_flow_analysis'] = create_cash_flow_analysis_chart(analysis)
        
    except Exception as e:
        logger.error(f"Error creating operational charts: {e}")
        charts['error'] = str(e)
    
    return charts

def create_operational_trends(company_id, analyzer):
    """Create operational trends analysis"""
    try:
        # Get historical data for trends
        company_data = analyzer.get_company_data(company_id)
        if not company_data:
            return {}
        
        return {
            'revenue_trend': 'stable',  # Would calculate from historical data
            'efficiency_trend': 'improving',
            'working_capital_trend': 'stable',
            'cash_flow_trend': 'stable'
        }
    except Exception as e:
        logger.error(f"Error creating operational trends: {e}")
        return {}

def generate_operational_insights(analysis):
    """Generate operational insights"""
    try:
        insights = []
        
        # Asset turnover insights
        asset_turnover = analysis['ratios']['efficiency'].get('asset_turnover', 0)
        if asset_turnover < 0.5:
            insights.append({
                'type': 'efficiency',
                'title': 'Low Asset Utilization',
                'description': 'Asset turnover is below industry standards',
                'recommendation': 'Consider strategies to improve asset utilization'
            })
        
        # Working capital insights
        current_ratio = analysis['ratios']['liquidity'].get('current_ratio', 0)
        if current_ratio > 3.0:
            insights.append({
                'type': 'liquidity',
                'title': 'Excess Working Capital',
                'description': 'High current ratio may indicate inefficient use of capital',
                'recommendation': 'Evaluate opportunities for better capital deployment'
            })
        
        return insights
    except Exception as e:
        logger.error(f"Error generating operational insights: {e}")
        return []

def create_risk_summary(analysis):
    """Create risk summary"""
    try:
        risk_data = analysis.get('risk_assessment', {})
        
        return {
            'overall_risk_level': risk_data.get('risk_levels', {}).get('overall_risk', 'unknown'),
            'risk_score': risk_data.get('risk_score', 0),
            'primary_risks': risk_data.get('risk_factors', [])[:3],
            'risk_trend': 'stable'  # Would calculate from historical data
        }
    except Exception as e:
        logger.error(f"Error creating risk summary: {e}")
        return {}

def create_risk_scenarios(analysis, predictions):
    """Create risk scenarios"""
    try:
        current_score = analysis['summary']['overall_score']
        predicted_score = predictions.get('predicted_health_score', current_score)
        
        scenarios = {
            'base_case': {
                'probability': 60,
                'health_score': predicted_score,
                'description': 'Current trends continue'
            },
            'optimistic': {
                'probability': 25,
                'health_score': min(100, predicted_score + 15),
                'description': 'Improvement initiatives successful'
            },
            'pessimistic': {
                'probability': 15,
                'health_score': max(0, predicted_score - 20),
                'description': 'Economic challenges impact performance'
            }
        }
        
        return scenarios
    except Exception as e:
        logger.error(f"Error creating risk scenarios: {e}")
        return {}

def create_risk_charts(company_id, analysis):
    """Create risk-focused charts"""
    charts = {}
    
    try:
        # Risk assessment chart
        charts['risk_assessment'] = create_risk_assessment_chart(analysis)
        
        # Risk trend chart
        charts['risk_trends'] = create_risk_trends_chart(analysis)
        
        # Scenario analysis chart
        charts['scenario_analysis'] = create_scenario_analysis_chart(analysis)
        
    except Exception as e:
        logger.error(f"Error creating risk charts: {e}")
        charts['error'] = str(e)
    
    return charts

def generate_risk_mitigation(analysis):
    """Generate risk mitigation strategies"""
    try:
        risk_levels = analysis['risk_assessment']['risk_levels']
        strategies = []
        
        for risk_type, level in risk_levels.items():
            if risk_type != 'overall_risk' and level in ['medium', 'high']:
                strategies.append({
                    'risk_type': risk_type,
                    'level': level,
                    'strategy': get_mitigation_strategy(risk_type, level),
                    'timeline': 'immediate' if level == 'high' else 'short_term'
                })
        
        return strategies
    except Exception as e:
        logger.error(f"Error generating risk mitigation: {e}")
        return []

def get_mitigation_strategy(risk_type, level):
    """Get specific mitigation strategy"""
    strategies = {
        'liquidity_risk': {
            'high': 'Immediate cash flow management and credit facility arrangement',
            'medium': 'Improve working capital cycle and receivables collection'
        },
        'solvency_risk': {
            'high': 'Debt restructuring and equity financing consideration',
            'medium': 'Debt reduction plan and profitability improvement'
        },
        'profitability_risk': {
            'high': 'Comprehensive cost reduction and revenue optimization',
            'medium': 'Focus on margin improvement and operational efficiency'
        },
        'operational_risk': {
            'high': 'Immediate operational review and process improvement',
            'medium': 'Enhance operational controls and efficiency measures'
        }
    }
    
    return strategies.get(risk_type, {}).get(level, 'Monitor and review regularly')

def create_health_radar_chart(analysis):
    """Create financial health radar chart"""
    try:
        categories = ['Liquidity', 'Profitability', 'Leverage', 'Efficiency']
        scores = []
        
        for category in ['liquidity', 'profitability', 'leverage', 'efficiency']:
            score = analysis['summary']['category_scores'].get(category, 50)
            scores.append(score)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=categories,
            fill='toself',
            name='Current Performance',
            line_color='#1f77b4'
        ))
        
        # Add benchmark
        benchmark_scores = [70] * len(categories)
        fig.add_trace(go.Scatterpolar(
            r=benchmark_scores,
            theta=categories,
            fill=None,
            name='Industry Benchmark',
            line=dict(color='#ff7f0e', dash='dash')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title="Financial Health Overview"
        )
        
        return {
            'type': 'radar',
            'config': fig.to_dict(),
            'title': 'Financial Health Radar'
        }
    except Exception as e:
        logger.error(f"Error creating radar chart: {e}")
        return {}

def create_risk_assessment_chart(analysis):
    """Create risk assessment chart"""
    try:
        risk_data = analysis.get('risk_assessment', {})
        risk_levels = risk_data.get('risk_levels', {})
        
        # Convert to scores
        risk_scores = {}
        risk_map = {'low': 20, 'medium': 60, 'high': 90}
        
        for risk_type, level in risk_levels.items():
            if risk_type != 'overall_risk':
                risk_scores[risk_type.replace('_risk', '').title()] = risk_map.get(level, 50)
        
        if not risk_scores:
            return {}
        
        fig = go.Figure(go.Bar(
            y=list(risk_scores.keys()),
            x=list(risk_scores.values()),
            orientation='h',
            marker=dict(
                color=['green' if score <= 30 else 'orange' if score <= 60 else 'red' for score in risk_scores.values()]
            )
        ))
        
        fig.update_layout(
            title="Risk Assessment",
            xaxis_title="Risk Score",
            xaxis=dict(range=[0, 100])
        )
        
        return {
            'type': 'bar',
            'config': fig.to_dict(),
            'title': 'Risk Assessment'
        }
    except Exception as e:
        logger.error(f"Error creating risk assessment chart: {e}")
        return {}

def create_performance_overview_chart(analysis):
    """Create performance overview chart"""
    try:
        categories = list(analysis['summary']['category_scores'].keys())
        scores = list(analysis['summary']['category_scores'].values())
        
        fig = go.Figure(go.Bar(
            x=[cat.title() for cat in categories],
            y=scores,
            marker_color=['#2E8B57' if score >= 70 else '#FFD700' if score >= 50 else '#FF6347' for score in scores],
            text=[f'{score:.1f}' for score in scores],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Performance Overview by Category",
            xaxis_title="Categories",
            yaxis_title="Score (0-100)",
            yaxis=dict(range=[0, 100])
        )
        
        return {
            'type': 'bar',
            'config': fig.to_dict(),
            'title': 'Performance Overview'
        }
    except Exception as e:
        logger.error(f"Error creating performance overview chart: {e}")
        return {}

def create_ratio_comparison_chart(analysis):
    """Create ratio comparison chart"""
    try:
        ratios_data = []
        benchmarks = {
            'current_ratio': 2.0,
            'quick_ratio': 1.0,
            'debt_to_equity': 0.5,
            'roa': 0.10,
            'asset_turnover': 1.0
        }
        
        for category, ratios in analysis['ratios'].items():
            for ratio_name, value in ratios.items():
                if ratio_name in benchmarks:
                    ratios_data.append({
                        'ratio': ratio_name.replace('_', ' ').title(),
                        'actual': value,
                        'benchmark': benchmarks[ratio_name]
                    })
        
        if not ratios_data:
            return {}
        
        fig = go.Figure()
        
        # Add actual values
        fig.add_trace(go.Bar(
            name='Actual',
            x=[r['ratio'] for r in ratios_data],
            y=[r['actual'] for r in ratios_data],
            marker_color='lightblue'
        ))
        
        # Add benchmarks
        fig.add_trace(go.Bar(
            name='Benchmark',
            x=[r['ratio'] for r in ratios_data],
            y=[r['benchmark'] for r in ratios_data],
            marker_color='orange'
        ))
        
        fig.update_layout(
            title="Ratio vs Benchmark Comparison",
            xaxis_title="Financial Ratios",
            yaxis_title="Value",
            barmode='group'
        )
        
        return {
            'type': 'grouped_bar',
            'config': fig.to_dict(),
            'title': 'Ratio Comparison'
        }
    except Exception as e:
        logger.error(f"Error creating ratio comparison chart: {e}")
        return {}

def create_cash_flow_waterfall(company_id, analyzer):
    """Create cash flow waterfall chart"""
    try:
        # Get cash flow data
        company_data = analyzer.get_company_data(company_id)
        if not company_data or not company_data.get('cash_flows'):
            return {}
        
        latest_cf = company_data['cash_flows'][0]
        
        # Extract cash flow components
        operating_cf = float(latest_cf.get('net_cash_from_operating_activities', 0) or 0)
        investing_cf = float(latest_cf.get('net_cash_from_investing_activities', 0) or 0)
        financing_cf = float(latest_cf.get('net_cash_from_financing_activities', 0) or 0)
        net_change = operating_cf + investing_cf + financing_cf
        
        fig = go.Figure(go.Waterfall(
            name="Cash Flow",
            orientation="v",
            measure=["relative", "relative", "relative", "total"],
            x=["Operating CF", "Investing CF", "Financing CF", "Net Change"],
            textposition="outside",
            text=[f"${operating_cf:,.0f}", f"${investing_cf:,.0f}", f"${financing_cf:,.0f}", f"${net_change:,.0f}"],
            y=[operating_cf, investing_cf, financing_cf, net_change],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title="Cash Flow Waterfall Analysis",
            showlegend=True
        )
        
        return {
            'type': 'waterfall',
            'config': fig.to_dict(),
            'title': 'Cash Flow Waterfall'
        }
    except Exception as e:
        logger.error(f"Error creating cash flow waterfall: {e}")
        return {}

def create_efficiency_trends_chart(analysis):
    """Create efficiency trends chart"""
    try:
        # Mock historical data - in real implementation, get from database
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        asset_turnover = [0.8, 0.85, 0.9, 0.88, 0.92, 0.95]
        inventory_turnover = [4.2, 4.5, 4.8, 4.6, 5.0, 5.2]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=months,
            y=asset_turnover,
            mode='lines+markers',
            name='Asset Turnover',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=months,
            y=inventory_turnover,
            mode='lines+markers',
            name='Inventory Turnover',
            yaxis='y2',
            line=dict(color='green')
        ))
        
        fig.update_layout(
            title="Efficiency Trends",
            xaxis_title="Month",
            yaxis=dict(title="Asset Turnover", side="left"),
            yaxis2=dict(title="Inventory Turnover", side="right", overlaying="y")
        )
        
        return {
            'type': 'line',
            'config': fig.to_dict(),
            'title': 'Efficiency Trends'
        }
    except Exception as e:
        logger.error(f"Error creating efficiency trends chart: {e}")
        return {}

def create_working_capital_chart(company_id, analysis):
    """Create working capital analysis chart"""
    try:
        # Get liquidity ratios
        liquidity = analysis['ratios']['liquidity']
        
        ratios = ['Current Ratio', 'Quick Ratio', 'Cash Ratio']
        values = [
            liquidity.get('current_ratio', 0),
            liquidity.get('quick_ratio', 0),
            liquidity.get('cash_ratio', 0)
        ]
        benchmarks = [2.0, 1.0, 0.5]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Actual',
            x=ratios,
            y=values,
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Scatter(
            name='Benchmark',
            x=ratios,
            y=benchmarks,
            mode='markers',
            marker=dict(color='red', size=10, symbol='diamond')
        ))
        
        fig.update_layout(
            title="Working Capital Analysis",
            xaxis_title="Liquidity Ratios",
            yaxis_title="Ratio Value"
        )
        
        return {
            'type': 'mixed',
            'config': fig.to_dict(),
            'title': 'Working Capital Analysis'
        }
    except Exception as e:
        logger.error(f"Error creating working capital chart: {e}")
        return {}

def create_cash_flow_analysis_chart(analysis):
    """Create cash flow analysis chart"""
    try:
        # Mock cash flow data - in real implementation, get from database
        quarters = ['Q1', 'Q2', 'Q3', 'Q4']
        operating_cf = [50000, 55000, 60000, 58000]
        investing_cf = [-20000, -25000, -15000, -30000]
        financing_cf = [-10000, -5000, -8000, -12000]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Operating CF',
            x=quarters,
            y=operating_cf,
            marker_color='green'
        ))
        
        fig.add_trace(go.Bar(
            name='Investing CF',
            x=quarters,
            y=investing_cf,
            marker_color='blue'
        ))
        
        fig.add_trace(go.Bar(
            name='Financing CF',
            x=quarters,
            y=financing_cf,
            marker_color='orange'
        ))
        
        fig.update_layout(
            title="Cash Flow Analysis by Quarter",
            xaxis_title="Quarter",
            yaxis_title="Cash Flow ($)",
            barmode='group'
        )
        
        return {
            'type': 'grouped_bar',
            'config': fig.to_dict(),
            'title': 'Cash Flow Analysis'
        }
    except Exception as e:
        logger.error(f"Error creating cash flow analysis chart: {e}")
        return {}

def create_risk_trends_chart(analysis):
    """Create risk trends chart"""
    try:
        # Mock risk trend data
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        overall_risk = [65, 62, 60, 58, 55, 53]
        liquidity_risk = [70, 68, 65, 63, 60, 58]
        solvency_risk = [60, 58, 55, 53, 50, 48]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=months,
            y=overall_risk,
            mode='lines+markers',
            name='Overall Risk',
            line=dict(color='red', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=months,
            y=liquidity_risk,
            mode='lines+markers',
            name='Liquidity Risk',
            line=dict(color='orange')
        ))
        
        fig.add_trace(go.Scatter(
            x=months,
            y=solvency_risk,
            mode='lines+markers',
            name='Solvency Risk',
            line=dict(color='yellow')
        ))
        
        fig.update_layout(
            title="Risk Trends Over Time",
            xaxis_title="Month",
            yaxis_title="Risk Score (Lower is Better)",
            yaxis=dict(range=[0, 100])
        )
        
        return {
            'type': 'line',
            'config': fig.to_dict(),
            'title': 'Risk Trends'
        }
    except Exception as e:
        logger.error(f"Error creating risk trends chart: {e}")
        return {}

def create_scenario_analysis_chart(analysis):
    """Create scenario analysis chart"""
    try:
        scenarios = ['Pessimistic', 'Base Case', 'Optimistic']
        probabilities = [15, 60, 25]
        health_scores = [45, 65, 85]
        
        fig = go.Figure()
        
        # Add probability bars
        fig.add_trace(go.Bar(
            name='Probability (%)',
            x=scenarios,
            y=probabilities,
            yaxis='y',
            marker_color='lightblue'
        ))
        
        # Add health score line
        fig.add_trace(go.Scatter(
            name='Health Score',
            x=scenarios,
            y=health_scores,
            yaxis='y2',
            mode='lines+markers',
            line=dict(color='red', width=3),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title="Scenario Analysis",
            xaxis_title="Scenarios",
            yaxis=dict(title="Probability (%)", side="left"),
            yaxis2=dict(title="Health Score", side="right", overlaying="y")
        )
        
        return {
            'type': 'mixed',
            'config': fig.to_dict(),
            'title': 'Scenario Analysis'
        }
    except Exception as e:
        logger.error(f"Error creating scenario analysis chart: {e}")
        return {}

def create_industry_overview(benchmark):
    """Create industry overview"""
    try:
        return {
            'total_companies': benchmark.get('total_companies', 0),
            'industry_average_score': benchmark.get('averages', {}).get('overall_score', {}).get('mean', 0),
            'top_performer_score': benchmark.get('top_performers', [{}])[0].get('overall_score', 0) if benchmark.get('top_performers') else 0,
            'industry_trends': 'stable'  # Would calculate from historical data
        }
    except Exception as e:
        logger.error(f"Error creating industry overview: {e}")
        return {}

def create_performance_distribution(benchmark):
    """Create performance distribution"""
    try:
        percentiles = benchmark.get('percentiles', {}).get('overall_score', {})
        
        return {
            'excellent': percentiles.get('90th', 90),
            'good': percentiles.get('75th', 75),
            'average': percentiles.get('50th', 50),
            'poor': percentiles.get('25th', 25),
            'critical': percentiles.get('10th', 10)
        }
    except Exception as e:
        logger.error(f"Error creating performance distribution: {e}")
        return {}

def create_benchmark_charts(benchmark, industry):
    """Create benchmark charts"""
    charts = {}
    
    try:
        # Industry distribution chart
        charts['distribution'] = create_industry_distribution_chart(benchmark)
        
        # Top performers chart
        charts['top_performers'] = create_top_performers_chart(benchmark)
        
        # Category averages chart
        charts['category_averages'] = create_category_averages_chart(benchmark)
        
    except Exception as e:
        logger.error(f"Error creating benchmark charts: {e}")
        charts['error'] = str(e)
    
    return charts

def create_industry_distribution_chart(benchmark):
    """Create industry distribution chart"""
    try:
        percentiles = benchmark.get('percentiles', {}).get('overall_score', {})
        
        categories = ['10th', '25th', '50th', '75th', '90th']
        values = [percentiles.get(cat, 0) for cat in categories]
        
        fig = go.Figure(go.Bar(
            x=categories,
            y=values,
            marker_color='skyblue',
            text=[f'{val:.1f}' for val in values],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Industry Performance Distribution",
            xaxis_title="Percentiles",
            yaxis_title="Health Score"
        )
        
        return {
            'type': 'bar',
            'config': fig.to_dict(),
            'title': 'Industry Distribution'
        }
    except Exception as e:
        logger.error(f"Error creating industry distribution chart: {e}")
        return {}

def create_top_performers_chart(benchmark):
    """Create top performers chart"""
    try:
        top_performers = benchmark.get('top_performers', [])[:10]  # Top 10
        
        if not top_performers:
            return {}
        
        companies = [f"Company {i+1}" for i in range(len(top_performers))]  # Anonymize
        scores = [perf.get('overall_score', 0) for perf in top_performers]
        
        fig = go.Figure(go.Bar(
            x=companies,
            y=scores,
            marker_color='gold',
            text=[f'{score:.1f}' for score in scores],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Top Performing Companies",
            xaxis_title="Companies",
            yaxis_title="Health Score"
        )
        
        return {
            'type': 'bar',
            'config': fig.to_dict(),
            'title': 'Top Performers'
        }
    except Exception as e:
        logger.error(f"Error creating top performers chart: {e}")
        return {}

def create_category_averages_chart(benchmark):
    """Create category averages chart"""
    try:
        averages = benchmark.get('averages', {})
        categories = []
        values = []
        
        for category, data in averages.items():
            if category != 'overall_score' and isinstance(data, dict):
                categories.append(category.title())
                values.append(data.get('mean', 0))
        
        if not categories:
            return {}
        
        fig = go.Figure(go.Bar(
            x=categories,
            y=values,
            marker_color='lightgreen',
            text=[f'{val:.1f}' for val in values],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Industry Average by Category",
            xaxis_title="Categories",
            yaxis_title="Average Score"
        )
        
        return {
            'type': 'bar',
            'config': fig.to_dict(),
            'title': 'Category Averages'
        }
    except Exception as e:
        logger.error(f"Error creating category averages chart: {e}")
        return {}

def generate_industry_insights(benchmark):
    """Generate industry insights"""
    try:
        insights = []
        
        # Industry average insight
        avg_score = benchmark.get('averages', {}).get('overall_score', {}).get('mean', 0)
        insights.append({
            'type': 'industry_average',
            'title': f'Industry Average: {avg_score:.1f}/100',
            'description': f'The industry maintains an average health score of {avg_score:.1f}'
        })
        
        # Top performer insight
        top_performers = benchmark.get('top_performers', [])
        if top_performers:
            top_score = top_performers[0].get('overall_score', 0)
            insights.append({
                'type': 'top_performance',
                'title': f'Top Performance: {top_score:.1f}/100',
                'description': f'Industry leaders achieve scores up to {top_score:.1f}'
            })
        
        return insights
    except Exception as e:
        logger.error(f"Error generating industry insights: {e}")
        return []

def create_comparison_summary(comparison):
    """Create comparison summary"""
    try:
        companies = comparison.get('companies', {})
        
        best_performer = None
        worst_performer = None
        best_score = 0
        worst_score = 100
        
        for company_id, data in companies.items():
            score = data.get('summary', {}).get('overall_score', 0)
            if score > best_score:
                best_score = score
                best_performer = company_id
            if score < worst_score:
                worst_score = score
                worst_performer = company_id
        
        return {
            'total_companies': len(companies),
            'best_performer': best_performer,
            'best_score': best_score,
            'worst_performer': worst_performer,
            'worst_score': worst_score,
            'score_range': best_score - worst_score
        }
    except Exception as e:
        logger.error(f"Error creating comparison summary: {e}")
        return {}

def create_comparison_charts(comparison):
    """Create comparison charts"""
    charts = {}
    
    try:
        # Overall comparison chart
        charts['overall_comparison'] = create_overall_comparison_chart(comparison)
        
        # Category comparison chart
        charts['category_comparison'] = create_category_comparison_chart(comparison)
        
        # Risk comparison chart
        charts['risk_comparison'] = create_risk_comparison_chart(comparison)
        
    except Exception as e:
        logger.error(f"Error creating comparison charts: {e}")
        charts['error'] = str(e)
    
    return charts

def create_overall_comparison_chart(comparison):
    """Create overall comparison chart"""
    try:
        companies = comparison.get('companies', {})
        
        company_names = list(companies.keys())
        scores = [companies[comp].get('summary', {}).get('overall_score', 0) for comp in company_names]
        
        fig = go.Figure(go.Bar(
            x=company_names,
            y=scores,
            marker_color=['#2E8B57' if score >= 70 else '#FFD700' if score >= 50 else '#FF6347' for score in scores],
            text=[f'{score:.1f}' for score in scores],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Overall Performance Comparison",
            xaxis_title="Companies",
            yaxis_title="Health Score",
            yaxis=dict(range=[0, 100])
        )
        
        return {
            'type': 'bar',
            'config': fig.to_dict(),
            'title': 'Overall Comparison'
        }
    except Exception as e:
        logger.error(f"Error creating overall comparison chart: {e}")
        return {}

def create_category_comparison_chart(comparison):
    """Create category comparison chart"""
    try:
        companies = comparison.get('companies', {})
        categories = ['liquidity', 'profitability', 'leverage', 'efficiency']
        
        fig = go.Figure()
        
        for company_id, data in companies.items():
            category_scores = data.get('summary', {}).get('category_scores', {})
            scores = [category_scores.get(cat, 0) for cat in categories]
            
            fig.add_trace(go.Bar(
                name=company_id,
                x=[cat.title() for cat in categories],
                y=scores
            ))
        
        fig.update_layout(
            title="Category Performance Comparison",
            xaxis_title="Categories",
            yaxis_title="Score",
            barmode='group'
        )
        
        return {
            'type': 'grouped_bar',
            'config': fig.to_dict(),
            'title': 'Category Comparison'
        }
    except Exception as e:
        logger.error(f"Error creating category comparison chart: {e}")
        return {}

def create_risk_comparison_chart(comparison):
    """Create risk comparison chart"""
    try:
        companies = comparison.get('companies', {})
        
        company_names = list(companies.keys())
        risk_scores = [companies[comp].get('risk_assessment', {}).get('risk_score', 0) for comp in company_names]
        
        fig = go.Figure(go.Bar(
            x=company_names,
            y=risk_scores,
            marker_color=['green' if score <= 30 else 'orange' if score <= 60 else 'red' for score in risk_scores],
            text=[f'{score:.1f}' for score in risk_scores],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Risk Score Comparison (Lower is Better)",
            xaxis_title="Companies",
            yaxis_title="Risk Score",
            yaxis=dict(range=[0, 100])
        )
        
        return {
            'type': 'bar',
            'config': fig.to_dict(),
            'title': 'Risk Comparison'
        }
    except Exception as e:
        logger.error(f"Error creating risk comparison chart: {e}")
        return {}

def generate_competitive_analysis(comparison):
    """Generate competitive analysis"""
    try:
        companies = comparison.get('companies', {})
        analysis = []
        
        for company_id, data in companies.items():
            overall_score = data.get('summary', {}).get('overall_score', 0)
            category_scores = data.get('summary', {}).get('category_scores', {})
            
            # Find strongest category
            strongest_category = max(category_scores.items(), key=lambda x: x[1]) if category_scores else ('unknown', 0)
            
            analysis.append({
                'company': company_id,
                'overall_score': overall_score,
                'competitive_position': get_competitive_position(overall_score),
                'strongest_area': strongest_category[0],
                'strength_score': strongest_category[1]
            })
        
        return analysis
    except Exception as e:
        logger.error(f"Error generating competitive analysis: {e}")
        return []

def get_competitive_position(score):
    """Get competitive position based on score"""
    if score >= 80:
        return 'Market Leader'
    elif score >= 65:
        return 'Strong Competitor'
    elif score >= 50:
        return 'Average Performer'
    else:
        return 'Underperformer'

def generate_investment_recommendations(comparison):
    """Generate investment recommendations"""
    try:
        companies = comparison.get('companies', {})
        recommendations = []
        
        for company_id, data in companies.items():
            overall_score = data.get('summary', {}).get('overall_score', 0)
            risk_score = data.get('risk_assessment', {}).get('risk_score', 50)
            
            # Simple investment logic
            if overall_score >= 75 and risk_score <= 30:
                recommendation = 'Strong Buy'
                rationale = 'High performance with low risk'
            elif overall_score >= 60 and risk_score <= 50:
                recommendation = 'Buy'
                rationale = 'Good performance with moderate risk'
            elif overall_score >= 45:
                recommendation = 'Hold'
                rationale = 'Average performance, monitor closely'
            else:
                recommendation = 'Avoid'
                rationale = 'Poor performance or high risk'
            
            recommendations.append({
                'company': company_id,
                'recommendation': recommendation,
                'rationale': rationale,
                'score': overall_score,
                'risk': risk_score
            })
        
        return recommendations
    except Exception as e:
        logger.error(f"Error generating investment recommendations: {e}")
        return []

def generate_executive_alerts(analysis):
    """Generate alerts for executive dashboard"""
    alerts = []
    
    try:
        # Critical alerts
        if analysis['summary']['overall_score'] < 40:
            alerts.append({
                'type': 'critical',
                'title': 'Poor Financial Health',
                'message': 'Overall financial health score is below acceptable levels',
                'action': 'Immediate review required'
            })
        
        # Liquidity alerts
        current_ratio = analysis['ratios']['liquidity'].get('current_ratio', 0)
        if current_ratio < 1.0:
            alerts.append({
                'type': 'warning',
                'title': 'Liquidity Risk',
                'message': f'Current ratio of {current_ratio:.2f} indicates potential liquidity issues',
                'action': 'Review working capital management'
            })
        
        # Profitability alerts
        roa = analysis['ratios']['profitability'].get('roa', 0)
        if roa < 0:
            alerts.append({
                'type': 'critical',
                'title': 'Negative Returns',
                'message': 'Return on assets is negative, indicating poor asset utilization',
                'action': 'Review operational efficiency'
            })
        
        # Leverage alerts
        debt_to_equity = analysis['ratios']['leverage'].get('debt_to_equity', 0)
        if debt_to_equity > 2.0:
            alerts.append({
                'type': 'warning',
                'title': 'High Leverage',
                'message': f'Debt-to-equity ratio of {debt_to_equity:.2f} indicates high financial risk',
                'action': 'Consider debt reduction strategies'
            })
    
    except Exception as e:
        logger.error(f"Error generating executive alerts: {e}")
    
    return alerts

def format_executive_recommendations(analysis):
    """Format recommendations for executive dashboard"""
    try:
        recommendations = analysis.get('recommendations', [])
        
        formatted = []
        for rec in recommendations[:5]:  # Top 5 recommendations
            formatted.append({
                'category': rec.get('category', 'General'),
                'priority': rec.get('priority', 'Medium'),
                'recommendation': rec.get('recommendation', ''),
                'impact': rec.get('impact', ''),
                'timeline': get_recommendation_timeline(rec.get('priority', 'Medium'))
            })
        
        return formatted
    except Exception as e:
        logger.error(f"Error formatting executive recommendations: {e}")
        return []

def get_recommendation_timeline(priority):
    """Get timeline based on priority"""
    timelines = {
        'High': 'Immediate (0-30 days)',
        'Medium': 'Short-term (1-3 months)',
        'Low': 'Long-term (3-6 months)'
    }
    return timelines.get(priority, 'Short-term (1-3 months)')

def get_status_from_score(score):
    """Get status based on score"""
    if score >= 80:
        return 'excellent'
    elif score >= 65:
        return 'good'
    elif score >= 50:
        return 'fair'
    else:
        return 'poor'

def get_ratio_status(value, benchmark, direction):
    """Get status based on ratio comparison"""
    try:
        if direction == 'higher_better':
            if value >= benchmark:
                return 'good'
            elif value >= benchmark * 0.8:
                return 'fair'
            else:
                return 'poor'
        else:  # lower_better
            if value <= benchmark:
                return 'good'
            elif value <= benchmark * 1.5:
                return 'fair'
            else:
                return 'poor'
    except:
        return 'unknown'

def identify_key_strengths(analysis):
    """Identify key financial strengths"""
    try:
        strengths = []
        
        for category, score in analysis['summary']['category_scores'].items():
            if score >= 75:
                strengths.append(f"Strong {category} performance")
        
        return strengths[:3]  # Top 3 strengths
    except Exception as e:
        logger.error(f"Error identifying key strengths: {e}")
        return []

def generate_financial_alerts(analysis):
    """Generate comprehensive financial alerts"""
    alerts = []
    
    try:
        # Overall health alerts
        overall_score = analysis['summary']['overall_score']
        if overall_score < 30:
            alerts.append({
                'type': 'critical',
                'severity': 'high',
                'title': 'Critical Financial Health',
                'message': f'Overall health score of {overall_score:.1f} indicates severe financial distress',
                'action': 'Immediate management intervention required',
                'category': 'financial_health'
            })
        elif overall_score < 50:
            alerts.append({
                'type': 'warning',
                'severity': 'medium',
                'title': 'Poor Financial Health',
                'message': f'Overall health score of {overall_score:.1f} is below acceptable levels',
                'action': 'Develop improvement plan',
                'category': 'financial_health'
            })
        
        # Liquidity alerts
        current_ratio = analysis['ratios']['liquidity'].get('current_ratio', 0)
        quick_ratio = analysis['ratios']['liquidity'].get('quick_ratio', 0)
        
        if current_ratio < 1.0:
            alerts.append({
                'type': 'critical',
                'severity': 'high',
                'title': 'Liquidity Crisis',
                'message': f'Current ratio of {current_ratio:.2f} indicates inability to meet short-term obligations',
                'action': 'Secure immediate funding or liquidate assets',
                'category': 'liquidity'
            })
        elif current_ratio < 1.5:
            alerts.append({
                'type': 'warning',
                'severity': 'medium',
                'title': 'Liquidity Concern',
                'message': f'Current ratio of {current_ratio:.2f} indicates tight liquidity',
                'action': 'Improve working capital management',
                'category': 'liquidity'
            })
        
        if quick_ratio < 0.5:
            alerts.append({
                'type': 'warning',
                'severity': 'medium',
                'title': 'Limited Quick Liquidity',
                'message': f'Quick ratio of {quick_ratio:.2f} indicates limited immediate liquidity',
                'action': 'Increase cash reserves or reduce inventory',
                'category': 'liquidity'
            })
        
        # Profitability alerts
        roa = analysis['ratios']['profitability'].get('roa', 0)
        roe = analysis['ratios']['profitability'].get('roe', 0)
        profit_margin = analysis['ratios']['profitability'].get('profit_margin', 0)
        
        if roa < 0:
            alerts.append({
                'type': 'critical',
                'severity': 'high',
                'title': 'Negative Asset Returns',
                'message': f'ROA of {roa:.1%} indicates poor asset utilization',
                'action': 'Review operational efficiency and asset management',
                'category': 'profitability'
            })
        elif roa < 0.02:
            alerts.append({
                'type': 'warning',
                'severity': 'medium',
                'title': 'Low Asset Returns',
                'message': f'ROA of {roa:.1%} is below industry standards',
                'action': 'Improve asset utilization strategies',
                'category': 'profitability'
            })
        
        if profit_margin < 0:
            alerts.append({
                'type': 'critical',
                'severity': 'high',
                'title': 'Operating Losses',
                'message': f'Negative profit margin of {profit_margin:.1%}',
                'action': 'Immediate cost reduction and revenue optimization required',
                'category': 'profitability'
            })
        
        # Leverage alerts
        debt_to_equity = analysis['ratios']['leverage'].get('debt_to_equity', 0)
        interest_coverage = analysis['ratios']['leverage'].get('interest_coverage', 0)
        
        if debt_to_equity > 3.0:
            alerts.append({
                'type': 'critical',
                'severity': 'high',
                'title': 'Excessive Leverage',
                'message': f'Debt-to-equity ratio of {debt_to_equity:.2f} indicates dangerous leverage levels',
                'action': 'Immediate debt reduction or equity financing required',
                'category': 'leverage'
            })
        elif debt_to_equity > 2.0:
            alerts.append({
                'type': 'warning',
                'severity': 'medium',
                'title': 'High Leverage',
                'message': f'Debt-to-equity ratio of {debt_to_equity:.2f} indicates high financial risk',
                'action': 'Develop debt reduction plan',
                'category': 'leverage'
            })
        
        if interest_coverage < 1.5:
            alerts.append({
                'type': 'critical',
                'severity': 'high',
                'title': 'Insufficient Interest Coverage',
                'message': f'Interest coverage of {interest_coverage:.2f} indicates difficulty servicing debt',
                'action': 'Renegotiate debt terms or improve earnings',
                'category': 'leverage'
            })
        
        # Risk-based alerts
        risk_levels = analysis['risk_assessment']['risk_levels']
        for risk_type, level in risk_levels.items():
            if risk_type != 'overall_risk' and level == 'high':
                alerts.append({
                    'type': 'warning',
                    'severity': 'high',
                    'title': f'High {risk_type.replace("_", " ").title()}',
                    'message': f'Company faces high {risk_type.replace("_", " ")} risk',
                    'action': f'Address {risk_type.replace("_", " ")} risk factors immediately',
                    'category': 'risk'
                })
        
    except Exception as e:
        logger.error(f"Error generating financial alerts: {e}")
    
    return alerts

def export_to_csv(dashboard_data):
    """Export dashboard data to CSV format"""
    try:
        # Extract key metrics for CSV export
        if 'dashboard' in dashboard_data:
            data = dashboard_data['dashboard']
        else:
            data = dashboard_data
        
        # Prepare data for CSV
        csv_data = []
        
        # Add summary information
        if 'summary' in data:
            summary = data['summary']
            csv_data.append(['Metric', 'Value', 'Category'])
            csv_data.append(['Overall Score', summary.get('overall_score', 0), 'Summary'])
            csv_data.append(['Overall Rating', summary.get('overall_rating', 'N/A'), 'Summary'])
            
            # Add category scores
            for category, score in summary.get('category_scores', {}).items():
                csv_data.append([f'{category.title()} Score', score, 'Category'])
        
        # Add KPIs
        if 'kpis' in data:
            for kpi_name, kpi_data in data['kpis'].items():
                value = kpi_data.get('value', 0)
                unit = kpi_data.get('unit', '')
                csv_data.append([f'{kpi_name.replace("_", " ").title()}', f'{value}{unit}', 'KPI'])
        
        # Convert to DataFrame and return as CSV
        df = pd.DataFrame(csv_data[1:], columns=csv_data[0])
        
        output = io.StringIO()
        df.to_csv(output, index=False)
        
        return output.getvalue()
        
    except Exception as e:
        logger.error(f"Error exporting to CSV: {e}")
        return "Error exporting data to CSV"

def export_to_excel(dashboard_data):
    """Export dashboard data to Excel format"""
    try:
        # Extract data
        if 'dashboard' in dashboard_data:
            data = dashboard_data['dashboard']
        else:
            data = dashboard_data
        
        # Create Excel file in memory
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Summary sheet
            if 'summary' in data:
                summary_data = []
                summary = data['summary']
                
                summary_data.append(['Overall Score', summary.get('overall_score', 0)])
                summary_data.append(['Overall Rating', summary.get('overall_rating', 'N/A')])
                
                for category, score in summary.get('category_scores', {}).items():
                    summary_data.append([f'{category.title()} Score', score])
                
                summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # KPIs sheet
            if 'kpis' in data:
                kpi_data = []
                for kpi_name, kpi_info in data['kpis'].items():
                    kpi_data.append([
                        kpi_name.replace('_', ' ').title(),
                        kpi_info.get('value', 0),
                        kpi_info.get('unit', ''),
                        kpi_info.get('status', 'N/A')
                    ])
                
                kpi_df = pd.DataFrame(kpi_data, columns=['KPI', 'Value', 'Unit', 'Status'])
                kpi_df.to_excel(writer, sheet_name='KPIs', index=False)
            
            # Alerts sheet
            if 'alerts' in data:
                alert_data = []
                for alert in data['alerts']:
                    alert_data.append([
                        alert.get('type', 'N/A'),
                        alert.get('title', 'N/A'),
                        alert.get('message', 'N/A'),
                        alert.get('action', 'N/A')
                    ])
                
                if alert_data:
                    alert_df = pd.DataFrame(alert_data, columns=['Type', 'Title', 'Message', 'Action'])
                    alert_df.to_excel(writer, sheet_name='Alerts', index=False)
            
            # Recommendations sheet
            if 'recommendations' in data:
                rec_data = []
                for rec in data['recommendations']:
                    rec_data.append([
                        rec.get('category', 'N/A'),
                        rec.get('priority', 'N/A'),
                        rec.get('recommendation', 'N/A'),
                        rec.get('impact', 'N/A')
                    ])
                
                if rec_data:
                    rec_df = pd.DataFrame(rec_data, columns=['Category', 'Priority', 'Recommendation', 'Impact'])
                    rec_df.to_excel(writer, sheet_name='Recommendations', index=False)
        
        output.seek(0)
        return send_file(
            io.BytesIO(output.read()),
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'dashboard_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        )
        
    except Exception as e:
        logger.error(f"Error exporting to Excel: {e}")
        return jsonify({'error': f'Excel export failed: {str(e)}'}), 500

# Health check endpoint
@dashboard_bp.route('/health')
def health_check():
    """Health check endpoint for dashboard service"""
    try:
        return jsonify({
            'status': 'healthy',
            'service': 'dashboard',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# Dashboard configuration endpoint
@dashboard_bp.route('/config')
def get_dashboard_config():
    """Get dashboard configuration"""
    try:
        config = {
            'available_dashboards': [
                'executive',
                'operational', 
                'risk',
                'industry_benchmark',
                'comparison'
            ],
            'chart_types': [
                'financial_health_radar',
                'trend_analysis',
                'ratio_comparison',
                'cash_flow_waterfall',
                'risk_assessment'
            ],
            'export_formats': ['json', 'csv', 'excel'],
            'refresh_interval': 300,  # 5 minutes
            'cache_duration': 600     # 10 minutes
        }
        
        return jsonify({
            'success': True,
            'config': config
        })
        
    except Exception as e:
        logger.error(f"Error getting dashboard config: {e}")
        return jsonify({'error': f'Config retrieval failed: {str(e)}'}), 500

# Error handlers
@dashboard_bp.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Dashboard endpoint not found',
        'available_endpoints': [
            '/executive/<company_id>',
            '/operational/<company_id>',
            '/risk/<company_id>',
            '/industry-benchmark/<industry>',
            '/comparison?companies=id1,id2',
            '/charts/<chart_type>/<company_id>',
            '/kpis/<company_id>',
            '/alerts/<company_id>',
            '/export/<dashboard_type>/<company_id>'
        ]
    }), 404

@dashboard_bp.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred while processing the dashboard request'
    }), 500

@dashboard_bp.errorhandler(400)
def bad_request(error):
    """Handle 400 errors"""
    return jsonify({
        'error': 'Bad request',
        'message': 'Invalid request parameters or missing required data'
    }), 400

# Utility function for dashboard caching (if needed)
def cache_dashboard_data(dashboard_type, company_id, data, duration=600):
    """Cache dashboard data for performance"""
    try:
        cache_key = f"dashboard_{dashboard_type}_{company_id}"
        # Implementation would depend on caching system (Redis, Memcached, etc.)
        logger.info(f"Caching dashboard data for {cache_key}")
        return True
    except Exception as e:
        logger.error(f"Error caching dashboard data: {e}")
        return False

def get_cached_dashboard_data(dashboard_type, company_id):
    """Retrieve cached dashboard data"""
    try:
        cache_key = f"dashboard_{dashboard_type}_{company_id}"
        # Implementation would depend on caching system
        logger.info(f"Retrieving cached data for {cache_key}")
        return None  # Return None if no cache or cache miss
    except Exception as e:
        logger.error(f"Error retrieving cached data: {e}")
        return None

# Main execution
if __name__ == "__main__":
    print("Dashboard Routes module loaded successfully")
    print("Available endpoints:")
    print("- /executive/<company_id> - Executive dashboard")
    print("- /operational/<company_id> - Operational dashboard") 
    print("- /risk/<company_id> - Risk assessment dashboard")
    print("- /industry-benchmark/<industry> - Industry benchmark")
    print("- /comparison?companies=id1,id2 - Company comparison")
    print("- /charts/<chart_type>/<company_id> - Individual charts")
    print("- /kpis/<company_id> - Key performance indicators")
    print("- /alerts/<company_id> - Financial alerts")
    print("- /export/<dashboard_type>/<company_id> - Export dashboard")
    print("- /health - Health check")
    print("- /config - Dashboard configuration")