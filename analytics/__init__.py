"""
Analytics Package for Financial Risk Assessment Platform

This package contains all analytics, reporting, and data visualization components
for the Financial Risk Assessment Platform.

Components:
- ai_insights: AI-powered financial insights and predictions
- chart_generator: Chart and visualization generation utilities
- dashboard_routes: Dashboard API endpoints and data providers
- financial_analyzer: Core financial analysis and metrics calculation
- report_generator: Automated report generation and formatting

Database Integration:
- Reads from balance_sheet_1 table
- Reads from cash_flow_statement table
- Generates insights and reports based on real financial data

Author: Prateek Dahiya
Course: COMP702 – M.Sc. project (2024/25)
University: University of Liverpool
"""

from .ai_insights import AIInsights
from .chart_generator import ChartGenerator
from .financial_analyzer import FinancialAnalyzer
from .report_generator import ReportGenerator
from .dashboard_routes import dashboard_bp

__version__ = "1.0.0"
__author__ = "Prateek Dahiya"
__email__ = "prateek.dahiya@student.liverpool.ac.uk"

# Export main classes
__all__ = [
    'AIInsights',
    'ChartGenerator', 
    'FinancialAnalyzer',
    'ReportGenerator',
    'dashboard_bp',
    'AnalyticsEngine'
]

# Analytics configuration
ANALYTICS_CONFIG = {
    'default_chart_colors': [
        '#3498db', '#e74c3c', '#2ecc71', '#f39c12', 
        '#9b59b6', '#34495e', '#16a085', '#e67e22'
    ],
    'risk_thresholds': {
        'low': 0.3,
        'medium': 0.6,
        'high': 0.8,
        'critical': 1.0
    },
    'financial_ratios': {
        'current_ratio': {'healthy': 2.0, 'warning': 1.5, 'critical': 1.0},
        'debt_to_equity': {'healthy': 0.3, 'warning': 0.6, 'critical': 1.0},
        'profit_margin': {'healthy': 0.15, 'warning': 0.05, 'critical': 0.0}
    },
    'industries': [
        'technology', 'manufacturing', 'retail', 'healthcare', 
        'financial', 'energy', 'construction', 'agriculture'
    ]
}

class AnalyticsEngine:
    """
    Main analytics engine that coordinates all analytics components
    """
    
    def __init__(self, db_config):
        """
        Initialize Analytics Engine
        
        Args:
            db_config (dict): Database configuration
        """
        self.db_config = db_config
        self.financial_analyzer = FinancialAnalyzer(db_config)
        self.chart_generator = ChartGenerator()
        self.ai_insights = AIInsights(db_config)
        self.report_generator = ReportGenerator(db_config)
    
    def generate_company_dashboard(self, company_id, years=None):
        """
        Generate complete dashboard data for a company
        
        Args:
            company_id (str): Company identifier
            years (list): Years to analyze (default: last 3 years)
            
        Returns:
            dict: Complete dashboard data
        """
        try:
            # Get financial analysis
            financial_data = self.financial_analyzer.analyze_company_performance(
                company_id, years
            )
            
            # Generate charts
            charts = self.chart_generator.generate_company_charts(financial_data)
            
            # Get AI insights
            insights = self.ai_insights.generate_company_insights(
                company_id, financial_data
            )
            
            # Compile dashboard
            dashboard = {
                'company_id': company_id,
                'generated_at': financial_data.get('generated_at'),
                'financial_summary': financial_data.get('summary', {}),
                'charts': charts,
                'ai_insights': insights,
                'risk_assessment': financial_data.get('risk_assessment', {}),
                'recommendations': insights.get('recommendations', [])
            }
            
            return dashboard
            
        except Exception as e:
            print(f"Error generating company dashboard: {e}")
            return None
    
    def generate_industry_analysis(self, industry, year=None):
        """
        Generate industry-wide analysis
        
        Args:
            industry (str): Industry name
            year (int): Year to analyze (default: current year)
            
        Returns:
            dict: Industry analysis data
        """
        try:
            # Get industry data
            industry_data = self.financial_analyzer.analyze_industry_performance(
                industry, year
            )
            
            # Generate industry charts
            charts = self.chart_generator.generate_industry_charts(industry_data)
            
            # Get industry insights
            insights = self.ai_insights.generate_industry_insights(
                industry, industry_data
            )
            
            # Compile analysis
            analysis = {
                'industry': industry,
                'year': year,
                'generated_at': industry_data.get('generated_at'),
                'industry_metrics': industry_data.get('metrics', {}),
                'company_rankings': industry_data.get('rankings', []),
                'charts': charts,
                'insights': insights,
                'benchmarks': industry_data.get('benchmarks', {})
            }
            
            return analysis
            
        except Exception as e:
            print(f"Error generating industry analysis: {e}")
            return None
    
    def generate_comprehensive_report(self, company_id, report_type='full'):
        """
        Generate comprehensive financial report
        
        Args:
            company_id (str): Company identifier
            report_type (str): Type of report ('full', 'summary', 'risk')
            
        Returns:
            dict: Generated report
        """
        try:
            return self.report_generator.generate_report(
                company_id, report_type
            )
        except Exception as e:
            print(f"Error generating report: {e}")
            return None

def get_analytics_config():
    """Get analytics configuration"""
    return ANALYTICS_CONFIG

def validate_analytics_input(data):
    """
    Validate input data for analytics functions
    
    Args:
        data (dict): Input data to validate
        
    Returns:
        tuple: (is_valid, error_message)
    """
    required_fields = ['company_id']
    
    for field in required_fields:
        if field not in data or not data[field]:
            return False, f"Missing required field: {field}"
    
    # Validate company_id format
    company_id = data['company_id']
    if not isinstance(company_id, str) or len(company_id) < 3:
        return False, "Invalid company_id format"
    
    # Validate year if provided
    if 'year' in data and data['year']:
        try:
            year = int(data['year'])
            if year < 2000 or year > 2030:
                return False, "Year must be between 2000 and 2030"
        except ValueError:
            return False, "Invalid year format"
    
    # Validate industry if provided
    if 'industry' in data and data['industry']:
        if data['industry'] not in ANALYTICS_CONFIG['industries']:
            return False, f"Invalid industry. Supported: {', '.join(ANALYTICS_CONFIG['industries'])}"
    
    return True, None

def calculate_risk_score(financial_metrics):
    """
    Calculate overall risk score based on financial metrics
    
    Args:
        financial_metrics (dict): Financial metrics data
        
    Returns:
        dict: Risk score and breakdown
    """
    try:
        risk_factors = []
        total_score = 0
        
        # Current ratio risk
        current_ratio = financial_metrics.get('current_ratio', 0)
        if current_ratio < 1.0:
            risk_factors.append('Low liquidity')
            total_score += 30
        elif current_ratio < 1.5:
            risk_factors.append('Moderate liquidity risk')
            total_score += 15
        
        # Debt to equity risk
        debt_ratio = financial_metrics.get('debt_to_equity_ratio', 0)
        if debt_ratio > 1.0:
            risk_factors.append('High debt levels')
            total_score += 25
        elif debt_ratio > 0.6:
            risk_factors.append('Moderate debt levels')
            total_score += 10
        
        # Profitability risk
        net_income = financial_metrics.get('net_income', 0)
        if net_income < 0:
            risk_factors.append('Negative profitability')
            total_score += 25
        
        # Cash flow risk
        operating_cf = financial_metrics.get('operating_cash_flow', 0)
        if operating_cf < 0:
            risk_factors.append('Negative operating cash flow')
            total_score += 20
        
        # Determine risk level
        if total_score >= 70:
            risk_level = 'critical'
        elif total_score >= 50:
            risk_level = 'high'
        elif total_score >= 30:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'risk_score': total_score,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'max_score': 100
        }
        
    except Exception as e:
        print(f"Error calculating risk score: {e}")
        return {
            'risk_score': 50,
            'risk_level': 'unknown',
            'risk_factors': ['Unable to calculate risk'],
            'max_score': 100
        }

def format_currency(amount, currency='USD'):
    """
    Format currency amounts for display
    
    Args:
        amount (float): Amount to format
        currency (str): Currency code
        
    Returns:
        str: Formatted currency string
    """
    try:
        if amount is None:
            return 'N/A'
        
        if abs(amount) >= 1000000000:
            return f"${amount/1000000000:.1f}B"
        elif abs(amount) >= 1000000:
            return f"${amount/1000000:.1f}M"
        elif abs(amount) >= 1000:
            return f"${amount/1000:.1f}K"
        else:
            return f"${amount:.2f}"
    except:
        return 'N/A'

def format_percentage(value, decimal_places=1):
    """
    Format percentage values for display
    
    Args:
        value (float): Percentage value (0.0 to 1.0)
        decimal_places (int): Number of decimal places
        
    Returns:
        str: Formatted percentage string
    """
    try:
        if value is None:
            return 'N/A'
        return f"{value * 100:.{decimal_places}f}%"
    except:
        return 'N/A'

def get_trend_direction(current_value, previous_value):
    """
    Determine trend direction between two values
    
    Args:
        current_value (float): Current period value
        previous_value (float): Previous period value
        
    Returns:
        dict: Trend information
    """
    try:
        if previous_value == 0:
            return {'direction': 'flat', 'change': 0, 'change_percent': 0}
        
        change = current_value - previous_value
        change_percent = (change / abs(previous_value)) * 100
        
        if change > 0:
            direction = 'up'
        elif change < 0:
            direction = 'down'
        else:
            direction = 'flat'
        
        return {
            'direction': direction,
            'change': change,
            'change_percent': change_percent
        }
        
    except Exception as e:
        print(f"Error calculating trend: {e}")
        return {'direction': 'flat', 'change': 0, 'change_percent': 0}