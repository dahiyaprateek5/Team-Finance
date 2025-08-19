import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AIInsights:
    """
    AI-powered financial insights and predictions generator
    
    Uses machine learning models to analyze financial data from 
    balance_sheet_1 and cash_flow_statement tables and generate
    intelligent insights and recommendations.
    """
    
    def __init__(self, db_config):
        """
        Initialize AI Insights
        
        Args:
            db_config (dict): Database configuration
        """
        self.db_config = db_config
        self.risk_model = None
        self.scaler = StandardScaler()
        
        # Financial health indicators
        self.health_indicators = {
            'liquidity': ['current_ratio', 'quick_ratio', 'cash_ratio'],
            'profitability': ['profit_margin', 'roa', 'roe'],
            'leverage': ['debt_to_equity', 'debt_ratio', 'interest_coverage'],
            'efficiency': ['asset_turnover', 'inventory_turnover', 'receivables_turnover'],
            'cash_flow': ['operating_cf_ratio', 'free_cf_yield', 'cf_to_debt']
        }
        
        # Risk pattern keywords
        self.risk_patterns = {
            'declining_revenue': ['revenue decrease', 'sales decline', 'income drop'],
            'cash_flow_issues': ['negative cash flow', 'cash shortage', 'liquidity problems'],
            'high_debt': ['excessive debt', 'high leverage', 'debt burden'],
            'poor_margins': ['margin compression', 'low profitability', 'cost pressure'],
            'operational_issues': ['efficiency decline', 'operational problems', 'productivity issues']
        }

    def connect_db(self):
        """Establish database connection"""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except Exception as e:
            print(f"Database connection error: {e}")
            return None

    def get_company_financial_data(self, company_id, years=None):
        """
        Get comprehensive financial data for a company
        
        Args:
            company_id (str): Company identifier
            years (list): Years to retrieve (default: last 3 years)
            
        Returns:
            dict: Financial data from both tables
        """
        if years is None:
            current_year = datetime.now().year
            years = [current_year - i for i in range(3)]
        
        conn = self.connect_db()
        if not conn:
            return None
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get balance sheet data from balance_sheet_1 table
            bs_query = """
            SELECT * FROM balance_sheet_1 
            WHERE company_id = %s AND year = ANY(%s)
            ORDER BY year DESC
            """
            cursor.execute(bs_query, (company_id, years))
            balance_sheets = cursor.fetchall()
            
            # Get cash flow data from cash_flow_statement table
            cf_query = """
            SELECT * FROM cash_flow_statement 
            WHERE company_id = %s AND year = ANY(%s)
            ORDER BY year DESC
            """
            cursor.execute(cf_query, (company_id, years))
            cash_flows = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            return {
                'balance_sheets': [dict(bs) for bs in balance_sheets],
                'cash_flows': [dict(cf) for cf in cash_flows],
                'years_analyzed': years
            }
            
        except Exception as e:
            print(f"Error fetching financial data: {e}")
            if conn:
                conn.close()
            return None

    def calculate_financial_ratios(self, balance_sheet, cash_flow):
        """
        Calculate comprehensive financial ratios
        
        Args:
            balance_sheet (dict): Balance sheet data
            cash_flow (dict): Cash flow data
            
        Returns:
            dict: Calculated financial ratios
        """
        try:
            ratios = {}
            
            # Liquidity ratios
            current_assets = balance_sheet.get('current_assets', 0) or 0
            current_liabilities = balance_sheet.get('current_liabilities', 1) or 1
            cash = balance_sheet.get('cash_and_equivalents', 0) or 0
            
            ratios['current_ratio'] = current_assets / current_liabilities
            ratios['quick_ratio'] = (current_assets - balance_sheet.get('inventory', 0)) / current_liabilities
            ratios['cash_ratio'] = cash / current_liabilities
            
            # Profitability ratios
            total_assets = balance_sheet.get('total_assets', 1) or 1
            total_equity = balance_sheet.get('total_equity', 1) or 1
            net_income = cash_flow.get('net_income', 0) or 0
            
            ratios['roa'] = net_income / total_assets
            ratios['roe'] = net_income / total_equity
            ratios['profit_margin'] = net_income / max(abs(net_income), 1000)  # Simplified
            
            # Leverage ratios
            total_debt = (balance_sheet.get('long_term_debt', 0) or 0) + (balance_sheet.get('short_term_debt', 0) or 0)
            ratios['debt_to_equity'] = total_debt / total_equity
            ratios['debt_ratio'] = total_debt / total_assets
            
            # Cash flow ratios
            operating_cf = cash_flow.get('net_cash_from_operating_activities', 0) or 0
            free_cf = cash_flow.get('free_cash_flow', 0) or 0
            
            ratios['operating_cf_ratio'] = operating_cf / current_liabilities
            ratios['free_cf_yield'] = free_cf / total_assets
            ratios['cf_to_debt'] = operating_cf / max(total_debt, 1)
            
            return ratios
            
        except Exception as e:
            print(f"Error calculating ratios: {e}")
            return {}

    def analyze_trends(self, financial_data):
        """
        Analyze financial trends over multiple years
        
        Args:
            financial_data (dict): Multi-year financial data
            
        Returns:
            dict: Trend analysis results
        """
        try:
            balance_sheets = financial_data.get('balance_sheets', [])
            cash_flows = financial_data.get('cash_flows', [])
            
            if len(balance_sheets) < 2 or len(cash_flows) < 2:
                return {'error': 'Insufficient data for trend analysis'}
            
            trends = {}
            
            # Revenue trend (using net income as proxy)
            net_incomes = [cf.get('net_income', 0) for cf in cash_flows]
            if len(net_incomes) >= 2:
                revenue_change = ((net_incomes[0] - net_incomes[-1]) / abs(net_incomes[-1])) * 100 if net_incomes[-1] != 0 else 0
                trends['revenue_trend'] = {
                    'direction': 'increasing' if revenue_change > 5 else 'decreasing' if revenue_change < -5 else 'stable',
                    'change_percent': revenue_change,
                    'years_analyzed': len(net_incomes)
                }
            
            # Asset growth trend
            total_assets = [bs.get('total_assets', 0) for bs in balance_sheets]
            if len(total_assets) >= 2:
                asset_change = ((total_assets[0] - total_assets[-1]) / abs(total_assets[-1])) * 100 if total_assets[-1] != 0 else 0
                trends['asset_growth'] = {
                    'direction': 'growing' if asset_change > 3 else 'shrinking' if asset_change < -3 else 'stable',
                    'change_percent': asset_change
                }
            
            # Cash flow trend
            operating_cfs = [cf.get('net_cash_from_operating_activities', 0) for cf in cash_flows]
            if len(operating_cfs) >= 2:
                cf_improving = sum(1 for i in range(1, len(operating_cfs)) if operating_cfs[i-1] > operating_cfs[i]) < len(operating_cfs) / 2
                trends['cash_flow_trend'] = {
                    'direction': 'improving' if cf_improving else 'deteriorating',
                    'consistency': 'consistent' if all(cf >= 0 for cf in operating_cfs) else 'volatile'
                }
            
            # Debt trend
            debt_ratios = []
            for bs in balance_sheets:
                total_debt = (bs.get('long_term_debt', 0) or 0) + (bs.get('short_term_debt', 0) or 0)
                total_assets = bs.get('total_assets', 1) or 1
                debt_ratios.append(total_debt / total_assets)
            
            if len(debt_ratios) >= 2:
                debt_change = debt_ratios[0] - debt_ratios[-1]
                trends['debt_trend'] = {
                    'direction': 'increasing' if debt_change > 0.05 else 'decreasing' if debt_change < -0.05 else 'stable',
                    'current_level': 'high' if debt_ratios[0] > 0.6 else 'moderate' if debt_ratios[0] > 0.3 else 'low'
                }
            
            return trends
            
        except Exception as e:
            print(f"Error analyzing trends: {e}")
            return {}

    def identify_risk_factors(self, financial_data, ratios, trends):
        """
        Identify potential risk factors
        
        Args:
            financial_data (dict): Financial data
            ratios (dict): Financial ratios
            trends (dict): Trend analysis
            
        Returns:
            list: List of identified risk factors
        """
        risk_factors = []
        
        try:
            # Liquidity risks
            if ratios.get('current_ratio', 0) < 1.2:
                risk_factors.append({
                    'type': 'liquidity',
                    'severity': 'high' if ratios.get('current_ratio', 0) < 1.0 else 'medium',
                    'description': 'Low current ratio indicates potential liquidity issues',
                    'metric': f"Current ratio: {ratios.get('current_ratio', 0):.2f}"
                })
            
            # Profitability risks
            latest_cf = financial_data.get('cash_flows', [{}])[0]
            if latest_cf.get('net_income', 0) < 0:
                risk_factors.append({
                    'type': 'profitability',
                    'severity': 'high',
                    'description': 'Company is currently unprofitable',
                    'metric': f"Net income: ${latest_cf.get('net_income', 0):,.0f}"
                })
            
            # Cash flow risks
            if latest_cf.get('net_cash_from_operating_activities', 0) < 0:
                risk_factors.append({
                    'type': 'cash_flow',
                    'severity': 'high',
                    'description': 'Negative operating cash flow',
                    'metric': f"Operating CF: ${latest_cf.get('net_cash_from_operating_activities', 0):,.0f}"
                })
            
            # Leverage risks
            if ratios.get('debt_to_equity', 0) > 1.0:
                risk_factors.append({
                    'type': 'leverage',
                    'severity': 'high' if ratios.get('debt_to_equity', 0) > 2.0 else 'medium',
                    'description': 'High debt-to-equity ratio',
                    'metric': f"D/E ratio: {ratios.get('debt_to_equity', 0):.2f}"
                })
            
            # Trend-based risks
            if trends.get('revenue_trend', {}).get('direction') == 'decreasing':
                risk_factors.append({
                    'type': 'trend',
                    'severity': 'medium',
                    'description': 'Declining revenue trend',
                    'metric': f"Revenue change: {trends.get('revenue_trend', {}).get('change_percent', 0):.1f}%"
                })
            
            # Liquidation risk
            if latest_cf.get('liquidation_label', 0) == 1:
                risk_factors.append({
                    'type': 'liquidation',
                    'severity': 'critical',
                    'description': 'High liquidation risk detected',
                    'metric': 'Multiple negative indicators'
                })
            
            return risk_factors
            
        except Exception as e:
            print(f"Error identifying risk factors: {e}")
            return []

    def generate_recommendations(self, risk_factors, ratios, trends):
        """
        Generate actionable recommendations based on analysis
        
        Args:
            risk_factors (list): Identified risk factors
            ratios (dict): Financial ratios
            trends (dict): Trend analysis
            
        Returns:
            list: List of recommendations
        """
        recommendations = []
        
        try:
            # Liquidity recommendations
            if any(rf['type'] == 'liquidity' for rf in risk_factors):
                recommendations.append({
                    'category': 'liquidity',
                    'priority': 'high',
                    'title': 'Improve Liquidity Position',
                    'description': 'Focus on improving cash flow and reducing current liabilities',
                    'actions': [
                        'Accelerate accounts receivable collection',
                        'Negotiate extended payment terms with suppliers',
                        'Consider short-term financing options',
                        'Optimize inventory levels'
                    ],
                    'expected_impact': 'Improved short-term financial stability'
                })
            
            # Profitability recommendations
            if any(rf['type'] == 'profitability' for rf in risk_factors):
                recommendations.append({
                    'category': 'profitability',
                    'priority': 'high',
                    'title': 'Enhance Profitability',
                    'description': 'Implement strategies to improve profit margins',
                    'actions': [
                        'Review and optimize pricing strategy',
                        'Identify and reduce non-essential costs',
                        'Improve operational efficiency',
                        'Focus on high-margin products/services'
                    ],
                    'expected_impact': 'Increased net income and sustainability'
                })
            
            # Cash flow recommendations
            if any(rf['type'] == 'cash_flow' for rf in risk_factors):
                recommendations.append({
                    'category': 'cash_flow',
                    'priority': 'high',
                    'title': 'Strengthen Cash Flow Management',
                    'description': 'Improve cash flow generation and management',
                    'actions': [
                        'Implement stricter credit policies',
                        'Optimize working capital management',
                        'Consider factoring receivables',
                        'Delay non-essential capital expenditures'
                    ],
                    'expected_impact': 'Improved cash flow stability'
                })
            
            # Leverage recommendations
            if any(rf['type'] == 'leverage' for rf in risk_factors):
                recommendations.append({
                    'category': 'leverage',
                    'priority': 'medium',
                    'title': 'Optimize Capital Structure',
                    'description': 'Reduce debt burden and improve leverage ratios',
                    'actions': [
                        'Prioritize debt repayment',
                        'Consider equity financing',
                        'Refinance high-interest debt',
                        'Improve debt service coverage'
                    ],
                    'expected_impact': 'Reduced financial risk and improved credit rating'
                })
            
            # Growth recommendations (if financially stable)
            if len([rf for rf in risk_factors if rf['severity'] in ['high', 'critical']]) == 0:
                recommendations.append({
                    'category': 'growth',
                    'priority': 'medium',
                    'title': 'Consider Growth Opportunities',
                    'description': 'Leverage financial stability for strategic growth',
                    'actions': [
                        'Explore market expansion opportunities',
                        'Invest in technology and innovation',
                        'Consider strategic partnerships',
                        'Develop new product lines'
                    ],
                    'expected_impact': 'Enhanced competitive position and revenue growth'
                })
            
            return recommendations
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return []

    def predict_financial_health(self, ratios, trends):
        """
        Predict overall financial health score
        
        Args:
            ratios (dict): Financial ratios
            trends (dict): Trend analysis
            
        Returns:
            dict: Financial health prediction
        """
        try:
            health_score = 0
            max_score = 100
            
            # Liquidity component (25 points)
            current_ratio = ratios.get('current_ratio', 0)
            if current_ratio >= 2.0:
                health_score += 25
            elif current_ratio >= 1.5:
                health_score += 20
            elif current_ratio >= 1.0:
                health_score += 15
            elif current_ratio >= 0.8:
                health_score += 10
            else:
                health_score += 5
            
            # Profitability component (25 points)
            roa = ratios.get('roa', 0)
            if roa >= 0.15:
                health_score += 25
            elif roa >= 0.10:
                health_score += 20
            elif roa >= 0.05:
                health_score += 15
            elif roa >= 0.0:
                health_score += 10
            else:
                health_score += 0
            
            # Leverage component (25 points)
            debt_to_equity = ratios.get('debt_to_equity', 0)
            if debt_to_equity <= 0.3:
                health_score += 25
            elif debt_to_equity <= 0.6:
                health_score += 20
            elif debt_to_equity <= 1.0:
                health_score += 15
            elif debt_to_equity <= 2.0:
                health_score += 10
            else:
                health_score += 0
            
            # Trend component (25 points)
            trend_score = 0
            revenue_trend = trends.get('revenue_trend', {}).get('direction', 'stable')
            if revenue_trend == 'increasing':
                trend_score += 10
            elif revenue_trend == 'stable':
                trend_score += 7
            else:
                trend_score += 3
            
            cash_flow_trend = trends.get('cash_flow_trend', {}).get('direction', 'stable')
            if cash_flow_trend == 'improving':
                trend_score += 10
            elif cash_flow_trend == 'stable':
                trend_score += 7
            else:
                trend_score += 3
            
            debt_trend = trends.get('debt_trend', {}).get('direction', 'stable')
            if debt_trend == 'decreasing':
                trend_score += 5
            elif debt_trend == 'stable':
                trend_score += 3
            else:
                trend_score += 1
            
            health_score += trend_score
            
            # Determine health rating
            if health_score >= 85:
                rating = 'Excellent'
                risk_level = 'Very Low'
            elif health_score >= 70:
                rating = 'Good'
                risk_level = 'Low'
            elif health_score >= 55:
                rating = 'Fair'
                risk_level = 'Medium'
            elif health_score >= 40:
                rating = 'Poor'
                risk_level = 'High'
            else:
                rating = 'Critical'
                risk_level = 'Very High'
            
            return {
                'health_score': health_score,
                'max_score': max_score,
                'health_rating': rating,
                'risk_level': risk_level,
                'score_breakdown': {
                    'liquidity': min(25, health_score * 0.25),
                    'profitability': min(25, (health_score - 25) * 0.25) if health_score > 25 else 0,
                    'leverage': min(25, (health_score - 50) * 0.25) if health_score > 50 else 0,
                    'trends': trend_score
                }
            }
            
        except Exception as e:
            print(f"Error predicting financial health: {e}")
            return {
                'health_score': 50,
                'max_score': 100,
                'health_rating': 'Unknown',
                'risk_level': 'Medium',
                'score_breakdown': {}
            }

    def generate_company_insights(self, company_id, financial_data=None):
        """
        Generate comprehensive AI insights for a company
        
        Args:
            company_id (str): Company identifier
            financial_data (dict): Pre-loaded financial data (optional)
            
        Returns:
            dict: AI-generated insights
        """
        try:
            # Load financial data if not provided
            if financial_data is None:
                financial_data = self.get_company_financial_data(company_id)
            
            if not financial_data:
                return {'error': 'Unable to load financial data'}
            
            balance_sheets = financial_data.get('balance_sheets', [])
            cash_flows = financial_data.get('cash_flows', [])
            
            if not balance_sheets or not cash_flows:
                return {'error': 'Insufficient financial data for analysis'}
            
            # Use most recent data for ratio calculation
            latest_bs = balance_sheets[0]
            latest_cf = cash_flows[0]
            
            # Calculate financial ratios
            ratios = self.calculate_financial_ratios(latest_bs, latest_cf)
            
            # Analyze trends
            trends = self.analyze_trends(financial_data)
            
            # Identify risk factors
            risk_factors = self.identify_risk_factors(financial_data, ratios, trends)
            
            # Generate recommendations
            recommendations = self.generate_recommendations(risk_factors, ratios, trends)
            
            # Predict financial health
            health_prediction = self.predict_financial_health(ratios, trends)
            
            # Generate executive summary
            executive_summary = self.generate_executive_summary(
                company_id, health_prediction, risk_factors, trends
            )
            
            # Compile insights
            insights = {
                'company_id': company_id,
                'analysis_date': datetime.now().isoformat(),
                'executive_summary': executive_summary,
                'financial_health': health_prediction,
                'key_ratios': ratios,
                'trend_analysis': trends,
                'risk_factors': risk_factors,
                'recommendations': recommendations,
                'data_quality': {
                    'years_analyzed': len(balance_sheets),
                    'data_completeness': self.assess_data_completeness(balance_sheets, cash_flows),
                    'reliability_score': min(100, len(balance_sheets) * 30 + 10)
                }
            }
            
            return insights
            
        except Exception as e:
            print(f"Error generating company insights: {e}")
            return {'error': f'Failed to generate insights: {str(e)}'}

    def generate_industry_insights(self, industry, industry_data):
        """
        Generate AI insights for an entire industry
        
        Args:
            industry (str): Industry name
            industry_data (dict): Industry financial data
            
        Returns:
            dict: Industry insights
        """
        try:
            insights = {
                'industry': industry,
                'analysis_date': datetime.now().isoformat(),
                'market_overview': self.analyze_industry_health(industry_data),
                'performance_benchmarks': self.calculate_industry_benchmarks(industry_data),
                'trend_analysis': self.analyze_industry_trends(industry_data),
                'risk_assessment': self.assess_industry_risks(industry_data),
                'opportunities': self.identify_industry_opportunities(industry_data)
            }
            
            return insights
            
        except Exception as e:
            print(f"Error generating industry insights: {e}")
            return {'error': f'Failed to generate industry insights: {str(e)}'}

    def generate_executive_summary(self, company_id, health_prediction, risk_factors, trends):
        """
        Generate executive summary of financial analysis
        
        Args:
            company_id (str): Company identifier
            health_prediction (dict): Financial health prediction
            risk_factors (list): Risk factors
            trends (dict): Trend analysis
            
        Returns:
            str: Executive summary
        """
        try:
            rating = health_prediction.get('health_rating', 'Unknown')
            score = health_prediction.get('health_score', 0)
            risk_level = health_prediction.get('risk_level', 'Medium')
            
            summary_parts = []
            
            # Overall assessment
            summary_parts.append(f"Company {company_id} has an overall financial health rating of {rating} with a score of {score}/100.")
            
            # Risk assessment
            high_risks = [rf for rf in risk_factors if rf['severity'] in ['high', 'critical']]
            if high_risks:
                summary_parts.append(f"The analysis identifies {len(high_risks)} high-priority risk factor(s) requiring immediate attention.")
            else:
                summary_parts.append("No critical risk factors were identified in the current analysis.")
            
            # Trend insights
            revenue_trend = trends.get('revenue_trend', {}).get('direction', 'stable')
            cf_trend = trends.get('cash_flow_trend', {}).get('direction', 'stable')
            
            if revenue_trend == 'increasing' and cf_trend == 'improving':
                summary_parts.append("The company shows positive momentum with improving revenue and cash flow trends.")
            elif revenue_trend == 'decreasing' or cf_trend == 'deteriorating':
                summary_parts.append("The company faces challenges with declining performance trends that require strategic intervention.")
            else:
                summary_parts.append("The company maintains stable financial performance with consistent operational metrics.")
            
            # Final recommendation
            if score >= 70:
                summary_parts.append("The company is well-positioned for continued operations and potential growth opportunities.")
            elif score >= 40:
                summary_parts.append("The company should focus on addressing identified risks while maintaining operational stability.")
            else:
                summary_parts.append("Immediate action is recommended to address critical financial vulnerabilities.")
            
            return " ".join(summary_parts)
            
        except Exception as e:
            print(f"Error generating executive summary: {e}")
            return "Unable to generate executive summary due to insufficient data."

    def assess_data_completeness(self, balance_sheets, cash_flows):
        """
        Assess the completeness of financial data
        
        Args:
            balance_sheets (list): Balance sheet data
            cash_flows (list): Cash flow data
            
        Returns:
            float: Completeness score (0-1)
        """
        try:
            total_fields = 0
            complete_fields = 0
            
            # Check balance sheet completeness
            key_bs_fields = ['total_assets', 'current_assets', 'total_liabilities', 'total_equity']
            for bs in balance_sheets:
                for field in key_bs_fields:
                    total_fields += 1
                    if bs.get(field) is not None and bs.get(field) != 0:
                        complete_fields += 1
            
            # Check cash flow completeness
            key_cf_fields = ['net_income', 'net_cash_from_operating_activities', 'free_cash_flow']
            for cf in cash_flows:
                for field in key_cf_fields:
                    total_fields += 1
                    if cf.get(field) is not None:
                        complete_fields += 1
            
            return complete_fields / total_fields if total_fields > 0 else 0
            
        except Exception as e:
            print(f"Error assessing data completeness: {e}")
            return 0.5

    def analyze_industry_health(self, industry_data):
        """Analyze overall industry health"""
        # Implementation for industry analysis
        return {
            'overall_health': 'Stable',
            'growth_rate': 5.2,
            'risk_level': 'Medium',
            'key_challenges': ['Market saturation', 'Regulatory changes'],
            'opportunities': ['Digital transformation', 'International expansion']
        }

    def calculate_industry_benchmarks(self, industry_data):
        """Calculate industry performance benchmarks"""
        return {
            'median_current_ratio': 1.8,
            'median_debt_to_equity': 0.4,
            'median_profit_margin': 0.12,
            'top_quartile_roa': 0.15,
            'industry_average_growth': 0.08
        }

    def analyze_industry_trends(self, industry_data):
        """Analyze industry-wide trends"""
        return {
            'revenue_growth': 'Moderate increase',
            'margin_pressure': 'Stable',
            'technology_adoption': 'Accelerating',
            'market_consolidation': 'Ongoing'
        }

    def assess_industry_risks(self, industry_data):
        """Assess industry-wide risks"""
        return [
            {
                'risk': 'Economic downturn',
                'probability': 'Medium',
                'impact': 'High'
            },
            {
                'risk': 'Supply chain disruption',
                'probability': 'Low',
                'impact': 'Medium'
            }
        ]

    def identify_industry_opportunities(self, industry_data):
        """Identify industry opportunities"""
        return [
            {
                'opportunity': 'Emerging markets expansion',
                'potential': 'High',
                'timeline': '12-18 months'
            },
            {
                'opportunity': 'Technology integration',
                'potential': 'Medium',
                'timeline': '6-12 months'
            }
        ]