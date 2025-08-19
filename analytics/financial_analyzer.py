import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import statistics
import warnings
warnings.filterwarnings('ignore')

class FinancialAnalyzer:
    """
    Complete financial analysis engine for the Financial Risk Assessment Platform
    
    Analyzes financial data from balance_sheet_1 and cash_flow_statement tables
    to provide comprehensive financial insights, ratios, and performance metrics.
    """
    
    def __init__(self, db_config):
        """
        Initialize Financial Analyzer
        
        Args:
            db_config (dict): Database configuration
        """
        self.db_config = db_config
        
        # Financial ratio categories and benchmarks
        self.ratio_benchmarks = {
            'liquidity': {
                'current_ratio': {'excellent': 2.5, 'good': 2.0, 'fair': 1.5, 'poor': 1.0},
                'quick_ratio': {'excellent': 1.5, 'good': 1.0, 'fair': 0.8, 'poor': 0.5},
                'cash_ratio': {'excellent': 0.5, 'good': 0.3, 'fair': 0.2, 'poor': 0.1}
            },
            'profitability': {
                'profit_margin': {'excellent': 0.20, 'good': 0.15, 'fair': 0.10, 'poor': 0.05},
                'roa': {'excellent': 0.15, 'good': 0.10, 'fair': 0.05, 'poor': 0.02},
                'roe': {'excellent': 0.20, 'good': 0.15, 'fair': 0.10, 'poor': 0.05}
            },
            'leverage': {
                'debt_to_equity': {'excellent': 0.3, 'good': 0.5, 'fair': 0.8, 'poor': 1.5},
                'debt_ratio': {'excellent': 0.2, 'good': 0.4, 'fair': 0.6, 'poor': 0.8},
                'interest_coverage': {'excellent': 10, 'good': 5, 'fair': 2.5, 'poor': 1.5}
            },
            'efficiency': {
                'asset_turnover': {'excellent': 2.0, 'good': 1.5, 'fair': 1.0, 'poor': 0.5},
                'inventory_turnover': {'excellent': 12, 'good': 8, 'fair': 6, 'poor': 4}
            }
        }
        
        # Industry multipliers for benchmarks
        self.industry_multipliers = {
            'technology': {'profitability': 1.2, 'liquidity': 1.1, 'leverage': 0.8, 'efficiency': 1.3},
            'manufacturing': {'profitability': 0.9, 'liquidity': 0.9, 'leverage': 1.1, 'efficiency': 1.0},
            'retail': {'profitability': 0.8, 'liquidity': 0.8, 'leverage': 1.0, 'efficiency': 1.5},
            'healthcare': {'profitability': 1.1, 'liquidity': 1.0, 'leverage': 0.9, 'efficiency': 0.9},
            'financial': {'profitability': 1.0, 'liquidity': 0.7, 'leverage': 1.5, 'efficiency': 0.8}
        }

    def connect_db(self):
        """Establish database connection"""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except Exception as e:
            print(f"Database connection error: {e}")
            return None

    def get_company_data(self, company_id, years=None):
        """
        Retrieve comprehensive company financial data
        
        Args:
            company_id (str): Company identifier
            years (list): Years to retrieve (default: last 3 years)
            
        Returns:
            dict: Complete financial data
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
                'company_id': company_id,
                'balance_sheets': [dict(bs) for bs in balance_sheets],
                'cash_flows': [dict(cf) for cf in cash_flows],
                'years_requested': years,
                'data_retrieved_at': datetime.now()
            }
            
        except Exception as e:
            print(f"Error retrieving company data: {e}")
            if conn:
                conn.close()
            return None

    def calculate_liquidity_ratios(self, balance_sheet):
        """
        Calculate liquidity ratios from balance sheet data
        
        Args:
            balance_sheet (dict): Balance sheet data
            
        Returns:
            dict: Liquidity ratios
        """
        try:
            current_assets = balance_sheet.get('current_assets', 0) or 0
            current_liabilities = balance_sheet.get('current_liabilities', 1) or 1
            cash = balance_sheet.get('cash_and_equivalents', 0) or 0
            inventory = balance_sheet.get('inventory', 0) or 0
            accounts_receivable = balance_sheet.get('accounts_receivable', 0) or 0
            
            ratios = {
                'current_ratio': current_assets / current_liabilities,
                'quick_ratio': (current_assets - inventory) / current_liabilities,
                'cash_ratio': cash / current_liabilities,
                'working_capital': current_assets - current_liabilities,
                'working_capital_ratio': (current_assets - current_liabilities) / current_assets if current_assets > 0 else 0
            }
            
            return ratios
            
        except Exception as e:
            print(f"Error calculating liquidity ratios: {e}")
            return {}

    def calculate_profitability_ratios(self, balance_sheet, cash_flow):
        """
        Calculate profitability ratios
        
        Args:
            balance_sheet (dict): Balance sheet data
            cash_flow (dict): Cash flow data
            
        Returns:
            dict: Profitability ratios
        """
        try:
            total_assets = balance_sheet.get('total_assets', 1) or 1
            total_equity = balance_sheet.get('total_equity', 1) or 1
            net_income = cash_flow.get('net_income', 0) or 0
            
            # Estimate revenue (simplified approach using operating cash flow as proxy)
            estimated_revenue = abs(cash_flow.get('net_cash_from_operating_activities', net_income) or net_income)
            if estimated_revenue == 0:
                estimated_revenue = abs(net_income) * 1.2  # Simple estimation
            
            ratios = {
                'profit_margin': net_income / max(estimated_revenue, 1),
                'roa': net_income / total_assets,
                'roe': net_income / total_equity,
                'gross_profit_margin': max(0, estimated_revenue - abs(net_income * 0.6)) / max(estimated_revenue, 1),  # Estimated
                'operating_margin': cash_flow.get('net_cash_from_operating_activities', 0) / max(estimated_revenue, 1)
            }
            
            return ratios
            
        except Exception as e:
            print(f"Error calculating profitability ratios: {e}")
            return {}

    def calculate_leverage_ratios(self, balance_sheet, cash_flow):
        """
        Calculate leverage and solvency ratios
        
        Args:
            balance_sheet (dict): Balance sheet data
            cash_flow (dict): Cash flow data
            
        Returns:
            dict: Leverage ratios
        """
        try:
            total_assets = balance_sheet.get('total_assets', 1) or 1
            total_equity = balance_sheet.get('total_equity', 1) or 1
            total_liabilities = balance_sheet.get('total_liabilities', 0) or 0
            long_term_debt = balance_sheet.get('long_term_debt', 0) or 0
            short_term_debt = balance_sheet.get('short_term_debt', 0) or 0
            
            total_debt = long_term_debt + short_term_debt
            
            # Estimate interest expense (simplified)
            estimated_interest = total_debt * 0.05  # Assume 5% average interest rate
            ebit = cash_flow.get('net_income', 0) + estimated_interest  # Simplified EBIT
            
            ratios = {
                'debt_to_equity': total_debt / total_equity,
                'debt_ratio': total_debt / total_assets,
                'equity_ratio': total_equity / total_assets,
                'debt_to_assets': total_liabilities / total_assets,
                'long_term_debt_to_equity': long_term_debt / total_equity,
                'interest_coverage': ebit / max(estimated_interest, 1),
                'debt_service_coverage': cash_flow.get('net_cash_from_operating_activities', 0) / max(estimated_interest, 1)
            }
            
            return ratios
            
        except Exception as e:
            print(f"Error calculating leverage ratios: {e}")
            return {}

    def calculate_efficiency_ratios(self, balance_sheet, cash_flow):
        """
        Calculate efficiency and activity ratios
        
        Args:
            balance_sheet (dict): Balance sheet data
            cash_flow (dict): Cash flow data
            
        Returns:
            dict: Efficiency ratios
        """
        try:
            total_assets = balance_sheet.get('total_assets', 1) or 1
            inventory = balance_sheet.get('inventory', 1) or 1
            accounts_receivable = balance_sheet.get('accounts_receivable', 1) or 1
            
            # Estimate revenue and cost of goods sold
            estimated_revenue = abs(cash_flow.get('net_cash_from_operating_activities', 0) or 0)
            if estimated_revenue == 0:
                estimated_revenue = abs(cash_flow.get('net_income', 0)) * 1.5
            
            estimated_cogs = estimated_revenue * 0.7  # Assume 70% cost ratio
            
            ratios = {
                'asset_turnover': estimated_revenue / total_assets,
                'inventory_turnover': estimated_cogs / inventory,
                'receivables_turnover': estimated_revenue / accounts_receivable,
                'days_sales_outstanding': 365 / max(estimated_revenue / accounts_receivable, 1),
                'days_inventory_outstanding': 365 / max(estimated_cogs / inventory, 1)
            }
            
            return ratios
            
        except Exception as e:
            print(f"Error calculating efficiency ratios: {e}")
            return {}

    def calculate_cash_flow_ratios(self, balance_sheet, cash_flow):
        """
        Calculate cash flow ratios
        
        Args:
            balance_sheet (dict): Balance sheet data
            cash_flow (dict): Cash flow data
            
        Returns:
            dict: Cash flow ratios
        """
        try:
            operating_cf = cash_flow.get('net_cash_from_operating_activities', 0) or 0
            free_cf = cash_flow.get('free_cash_flow', 0) or 0
            net_income = cash_flow.get('net_income', 1) or 1
            current_liabilities = balance_sheet.get('current_liabilities', 1) or 1
            total_debt = (balance_sheet.get('long_term_debt', 0) or 0) + (balance_sheet.get('short_term_debt', 0) or 0)
            
            ratios = {
                'operating_cf_ratio': operating_cf / current_liabilities,
                'free_cf_ratio': free_cf / operating_cf if operating_cf != 0 else 0,
                'cf_to_net_income': operating_cf / net_income,
                'cf_coverage_ratio': operating_cf / max(total_debt, 1),
                'cash_return_on_assets': operating_cf / balance_sheet.get('total_assets', 1)
            }
            
            return ratios
            
        except Exception as e:
            print(f"Error calculating cash flow ratios: {e}")
            return {}

    def assess_ratio_performance(self, ratio_value, ratio_name, category, industry='general'):
        """
        Assess ratio performance against benchmarks
        
        Args:
            ratio_value (float): Calculated ratio value
            ratio_name (str): Name of the ratio
            category (str): Ratio category (liquidity, profitability, etc.)
            industry (str): Industry for adjusted benchmarks
            
        Returns:
            dict: Performance assessment
        """
        try:
            benchmarks = self.ratio_benchmarks.get(category, {}).get(ratio_name, {})
            if not benchmarks:
                return {'rating': 'unknown', 'score': 50, 'benchmark': 'N/A'}
            
            # Apply industry multipliers
            industry_mult = self.industry_multipliers.get(industry, {}).get(category, 1.0)
            adjusted_benchmarks = {k: v * industry_mult for k, v in benchmarks.items()}
            
            # Determine rating
            if ratio_value >= adjusted_benchmarks.get('excellent', 999):
                rating = 'excellent'
                score = 90
            elif ratio_value >= adjusted_benchmarks.get('good', 999):
                rating = 'good'
                score = 75
            elif ratio_value >= adjusted_benchmarks.get('fair', 999):
                rating = 'fair'
                score = 60
            elif ratio_value >= adjusted_benchmarks.get('poor', 0):
                rating = 'poor'
                score = 40
            else:
                rating = 'critical'
                score = 20
            
            return {
                'rating': rating,
                'score': score,
                'benchmark': adjusted_benchmarks,
                'industry_adjusted': industry != 'general'
            }
            
        except Exception as e:
            print(f"Error assessing ratio performance: {e}")
            return {'rating': 'unknown', 'score': 50, 'benchmark': 'N/A'}

    def analyze_financial_trends(self, balance_sheets, cash_flows):
        """
        Analyze financial trends across multiple years
        
        Args:
            balance_sheets (list): Historical balance sheet data
            cash_flows (list): Historical cash flow data
            
        Returns:
            dict: Trend analysis
        """
        try:
            if len(balance_sheets) < 2 or len(cash_flows) < 2:
                return {'error': 'Insufficient data for trend analysis'}
            
            trends = {
                'revenue_growth': [],
                'asset_growth': [],
                'equity_growth': [],
                'cash_flow_trend': [],
                'debt_trend': []
            }
            
            # Sort by year
            balance_sheets = sorted(balance_sheets, key=lambda x: x['year'])
            cash_flows = sorted(cash_flows, key=lambda x: x['year'])
            
            # Calculate year-over-year changes
            for i in range(1, len(balance_sheets)):
                prev_bs = balance_sheets[i-1]
                curr_bs = balance_sheets[i]
                
                # Asset growth
                asset_growth = ((curr_bs.get('total_assets', 0) - prev_bs.get('total_assets', 0)) / 
                               max(prev_bs.get('total_assets', 1), 1)) * 100
                trends['asset_growth'].append(asset_growth)
                
                # Equity growth
                equity_growth = ((curr_bs.get('total_equity', 0) - prev_bs.get('total_equity', 0)) / 
                                max(prev_bs.get('total_equity', 1), 1)) * 100
                trends['equity_growth'].append(equity_growth)
                
                # Debt trend
                prev_debt = (prev_bs.get('long_term_debt', 0) + prev_bs.get('short_term_debt', 0))
                curr_debt = (curr_bs.get('long_term_debt', 0) + curr_bs.get('short_term_debt', 0))
                debt_trend = ((curr_debt - prev_debt) / max(prev_debt, 1)) * 100
                trends['debt_trend'].append(debt_trend)
            
            for i in range(1, len(cash_flows)):
                prev_cf = cash_flows[i-1]
                curr_cf = cash_flows[i]
                
                # Cash flow trend
                cf_trend = ((curr_cf.get('net_cash_from_operating_activities', 0) - 
                           prev_cf.get('net_cash_from_operating_activities', 0)) / 
                          max(abs(prev_cf.get('net_cash_from_operating_activities', 1)), 1)) * 100
                trends['cash_flow_trend'].append(cf_trend)
            
            # Calculate average trends
            trend_summary = {}
            for key, values in trends.items():
                if values:
                    trend_summary[key] = {
                        'average': statistics.mean(values),
                        'values': values,
                        'direction': 'improving' if statistics.mean(values) > 0 else 'declining'
                    }
            
            return trend_summary
            
        except Exception as e:
            print(f"Error analyzing trends: {e}")
            return {}

    def assess_financial_risks(self, balance_sheet, cash_flow, ratio_assessments):
        """
        Assess financial risks based on ratios and financial data
        
        Args:
            balance_sheet (dict): Latest balance sheet data
            cash_flow (dict): Latest cash flow data
            ratio_assessments (dict): Calculated ratio assessments
            
        Returns:
            dict: Risk assessment
        """
        try:
            risks = {
                'liquidity_risk': 'low',
                'solvency_risk': 'low',
                'profitability_risk': 'low',
                'operational_risk': 'low',
                'overall_risk': 'low'
            }
            
            risk_factors = []
            
            # Liquidity risk assessment
            current_ratio = ratio_assessments.get('liquidity', {}).get('current_ratio', {}).get('assessment', {}).get('score', 50)
            if current_ratio < 40:
                risks['liquidity_risk'] = 'high'
                risk_factors.append('Poor current ratio indicates liquidity problems')
            elif current_ratio < 60:
                risks['liquidity_risk'] = 'medium'
            
            # Solvency risk assessment
            debt_to_equity = ratio_assessments.get('leverage', {}).get('debt_to_equity', {}).get('value', 0)
            if debt_to_equity > 1.5:
                risks['solvency_risk'] = 'high'
                risk_factors.append('High debt-to-equity ratio indicates solvency risk')
            elif debt_to_equity > 0.8:
                risks['solvency_risk'] = 'medium'
            
            # Profitability risk
            net_income = cash_flow.get('net_income', 0)
            if net_income < 0:
                risks['profitability_risk'] = 'high'
                risk_factors.append('Negative net income indicates profitability concerns')
            
            # Operational risk
            operating_cf = cash_flow.get('net_cash_from_operating_activities', 0)
            if operating_cf < 0:
                risks['operational_risk'] = 'high'
                risk_factors.append('Negative operating cash flow indicates operational problems')
            
            # Overall risk assessment
            high_risks = sum(1 for risk in risks.values() if risk == 'high')
            medium_risks = sum(1 for risk in risks.values() if risk == 'medium')
            
            if high_risks >= 2:
                risks['overall_risk'] = 'high'
            elif high_risks >= 1 or medium_risks >= 2:
                risks['overall_risk'] = 'medium'
            
            return {
                'risk_levels': risks,
                'risk_factors': risk_factors,
                'risk_score': self.calculate_risk_score(risks)
            }
            
        except Exception as e:
            print(f"Error assessing risks: {e}")
            return {}

    def calculate_risk_score(self, risks):
        """Calculate overall risk score"""
        try:
            risk_weights = {
                'liquidity_risk': 0.25,
                'solvency_risk': 0.3,
                'profitability_risk': 0.25,
                'operational_risk': 0.2
            }
            
            risk_values = {'low': 20, 'medium': 60, 'high': 90}
            
            weighted_score = 0
            for risk_type, weight in risk_weights.items():
                risk_level = risks.get(risk_type, 'low')
                weighted_score += risk_values[risk_level] * weight
            
            return round(weighted_score, 1)
            
        except Exception as e:
            print(f"Error calculating risk score: {e}")
            return 50

    def score_to_rating(self, score):
        """Convert numeric score to rating"""
        if score >= 85:
            return 'Excellent'
        elif score >= 70:
            return 'Good'
        elif score >= 55:
            return 'Fair'
        elif score >= 40:
            return 'Poor'
        else:
            return 'Critical'

    def generate_recommendations(self, analysis):
        """
        Generate actionable recommendations based on analysis
        
        Args:
            analysis (dict): Complete financial analysis
            
        Returns:
            list: Recommendations
        """
        try:
            recommendations = []
            
            # Get ratio assessments
            ratios = analysis.get('ratios', {})
            risk_assessment = analysis.get('risk_assessment', {})
            
            # Liquidity recommendations
            liquidity_ratios = ratios.get('liquidity', {})
            if liquidity_ratios.get('current_ratio', {}).get('assessment', {}).get('score', 50) < 60:
                recommendations.append({
                    'category': 'Liquidity',
                    'priority': 'High',
                    'recommendation': 'Improve current ratio by reducing current liabilities or increasing current assets',
                    'impact': 'Better short-term financial stability'
                })
            
            # Profitability recommendations
            profitability_ratios = ratios.get('profitability', {})
            if profitability_ratios.get('profit_margin', {}).get('assessment', {}).get('score', 50) < 60:
                recommendations.append({
                    'category': 'Profitability',
                    'priority': 'High',
                    'recommendation': 'Focus on cost reduction and revenue optimization strategies',
                    'impact': 'Improved profit margins and financial performance'
                })
            
            # Leverage recommendations
            leverage_ratios = ratios.get('leverage', {})
            if leverage_ratios.get('debt_to_equity', {}).get('value', 0) > 1.0:
                recommendations.append({
                    'category': 'Leverage',
                    'priority': 'Medium',
                    'recommendation': 'Consider debt reduction strategies or equity financing',
                    'impact': 'Reduced financial risk and improved solvency'
                })
            
            # Cash flow recommendations
            cash_flow_ratios = ratios.get('cash_flow', {})
            if cash_flow_ratios.get('operating_cf_ratio', {}).get('value', 0) < 0.5:
                recommendations.append({
                    'category': 'Cash Flow',
                    'priority': 'High',
                    'recommendation': 'Improve cash flow management and working capital efficiency',
                    'impact': 'Better operational cash generation'
                })
            
            return recommendations
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return []

    def analyze_company_performance(self, company_id, years=None):
        """
        Comprehensive company performance analysis
        
        Args:
            company_id (str): Company identifier
            years (list): Years to analyze
            
        Returns:
            dict: Complete performance analysis
        """
        try:
            # Get company data
            company_data = self.get_company_data(company_id, years)
            if not company_data:
                return {'error': 'Unable to retrieve company data'}
            
            balance_sheets = company_data.get('balance_sheets', [])
            cash_flows = company_data.get('cash_flows', [])
            
            if not balance_sheets or not cash_flows:
                return {'error': 'Insufficient financial data'}
            
            # Use most recent data for analysis
            latest_bs = balance_sheets[0]
            latest_cf = cash_flows[0]
            
            # Determine industry (simplified - would normally be in company master data)
            industry = cash_flows[0].get('industry', 'general')
            
            # Calculate all ratio categories
            liquidity_ratios = self.calculate_liquidity_ratios(latest_bs)
            profitability_ratios = self.calculate_profitability_ratios(latest_bs, latest_cf)
            leverage_ratios = self.calculate_leverage_ratios(latest_bs, latest_cf)
            efficiency_ratios = self.calculate_efficiency_ratios(latest_bs, latest_cf)
            cash_flow_ratios = self.calculate_cash_flow_ratios(latest_bs, latest_cf)
            
            # Assess performance for each ratio
            ratio_assessments = {}
            
            for category, ratios in [
                ('liquidity', liquidity_ratios),
                ('profitability', profitability_ratios),
                ('leverage', leverage_ratios),
                ('efficiency', efficiency_ratios),
                ('cash_flow', cash_flow_ratios)
            ]:
                ratio_assessments[category] = {}
                for ratio_name, ratio_value in ratios.items():
                    assessment = self.assess_ratio_performance(ratio_value, ratio_name, category, industry)
                    ratio_assessments[category][ratio_name] = {
                        'value': ratio_value,
                        'assessment': assessment
                    }
            
            # Calculate overall scores
            category_scores = {}
            for category, ratios in ratio_assessments.items():
                scores = [r['assessment']['score'] for r in ratios.values() if r['assessment']['score']]
                category_scores[category] = statistics.mean(scores) if scores else 50
            
            overall_score = statistics.mean(category_scores.values()) if category_scores else 50
            
            # Analyze trends if multiple years available
            trends = self.analyze_financial_trends(balance_sheets, cash_flows)
            
            # Risk assessment
            risk_assessment = self.assess_financial_risks(latest_bs, latest_cf, ratio_assessments)
            
            # Generate recommendations
            analysis_data = {
                'ratios': ratio_assessments,
                'risk_assessment': risk_assessment
            }
            recommendations = self.generate_recommendations(analysis_data)
            
            # Compile analysis
            analysis = {
                'company_id': company_id,
                'analysis_date': datetime.now().isoformat(),
                'industry': industry,
                'years_analyzed': [bs['year'] for bs in balance_sheets],
                'summary': {
                    'overall_score': round(overall_score, 1),
                    'overall_rating': self.score_to_rating(overall_score),
                    'category_scores': {k: round(v, 1) for k, v in category_scores.items()}
                },
                'ratios': {
                    'liquidity': liquidity_ratios,
                    'profitability': profitability_ratios,
                    'leverage': leverage_ratios,
                    'efficiency': efficiency_ratios,
                    'cash_flow': cash_flow_ratios
                },
                'ratio_assessments': ratio_assessments,
                'trends': trends,
                'risk_assessment': risk_assessment,
                'recommendations': recommendations,
                'data_quality': {
                    'balance_sheet_completeness': self.assess_data_completeness(latest_bs),
                    'cash_flow_completeness': self.assess_data_completeness(latest_cf),
                    'years_available': len(balance_sheets)
                }
            }
            
            return analysis
            
        except Exception as e:
            print(f"Error in company performance analysis: {e}")
            return {'error': f'Analysis failed: {str(e)}'}

    def assess_data_completeness(self, financial_data):
        """
        Assess the completeness of financial data
        
        Args:
            financial_data (dict): Financial data to assess
            
        Returns:
            float: Completeness percentage
        """
        try:
            total_fields = len(financial_data)
            non_null_fields = sum(1 for value in financial_data.values() if value is not None and value != 0)
            
            return round((non_null_fields / total_fields) * 100, 1) if total_fields > 0 else 0
            
        except Exception as e:
            print(f"Error assessing data completeness: {e}")
            return 0

    def compare_companies(self, company_ids, years=None):
        """
        Compare multiple companies
        
        Args:
            company_ids (list): List of company identifiers
            years (list): Years to compare
            
        Returns:
            dict: Comparative analysis
        """
        try:
            comparison = {
                'companies': {},
                'comparison_metrics': {},
                'rankings': {}
            }
            
            # Analyze each company
            for company_id in company_ids:
                analysis = self.analyze_company_performance(company_id, years)
                if 'error' not in analysis:
                    comparison['companies'][company_id] = analysis
            
            if not comparison['companies']:
                return {'error': 'No valid company data for comparison'}
            
            # Generate comparison metrics
            metrics_to_compare = ['overall_score', 'liquidity', 'profitability', 'leverage', 'efficiency']
            
            for metric in metrics_to_compare:
                metric_values = {}
                for company_id, analysis in comparison['companies'].items():
                    if metric == 'overall_score':
                        metric_values[company_id] = analysis['summary']['overall_score']
                    else:
                        metric_values[company_id] = analysis['summary']['category_scores'].get(metric, 0)
                
                # Rank companies for this metric
                ranked = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
                comparison['comparison_metrics'][metric] = {
                    'values': metric_values,
                    'ranking': ranked,
                    'best': ranked[0] if ranked else None,
                    'worst': ranked[-1] if ranked else None
                }
            
            # Overall rankings
            overall_scores = comparison['comparison_metrics']['overall_score']['ranking']
            comparison['rankings'] = {
                'overall': overall_scores,
                'best_performer': overall_scores[0] if overall_scores else None,
                'needs_attention': [comp for comp, score in overall_scores if score < 50]
            }
            
            return comparison
            
        except Exception as e:
            print(f"Error in company comparison: {e}")
            return {'error': f'Comparison failed: {str(e)}'}

    def generate_industry_benchmark(self, industry, years=None):
        """
        Generate industry benchmark analysis
        
        Args:
            industry (str): Industry to benchmark
            years (list): Years to include
            
        Returns:
            dict: Industry benchmark data
        """
        try:
            conn = self.connect_db()
            if not conn:
                return {'error': 'Database connection failed'}
            
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get all companies in the industry
            if years is None:
                current_year = datetime.now().year
                years = [current_year - i for i in range(3)]
            
            industry_query = """
            SELECT DISTINCT company_id FROM cash_flow_statement 
            WHERE industry = %s AND year = ANY(%s)
            """
            cursor.execute(industry_query, (industry, years))
            companies = [row['company_id'] for row in cursor.fetchall()]
            
            cursor.close()
            conn.close()
            
            if not companies:
                return {'error': f'No companies found for industry: {industry}'}
            
            # Analyze all companies in the industry
            industry_analyses = []
            for company_id in companies:
                analysis = self.analyze_company_performance(company_id, years)
                if 'error' not in analysis:
                    industry_analyses.append(analysis)
            
            if not industry_analyses:
                return {'error': 'No valid analysis data for industry benchmark'}
            
            # Calculate industry averages
            benchmark = {
                'industry': industry,
                'companies_analyzed': len(industry_analyses),
                'benchmark_date': datetime.now().isoformat(),
                'averages': {},
                'percentiles': {},
                'top_performers': [],
                'industry_trends': {}
            }
            
            # Calculate averages for key metrics
            metrics = ['overall_score']
            for analysis in industry_analyses:
                for category in analysis['summary']['category_scores']:
                    metrics.append(category)
            
            unique_metrics = list(set(metrics))
            
            for metric in unique_metrics:
                values = []
                for analysis in industry_analyses:
                    if metric == 'overall_score':
                        values.append(analysis['summary']['overall_score'])
                    else:
                        values.append(analysis['summary']['category_scores'].get(metric, 0))
                
                if values:
                    benchmark['averages'][metric] = {
                        'mean': round(statistics.mean(values), 2),
                        'median': round(statistics.median(values), 2),
                        'std_dev': round(statistics.stdev(values) if len(values) > 1 else 0, 2)
                    }
                    
                    # Calculate percentiles
                    benchmark['percentiles'][metric] = {
                        '25th': round(np.percentile(values, 25), 2),
                        '50th': round(np.percentile(values, 50), 2),
                        '75th': round(np.percentile(values, 75), 2),
                        '90th': round(np.percentile(values, 90), 2)
                    }
            
            # Identify top performers (top 10% by overall score)
            sorted_analyses = sorted(industry_analyses, 
                                   key=lambda x: x['summary']['overall_score'], 
                                   reverse=True)
            
            top_count = max(1, len(sorted_analyses) // 10)
            benchmark['top_performers'] = [
                {
                    'company_id': analysis['company_id'],
                    'overall_score': analysis['summary']['overall_score'],
                    'rating': analysis['summary']['overall_rating']
                }
                for analysis in sorted_analyses[:top_count]
            ]
            
            return benchmark
            
        except Exception as e:
            print(f"Error generating industry benchmark: {e}")
            return {'error': f'Benchmark generation failed: {str(e)}'}

    def predict_financial_health(self, company_id, prediction_months=12):
        """
        Predict future financial health based on trends
        
        Args:
            company_id (str): Company identifier
            prediction_months (int): Months to predict ahead
            
        Returns:
            dict: Financial health predictions
        """
        try:
            # Get historical data for trend analysis
            company_data = self.get_company_data(company_id)
            if not company_data:
                return {'error': 'Unable to retrieve company data for prediction'}
            
            balance_sheets = company_data.get('balance_sheets', [])
            cash_flows = company_data.get('cash_flows', [])
            
            if len(balance_sheets) < 2 or len(cash_flows) < 2:
                return {'error': 'Insufficient historical data for prediction'}
            
            # Analyze current trends
            trends = self.analyze_financial_trends(balance_sheets, cash_flows)
            current_analysis = self.analyze_company_performance(company_id)
            
            if 'error' in current_analysis:
                return {'error': 'Current analysis failed'}
            
            # Simple linear projection based on trends
            predictions = {
                'company_id': company_id,
                'prediction_date': datetime.now().isoformat(),
                'prediction_horizon_months': prediction_months,
                'current_health_score': current_analysis['summary']['overall_score'],
                'predicted_health_score': None,
                'trend_analysis': trends,
                'risk_factors': [],
                'confidence_level': 'medium'
            }
            
            # Calculate predicted health score based on trends
            current_score = current_analysis['summary']['overall_score']
            
            # Average trend impact
            trend_impacts = []
            for trend_key, trend_data in trends.items():
                if isinstance(trend_data, dict) and 'average' in trend_data:
                    # Convert trend percentage to score impact
                    impact = trend_data['average'] * 0.1  # Scale factor
                    trend_impacts.append(impact)
            
            if trend_impacts:
                avg_trend_impact = statistics.mean(trend_impacts)
                # Project forward (simplified linear projection)
                projection_factor = (prediction_months / 12) * avg_trend_impact
                predicted_score = max(0, min(100, current_score + projection_factor))
                predictions['predicted_health_score'] = round(predicted_score, 1)
                
                # Determine confidence based on trend consistency
                trend_consistency = 1 - (statistics.stdev(trend_impacts) / 10 if len(trend_impacts) > 1 else 0)
                if trend_consistency > 0.8:
                    predictions['confidence_level'] = 'high'
                elif trend_consistency > 0.5:
                    predictions['confidence_level'] = 'medium'
                else:
                    predictions['confidence_level'] = 'low'
            else:
                predictions['predicted_health_score'] = current_score
                predictions['confidence_level'] = 'low'
            
            # Identify risk factors
            if predictions['predicted_health_score'] < current_score:
                predictions['risk_factors'].append('Declining financial health trend detected')
            
            if current_analysis['risk_assessment']['risk_levels']['overall_risk'] == 'high':
                predictions['risk_factors'].append('Current high risk level may persist')
            
            # Add scenario analysis
            predictions['scenarios'] = {
                'optimistic': min(100, predictions['predicted_health_score'] + 10),
                'pessimistic': max(0, predictions['predicted_health_score'] - 15),
                'most_likely': predictions['predicted_health_score']
            }
            
            return predictions
            
        except Exception as e:
            print(f"Error in financial health prediction: {e}")
            return {'error': f'Prediction failed: {str(e)}'}

    def export_analysis_report(self, company_id, format='json'):
        """
        Export comprehensive analysis report
        
        Args:
            company_id (str): Company identifier
            format (str): Export format ('json', 'dict')
            
        Returns:
            dict/str: Formatted report
        """
        try:
            # Get comprehensive analysis
            analysis = self.analyze_company_performance(company_id)
            if 'error' in analysis:
                return analysis
            
            # Get predictions
            predictions = self.predict_financial_health(company_id)
            
            # Compile comprehensive report
            report = {
                'report_metadata': {
                    'company_id': company_id,
                    'report_date': datetime.now().isoformat(),
                    'report_type': 'Comprehensive Financial Analysis',
                    'analysis_engine': 'FinancialAnalyzer v1.0'
                },
                'executive_summary': {
                    'overall_score': analysis['summary']['overall_score'],
                    'overall_rating': analysis['summary']['overall_rating'],
                    'key_strengths': self.identify_strengths(analysis),
                    'key_concerns': analysis['risk_assessment']['risk_factors'],
                    'primary_recommendation': self.get_primary_recommendation(analysis)
                },
                'financial_analysis': analysis,
                'future_outlook': predictions,
                'action_plan': analysis.get('recommendations', [])
            }
            
            if format == 'json':
                import json
                return json.dumps(report, indent=2, default=str)
            else:
                return report
                
        except Exception as e:
            print(f"Error exporting analysis report: {e}")
            return {'error': f'Export failed: {str(e)}'}

    def identify_strengths(self, analysis):
        """Identify key financial strengths"""
        try:
            strengths = []
            
            for category, scores in analysis['summary']['category_scores'].items():
                if scores >= 75:
                    strengths.append(f"Strong {category} performance")
            
            if analysis['summary']['overall_score'] >= 80:
                strengths.append("Excellent overall financial health")
            
            return strengths if strengths else ["Areas for improvement identified"]
            
        except Exception as e:
            return ["Analysis unavailable"]

    def get_primary_recommendation(self, analysis):
        """Get the most important recommendation"""
        try:
            recommendations = analysis.get('recommendations', [])
            if not recommendations:
                return "Continue monitoring financial performance"
            
            # Return highest priority recommendation
            high_priority = [rec for rec in recommendations if rec.get('priority') == 'High']
            if high_priority:
                return high_priority[0]['recommendation']
            
            return recommendations[0]['recommendation']
            
        except Exception as e:
            return "Regular financial monitoring recommended"

    def get_company_list_by_industry(self, industry):
        """
        Get list of companies in a specific industry
        
        Args:
            industry (str): Industry name
            
        Returns:
            list: List of company IDs
        """
        try:
            conn = self.connect_db()
            if not conn:
                return []
            
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
            SELECT DISTINCT company_id, company_name 
            FROM cash_flow_statement 
            WHERE industry = %s 
            ORDER BY company_name
            """
            cursor.execute(query, (industry,))
            companies = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            return [{'company_id': row['company_id'], 'company_name': row['company_name']} 
                    for row in companies]
            
        except Exception as e:
            print(f"Error getting company list: {e}")
            return []

    def get_available_years(self, company_id):
        """
        Get available years of data for a company
        
        Args:
            company_id (str): Company identifier
            
        Returns:
            list: Available years
        """
        try:
            conn = self.connect_db()
            if not conn:
                return []
            
            cursor = conn.cursor()
            
            query = """
            SELECT DISTINCT year FROM balance_sheet_1 
            WHERE company_id = %s 
            ORDER BY year DESC
            """
            cursor.execute(query, (company_id,))
            years = [row[0] for row in cursor.fetchall()]
            
            cursor.close()
            conn.close()
            
            return years
            
        except Exception as e:
            print(f"Error getting available years: {e}")
            return []

# Usage Example and Testing Functions
if __name__ == "__main__":
    # Database configuration
    db_config = {
        'host': 'localhost',
        'database': 'financial_db',
        'user': 'postgres',
        'password': 'Prateek@2003',
        'port': 5432
    }
    
    # Initialize analyzer
    analyzer = FinancialAnalyzer(db_config)
    
    # Example usage
    try:
        # Analyze a single company
        company_analysis = analyzer.analyze_company_performance('COMP001')
        print("Company Analysis:", company_analysis)
        
        # Compare multiple companies
        comparison = analyzer.compare_companies(['COMP001', 'COMP002', 'COMP003'])
        print("Company Comparison:", comparison)
        
        # Generate industry benchmark
        benchmark = analyzer.generate_industry_benchmark('technology')
        print("Industry Benchmark:", benchmark)
        
        # Predict financial health
        prediction = analyzer.predict_financial_health('COMP001', 12)
        print("Financial Health Prediction:", prediction)
        
        # Export comprehensive report
        report = analyzer.export_analysis_report('COMP001', format='dict')
        print("Comprehensive Report Generated")
        
    except Exception as e:
        print(f"Error in analysis: {e}")