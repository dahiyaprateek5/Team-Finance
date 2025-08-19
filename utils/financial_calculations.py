"""
Financial Calculations Module
============================

This module provides comprehensive financial analysis and risk calculation functions
for the Financial Risk Assessment Platform.

Functions:
----------
- calculate_financial_ratios: Calculate key financial ratios
- calculate_cash_flow_metrics: Calculate cash flow related metrics
- assess_liquidation_risk: Assess company's liquidation risk
- calculate_z_score: Calculate Altman Z-Score
- generate_financial_health_score: Generate overall financial health score

Author: Prateek Dahiya
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)


class FinancialCalculator:
    """
    Comprehensive financial calculations and risk assessment
    """
    
    def __init__(self):
        self.industry_benchmarks = self._load_industry_benchmarks()
        self.risk_thresholds = self._load_risk_thresholds()
    
    def _load_industry_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """Load industry benchmark data"""
        return {
            'technology': {
                'current_ratio': 2.5,
                'debt_to_equity': 0.3,
                'roa': 0.12,
                'roe': 0.18,
                'profit_margin': 0.15,
                'asset_turnover': 0.8
            },
            'healthcare': {
                'current_ratio': 2.0,
                'debt_to_equity': 0.4,
                'roa': 0.08,
                'roe': 0.14,
                'profit_margin': 0.12,
                'asset_turnover': 0.7
            },
            'industrial': {
                'current_ratio': 1.8,
                'debt_to_equity': 0.6,
                'roa': 0.06,
                'roe': 0.12,
                'profit_margin': 0.08,
                'asset_turnover': 1.2
            },
            'financial': {
                'current_ratio': 1.1,
                'debt_to_equity': 5.0,
                'roa': 0.02,
                'roe': 0.12,
                'profit_margin': 0.25,
                'asset_turnover': 0.1
            },
            'default': {
                'current_ratio': 2.0,
                'debt_to_equity': 0.5,
                'roa': 0.08,
                'roe': 0.15,
                'profit_margin': 0.10,
                'asset_turnover': 1.0
            }
        }
    
    def _load_risk_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Load risk assessment thresholds"""
        return {
            'liquidation_risk': {
                'low': 0.3,
                'medium': 0.6,
                'high': 0.8
            },
            'z_score': {
                'safe': 3.0,
                'caution': 1.8,
                'distress': 1.8
            },
            'cash_flow': {
                'strong': 0.15,
                'adequate': 0.08,
                'weak': 0.05
            }
        }


def calculate_financial_ratios(financial_data: Dict[str, float]) -> Dict[str, Any]:
    """
    Calculate comprehensive financial ratios from financial data
    
    Args:
        financial_data: Dictionary containing financial statement data
        
    Returns:
        Dictionary containing calculated ratios and analysis
    """
    try:
        ratios = {}
        warnings = []
        
        # Extract key financial statement items
        total_assets = financial_data.get('total_assets', 0)
        current_assets = financial_data.get('current_assets', 0)
        cash = financial_data.get('cash_and_equivalents', 0)
        accounts_receivable = financial_data.get('accounts_receivable', 0)
        inventory = financial_data.get('inventory', 0)
        
        total_liabilities = financial_data.get('total_liabilities', 0)
        current_liabilities = financial_data.get('current_liabilities', 0)
        long_term_debt = financial_data.get('long_term_debt', 0)
        
        total_equity = financial_data.get('total_equity', 0)
        retained_earnings = financial_data.get('retained_earnings', 0)
        
        revenue = financial_data.get('revenue', 0)
        net_income = financial_data.get('net_income', 0)
        operating_income = financial_data.get('operating_income', 0)
        interest_expense = financial_data.get('interest_expense', 0)
        
        # Liquidity Ratios
        if current_liabilities != 0:
            ratios['current_ratio'] = current_assets / current_liabilities
            ratios['quick_ratio'] = (current_assets - inventory) / current_liabilities
            ratios['cash_ratio'] = cash / current_liabilities
        else:
            ratios['current_ratio'] = float('inf') if current_assets > 0 else 0
            ratios['quick_ratio'] = float('inf') if (current_assets - inventory) > 0 else 0
            ratios['cash_ratio'] = float('inf') if cash > 0 else 0
            warnings.append("Current liabilities is zero - liquidity ratios may be misleading")
        
        # Efficiency Ratios
        if total_assets != 0:
            ratios['asset_turnover'] = revenue / total_assets
            ratios['roa'] = net_income / total_assets  # Return on Assets
        else:
            ratios['asset_turnover'] = 0
            ratios['roa'] = 0
            warnings.append("Total assets is zero - efficiency ratios cannot be calculated")
        
        if total_equity != 0:
            ratios['roe'] = net_income / total_equity  # Return on Equity
            ratios['equity_multiplier'] = total_assets / total_equity
        else:
            ratios['roe'] = 0
            ratios['equity_multiplier'] = 0
            warnings.append("Total equity is zero - ROE cannot be calculated")
        
        # Leverage/Solvency Ratios
        if total_equity != 0:
            ratios['debt_to_equity'] = total_liabilities / total_equity
        else:
            ratios['debt_to_equity'] = float('inf') if total_liabilities > 0 else 0
            
        if total_assets != 0:
            ratios['debt_to_assets'] = total_liabilities / total_assets
        else:
            ratios['debt_to_assets'] = 0
        
        # Coverage Ratios
        if interest_expense != 0:
            ratios['interest_coverage'] = operating_income / interest_expense
        else:
            ratios['interest_coverage'] = float('inf') if operating_income > 0 else 0
        
        # Profitability Ratios
        if revenue != 0:
            ratios['profit_margin'] = net_income / revenue
            ratios['operating_margin'] = operating_income / revenue
        else:
            ratios['profit_margin'] = 0
            ratios['operating_margin'] = 0
            warnings.append("Revenue is zero - profitability ratios cannot be calculated")
        
        # Working Capital
        ratios['working_capital'] = current_assets - current_liabilities
        ratios['working_capital_ratio'] = ratios['working_capital'] / total_assets if total_assets != 0 else 0
        
        # DuPont Analysis
        ratios['dupont_roe'] = ratios.get('profit_margin', 0) * ratios.get('asset_turnover', 0) * ratios.get('equity_multiplier', 0)
        
        return {
            'success': True,
            'ratios': ratios,
            'warnings': warnings,
            'calculation_date': datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error calculating financial ratios: {e}")
        return {
            'success': False,
            'error': str(e),
            'ratios': {},
            'warnings': []
        }


def calculate_cash_flow_metrics(cash_flow_data: Dict[str, float], balance_sheet_data: Dict[str, float]) -> Dict[str, Any]:
    """
    Calculate cash flow related metrics
    
    Args:
        cash_flow_data: Cash flow statement data
        balance_sheet_data: Balance sheet data for additional calculations
        
    Returns:
        Dictionary containing cash flow metrics
    """
    try:
        metrics = {}
        warnings = []
        
        # Extract cash flow items
        operating_cash_flow = cash_flow_data.get('net_cash_from_operating_activities', 0)
        investing_cash_flow = cash_flow_data.get('net_cash_from_investing_activities', 0)
        financing_cash_flow = cash_flow_data.get('net_cash_from_financing_activities', 0)
        capital_expenditures = cash_flow_data.get('capital_expenditures', 0)
        net_income = cash_flow_data.get('net_income', 0)
        
        # Extract balance sheet items
        total_assets = balance_sheet_data.get('total_assets', 0)
        total_debt = balance_sheet_data.get('total_liabilities', 0)
        revenue = balance_sheet_data.get('revenue', 0)
        
        # Basic Cash Flow Metrics
        metrics['free_cash_flow'] = operating_cash_flow - capital_expenditures
        metrics['net_cash_flow'] = operating_cash_flow + investing_cash_flow + financing_cash_flow
        
        # Cash Flow Ratios
        if net_income != 0:
            metrics['operating_cash_flow_to_net_income'] = operating_cash_flow / net_income
        else:
            metrics['operating_cash_flow_to_net_income'] = 0
            warnings.append("Net income is zero - OCF to NI ratio cannot be calculated")
        
        if total_assets != 0:
            metrics['cash_flow_to_assets'] = operating_cash_flow / total_assets
        else:
            metrics['cash_flow_to_assets'] = 0
        
        if total_debt != 0:
            metrics['cash_flow_to_debt'] = operating_cash_flow / total_debt
        else:
            metrics['cash_flow_to_debt'] = 0
        
        if revenue != 0:
            metrics['cash_flow_margin'] = operating_cash_flow / revenue
        else:
            metrics['cash_flow_margin'] = 0
        
        # Cash Flow Quality Indicators
        metrics['cash_flow_stability'] = _calculate_cash_flow_stability(cash_flow_data)
        metrics['reinvestment_ratio'] = capital_expenditures / operating_cash_flow if operating_cash_flow != 0 else 0
        
        # Cash Flow Trend Analysis (if historical data available)
        metrics['cash_flow_growth'] = _calculate_cash_flow_growth(cash_flow_data)
        
        return {
            'success': True,
            'metrics': metrics,
            'warnings': warnings,
            'calculation_date': datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error calculating cash flow metrics: {e}")
        return {
            'success': False,
            'error': str(e),
            'metrics': {},
            'warnings': []
        }


def _calculate_cash_flow_stability(cash_flow_data: Dict[str, float]) -> float:
    """Calculate cash flow stability score"""
    try:
        operating_cf = cash_flow_data.get('net_cash_from_operating_activities', 0)
        net_income = cash_flow_data.get('net_income', 0)
        
        if net_income != 0:
            cf_ni_ratio = operating_cf / net_income
            # Higher stability when operating cash flow is close to or exceeds net income
            if cf_ni_ratio >= 1.0:
                return min(1.0, cf_ni_ratio / 1.5)  # Cap at 1.0
            else:
                return max(0.0, cf_ni_ratio)
        
        return 0.5  # Neutral stability score when net income is zero
        
    except Exception:
        return 0.0


def _calculate_cash_flow_growth(cash_flow_data: Dict[str, float]) -> float:
    """Calculate cash flow growth (simplified - would need historical data for full calculation)"""
    # This is a simplified version - in practice, you'd need multiple years of data
    operating_cf = cash_flow_data.get('net_cash_from_operating_activities', 0)
    
    # Return a normalized growth indicator based on current cash flow strength
    if operating_cf > 0:
        return min(1.0, operating_cf / 1000000)  # Normalize to millions
    else:
        return -1.0  # Negative growth indicator


def calculate_z_score(financial_data: Dict[str, float]) -> Dict[str, Any]:
    """
    Calculate Altman Z-Score for bankruptcy prediction
    
    Args:
        financial_data: Combined financial statement data
        
    Returns:
        Dictionary containing Z-score and interpretation
    """
    try:
        # Extract required data
        total_assets = financial_data.get('total_assets', 0)
        current_assets = financial_data.get('current_assets', 0)
        current_liabilities = financial_data.get('current_liabilities', 0)
        retained_earnings = financial_data.get('retained_earnings', 0)
        operating_income = financial_data.get('operating_income', 0)
        revenue = financial_data.get('revenue', 0)
        market_value_equity = financial_data.get('market_value_equity', total_assets * 0.6)  # Estimate if not available
        total_liabilities = financial_data.get('total_liabilities', 0)
        
        if total_assets == 0:
            return {
                'success': False,
                'error': 'Cannot calculate Z-score: total assets is zero',
                'z_score': 0,
                'risk_category': 'unknown'
            }
        
        # Calculate Z-score components
        working_capital = current_assets - current_liabilities
        
        z1 = working_capital / total_assets
        z2 = retained_earnings / total_assets
        z3 = operating_income / total_assets
        z4 = market_value_equity / total_liabilities if total_liabilities != 0 else 0
        z5 = revenue / total_assets
        
        # Original Altman Z-score formula
        z_score = 1.2 * z1 + 1.4 * z2 + 3.3 * z3 + 0.6 * z4 + 1.0 * z5
        
        # Determine risk category
        if z_score > 3.0:
            risk_category = 'safe'
            risk_description = 'Low bankruptcy risk'
        elif z_score > 1.8:
            risk_category = 'caution'
            risk_description = 'Moderate bankruptcy risk - monitor closely'
        else:
            risk_category = 'distress'
            risk_description = 'High bankruptcy risk'
        
        return {
            'success': True,
            'z_score': round(z_score, 3),
            'risk_category': risk_category,
            'risk_description': risk_description,
            'components': {
                'working_capital_to_assets': round(z1, 3),
                'retained_earnings_to_assets': round(z2, 3),
                'operating_income_to_assets': round(z3, 3),
                'market_value_equity_to_liabilities': round(z4, 3),
                'sales_to_assets': round(z5, 3)
            },
            'calculation_date': datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error calculating Z-score: {e}")
        return {
            'success': False,
            'error': str(e),
            'z_score': 0,
            'risk_category': 'unknown'
        }


def assess_liquidation_risk(financial_data: Dict[str, float], historical_data: List[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Assess company's liquidation risk based on multiple factors
    
    Args:
        financial_data: Current financial statement data
        historical_data: Optional historical financial data for trend analysis
        
    Returns:
        Dictionary containing liquidation risk assessment
    """
    try:
        risk_factors = {}
        risk_score = 0.0
        warnings = []
        
        # Extract key metrics
        net_income = financial_data.get('net_income', 0)
        operating_cash_flow = financial_data.get('net_cash_from_operating_activities', 0)
        current_ratio = financial_data.get('current_ratio', 0)
        debt_to_equity = financial_data.get('debt_to_equity', 0)
        cash = financial_data.get('cash_and_equivalents', 0)
        total_assets = financial_data.get('total_assets', 0)
        revenue = financial_data.get('revenue', 0)
        
        # Risk Factor 1: Profitability
        if net_income < 0:
            risk_factors['negative_net_income'] = True
            risk_score += 0.15
        else:
            risk_factors['negative_net_income'] = False
        
        # Risk Factor 2: Cash Flow
        if operating_cash_flow < 0:
            risk_factors['negative_operating_cash_flow'] = True
            risk_score += 0.20
        else:
            risk_factors['negative_operating_cash_flow'] = False
        
        # Risk Factor 3: Liquidity
        if current_ratio < 1.0:
            risk_factors['poor_liquidity'] = True
            risk_score += 0.15
        elif current_ratio < 1.5:
            risk_score += 0.05
        
        # Risk Factor 4: Leverage
        if debt_to_equity > 2.0:
            risk_factors['high_leverage'] = True
            risk_score += 0.10
        elif debt_to_equity > 1.0:
            risk_score += 0.05
        
        # Risk Factor 5: Cash Position
        cash_to_assets = cash / total_assets if total_assets > 0 else 0
        if cash_to_assets < 0.05:  # Less than 5% cash
            risk_factors['low_cash_reserves'] = True
            risk_score += 0.10
        
        # Risk Factor 6: Revenue Decline (if historical data available)
        if historical_data and len(historical_data) > 1:
            revenue_trend = _analyze_revenue_trend(historical_data)
            if revenue_trend < -0.1:  # 10% decline
                risk_factors['declining_revenue'] = True
                risk_score += 0.15
        
        # Risk Factor 7: Combined Loss Pattern
        if net_income < 0 and operating_cash_flow < 0:
            risk_factors['combined_losses'] = True
            risk_score += 0.10
        
        # Calculate Z-score and incorporate it
        z_score_result = calculate_z_score(financial_data)
        if z_score_result['success']:
            z_score = z_score_result['z_score']
            if z_score < 1.8:
                risk_score += 0.20
            elif z_score < 3.0:
                risk_score += 0.10
            risk_factors['z_score'] = z_score
        
        # Normalize risk score to 0-1 range
        risk_score = min(1.0, risk_score)
        
        # Determine risk level
        if risk_score < 0.3:
            risk_level = 'low'
            risk_description = 'Low liquidation risk - company appears financially stable'
        elif risk_score < 0.6:
            risk_level = 'medium'
            risk_description = 'Moderate liquidation risk - monitor financial health closely'
        elif risk_score < 0.8:
            risk_level = 'high'
            risk_description = 'High liquidation risk - immediate attention required'
        else:
            risk_level = 'critical'
            risk_description = 'Critical liquidation risk - company may be in financial distress'
        
        # Generate recommendations
        recommendations = _generate_risk_recommendations(risk_factors, risk_score)
        
        return {
            'success': True,
            'liquidation_risk_score': round(risk_score, 3),
            'risk_level': risk_level,
            'risk_description': risk_description,
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            'warnings': warnings,
            'assessment_date': datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error assessing liquidation risk: {e}")
        return {
            'success': False,
            'error': str(e),
            'liquidation_risk_score': 0,
            'risk_level': 'unknown'
        }


def _analyze_revenue_trend(historical_data: List[Dict[str, float]]) -> float:
    """Analyze revenue trend from historical data"""
    try:
        revenues = [data.get('revenue', 0) for data in historical_data if data.get('revenue', 0) > 0]
        
        if len(revenues) < 2:
            return 0.0
        
        # Calculate simple growth rate
        latest_revenue = revenues[-1]
        previous_revenue = revenues[-2]
        
        if previous_revenue > 0:
            growth_rate = (latest_revenue - previous_revenue) / previous_revenue
            return growth_rate
        
        return 0.0
        
    except Exception:
        return 0.0


def _generate_risk_recommendations(risk_factors: Dict[str, Any], risk_score: float) -> List[str]:
    """Generate recommendations based on risk factors"""
    recommendations = []
    
    if risk_factors.get('negative_net_income', False):
        recommendations.append("Focus on improving profitability through cost reduction or revenue enhancement")
    
    if risk_factors.get('negative_operating_cash_flow', False):
        recommendations.append("Improve cash flow management and collection processes")
    
    if risk_factors.get('poor_liquidity', False):
        recommendations.append("Strengthen liquidity position through better working capital management")
    
    if risk_factors.get('high_leverage', False):
        recommendations.append("Consider debt reduction strategies to improve financial flexibility")
    
    if risk_factors.get('low_cash_reserves', False):
        recommendations.append("Build cash reserves to provide financial buffer for operations")
    
    if risk_factors.get('declining_revenue', False):
        recommendations.append("Develop strategies to stabilize and grow revenue streams")
    
    if risk_factors.get('combined_losses', False):
        recommendations.append("Implement comprehensive financial restructuring plan")
    
    if risk_score > 0.7:
        recommendations.append("Consider engaging financial advisors for immediate assistance")
        recommendations.append("Evaluate all non-essential expenses and investments")
    
    return recommendations


def generate_financial_health_score(financial_data: Dict[str, float], industry: str = 'default') -> Dict[str, Any]:
    """
    Generate comprehensive financial health score
    
    Args:
        financial_data: Complete financial statement data
        industry: Industry classification for benchmarking
        
    Returns:
        Dictionary containing financial health score and analysis
    """
    try:
        calculator = FinancialCalculator()
        health_components = {}
        weights = {
            'liquidity': 0.20,
            'profitability': 0.25,
            'efficiency': 0.20,
            'leverage': 0.15,
            'cash_flow': 0.20
        }
        
        # Get industry benchmarks
        benchmarks = calculator.industry_benchmarks.get(industry.lower(), 
                                                       calculator.industry_benchmarks['default'])
        
        # Calculate component scores
        ratios_result = calculate_financial_ratios(financial_data)
        if not ratios_result['success']:
            return ratios_result
        
        ratios = ratios_result['ratios']
        
        # Liquidity Score
        current_ratio = ratios.get('current_ratio', 0)
        quick_ratio = ratios.get('quick_ratio', 0)
        
        liquidity_score = 0
        if current_ratio >= benchmarks['current_ratio']:
            liquidity_score += 50
        else:
            liquidity_score += (current_ratio / benchmarks['current_ratio']) * 50
        
        if quick_ratio >= benchmarks['current_ratio'] * 0.8:  # 80% of current ratio benchmark
            liquidity_score += 50
        else:
            liquidity_score += (quick_ratio / (benchmarks['current_ratio'] * 0.8)) * 50
        
        health_components['liquidity'] = min(100, liquidity_score)
        
        # Profitability Score
        profit_margin = ratios.get('profit_margin', 0)
        roa = ratios.get('roa', 0)
        roe = ratios.get('roe', 0)
        
        profitability_score = 0
        if profit_margin >= benchmarks['profit_margin']:
            profitability_score += 40
        else:
            profitability_score += max(0, (profit_margin / benchmarks['profit_margin']) * 40)
        
        if roa >= benchmarks['roa']:
            profitability_score += 30
        else:
            profitability_score += max(0, (roa / benchmarks['roa']) * 30)
        
        if roe >= benchmarks['roe']:
            profitability_score += 30
        else:
            profitability_score += max(0, (roe / benchmarks['roe']) * 30)
        
        health_components['profitability'] = min(100, profitability_score)
        
        # Efficiency Score
        asset_turnover = ratios.get('asset_turnover', 0)
        
        if asset_turnover >= benchmarks['asset_turnover']:
            efficiency_score = 100
        else:
            efficiency_score = (asset_turnover / benchmarks['asset_turnover']) * 100
        
        health_components['efficiency'] = min(100, efficiency_score)
        
        # Leverage Score (inverted - lower leverage is better)
        debt_to_equity = ratios.get('debt_to_equity', 0)
        
        if debt_to_equity <= benchmarks['debt_to_equity']:
            leverage_score = 100
        elif debt_to_equity <= benchmarks['debt_to_equity'] * 2:
            leverage_score = 100 - ((debt_to_equity - benchmarks['debt_to_equity']) / benchmarks['debt_to_equity']) * 50
        else:
            leverage_score = max(0, 50 - (debt_to_equity - benchmarks['debt_to_equity'] * 2) * 10)
        
        health_components['leverage'] = max(0, min(100, leverage_score))
        
        # Cash Flow Score
        cash_flow_result = calculate_cash_flow_metrics(financial_data, financial_data)
        if cash_flow_result['success']:
            cash_flow_metrics = cash_flow_result['metrics']
            
            ocf_to_ni = cash_flow_metrics.get('operating_cash_flow_to_net_income', 0)
            cash_flow_margin = cash_flow_metrics.get('cash_flow_margin', 0)
            
            cash_flow_score = 0
            if ocf_to_ni >= 1.0:
                cash_flow_score += 50
            else:
                cash_flow_score += max(0, ocf_to_ni * 50)
            
            if cash_flow_margin >= 0.1:  # 10% cash flow margin
                cash_flow_score += 50
            else:
                cash_flow_score += max(0, cash_flow_margin * 500)
            
            health_components['cash_flow'] = min(100, cash_flow_score)
        else:
            health_components['cash_flow'] = 50  # Neutral score if can't calculate
        
        # Calculate weighted overall score
        overall_score = sum(health_components[component] * weights[component] 
                           for component in health_components)
        
        # Determine health grade
        if overall_score >= 80:
            health_grade = 'A'
            health_description = 'Excellent financial health'
        elif overall_score >= 70:
            health_grade = 'B'
            health_description = 'Good financial health'
        elif overall_score >= 60:
            health_grade = 'C'
            health_description = 'Fair financial health - some areas need attention'
        elif overall_score >= 50:
            health_grade = 'D'
            health_description = 'Poor financial health - significant improvements needed'
        else:
            health_grade = 'F'
            health_description = 'Critical financial condition - immediate action required'
        
        return {
            'success': True,
            'overall_score': round(overall_score, 1),
            'health_grade': health_grade,
            'health_description': health_description,
            'component_scores': {k: round(v, 1) for k, v in health_components.items()},
            'benchmarks_used': benchmarks,
            'industry': industry,
            'calculation_date': datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error generating financial health score: {e}")
        return {
            'success': False,
            'error': str(e),
            'overall_score': 0,
            'health_grade': 'F'
        }


def calculate_working_capital_metrics(financial_data: Dict[str, float]) -> Dict[str, Any]:
    """
    Calculate detailed working capital metrics
    
    Args:
        financial_data: Financial statement data
        
    Returns:
        Dictionary containing working capital analysis
    """
    try:
        metrics = {}
        
        # Extract data
        current_assets = financial_data.get('current_assets', 0)
        current_liabilities = financial_data.get('current_liabilities', 0)
        cash = financial_data.get('cash_and_equivalents', 0)
        accounts_receivable = financial_data.get('accounts_receivable', 0)
        inventory = financial_data.get('inventory', 0)
        accounts_payable = financial_data.get('accounts_payable', 0)
        revenue = financial_data.get('revenue', 0)
        cost_of_goods_sold = financial_data.get('cost_of_goods_sold', revenue * 0.7)  # Estimate if not available
        
        # Basic Working Capital
        metrics['working_capital'] = current_assets - current_liabilities
        metrics['net_working_capital_ratio'] = metrics['working_capital'] / revenue if revenue != 0 else 0
        
        # Working Capital Components
        metrics['cash_component'] = cash
        metrics['receivables_component'] = accounts_receivable
        metrics['inventory_component'] = inventory
        metrics['payables_component'] = accounts_payable
        
        # Turnover Ratios
        if accounts_receivable != 0:
            metrics['receivables_turnover'] = revenue / accounts_receivable
            metrics['days_sales_outstanding'] = 365 / metrics['receivables_turnover']
        else:
            metrics['receivables_turnover'] = 0
            metrics['days_sales_outstanding'] = 0
        
        if inventory != 0:
            metrics['inventory_turnover'] = cost_of_goods_sold / inventory
            metrics['days_inventory_outstanding'] = 365 / metrics['inventory_turnover']
        else:
            metrics['inventory_turnover'] = 0
            metrics['days_inventory_outstanding'] = 0
        
        if accounts_payable != 0:
            metrics['payables_turnover'] = cost_of_goods_sold / accounts_payable
            metrics['days_payable_outstanding'] = 365 / metrics['payables_turnover']
        else:
            metrics['payables_turnover'] = 0
            metrics['days_payable_outstanding'] = 0
        
        # Cash Conversion Cycle
        metrics['cash_conversion_cycle'] = (metrics['days_sales_outstanding'] + 
                                          metrics['days_inventory_outstanding'] - 
                                          metrics['days_payable_outstanding'])
        
        return {
            'success': True,
            'metrics': metrics,
            'calculation_date': datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error calculating working capital metrics: {e}")
        return {
            'success': False,
            'error': str(e),
            'metrics': {}
        }


def perform_ratio_analysis(financial_data: Dict[str, float], industry: str = 'default') -> Dict[str, Any]:
    """
    Perform comprehensive ratio analysis with industry comparison
    
    Args:
        financial_data: Complete financial statement data
        industry: Industry for benchmarking
        
    Returns:
        Complete ratio analysis with benchmarks and recommendations
    """
    try:
        calculator = FinancialCalculator()
        
        # Calculate all ratios
        ratios_result = calculate_financial_ratios(financial_data)
        if not ratios_result['success']:
            return ratios_result
        
        ratios = ratios_result['ratios']
        
        # Get industry benchmarks
        benchmarks = calculator.industry_benchmarks.get(industry.lower(),
                                                       calculator.industry_benchmarks['default'])
        
        # Compare with benchmarks
        ratio_comparison = {}
        recommendations = []
        
        for ratio_name, ratio_value in ratios.items():
            if ratio_name in benchmarks:
                benchmark = benchmarks[ratio_name]
                
                if ratio_value == float('inf'):
                    comparison = 'undefined'
                    variance = 0
                elif benchmark == 0:
                    comparison = 'no_benchmark'
                    variance = 0
                else:
                    variance = (ratio_value - benchmark) / benchmark
                    if abs(variance) < 0.1:  # Within 10%
                        comparison = 'on_target'
                    elif variance > 0:
                        comparison = 'above_benchmark'
                    else:
                        comparison = 'below_benchmark'
                
                ratio_comparison[ratio_name] = {
                    'value': ratio_value,
                    'benchmark': benchmark,
                    'variance': variance,
                    'comparison': comparison
                }
                
                # Generate recommendations
                if comparison == 'below_benchmark' and abs(variance) > 0.2:  # 20% below
                    recommendations.append(f"Improve {ratio_name.replace('_', ' ')}: currently {variance:.1%} below industry benchmark")
        
        return {
            'success': True,
            'ratios': ratios,
            'benchmarks': benchmarks,
            'ratio_comparison': ratio_comparison,
            'recommendations': recommendations,
            'industry': industry,
            'analysis_date': datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error performing ratio analysis: {e}")
        return {
            'success': False,
            'error': str(e)
        }