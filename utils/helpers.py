"""
Helpers Module
==============

This module provides utility functions and helper classes for the 
Financial Risk Assessment Platform.

Functions:
----------
- format_currency: Format numbers as currency with international support
- validate_financial_data: Comprehensive financial data validation
- calculate_percentage_change: Calculate percentage changes between periods
- normalize_company_name: Standardize company names for consistency
- generate_report_summary: Generate professional executive summaries
- export_analysis_results: Export complete analysis to multiple formats

Classes:
--------
- DataValidator: Advanced data validation with business rules
- ReportGenerator: Professional report generation with templates
- DateHelper: Financial date manipulation utilities
- ChartDataGenerator: Visualization data preparation
- ExcelReportBuilder: Advanced Excel report generation

Author: Prateek Dahiya
"""

import pandas as pd
import numpy as np
import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import json
from pathlib import Path
import warnings
from decimal import Decimal, ROUND_HALF_UP
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)


class DataValidator:
    """Advanced data validation for financial data with business rules"""
    
    def __init__(self):
        self.required_fields = {
            'balance_sheet': [
                'total_assets', 'current_assets', 'cash_and_equivalents',
                'total_liabilities', 'current_liabilities', 'total_equity'
            ],
            'income_statement': [
                'revenue', 'net_income', 'operating_income'
            ],
            'cash_flow': [
                'net_cash_from_operating_activities', 'net_cash_from_investing_activities',
                'net_cash_from_financing_activities'
            ]
        }
        
        self.validation_rules = {
            'total_assets': lambda x: x >= 0,
            'current_assets': lambda x: x >= 0,
            'cash_and_equivalents': lambda x: x >= 0,
            'total_liabilities': lambda x: x >= 0,
            'current_liabilities': lambda x: x >= 0,
            'total_equity': lambda x: True,  # Can be negative in distressed companies
            'revenue': lambda x: x >= 0,
            'inventory': lambda x: x >= 0,
            'accounts_receivable': lambda x: x >= 0,
            'accounts_payable': lambda x: x >= 0
        }
        
        # Business logic validation rules
        self.business_rules = {
            'current_assets_vs_total': lambda ca, ta: ca <= ta if ta > 0 else True,
            'current_liabilities_vs_total': lambda cl, tl: cl <= tl if tl > 0 else True,
            'cash_vs_current_assets': lambda cash, ca: cash <= ca if ca > 0 else True,
            'reasonable_profit_margin': lambda ni, rev: abs(ni/rev) <= 2.0 if rev > 0 else True
        }
    
    def validate_data_completeness(self, data: Dict[str, Any], data_type: str = 'general') -> Dict[str, Any]:
        """Advanced data completeness validation with scoring"""
        try:
            validation_result = {
                'is_valid': True,
                'missing_fields': [],
                'invalid_fields': [],
                'completeness_score': 0.0,
                'warnings': [],
                'critical_missing': [],
                'recommendations': []
            }
            
            required_fields = self.required_fields.get(data_type, ['total_assets', 'revenue', 'net_income'])
            
            missing_count = 0
            invalid_count = 0
            critical_fields = ['total_assets', 'revenue', 'net_income']  # Always critical
            
            for field in required_fields:
                if field not in data:
                    validation_result['missing_fields'].append(field)
                    missing_count += 1
                    if field in critical_fields:
                        validation_result['critical_missing'].append(field)
                else:
                    value = data[field]
                    if pd.isna(value) or value is None:
                        validation_result['missing_fields'].append(field)
                        missing_count += 1
                        if field in critical_fields:
                            validation_result['critical_missing'].append(field)
                    elif field in self.validation_rules:
                        if not self.validation_rules[field](value):
                            validation_result['invalid_fields'].append(field)
                            invalid_count += 1
            
            # Calculate completeness score with weights
            total_fields = len(required_fields)
            valid_fields = total_fields - missing_count - invalid_count
            validation_result['completeness_score'] = valid_fields / total_fields if total_fields > 0 else 0
            
            # Generate recommendations
            if missing_count > 0:
                validation_result['recommendations'].append(
                    f"Gather missing data for {missing_count} fields to improve analysis accuracy"
                )
            
            if validation_result['critical_missing']:
                validation_result['recommendations'].append(
                    "Critical financial data missing - consider alternative data sources"
                )
                validation_result['is_valid'] = False
            
            # Warnings based on completeness
            if validation_result['completeness_score'] < 0.5:
                validation_result['warnings'].append("Very low data completeness - analysis reliability compromised")
            elif validation_result['completeness_score'] < 0.7:
                validation_result['warnings'].append("Low data completeness - some analysis limitations expected")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating data completeness: {e}")
            return {
                'is_valid': False,
                'error': str(e),
                'missing_fields': [],
                'invalid_fields': [],
                'completeness_score': 0.0
            }
    
    def validate_data_consistency(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced data consistency validation with business rules"""
        try:
            consistency_result = {
                'is_consistent': True,
                'inconsistencies': [],
                'warnings': [],
                'business_rule_violations': [],
                'severity_scores': {}
            }
            
            # Balance Sheet Equation: Assets = Liabilities + Equity
            total_assets = data.get('total_assets', 0)
            total_liabilities = data.get('total_liabilities', 0)
            total_equity = data.get('total_equity', 0)
            
            if all(v != 0 for v in [total_assets, total_liabilities, total_equity]):
                balance_check = abs(total_assets - (total_liabilities + total_equity))
                tolerance = max(total_assets * 0.02, 1000)  # 2% or $1000 tolerance (tighter)
                
                if balance_check > tolerance:
                    severity = 'high' if balance_check > total_assets * 0.1 else 'medium'
                    consistency_result['inconsistencies'].append({
                        'type': 'balance_equation',
                        'message': f"Balance sheet equation imbalance: Assets ({total_assets:,.0f}) ≠ Liabilities + Equity ({total_liabilities + total_equity:,.0f})",
                        'difference': balance_check,
                        'severity': severity
                    })
                    consistency_result['is_consistent'] = False
            
            # Business Rules Validation
            current_assets = data.get('current_assets', 0)
            current_liabilities = data.get('current_liabilities', 0)
            cash = data.get('cash_and_equivalents', 0)
            revenue = data.get('revenue', 0)
            net_income = data.get('net_income', 0)
            
            # Rule: Current Assets <= Total Assets
            if not self.business_rules['current_assets_vs_total'](current_assets, total_assets):
                consistency_result['business_rule_violations'].append(
                    "Current assets exceed total assets - data integrity issue"
                )
            
            # Rule: Current Liabilities <= Total Liabilities  
            if not self.business_rules['current_liabilities_vs_total'](current_liabilities, total_liabilities):
                consistency_result['business_rule_violations'].append(
                    "Current liabilities exceed total liabilities - classification error"
                )
            
            # Rule: Cash <= Current Assets
            if not self.business_rules['cash_vs_current_assets'](cash, current_assets):
                consistency_result['business_rule_violations'].append(
                    "Cash exceeds current assets - impossible scenario"
                )
            
            # Rule: Reasonable Profit Margin
            if revenue > 0 and not self.business_rules['reasonable_profit_margin'](net_income, revenue):
                consistency_result['warnings'].append(
                    f"Extreme profit margin detected: {(net_income/revenue)*100:.1f}% - verify data accuracy"
                )
            
            # Additional consistency checks
            self._validate_ratio_reasonableness(data, consistency_result)
            
            return consistency_result
            
        except Exception as e:
            logger.error(f"Error validating data consistency: {e}")
            return {
                'is_consistent': False,
                'error': str(e),
                'inconsistencies': [],
                'warnings': []
            }
    
    def _validate_ratio_reasonableness(self, data: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Validate that calculated ratios are within reasonable ranges"""
        current_assets = data.get('current_assets', 0)
        current_liabilities = data.get('current_liabilities', 0)
        total_assets = data.get('total_assets', 0)
        revenue = data.get('revenue', 0)
        
        # Current Ratio reasonableness (0.1 to 20 is reasonable range)
        if current_liabilities > 0:
            current_ratio = current_assets / current_liabilities
            if current_ratio > 20:
                result['warnings'].append(f"Unusually high current ratio: {current_ratio:.1f} - verify current liabilities")
            elif current_ratio < 0.1:
                result['warnings'].append(f"Extremely low current ratio: {current_ratio:.1f} - potential liquidity crisis")
        
        # Asset Turnover reasonableness (0.01 to 10 is reasonable range)
        if total_assets > 0:
            asset_turnover = revenue / total_assets
            if asset_turnover > 10:
                result['warnings'].append(f"Unusually high asset turnover: {asset_turnover:.1f} - verify total assets")
            elif asset_turnover < 0.01 and revenue > 0:
                result['warnings'].append(f"Extremely low asset turnover: {asset_turnover:.3f} - verify asset valuation")


class ReportGenerator:
    """Professional report generation with enhanced templates"""
    
    def __init__(self):
        self.report_templates = {
            'executive_summary': self._get_executive_summary_template(),
            'detailed_analysis': self._get_detailed_analysis_template(),
            'risk_assessment': self._get_risk_assessment_template(),
            'trend_analysis': self._get_trend_analysis_template()
        }
    
    def _get_executive_summary_template(self) -> str:
        """Enhanced executive summary template"""
        return """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    EXECUTIVE SUMMARY - FINANCIAL RISK ASSESSMENT            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Company: {company_name}
Assessment Date: {assessment_date}
Report Period: {report_period}
Analysis Confidence: {confidence_level}%

╔══════════════════════════════════════════════════════════════════════════════╗
║                              OVERALL ASSESSMENT                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

• Financial Health Score: {health_score}/100 (Grade: {health_grade})
• Liquidation Risk Level: {risk_level} ({risk_probability:.1%} probability)
• Altman Z-Score: {z_score} ({z_score_interpretation})
• Industry Benchmark: {industry_comparison}

╔══════════════════════════════════════════════════════════════════════════════╗
║                                KEY FINDINGS                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

{key_findings}

╔══════════════════════════════════════════════════════════════════════════════╗
║                          IMMEDIATE RECOMMENDATIONS                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Priority Level: {priority_level}

{recommendations}

╔══════════════════════════════════════════════════════════════════════════════╗
║                           RISK FACTORS IDENTIFIED                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

{risk_factors}

╔══════════════════════════════════════════════════════════════════════════════╗
║                              MONITORING PLAN                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

{monitoring_plan}

DISCLAIMER: This assessment is based on financial data as of {data_date} and should be 
reviewed regularly as financial conditions change. Consult qualified financial 
professionals for critical business decisions.

Report Generated: {generation_timestamp}
        """
    
    def _get_detailed_analysis_template(self) -> str:
        """Enhanced detailed analysis template"""
        return """
╔══════════════════════════════════════════════════════════════════════════════╗
║                      DETAILED FINANCIAL ANALYSIS REPORT                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Company: {company_name}
Analysis Date: {analysis_date}
Reporting Currency: {currency}

╔══════════════════════════════════════════════════════════════════════════════╗
║                            LIQUIDITY ANALYSIS                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

{liquidity_analysis}

╔══════════════════════════════════════════════════════════════════════════════╗
║                          PROFITABILITY ANALYSIS                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

{profitability_analysis}

╔══════════════════════════════════════════════════════════════════════════════╗
║                            LEVERAGE ANALYSIS                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

{leverage_analysis}

╔══════════════════════════════════════════════════════════════════════════════╗
║                           EFFICIENCY ANALYSIS                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

{efficiency_analysis}

╔══════════════════════════════════════════════════════════════════════════════╗
║                            CASH FLOW ANALYSIS                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

{cash_flow_analysis}

╔══════════════════════════════════════════════════════════════════════════════╗
║                         COMPARATIVE ANALYSIS                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

{comparative_analysis}

╔══════════════════════════════════════════════════════════════════════════════╗
║                          DETAILED RECOMMENDATIONS                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

{detailed_recommendations}
        """
    
    def validate_data_completeness(self, data: Dict[str, Any], data_type: str = 'general') -> Dict[str, Any]:
        """Validate data completeness"""
        try:
            validation_result = {
                'is_valid': True,
                'missing_fields': [],
                'invalid_fields': [],
                'completeness_score': 0.0,
                'warnings': []
            }
            
            required_fields = self.required_fields.get(data_type, [])
            
            if not required_fields:
                # General validation - check for basic financial data
                required_fields = ['total_assets', 'revenue', 'net_income']
            
            missing_count = 0
            invalid_count = 0
            
            for field in required_fields:
                if field not in data:
                    validation_result['missing_fields'].append(field)
                    missing_count += 1
                else:
                    value = data[field]
                    if pd.isna(value) or value is None:
                        validation_result['missing_fields'].append(field)
                        missing_count += 1
                    elif field in self.validation_rules:
                        if not self.validation_rules[field](value):
                            validation_result['invalid_fields'].append(field)
                            invalid_count += 1
            
            # Calculate completeness score
            total_fields = len(required_fields)
            valid_fields = total_fields - missing_count - invalid_count
            validation_result['completeness_score'] = valid_fields / total_fields if total_fields > 0 else 0
            
            # Set overall validity
            if missing_count > 0 or invalid_count > 0:
                validation_result['is_valid'] = False
            
            # Add warnings for low completeness
            if validation_result['completeness_score'] < 0.7:
                validation_result['warnings'].append("Low data completeness - analysis may be limited")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating data completeness: {e}")
            return {
                'is_valid': False,
                'error': str(e),
                'missing_fields': [],
                'invalid_fields': [],
                'completeness_score': 0.0
            }
    
    def validate_data_consistency(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data consistency and logical relationships"""
        try:
            consistency_result = {
                'is_consistent': True,
                'inconsistencies': [],
                'warnings': []
            }
            
            # Balance Sheet Equation: Assets = Liabilities + Equity
            total_assets = data.get('total_assets', 0)
            total_liabilities = data.get('total_liabilities', 0)
            total_equity = data.get('total_equity', 0)
            
            if all(v != 0 for v in [total_assets, total_liabilities, total_equity]):
                balance_check = abs(total_assets - (total_liabilities + total_equity))
                tolerance = max(total_assets * 0.05, 1000)  # 5% or $1000 tolerance
                
                if balance_check > tolerance:
                    consistency_result['inconsistencies'].append(
                        f"Balance sheet equation imbalance: Assets ({total_assets:,.0f}) ≠ Liabilities + Equity ({total_liabilities + total_equity:,.0f})"
                    )
                    consistency_result['is_consistent'] = False
            
            # Current Assets should be <= Total Assets
            current_assets = data.get('current_assets', 0)
            if current_assets > 0 and total_assets > 0:
                if current_assets > total_assets:
                    consistency_result['inconsistencies'].append(
                        "Current assets cannot exceed total assets"
                    )
                    consistency_result['is_consistent'] = False
            
            # Current Liabilities should be <= Total Liabilities
            current_liabilities = data.get('current_liabilities', 0)
            if current_liabilities > 0 and total_liabilities > 0:
                if current_liabilities > total_liabilities:
                    consistency_result['inconsistencies'].append(
                        "Current liabilities cannot exceed total liabilities"
                    )
                    consistency_result['is_consistent'] = False
            
            # Cash should be <= Current Assets
            cash = data.get('cash_and_equivalents', 0)
            if cash > 0 and current_assets > 0:
                if cash > current_assets:
                    consistency_result['inconsistencies'].append(
                        "Cash cannot exceed current assets"
                    )
                    consistency_result['is_consistent'] = False
            
            # Revenue consistency checks
            revenue = data.get('revenue', 0)
            net_income = data.get('net_income', 0)
            
            if revenue != 0 and net_income != 0:
                profit_margin = net_income / revenue
                if profit_margin > 0.5:  # 50% profit margin is unusually high
                    consistency_result['warnings'].append(
                        f"Unusually high profit margin: {profit_margin:.1%}"
                    )
                elif profit_margin < -2.0:  # Loss exceeding 200% of revenue
                    consistency_result['warnings'].append(
                        f"Extremely high loss relative to revenue: {profit_margin:.1%}"
                    )
            
            return consistency_result
            
        except Exception as e:
            logger.error(f"Error validating data consistency: {e}")
            return {
                'is_consistent': False,
                'error': str(e),
                'inconsistencies': [],
                'warnings': []
            }


class ReportGenerator:
    """Generate formatted reports and summaries"""
    
    def __init__(self):
        self.report_templates = {
            'executive_summary': self._get_executive_summary_template(),
            'detailed_analysis': self._get_detailed_analysis_template(),
            'risk_assessment': self._get_risk_assessment_template()
        }
    
    def _get_executive_summary_template(self) -> str:
        """Get executive summary template"""
        return """
EXECUTIVE SUMMARY - FINANCIAL RISK ASSESSMENT
=============================================

Company: {company_name}
Assessment Date: {assessment_date}
Report Period: {report_period}

OVERALL ASSESSMENT:
• Financial Health Score: {health_score}/100 (Grade: {health_grade})
• Liquidation Risk Level: {risk_level}
• Z-Score: {z_score} ({z_score_interpretation})

KEY FINDINGS:
{key_findings}

IMMEDIATE RECOMMENDATIONS:
{recommendations}

RISK FACTORS IDENTIFIED:
{risk_factors}

This assessment is based on financial data as of {data_date} and should be 
reviewed regularly as financial conditions change.
        """
    
    def _get_detailed_analysis_template(self) -> str:
        """Get detailed analysis template"""
        return """
DETAILED FINANCIAL ANALYSIS REPORT
==================================

Company: {company_name}
Analysis Date: {analysis_date}

FINANCIAL RATIOS ANALYSIS:
{ratio_analysis}

CASH FLOW ANALYSIS:
{cash_flow_analysis}

WORKING CAPITAL ANALYSIS:
{working_capital_analysis}

COMPARATIVE ANALYSIS:
{comparative_analysis}

TREND ANALYSIS:
{trend_analysis}

DETAILED RECOMMENDATIONS:
{detailed_recommendations}
        """
    
    def _get_risk_assessment_template(self) -> str:
        """Get risk assessment template"""
        return """
FINANCIAL RISK ASSESSMENT REPORT
================================

Company: {company_name}
Risk Assessment Date: {assessment_date}

OVERALL RISK SCORE: {overall_risk_score}
RISK CATEGORY: {risk_category}

RISK BREAKDOWN:
• Credit Risk: {credit_risk}
• Liquidity Risk: {liquidity_risk}  
• Operational Risk: {operational_risk}
• Market Risk: {market_risk}

SPECIFIC RISK FACTORS:
{specific_risk_factors}

MITIGATION STRATEGIES:
{mitigation_strategies}

MONITORING RECOMMENDATIONS:
{monitoring_recommendations}
        """
    
    def generate_executive_summary(self, analysis_results: Dict[str, Any], 
                                 company_name: str = "Unknown Company") -> str:
        """Generate executive summary report"""
        try:
            # Extract key information
            health_score = analysis_results.get('financial_health', {}).get('overall_score', 0)
            health_grade = analysis_results.get('financial_health', {}).get('health_grade', 'N/A')
            risk_level = analysis_results.get('liquidation_risk', {}).get('risk_level', 'unknown')
            z_score = analysis_results.get('z_score', {}).get('z_score', 'N/A')
            
            # Generate key findings
            key_findings = self._generate_key_findings(analysis_results)
            recommendations = self._generate_recommendations(analysis_results)
            risk_factors = self._generate_risk_factors(analysis_results)
            
            # Format template
            summary = self.report_templates['executive_summary'].format(
                company_name=company_name,
                assessment_date=datetime.now().strftime('%B %d, %Y'),
                report_period=f"Year ending {datetime.now().year}",
                health_score=health_score,
                health_grade=health_grade,
                risk_level=risk_level.title(),
                z_score=z_score,
                z_score_interpretation=self._interpret_z_score(z_score),
                key_findings=key_findings,
                recommendations=recommendations,
                risk_factors=risk_factors,
                data_date=datetime.now().strftime('%B %d, %Y')
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return f"Error generating executive summary: {str(e)}"
    
    def _generate_key_findings(self, analysis_results: Dict[str, Any]) -> str:
        """Generate key findings section"""
        findings = []
        
        # Financial health findings
        health_data = analysis_results.get('financial_health', {})
        if health_data:
            health_score = health_data.get('overall_score', 0)
            if health_score >= 80:
                findings.append("• Company demonstrates excellent financial health with strong performance across all metrics")
            elif health_score >= 60:
                findings.append("• Company shows reasonable financial health with some areas needing attention")
            else:
                findings.append("• Company faces significant financial challenges requiring immediate attention")
        
        # Liquidity findings
        component_scores = health_data.get('component_scores', {})
        liquidity_score = component_scores.get('liquidity', 0)
        if liquidity_score < 50:
            findings.append("• Liquidity position is concerning and may impact short-term operations")
        elif liquidity_score > 80:
            findings.append("• Strong liquidity position provides good operational flexibility")
        
        # Profitability findings
        profitability_score = component_scores.get('profitability', 0)
        if profitability_score < 40:
            findings.append("• Profitability is below industry standards and requires improvement")
        elif profitability_score > 80:
            findings.append("• Excellent profitability performance exceeds industry benchmarks")
        
        return '\n'.join(findings) if findings else "• No significant findings identified"
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> str:
        """Generate recommendations section"""
        recommendations = []
        
        # Based on component scores
        component_scores = analysis_results.get('financial_health', {}).get('component_scores', {})
        
        if component_scores.get('liquidity', 0) < 60:
            recommendations.append("• Improve liquidity through better cash management and working capital optimization")
        
        if component_scores.get('profitability', 0) < 60:
            recommendations.append("• Focus on margin improvement through cost reduction or revenue enhancement initiatives")
        
        if component_scores.get('leverage', 0) < 60:
            recommendations.append("• Consider debt reduction strategies to improve financial flexibility")
        
        if component_scores.get('cash_flow', 0) < 60:
            recommendations.append("• Strengthen cash flow generation through operational improvements")
        
        # Liquidation risk recommendations
        risk_data = analysis_results.get('liquidation_risk', {})
        risk_level = risk_data.get('risk_level', '')
        if risk_level in ['high', 'critical']:
            recommendations.append("• Implement immediate financial restructuring plan")
            recommendations.append("• Consider engaging financial advisors for turnaround strategy")
        
        return '\n'.join(recommendations) if recommendations else "• Continue current financial management practices"
    
    def _generate_risk_factors(self, analysis_results: Dict[str, Any]) -> str:
        """Generate risk factors section"""
        risk_factors = []
        
        # Extract risk factors from liquidation risk analysis
        liquidation_risk = analysis_results.get('liquidation_risk', {})
        if liquidation_risk.get('risk_factors'):
            for factor in liquidation_risk['risk_factors']:
                risk_factors.append(f"• {factor}")
        
        # Add component-based risk factors
        component_scores = analysis_results.get('financial_health', {}).get('component_scores', {})
        
        if component_scores.get('liquidity', 100) < 50:
            risk_factors.append("• Poor liquidity ratios indicate potential cash flow problems")
        
        if component_scores.get('leverage', 100) < 50:
            risk_factors.append("• High debt levels create financial vulnerability")
        
        return '\n'.join(risk_factors) if risk_factors else "• No significant risk factors identified"
    
    def _interpret_z_score(self, z_score) -> str:
        """Interpret Z-Score value"""
        try:
            z_value = float(z_score)
            if z_value > 3.0:
                return "Safe Zone"
            elif z_value > 1.8:
                return "Grey Zone - Monitor"
            else:
                return "Distress Zone"
        except (ValueError, TypeError):
            return "Unable to interpret"


class DateHelper:
    """Date manipulation utilities for financial analysis"""
    
    @staticmethod
    def get_financial_year_range(year: int) -> Tuple[datetime, datetime]:
        """Get start and end dates for a financial year"""
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)
        return start_date, end_date
    
    @staticmethod
    def get_quarter_range(year: int, quarter: int) -> Tuple[datetime, datetime]:
        """Get start and end dates for a specific quarter"""
        quarter_months = {
            1: (1, 3),
            2: (4, 6),
            3: (7, 9),
            4: (10, 12)
        }
        
        start_month, end_month = quarter_months.get(quarter, (1, 3))
        start_date = datetime(year, start_month, 1)
        
        # Get last day of end month
        if end_month == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(year, end_month + 1, 1) - timedelta(days=1)
        
        return start_date, end_date
    
    @staticmethod
    def format_date_range(start_date: datetime, end_date: datetime) -> str:
        """Format date range for display"""
        if start_date.year == end_date.year:
            return f"{start_date.strftime('%B %d')} - {end_date.strftime('%B %d, %Y')}"
        else:
            return f"{start_date.strftime('%B %d, %Y')} - {end_date.strftime('%B %d, %Y')}"
    
    @staticmethod
    def get_periods_between_dates(start_date: datetime, end_date: datetime, 
                                 period_type: str = 'months') -> int:
        """Calculate number of periods between two dates"""
        if period_type == 'days':
            return (end_date - start_date).days
        elif period_type == 'months':
            return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
        elif period_type == 'years':
            return end_date.year - start_date.year
        else:
            return 0


# Utility Functions

def format_currency(amount: Union[int, float], currency: str = 'USD', 
                   show_cents: bool = True) -> str:
    """
    Format number as currency
    
    Args:
        amount: Numeric amount
        currency: Currency code (USD, EUR, GBP, etc.)
        show_cents: Whether to show decimal places
        
    Returns:
        Formatted currency string
    """
    try:
        if pd.isna(amount) or amount is None:
            return "N/A"
        
        currency_symbols = {
            'USD': '$',
            'EUR': '€',
            'GBP': '£',
            'CAD': 'C',
            'AUD': 'A',
            'JPY': '¥'
        }
        
        symbol = currency_symbols.get(currency, currency + ' ')
        
        if show_cents and currency != 'JPY':
            if abs(amount) >= 1_000_000_000:
                return f"{symbol}{amount/1_000_000_000:.2f}B"
            elif abs(amount) >= 1_000_000:
                return f"{symbol}{amount/1_000_000:.2f}M"
            elif abs(amount) >= 1_000:
                return f"{symbol}{amount/1_000:.2f}K"
            else:
                return f"{symbol}{amount:,.2f}"
        else:
            if abs(amount) >= 1_000_000_000:
                return f"{symbol}{amount/1_000_000_000:.0f}B"
            elif abs(amount) >= 1_000_000:
                return f"{symbol}{amount/1_000_000:.0f}M"
            elif abs(amount) >= 1_000:
                return f"{symbol}{amount/1_000:.0f}K"
            else:
                return f"{symbol}{amount:,.0f}"
                
    except (TypeError, ValueError):
        return str(amount)


def format_percentage(value: Union[int, float], decimal_places: int = 1) -> str:
    """
    Format number as percentage
    
    Args:
        value: Numeric value (0.15 = 15%)
        decimal_places: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    try:
        if pd.isna(value) or value is None:
            return "N/A"
        
        return f"{value * 100:.{decimal_places}f}%"
        
    except (TypeError, ValueError):
        return str(value)


def calculate_percentage_change(old_value: Union[int, float], 
                              new_value: Union[int, float]) -> float:
    """
    Calculate percentage change between two values
    
    Args:
        old_value: Previous value
        new_value: Current value
        
    Returns:
        Percentage change (decimal form)
    """
    try:
        if pd.isna(old_value) or pd.isna(new_value) or old_value == 0:
            return 0.0
        
        return (new_value - old_value) / abs(old_value)
        
    except (TypeError, ZeroDivisionError):
        return 0.0


def normalize_company_name(company_name: str) -> str:
    """
    Normalize company name for consistency
    
    Args:
        company_name: Raw company name
        
    Returns:
        Normalized company name
    """
    try:
        if not company_name or pd.isna(company_name):
            return "Unknown Company"
        
        # Convert to string and strip whitespace
        name = str(company_name).strip()
        
        # Remove common suffixes and normalize
        suffixes = [
            'inc', 'incorporated', 'corp', 'corporation', 'ltd', 'limited',
            'llc', 'plc', 'co', 'company', 'group', 'holdings'
        ]
        
        # Split name into words
        words = name.split()
        
        # Remove suffix if it's the last word
        if len(words) > 1 and words[-1].lower().replace('.', '') in suffixes:
            words = words[:-1]
        
        # Join and title case
        normalized_name = ' '.join(words)
        normalized_name = ' '.join([word.capitalize() for word in normalized_name.split()])
        
        return normalized_name
        
    except Exception as e:
        logger.error(f"Error normalizing company name: {e}")
        return str(company_name)


def validate_financial_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Comprehensive financial data validation
    
    Args:
        data: Financial data dictionary
        
    Returns:
        Validation results
    """
    validator = DataValidator()
    
    # Validate completeness
    completeness_result = validator.validate_data_completeness(data)
    
    # Validate consistency
    consistency_result = validator.validate_data_consistency(data)
    
    # Combine results
    overall_valid = completeness_result['is_valid'] and consistency_result['is_consistent']
    
    return {
        'is_valid': overall_valid,
        'completeness': completeness_result,
        'consistency': consistency_result,
        'validation_date': datetime.now()
    }


def generate_report_summary(analysis_results: Dict[str, Any], 
                          company_name: str = "Unknown Company") -> str:
    """
    Generate executive summary report
    
    Args:
        analysis_results: Complete analysis results
        company_name: Company name
        
    Returns:
        Executive summary report
    """
    generator = ReportGenerator()
    return generator.generate_executive_summary(analysis_results, company_name)


def calculate_financial_ratios(financial_data: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate common financial ratios
    
    Args:
        financial_data: Dictionary of financial statement data
        
    Returns:
        Dictionary of calculated ratios
    """
    try:
        ratios = {}
        
        # Extract key values
        current_assets = financial_data.get('current_assets', 0)
        current_liabilities = financial_data.get('current_liabilities', 0)
        total_assets = financial_data.get('total_assets', 0)
        total_liabilities = financial_data.get('total_liabilities', 0)
        total_equity = financial_data.get('total_equity', 0)
        revenue = financial_data.get('revenue', 0)
        net_income = financial_data.get('net_income', 0)
        inventory = financial_data.get('inventory', 0)
        cash = financial_data.get('cash_and_equivalents', 0)
        
        # Liquidity Ratios
        if current_liabilities != 0:
            ratios['current_ratio'] = current_assets / current_liabilities
            ratios['quick_ratio'] = (current_assets - inventory) / current_liabilities
            ratios['cash_ratio'] = cash / current_liabilities
        
        # Leverage Ratios
        if total_equity != 0:
            ratios['debt_to_equity'] = total_liabilities / total_equity
        
        if total_assets != 0:
            ratios['debt_to_assets'] = total_liabilities / total_assets
            ratios['asset_turnover'] = revenue / total_assets
            ratios['roa'] = net_income / total_assets  # Return on Assets
        
        if total_equity != 0:
            ratios['roe'] = net_income / total_equity  # Return on Equity
        
        # Profitability Ratios
        if revenue != 0:
            ratios['profit_margin'] = net_income / revenue
        
        # Working Capital
        ratios['working_capital'] = current_assets - current_liabilities
        
        return ratios
        
    except Exception as e:
        logger.error(f"Error calculating financial ratios: {e}")
        return {}


def clean_numeric_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean and standardize numeric data
    
    Args:
        data: Dictionary with potentially dirty numeric data
        
    Returns:
        Dictionary with cleaned numeric data
    """
    try:
        cleaned_data = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                # Try to convert string to numeric
                cleaned_value = clean_currency_string(value)
                cleaned_data[key] = cleaned_value
            elif isinstance(value, (int, float)):
                # Handle NaN and infinity
                if pd.isna(value) or np.isinf(value):
                    cleaned_data[key] = 0.0
                else:
                    cleaned_data[key] = float(value)
            else:
                cleaned_data[key] = value
        
        return cleaned_data
        
    except Exception as e:
        logger.error(f"Error cleaning numeric data: {e}")
        return data


def clean_currency_string(currency_str: str) -> float:
    """
    Clean currency string and convert to float
    
    Args:
        currency_str: String containing currency amount
        
    Returns:
        Numeric value
    """
    try:
        if not currency_str or pd.isna(currency_str):
            return 0.0
        
        # Convert to string and clean
        clean_str = str(currency_str).strip()
        
        # Remove currency symbols and common formatting
        clean_str = re.sub(r'[,$€£¥\s()]', '', clean_str)
        
        # Handle parentheses (negative values)
        is_negative = '(' in str(currency_str) and ')' in str(currency_str)
        
        # Handle minus signs
        if clean_str.startswith('-'):
            is_negative = True
            clean_str = clean_str[1:]
        
        # Handle suffixes (K, M, B)
        multiplier = 1
        clean_str = clean_str.upper()
        if clean_str.endswith('K'):
            multiplier = 1_000
            clean_str = clean_str[:-1]
        elif clean_str.endswith('M'):
            multiplier = 1_000_000
            clean_str = clean_str[:-1]
        elif clean_str.endswith('B'):
            multiplier = 1_000_000_000
            clean_str = clean_str[:-1]
        
        # Remove any remaining non-numeric characters except decimal point
        clean_str = re.sub(r'[^\d.]', '', clean_str)
        
        if not clean_str or clean_str == '.':
            return 0.0
        
        # Convert to float
        value = float(clean_str) * multiplier
        return -value if is_negative else value
        
    except (ValueError, TypeError):
        return 0.0


def get_industry_benchmarks(industry: str) -> Dict[str, float]:
    """
    Get industry benchmark ratios
    
    Args:
        industry: Industry name
        
    Returns:
        Dictionary of benchmark ratios
    """
    benchmarks = {
        'technology': {
            'current_ratio': 2.5,
            'debt_to_equity': 0.3,
            'profit_margin': 0.15,
            'roa': 0.12,
            'roe': 0.18
        },
        'healthcare': {
            'current_ratio': 2.0,
            'debt_to_equity': 0.4,
            'profit_margin': 0.12,
            'roa': 0.08,
            'roe': 0.14
        },
        'industrial': {
            'current_ratio': 1.8,
            'debt_to_equity': 0.6,
            'profit_margin': 0.08,
            'roa': 0.06,
            'roe': 0.12
        },
        'financial': {
            'current_ratio': 1.1,
            'debt_to_equity': 5.0,
            'profit_margin': 0.25,
            'roa': 0.02,
            'roe': 0.12
        },
        'retail': {
            'current_ratio': 1.5,
            'debt_to_equity': 0.8,
            'profit_margin': 0.05,
            'roa': 0.04,
            'roe': 0.10
        }
    }
    
    return benchmarks.get(industry.lower(), benchmarks['industrial'])  # Default to industrial


def create_performance_summary(financial_data: Dict[str, float],
                             industry: str = 'industrial') -> Dict[str, Any]:
    """
    Create performance summary with benchmarks
    
    Args:
        financial_data: Financial data
        industry: Industry for benchmarking
        
    Returns:
        Performance summary
    """
    try:
        # Calculate ratios
        ratios = calculate_financial_ratios(financial_data)
        
        # Get benchmarks
        benchmarks = get_industry_benchmarks(industry)
        
        # Compare with benchmarks
        performance = {}
        
        for ratio_name, ratio_value in ratios.items():
            if ratio_name in benchmarks:
                benchmark = benchmarks[ratio_name]
                
                if benchmark != 0:
                    performance_ratio = ratio_value / benchmark
                    
                    if performance_ratio >= 1.2:
                        status = 'excellent'
                    elif performance_ratio >= 1.0:
                        status = 'good'
                    elif performance_ratio >= 0.8:
                        status = 'fair'
                    else:
                        status = 'poor'
                else:
                    status = 'unknown'
                    performance_ratio = 0
                
                performance[ratio_name] = {
                    'value': ratio_value,
                    'benchmark': benchmark,
                    'performance_ratio': performance_ratio,
                    'status': status
                }
        
        return {
            'success': True,
            'performance': performance,
            'industry': industry,
            'summary_date': datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error creating performance summary: {e}")
        return {
            'success': False,
            'error': str(e),
            'performance': {}
        }


def export_to_excel(data: Dict[str, Any], filename: str = None) -> str:
    """
    Export analysis results to Excel file
    
    Args:
        data: Analysis results dictionary
        filename: Optional filename
        
    Returns:
        Generated filename
    """
    try:
        if filename is None:
            filename = f"financial_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = {
                'Metric': [],
                'Value': [],
                'Status': []
            }
            
            # Add financial health data
            if 'financial_health' in data:
                health_data = data['financial_health']
                summary_data['Metric'].extend([
                    'Overall Health Score',
                    'Health Grade',
                    'Liquidity Score',
                    'Profitability Score',
                    'Leverage Score'
                ])
                summary_data['Value'].extend([
                    health_data.get('overall_score', 0),
                    health_data.get('health_grade', 'N/A'),
                    health_data.get('component_scores', {}).get('liquidity', 0),
                    health_data.get('component_scores', {}).get('profitability', 0),
                    health_data.get('component_scores', {}).get('leverage', 0)
                ])
                summary_data['Status'].extend([
                    health_data.get('health_status', 'N/A'),
                    'Grade',
                    'Component',
                    'Component',
                    'Component'
                ])
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Ratios sheet (if available)
            if 'ratios' in data:
                ratios_df = pd.DataFrame(list(data['ratios'].items()), 
                                       columns=['Ratio', 'Value'])
                ratios_df.to_excel(writer, sheet_name='Ratios', index=False)
            
            # Risk assessment sheet (if available)
            if 'liquidation_risk' in data:
                risk_data = data['liquidation_risk']
                risk_df = pd.DataFrame([
                    ['Risk Probability', risk_data.get('risk_probability', 0)],
                    ['Risk Level', risk_data.get('risk_level', 'unknown')],
                    ['Risk Description', risk_data.get('risk_description', 'N/A')]
                ], columns=['Risk Factor', 'Value'])
                risk_df.to_excel(writer, sheet_name='Risk Assessment', index=False)
        
        return filename
        
    except Exception as e:
        logger.error(f"Error exporting to Excel: {e}")
        return None


def generate_chart_data(financial_data: Dict[str, float]) -> Dict[str, Any]:
    """
    Generate data structure for chart visualization
    
    Args:
        financial_data: Financial data dictionary
        
    Returns:
        Chart data structure
    """
    try:
        # Calculate ratios
        ratios = calculate_financial_ratios(financial_data)
        
        # Prepare chart data
        chart_data = {
            'ratios_chart': {
                'labels': [],
                'values': []
            },
            'balance_sheet_chart': {
                'assets': financial_data.get('total_assets', 0),
                'liabilities': financial_data.get('total_liabilities', 0),
                'equity': financial_data.get('total_equity', 0)
            },
            'performance_indicators': {}
        }
        
        # Add key ratios for chart
        key_ratios = ['current_ratio', 'debt_to_equity', 'profit_margin', 'roa', 'roe']
        
        for ratio in key_ratios:
            if ratio in ratios:
                chart_data['ratios_chart']['labels'].append(ratio.replace('_', ' ').title())
                chart_data['ratios_chart']['values'].append(ratios[ratio])
        
        # Add performance indicators
        current_ratio = ratios.get('current_ratio', 0)
        if current_ratio >= 2.0:
            chart_data['performance_indicators']['liquidity'] = 'excellent'
        elif current_ratio >= 1.5:
            chart_data['performance_indicators']['liquidity'] = 'good'
        elif current_ratio >= 1.0:
            chart_data['performance_indicators']['liquidity'] = 'fair'
        else:
            chart_data['performance_indicators']['liquidity'] = 'poor'
        
        return chart_data
        
    except Exception as e:
        logger.error(f"Error generating chart data: {e}")
        return {}


# Configuration and Constants
CURRENCY_SYMBOLS = {
    'USD': '$',      
    'EUR': '€',
    'GBP': '£',
    'CAD': 'C$',     
    'AUD': 'A$',     
    'JPY': '¥',
    'CHF': 'CHF',
    'CNY': '¥'       
}

DEFAULT_INDUSTRY_BENCHMARKS = {
    'current_ratio': 2.0,
    'debt_to_equity': 0.5,
    'profit_margin': 0.10,
    'roa': 0.08,
    'roe': 0.15
}

RISK_LEVEL_COLORS = {
    'low': '#28a745',
    'medium': '#ffc107',
    'high': '#fd7e14',
    'critical': '#dc3545'
}

HEALTH_GRADE_COLORS = {
    'A': '#28a745',
    'B': '#6c757d',
    'C': '#ffc107',
    'D': '#fd7e14',
    'F': '#dc3545'
}

# ADD THESE FUNCTIONS TO YOUR utils/helpers.py FILE
# Just copy-paste these functions at the end of your existing helpers.py

def calculate_cash_flow_health_score(company_data: Dict[str, Any]) -> float:
    """
    Calculate comprehensive cash flow health score for a company
    
    Args:
        company_data: Dictionary containing company financial data
        
    Returns:
        Health score (0-100)
    """
    try:
        score = 0
        max_score = 100
        
        # Extract financial metrics (handle None values)
        net_income = company_data.get('net_income', 0) or 0
        operating_cash_flow = company_data.get('net_cash_from_operating_activities', 0) or 0
        free_cash_flow = company_data.get('free_cash_flow', 0) or 0
        revenue = company_data.get('revenue', 0) or 0
        total_assets = company_data.get('total_assets', 0) or 0
        current_ratio = company_data.get('current_ratio', 0) or 0
        debt_to_equity = company_data.get('debt_to_equity_ratio', 0) or 0
        liquidation_label = company_data.get('liquidation_label', 0) or 0
        
        # 1. Profitability Assessment (25 points)
        if net_income > 0:
            score += 20
            if revenue > 0:
                profit_margin = net_income / revenue
                if profit_margin > 0.15:
                    score += 5  # Excellent margin
                elif profit_margin > 0.05:
                    score += 3  # Good margin
        elif net_income == 0:
            score += 5  # Break-even is better than loss
        
        # 2. Operating Cash Flow Assessment (25 points)
        if operating_cash_flow > 0:
            score += 20
            if net_income > 0:
                ocf_to_ni_ratio = operating_cash_flow / net_income if net_income != 0 else 0
                if ocf_to_ni_ratio > 1.2:
                    score += 5  # Strong cash conversion
                elif ocf_to_ni_ratio > 0.8:
                    score += 3  # Good cash conversion
        elif operating_cash_flow == 0:
            score += 5  # Break-even is better than negative
        
        # 3. Free Cash Flow Assessment (20 points)
        if free_cash_flow > 0:
            score += 15
            if revenue > 0:
                fcf_margin = free_cash_flow / revenue
                if fcf_margin > 0.1:
                    score += 5  # Excellent FCF margin
                elif fcf_margin > 0.05:
                    score += 3  # Good FCF margin
        elif free_cash_flow == 0:
            score += 3  # Break-even
        
        # 4. Liquidity Assessment (15 points)
        if current_ratio >= 2.0:
            score += 15  # Excellent liquidity
        elif current_ratio >= 1.5:
            score += 12  # Good liquidity
        elif current_ratio >= 1.0:
            score += 8   # Fair liquidity
        elif current_ratio >= 0.5:
            score += 4   # Poor liquidity
        # Below 0.5 gets 0 points
        
        # 5. Leverage Assessment (10 points)
        if debt_to_equity <= 0.3:
            score += 10  # Conservative leverage
        elif debt_to_equity <= 0.6:
            score += 8   # Moderate leverage
        elif debt_to_equity <= 1.0:
            score += 5   # Higher leverage
        elif debt_to_equity <= 2.0:
            score += 2   # High leverage
        # Above 2.0 gets 0 points
        
        # 6. Risk Penalty (5 points deduction)
        if liquidation_label == 1:
            score -= 15  # High risk penalty
        
        # Ensure score is within bounds
        final_score = max(0, min(score, max_score))
        
        return round(final_score, 1)
        
    except Exception as e:
        logger.error(f"Error calculating cash flow health score: {e}")
        return 50.0  # Default neutral score


def calculate_enhanced_health_score(company_data: Dict[str, Any]) -> float:
    """
    Enhanced health score calculation using both cash flow and balance sheet data
    
    Args:
        company_data: Dictionary containing comprehensive financial data
        
    Returns:
        Enhanced health score (0-100)
    """
    try:
        # Use the existing cash flow health score as base
        base_score = calculate_cash_flow_health_score(company_data)
        
        # Add balance sheet enhancements if data available
        enhancement_score = 0
        
        # Balance sheet ratios enhancement (max 10 additional points)
        current_ratio = company_data.get('current_ratio', 0) or 0
        debt_ratio = company_data.get('debt_to_equity_ratio', 0) or 0
        total_assets = company_data.get('total_assets', 0) or 0
        
        # Current ratio bonus
        if current_ratio >= 2.5:
            enhancement_score += 3
        elif current_ratio >= 2.0:
            enhancement_score += 2
        elif current_ratio >= 1.5:
            enhancement_score += 1
        
        # Debt ratio bonus
        if debt_ratio <= 0.2:
            enhancement_score += 3
        elif debt_ratio <= 0.5:
            enhancement_score += 2
        elif debt_ratio <= 1.0:
            enhancement_score += 1
        
        # Asset size stability bonus
        if total_assets > 1_000_000:  # $1M+
            enhancement_score += 2
        elif total_assets > 100_000:  # $100K+
            enhancement_score += 1
        
        # Combine scores
        final_score = min(100, base_score + enhancement_score)
        
        return round(final_score, 1)
        
    except Exception as e:
        logger.error(f"Error calculating enhanced health score: {e}")
        return calculate_cash_flow_health_score(company_data)


def get_risk_level_from_score(health_score: float) -> str:
    """
    Convert health score to risk level
    
    Args:
        health_score: Numeric health score (0-100)
        
    Returns:
        Risk level string
    """
    if health_score >= 85:
        return 'Very Low'
    elif health_score >= 70:
        return 'Low'
    elif health_score >= 55:
        return 'Moderate'
    elif health_score >= 40:
        return 'Medium'
    elif health_score >= 25:
        return 'High'
    else:
        return 'Critical'


def format_health_score_display(health_score: float) -> Dict[str, Any]:
    """
    Format health score for display with color coding
    
    Args:
        health_score: Numeric health score
        
    Returns:
        Display information dictionary
    """
    risk_level = get_risk_level_from_score(health_score)
    
    # Color mapping
    color_map = {
        'Very Low': '#28a745',   # Green
        'Low': '#6c757d',        # Light green  
        'Moderate': '#ffc107',   # Yellow
        'Medium': '#fd7e14',     # Orange
        'High': '#dc3545',       # Red
        'Critical': '#8b0000'    # Dark red
    }
    
    # Grade mapping
    grade_map = {
        'Very Low': 'A+',
        'Low': 'A',
        'Moderate': 'B',
        'Medium': 'C',
        'High': 'D',
        'Critical': 'F'
    }
    
    return {
        'score': health_score,
        'risk_level': risk_level,
        'grade': grade_map.get(risk_level, 'F'),
        'color': color_map.get(risk_level, '#8b0000'),
        'description': f"{health_score:.1f}/100 ({risk_level} Risk)"
    }


def analyze_financial_trends(historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze financial trends from historical data
    
    Args:
        historical_data: List of financial data dictionaries sorted by year
        
    Returns:
        Trend analysis results
    """
    try:
        if len(historical_data) < 2:
            return {
                'trends_available': False,
                'message': 'Insufficient data for trend analysis'
            }
        
        trends = {
            'trends_available': True,
            'revenue_trend': 'stable',
            'profitability_trend': 'stable',
            'liquidity_trend': 'stable',
            'health_score_trend': 'stable',
            'trend_details': {}
        }
        
        # Sort by year to ensure proper ordering
        sorted_data = sorted(historical_data, key=lambda x: x.get('year', 0))
        
        # Calculate trends for key metrics
        metrics = ['net_income', 'net_cash_from_operating_activities', 'current_ratio']
        
        for metric in metrics:
            values = [float(data.get(metric, 0) or 0) for data in sorted_data]
            if len(values) >= 2:
                # Simple trend calculation
                first_value = values[0]
                last_value = values[-1]
                
                if first_value != 0:
                    change_percent = ((last_value - first_value) / abs(first_value)) * 100
                else:
                    change_percent = 0
                
                if change_percent > 10:
                    trend = 'improving'
                elif change_percent < -10:
                    trend = 'declining'
                else:
                    trend = 'stable'
                
                trends['trend_details'][metric] = {
                    'trend': trend,
                    'change_percent': round(change_percent, 1),
                    'first_value': first_value,
                    'last_value': last_value
                }
        
        # Overall trend assessment
        improving_count = sum(1 for t in trends['trend_details'].values() if t['trend'] == 'improving')
        declining_count = sum(1 for t in trends['trend_details'].values() if t['trend'] == 'declining')
        
        if improving_count > declining_count:
            trends['overall_trend'] = 'improving'
        elif declining_count > improving_count:
            trends['overall_trend'] = 'declining'
        else:
            trends['overall_trend'] = 'stable'
        
        return trends
        
    except Exception as e:
        logger.error(f"Error analyzing financial trends: {e}")
        return {
            'trends_available': False,
            'error': str(e)
        }


def generate_financial_insights(company_data: Dict[str, Any]) -> List[str]:
    """
    Generate actionable financial insights
    
    Args:
        company_data: Company financial data
        
    Returns:
        List of insight strings
    """
    insights = []
    
    try:
        # Calculate health score and metrics
        health_score = calculate_enhanced_health_score(company_data)
        net_income = company_data.get('net_income', 0) or 0
        operating_cf = company_data.get('net_cash_from_operating_activities', 0) or 0
        current_ratio = company_data.get('current_ratio', 0) or 0
        debt_ratio = company_data.get('debt_to_equity_ratio', 0) or 0
        
        # Health score insights
        if health_score >= 80:
            insights.append("💚 Excellent financial health - company is performing well across all metrics")
        elif health_score >= 60:
            insights.append("💛 Good financial health with some areas for improvement")
        elif health_score >= 40:
            insights.append("🧡 Moderate financial health - requires attention to key areas")
        else:
            insights.append("🔴 Poor financial health - immediate action required")
        
        # Cash flow insights
        if net_income > 0 and operating_cf > 0:
            if operating_cf > net_income * 1.2:
                insights.append("💰 Strong cash conversion - generating more cash than reported profits")
            else:
                insights.append("✅ Positive profitability and cash flow")
        elif net_income > 0 and operating_cf <= 0:
            insights.append("⚠️ Profitable but poor cash flow - potential collection issues")
        elif net_income <= 0 and operating_cf > 0:
            insights.append("💵 Losses but positive cash flow - may indicate non-cash charges")
        else:
            insights.append("🚨 Both profitability and cash flow are negative - urgent attention needed")
        
        # Liquidity insights
        if current_ratio >= 2.0:
            insights.append("🏦 Strong liquidity position - good short-term financial flexibility")
        elif current_ratio >= 1.5:
            insights.append("💧 Adequate liquidity - reasonable short-term coverage")
        elif current_ratio >= 1.0:
            insights.append("⚡ Tight liquidity - monitor working capital closely")
        else:
            insights.append("🆘 Liquidity crisis - current liabilities exceed current assets")
        
        # Leverage insights
        if debt_ratio <= 0.3:
            insights.append("🛡️ Conservative debt levels - low financial risk")
        elif debt_ratio <= 0.6:
            insights.append("⚖️ Moderate debt levels - manageable leverage")
        elif debt_ratio <= 1.0:
            insights.append("📈 Higher debt levels - monitor debt service capability")
        else:
            insights.append("🔴 High debt burden - potential financial stress")
        
        # Risk insights
        liquidation_risk = company_data.get('liquidation_label', 0)
        if liquidation_risk == 1:
            insights.append("🚨 High liquidation risk detected - consider restructuring options")
        
        return insights
        
    except Exception as e:
        logger.error(f"Error generating financial insights: {e}")
        return ["❓ Unable to generate insights due to data issues"]


def create_financial_dashboard_data(company_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create comprehensive dashboard data structure
    
    Args:
        company_data: Company financial data
        
    Returns:
        Dashboard data dictionary
    """
    try:
        # Calculate key metrics
        health_score = calculate_enhanced_health_score(company_data)
        health_display = format_health_score_display(health_score)
        insights = generate_financial_insights(company_data)
        
        # Extract key financial figures
        net_income = company_data.get('net_income', 0) or 0
        operating_cf = company_data.get('net_cash_from_operating_activities', 0) or 0
        free_cf = company_data.get('free_cash_flow', 0) or 0
        total_assets = company_data.get('total_assets', 0) or 0
        current_ratio = company_data.get('current_ratio', 0) or 0
        debt_ratio = company_data.get('debt_to_equity_ratio', 0) or 0
        
        dashboard_data = {
            'company_name': company_data.get('company_name', 'Unknown Company'),
            'industry': company_data.get('industry', 'General'),
            'year': company_data.get('year', 'N/A'),
            
            # Health metrics
            'health_score': health_display,
            'insights': insights,
            
            # Financial figures
            'financial_figures': {
                'net_income': {
                    'value': net_income,
                    'formatted': format_currency(net_income),
                    'status': 'positive' if net_income > 0 else 'negative'
                },
                'operating_cash_flow': {
                    'value': operating_cf,
                    'formatted': format_currency(operating_cf),
                    'status': 'positive' if operating_cf > 0 else 'negative'
                },
                'free_cash_flow': {
                    'value': free_cf,
                    'formatted': format_currency(free_cf),
                    'status': 'positive' if free_cf > 0 else 'negative'
                },
                'total_assets': {
                    'value': total_assets,
                    'formatted': format_currency(total_assets),
                    'status': 'neutral'
                }
            },
            
            # Key ratios
            'key_ratios': {
                'current_ratio': {
                    'value': round(current_ratio, 2),
                    'status': 'good' if current_ratio >= 1.5 else 'fair' if current_ratio >= 1.0 else 'poor',
                    'benchmark': 1.5
                },
                'debt_to_equity': {
                    'value': round(debt_ratio, 2),
                    'status': 'good' if debt_ratio <= 0.5 else 'fair' if debt_ratio <= 1.0 else 'poor',
                    'benchmark': 0.5
                }
            },
            
            # Risk assessment
            'risk_assessment': {
                'liquidation_risk': company_data.get('liquidation_label', 0),
                'risk_level': health_display['risk_level'],
                'risk_color': health_display['color']
            },
            
            # Metadata
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_completeness': calculate_data_completeness(company_data)
        }
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Error creating dashboard data: {e}")
        return {
            'error': str(e),
            'company_name': company_data.get('company_name', 'Unknown'),
            'health_score': {'score': 0, 'risk_level': 'Unknown', 'grade': 'F'}
        }


def calculate_data_completeness(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate data completeness percentage
    
    Args:
        data: Data dictionary
        
    Returns:
        Completeness information
    """
    required_fields = [
        'net_income', 'net_cash_from_operating_activities', 'current_ratio',
        'debt_to_equity_ratio', 'total_assets', 'company_name'
    ]
    
    present_count = 0
    for field in required_fields:
        value = data.get(field)
        if value is not None and value != '' and not pd.isna(value):
            present_count += 1
    
    completeness_percent = (present_count / len(required_fields)) * 100
    
    return {
        'percentage': round(completeness_percent, 1),
        'present_fields': present_count,
        'total_fields': len(required_fields),
        'status': 'complete' if completeness_percent >= 80 else 'partial' if completeness_percent >= 60 else 'incomplete'
    }