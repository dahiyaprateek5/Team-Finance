"""
services/cash_flow_service.py
============================

Cash Flow Service - Complete Implementation
==========================================

This service handles cash flow statement processing, analysis, and forecasting.
It can generate cash flow statements from balance sheet data, validate existing
statements, and provide cash flow analysis using database data.

Author: Prateek Dahiya
Project: Financial Risk Assessment Model for Small Companies and Startups
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CashFlowAnalysis:
    """Analysis results for cash flow data."""
    company_id: str
    analysis_date: datetime
    operating_cash_flow: float
    investing_cash_flow: float
    financing_cash_flow: float
    net_cash_change: float
    free_cash_flow: float
    cash_flow_ratios: Dict[str, float]
    quality_metrics: Dict[str, float]
    trends: Dict[str, Any]
    forecasts: Dict[str, Any]
    risk_indicators: List[str]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['analysis_date'] = self.analysis_date.isoformat()
        return data

class CashFlowService:
    """
    Service for processing and analyzing cash flow statements.
    Handles cash flow generation, validation, analysis, and forecasting using database data.
    """
    
    def __init__(self, 
                 forecasting_enabled: bool = True,
                 historical_periods: int = 3,
                 validation_enabled: bool = True):
        """
        Initialize Cash Flow Service.
        
        Args:
            forecasting_enabled (bool): Enable cash flow forecasting
            historical_periods (int): Number of historical periods for analysis
            validation_enabled (bool): Enable data validation
        """
        self.forecasting_enabled = forecasting_enabled
        self.historical_periods = historical_periods
        self.validation_enabled = validation_enabled
        
        # Database connection (to be injected)
        self.db_connection = None
        
        # Cash flow statement structure
        self.cash_flow_structure = {
            'operating_activities': [
                'net_income',
                'depreciation_amortization',
                'stock_based_compensation',
                'accounts_receivable_change',
                'inventory_change',
                'accounts_payable_change',
                'accrued_liabilities_change',
                'other_operating_changes'
            ],
            'investing_activities': [
                'capital_expenditures',
                'acquisitions',
                'asset_sales',
                'investment_purchases',
                'investment_sales',
                'other_investing_changes'
            ],
            'financing_activities': [
                'debt_issued',
                'debt_repaid',
                'equity_issued',
                'equity_repurchased',
                'dividends_paid',
                'other_financing_changes'
            ]
        }
        
        # Quality metrics thresholds
        self.quality_thresholds = {
            'ocf_to_net_income_min': 0.8,
            'ocf_to_net_income_max': 1.5,
            'free_cash_flow_margin_min': 0.05,
            'cash_conversion_cycle_max': 90
        }
        
        logger.info("Cash Flow Service initialized")
    
    def set_database_connection(self, db_connection):
        """Set database connection for data fetching."""
        self.db_connection = db_connection
        logger.info("Database connection set for Cash Flow Service")
    
    def fetch_cash_flow_data(self, 
                            company_id: str, 
                            year: int = None,
                            periods: int = None) -> pd.DataFrame:
        """
        Fetch cash flow data from database.
        
        Args:
            company_id (str): Company identifier
            year (int): Specific year for data retrieval
            periods (int): Number of periods to fetch
            
        Returns:
            pd.DataFrame: Cash flow data from database
        """
        try:
            if not self.db_connection:
                raise ValueError("Database connection not set")
            
            # Build query
            query = """
                SELECT * FROM cash_flow_data 
                WHERE company_id = %s
            """
            params = [company_id]
            
            if year:
                query += " AND EXTRACT(YEAR FROM date) = %s"
                params.append(year)
            
            query += " ORDER BY date DESC"
            
            if periods:
                query += f" LIMIT {periods}"
            
            # Execute query
            df = pd.read_sql_query(query, self.db_connection, params=params)
            
            if df.empty:
                logger.warning(f"No cash flow data found for company {company_id}")
                return pd.DataFrame()
            
            logger.info(f"Fetched {len(df)} cash flow records for company {company_id}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching cash flow data: {e}")
            raise
    
    def fetch_balance_sheet_data(self, company_id: str, periods: int = 2) -> pd.DataFrame:
        """
        Fetch balance sheet data for cash flow generation.
        
        Args:
            company_id (str): Company identifier
            periods (int): Number of periods to fetch
            
        Returns:
            pd.DataFrame: Balance sheet data from database
        """
        try:
            if not self.db_connection:
                raise ValueError("Database connection not set")
            
            query = """
                SELECT * FROM balance_sheet_data 
                WHERE company_id = %s 
                ORDER BY date DESC 
                LIMIT %s
            """
            
            df = pd.read_sql_query(query, self.db_connection, params=[company_id, periods])
            
            if df.empty:
                logger.warning(f"No balance sheet data found for company {company_id}")
                return pd.DataFrame()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching balance sheet data: {e}")
            raise
    
    def fetch_income_statement_data(self, company_id: str, periods: int = 1) -> pd.DataFrame:
        """
        Fetch income statement data for cash flow generation.
        
        Args:
            company_id (str): Company identifier
            periods (int): Number of periods to fetch
            
        Returns:
            pd.DataFrame: Income statement data from database
        """
        try:
            if not self.db_connection:
                raise ValueError("Database connection not set")
            
            query = """
                SELECT * FROM income_statement_data 
                WHERE company_id = %s 
                ORDER BY date DESC 
                LIMIT %s
            """
            
            df = pd.read_sql_query(query, self.db_connection, params=[company_id, periods])
            
            if df.empty:
                logger.warning(f"No income statement data found for company {company_id}")
                return pd.DataFrame()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching income statement data: {e}")
            raise
    
    def generate_cash_flow_from_balance_sheet(self, company_id: str) -> Dict[str, Any]:
        """
        Generate cash flow statement from balance sheet data using indirect method.
        
        Args:
            company_id (str): Company identifier
            
        Returns:
            Dict[str, Any]: Generated cash flow statement
        """
        try:
            logger.info(f"Generating cash flow statement for company {company_id}")
            
            # Fetch required data from database
            balance_sheet_data = self.fetch_balance_sheet_data(company_id, periods=2)
            income_statement_data = self.fetch_income_statement_data(company_id, periods=1)
            
            if balance_sheet_data.empty or len(balance_sheet_data) < 2:
                raise ValueError("Insufficient balance sheet data for cash flow generation")
            
            if income_statement_data.empty:
                raise ValueError("Income statement data required for cash flow generation")
            
            # Sort by date (newest first)
            balance_sheet_data = balance_sheet_data.sort_values('date', ascending=False)
            current_period = balance_sheet_data.iloc[0]
            previous_period = balance_sheet_data.iloc[1]
            income_data = income_statement_data.iloc[0]
            
            # Generate cash flow statement using indirect method
            cash_flow_statement = self._calculate_cash_flow_indirect_method(
                current_period, previous_period, income_data
            )
            
            # Validate generated statement
            if self.validation_enabled:
                validation_results = self._validate_cash_flow_statement(cash_flow_statement)
                cash_flow_statement['validation_results'] = validation_results
            
            # Save to database
            success = self._save_cash_flow_to_database(cash_flow_statement, company_id)
            
            result = {
                'success': success,
                'company_id': company_id,
                'generation_method': 'indirect_method',
                'generation_timestamp': datetime.now().isoformat(),
                'cash_flow_statement': cash_flow_statement,
                'data_sources': {
                    'balance_sheet_periods': len(balance_sheet_data),
                    'income_statement_available': True
                }
            }
            
            logger.info(f"Cash flow statement generated successfully for {company_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating cash flow statement: {e}")
            raise
    
    def _calculate_cash_flow_indirect_method(self, 
                                           current_bs: pd.Series, 
                                           previous_bs: pd.Series, 
                                           income_data: pd.Series) -> Dict[str, Any]:
        """Calculate cash flow using indirect method from database data."""
        try:
            cash_flow = {
                'date': current_bs.get('date'),
                'period': current_bs.get('period', 'Annual'),
                'operating_activities': {},
                'investing_activities': {},
                'financing_activities': {},
                'net_cash_change': 0,
                'cash_beginning': 0,
                'cash_ending': 0
            }
            
            # Operating Activities (Indirect Method)
            net_income = float(income_data.get('net_income', 0))
            cash_flow['operating_activities']['net_income'] = net_income
            
            # Add back non-cash expenses
            depreciation = float(income_data.get('depreciation_amortization', 0))
            stock_compensation = float(income_data.get('stock_based_compensation', 0))
            
            cash_flow['operating_activities']['depreciation_amortization'] = depreciation
            cash_flow['operating_activities']['stock_based_compensation'] = stock_compensation
            
            # Working capital changes
            ar_change = self._calculate_change(current_bs, previous_bs, 'accounts_receivable')
            inventory_change = self._calculate_change(current_bs, previous_bs, 'inventory')
            ap_change = self._calculate_change(current_bs, previous_bs, 'accounts_payable')
            accrued_change = self._calculate_change(current_bs, previous_bs, 'accrued_liabilities')
            
            cash_flow['operating_activities']['accounts_receivable_change'] = -ar_change  # Negative because increase in AR reduces cash
            cash_flow['operating_activities']['inventory_change'] = -inventory_change
            cash_flow['operating_activities']['accounts_payable_change'] = ap_change  # Positive because increase in AP increases cash
            cash_flow['operating_activities']['accrued_liabilities_change'] = accrued_change
            
            # Calculate total operating cash flow
            operating_cash_flow = (
                net_income + depreciation + stock_compensation - 
                ar_change - inventory_change + ap_change + accrued_change
            )
            cash_flow['operating_activities']['total_operating_cash_flow'] = operating_cash_flow
            
            # Investing Activities
            ppe_change = self._calculate_change(current_bs, previous_bs, 'property_plant_equipment')
            capex = ppe_change + depreciation  # Approximate CapEx
            
            cash_flow['investing_activities']['capital_expenditures'] = -capex if capex > 0 else 0
            
            # Investment changes
            investment_change = self._calculate_change(current_bs, previous_bs, 'long_term_investments')
            cash_flow['investing_activities']['investment_changes'] = -investment_change
            
            total_investing_cash_flow = -capex - investment_change
            cash_flow['investing_activities']['total_investing_cash_flow'] = total_investing_cash_flow
            
            # Financing Activities
            debt_change = (
                self._calculate_change(current_bs, previous_bs, 'short_term_debt') +
                self._calculate_change(current_bs, previous_bs, 'long_term_debt')
            )
            equity_change = self._calculate_change(current_bs, previous_bs, 'total_equity')
            
            # Estimate dividends paid from retained earnings change
            retained_earnings_change = self._calculate_change(current_bs, previous_bs, 'retained_earnings')
            dividends_paid = net_income - retained_earnings_change if retained_earnings_change < net_income else 0
            
            cash_flow['financing_activities']['debt_changes'] = debt_change
            cash_flow['financing_activities']['equity_changes'] = equity_change - net_income + dividends_paid  # Adjust for earnings
            cash_flow['financing_activities']['dividends_paid'] = -dividends_paid
            
            total_financing_cash_flow = debt_change + (equity_change - net_income + dividends_paid) - dividends_paid
            cash_flow['financing_activities']['total_financing_cash_flow'] = total_financing_cash_flow
            
            # Net cash change
            net_cash_change = operating_cash_flow + total_investing_cash_flow + total_financing_cash_flow
            cash_flow['net_cash_change'] = net_cash_change
            
            # Cash balances
            cash_beginning = float(previous_bs.get('cash_and_equivalents', 0))
            cash_ending = float(current_bs.get('cash_and_equivalents', 0))
            
            cash_flow['cash_beginning'] = cash_beginning
            cash_flow['cash_ending'] = cash_ending
            
            # Verify cash reconciliation
            calculated_ending_cash = cash_beginning + net_cash_change
            cash_flow['cash_reconciliation_difference'] = cash_ending - calculated_ending_cash
            
            return cash_flow
            
        except Exception as e:
            logger.error(f"Error in indirect method calculation: {e}")
            raise
    
    def _calculate_change(self, current: pd.Series, previous: pd.Series, field: str) -> float:
        """Calculate change in a field between two periods."""
        try:
            current_value = float(current.get(field, 0))
            previous_value = float(previous.get(field, 0))
            return current_value - previous_value
        except:
            return 0.0
    
    def analyze_cash_flow(self, company_id: str, period: str = None) -> CashFlowAnalysis:
        """
        Perform comprehensive cash flow analysis using database data.
        
        Args:
            company_id (str): Company identifier
            period (str): Analysis period
            
        Returns:
            CashFlowAnalysis: Analysis results
        """
        try:
            # Fetch cash flow data from database
            cash_flow_data = self.fetch_cash_flow_data(company_id, periods=self.historical_periods)
            
            if cash_flow_data.empty:
                raise ValueError(f"No cash flow data available for company {company_id}")
            
            # Use latest record for analysis
            latest_record = cash_flow_data.iloc[0]
            
            # Extract key cash flow components
            operating_cf = float(latest_record.get('total_operating_cash_flow', 0))
            investing_cf = float(latest_record.get('total_investing_cash_flow', 0))
            financing_cf = float(latest_record.get('total_financing_cash_flow', 0))
            net_cash_change = float(latest_record.get('net_cash_change', 0))
            
            # Calculate free cash flow
            capex = abs(float(latest_record.get('capital_expenditures', 0)))
            free_cash_flow = operating_cf - capex
            
            # Calculate cash flow ratios
            cash_flow_ratios = self._calculate_cash_flow_ratios(latest_record, company_id)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(latest_record, company_id)
            
            # Analyze trends if multiple periods available
            trends = self._analyze_cash_flow_trends(cash_flow_data) if len(cash_flow_data) > 1 else {}
            
            # Generate forecasts if enabled
            forecasts = {}
            if self.forecasting_enabled and len(cash_flow_data) >= 3:
                forecasts = self._generate_cash_flow_forecasts(cash_flow_data)
            
            # Identify risk indicators
            risk_indicators = self._identify_cash_flow_risks(latest_record, trends, quality_metrics)
            
            # Generate recommendations
            recommendations = self._generate_cash_flow_recommendations(
                latest_record, cash_flow_ratios, quality_metrics, trends
            )
            
            analysis = CashFlowAnalysis(
                company_id=company_id,
                analysis_date=datetime.now(),
                operating_cash_flow=operating_cf,
                investing_cash_flow=investing_cf,
                financing_cash_flow=financing_cf,
                net_cash_change=net_cash_change,
                free_cash_flow=free_cash_flow,
                cash_flow_ratios=cash_flow_ratios,
                quality_metrics=quality_metrics,
                trends=trends,
                forecasts=forecasts,
                risk_indicators=risk_indicators,
                recommendations=recommendations
            )
            
            logger.info(f"Cash flow analysis completed for {company_id}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in cash flow analysis: {e}")
            raise
    
    def _calculate_cash_flow_ratios(self, record: pd.Series, company_id: str) -> Dict[str, float]:
        """Calculate cash flow ratios using database data."""
        try:
            ratios = {}
            
            # Fetch additional data needed for ratios
            income_data = self.fetch_income_statement_data(company_id, periods=1)
            balance_sheet_data = self.fetch_balance_sheet_data(company_id, periods=1)
            
            operating_cf = float(record.get('total_operating_cash_flow', 0))
            
            # Operating cash flow ratios
            if not income_data.empty:
                net_income = float(income_data.iloc[0].get('net_income', 1))
                revenue = float(income_data.iloc[0].get('revenue', 1))
                
                if net_income != 0:
                    ratios['ocf_to_net_income'] = operating_cf / net_income
                
                if revenue != 0:
                    ratios['operating_cash_margin'] = operating_cf / revenue
            
            # Free cash flow ratios
            capex = abs(float(record.get('capital_expenditures', 0)))
            free_cash_flow = operating_cf - capex
            ratios['free_cash_flow'] = free_cash_flow
            
            if not income_data.empty:
                revenue = float(income_data.iloc[0].get('revenue', 1))
                if revenue != 0:
                    ratios['free_cash_flow_margin'] = free_cash_flow / revenue
            
            # Cash coverage ratios
            if not balance_sheet_data.empty:
                current_liabilities = float(balance_sheet_data.iloc[0].get('current_liabilities', 1))
                total_debt = float(balance_sheet_data.iloc[0].get('total_debt', 1))
                
                if current_liabilities != 0:
                    ratios['cash_coverage_ratio'] = operating_cf / current_liabilities
                
                if total_debt != 0:
                    ratios['cash_debt_coverage'] = operating_cf / total_debt
            
            return ratios
            
        except Exception as e:
            logger.warning(f"Error calculating cash flow ratios: {e}")
            return {}
    
    def _calculate_quality_metrics(self, record: pd.Series, company_id: str) -> Dict[str, float]:
        """Calculate cash flow quality metrics."""
        try:
            quality_metrics = {}
            
            # Fetch additional data
            income_data = self.fetch_income_statement_data(company_id, periods=1)
            
            operating_cf = float(record.get('total_operating_cash_flow', 0))
            
            if not income_data.empty:
                net_income = float(income_data.iloc[0].get('net_income', 1))
                revenue = float(income_data.iloc[0].get('revenue', 1))
                
                # Quality of earnings
                if net_income != 0:
                    quality_metrics['earnings_quality'] = operating_cf / net_income
                
                # Cash conversion efficiency
                if revenue != 0:
                    quality_metrics['cash_conversion_efficiency'] = operating_cf / revenue
            
            # Working capital efficiency
            ar_change = float(record.get('accounts_receivable_change', 0))
            inventory_change = float(record.get('inventory_change', 0))
            ap_change = float(record.get('accounts_payable_change', 0))
            
            working_capital_change = ar_change + inventory_change - ap_change
            quality_metrics['working_capital_efficiency'] = -working_capital_change / operating_cf if operating_cf != 0 else 0
            
            # Cash flow stability (would require multiple periods)
            quality_metrics['cash_flow_volatility'] = 0  # Placeholder - would calculate from historical data
            
            return quality_metrics
            
        except Exception as e:
            logger.warning(f"Error calculating quality metrics: {e}")
            return {}
    
    def _analyze_cash_flow_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze cash flow trends over time using database data."""
        try:
            if len(data) < 2:
                return {}
            
            # Sort by date
            data_sorted = data.sort_values('date')
            
            trends = {}
            metrics_to_analyze = [
                'total_operating_cash_flow', 
                'total_investing_cash_flow', 
                'total_financing_cash_flow',
                'net_cash_change'
            ]
            
            for metric in metrics_to_analyze:
                if metric in data_sorted.columns:
                    values = data_sorted[metric].dropna()
                    if len(values) >= 2:
                        # Calculate trend
                        pct_change = ((values.iloc[-1] - values.iloc[0]) / abs(values.iloc[0])) * 100 if values.iloc[0] != 0 else 0
                        
                        # Calculate volatility
                        volatility = values.std() / abs(values.mean()) if values.mean() != 0 else 0
                        
                        trends[metric] = {
                            'total_change_pct': float(pct_change),
                            'volatility': float(volatility),
                            'trend_direction': 'improving' if pct_change > 5 else 'declining' if pct_change < -5 else 'stable',
                            'latest_value': float(values.iloc[-1]),
                            'average_value': float(values.mean())
                        }
            
            return trends
            
        except Exception as e:
            logger.warning(f"Error analyzing cash flow trends: {e}")
            return {}
    
    def _generate_cash_flow_forecasts(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate cash flow forecasts using database data."""
        try:
            forecasts = {}
            
            # Simple trend-based forecasting
            data_sorted = data.sort_values('date')
            
            # Operating cash flow forecast
            ocf_values = data_sorted['total_operating_cash_flow'].dropna()
            if len(ocf_values) >= 3:
                # Simple linear trend
                trend = np.polyfit(range(len(ocf_values)), ocf_values, 1)
                next_period_forecast = trend[0] * len(ocf_values) + trend[1]
                
                forecasts['operating_cash_flow'] = {
                    'next_period': float(next_period_forecast),
                    'confidence': 0.7,  # Medium confidence for simple model
                    'method': 'linear_trend'
                }
            
            # Free cash flow forecast
            if 'capital_expenditures' in data_sorted.columns:
                capex_avg = data_sorted['capital_expenditures'].mean()
                if 'operating_cash_flow' in forecasts:
                    forecasted_ocf = forecasts['operating_cash_flow']['next_period']
                    forecasted_fcf = forecasted_ocf - abs(capex_avg)
                    
                    forecasts['free_cash_flow'] = {
                        'next_period': float(forecasted_fcf),
                        'confidence': 0.6,
                        'method': 'derived_from_ocf'
                    }
            
            return forecasts
            
        except Exception as e:
            logger.warning(f"Error generating cash flow forecasts: {e}")
            return {}
    
    def _identify_cash_flow_risks(self, 
                                 record: pd.Series, 
                                 trends: Dict[str, Any], 
                                 quality_metrics: Dict[str, float]) -> List[str]:
        """Identify cash flow risk indicators."""
        risks = []
        
        try:
            # Operating cash flow risks
            operating_cf = float(record.get('total_operating_cash_flow', 0))
            if operating_cf < 0:
                risks.append("🚨 Negative operating cash flow indicates operational difficulties")
            
            # Free cash flow risks
            capex = abs(float(record.get('capital_expenditures', 0)))
            free_cf = operating_cf - capex
            if free_cf < 0:
                risks.append("⚠️ Negative free cash flow suggests cash burn from operations and investments")
            
            # Quality risks
            earnings_quality = quality_metrics.get('earnings_quality', 1.0)
            if earnings_quality < self.quality_thresholds['ocf_to_net_income_min']:
                risks.append("📉 Low earnings quality - operating cash flow significantly below net income")
            
            # Trend risks
            ocf_trend = trends.get('total_operating_cash_flow', {})
            if ocf_trend.get('trend_direction') == 'declining':
                risks.append("📊 Declining operating cash flow trend")
            
            # High volatility risks
            if ocf_trend.get('volatility', 0) > 0.5:
                risks.append("📈 High cash flow volatility indicates unpredictable operations")
            
            # Working capital risks
            working_capital_efficiency = quality_metrics.get('working_capital_efficiency', 0)
            if working_capital_efficiency < -0.2:
                risks.append("💧 Poor working capital management affecting cash flow")
            
            return risks[:6]  # Return top 6 risks
            
        except Exception as e:
            logger.warning(f"Error identifying cash flow risks: {e}")
            return ["Unable to assess cash flow risks"]
    
    def _generate_cash_flow_recommendations(self, 
                                           record: pd.Series,
                                           ratios: Dict[str, float],
                                           quality_metrics: Dict[str, float],
                                           trends: Dict[str, Any]) -> List[str]:
        """Generate cash flow improvement recommendations."""
        recommendations = []
        
        try:
            # Operating cash flow recommendations
            operating_cf = float(record.get('total_operating_cash_flow', 0))
            if operating_cf < 0:
                recommendations.extend([
                    "🚨 URGENT: Focus on improving operational efficiency and cost management",
                    "💰 Accelerate accounts receivable collection processes",
                    "📦 Optimize inventory management to free up working capital"
                ])
            
            # Free cash flow recommendations
            capex = abs(float(record.get('capital_expenditures', 0)))
            free_cf = operating_cf - capex
            if free_cf < 0 and operating_cf > 0:
                recommendations.append("🏗️ Review capital expenditure priorities and timing")
            
            # Quality-based recommendations
            earnings_quality = quality_metrics.get('earnings_quality', 1.0)
            if earnings_quality < 0.8:
                recommendations.extend([
                    "📊 Investigate discrepancies between earnings and cash flow",
                    "🔍 Review accrual accounting practices and working capital management"
                ])
            
            working_capital_efficiency = quality_metrics.get('working_capital_efficiency', 0)
            if working_capital_efficiency < -0.1:
                recommendations.extend([
                    "⚡ Implement working capital optimization program",
                    "📋 Establish better payment terms with customers and suppliers"
                ])
            
            # Trend-based recommendations
            ocf_trend = trends.get('total_operating_cash_flow', {})
            if ocf_trend.get('trend_direction') == 'declining':
                recommendations.extend([
                    "📈 Develop action plan to reverse declining cash flow trend",
                    "🎯 Focus on core business profitability and operational efficiency"
                ])
            
            # Ratio-based recommendations
            cash_coverage = ratios.get('cash_coverage_ratio', 1.0)
            if cash_coverage < 1.0:
                recommendations.append("🛡️ Improve cash coverage of current liabilities")
            
            # General cash flow management recommendations
            recommendations.extend([
                "📊 Implement monthly cash flow forecasting and monitoring",
                "💡 Consider factoring or invoice financing for immediate cash needs",
                "🤝 Negotiate better payment terms with suppliers and customers",
                "📈 Focus on high-margin, cash-generating business segments"
            ])
            
            return recommendations[:10]  # Return top 10 recommendations
            
        except Exception as e:
            logger.warning(f"Error generating cash flow recommendations: {e}")
            return ["Consult with financial advisor for detailed cash flow improvement strategies"]
    
    def _validate_cash_flow_statement(self, cash_flow: Dict[str, Any]) -> Dict[str, Any]:
        """Validate cash flow statement integrity."""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'validation_timestamp': datetime.now().isoformat()
        }
        
        try:
            # Check cash reconciliation
            calculated_ending = cash_flow.get('cash_beginning', 0) + cash_flow.get('net_cash_change', 0)
            actual_ending = cash_flow.get('cash_ending', 0)
            reconciliation_diff = abs(calculated_ending - actual_ending)
            
            if reconciliation_diff > 1000:  # Threshold for material difference
                validation_results['errors'].append(
                    f"Cash reconciliation error: Calculated ending cash ({calculated_ending:,.2f}) "
                    f"differs from actual ({actual_ending:,.2f}) by {reconciliation_diff:,.2f}"
                )
                validation_results['is_valid'] = False
            
            # Validate operating activities
            operating_activities = cash_flow.get('operating_activities', {})
            net_income = operating_activities.get('net_income', 0)
            total_ocf = operating_activities.get('total_operating_cash_flow', 0)
            
            # Check for extreme OCF to Net Income ratios
            if net_income != 0:
                ocf_ni_ratio = total_ocf / net_income
                if ocf_ni_ratio < -5 or ocf_ni_ratio > 10:
                    validation_results['warnings'].append(
                        f"Unusual OCF to Net Income ratio: {ocf_ni_ratio:.2f}. "
                        "This may indicate data quality issues or exceptional circumstances."
                    )
            
            # Validate investing activities
            investing_activities = cash_flow.get('investing_activities', {})
            capex = investing_activities.get('capital_expenditures', 0)
            
            # CapEx should typically be negative (cash outflow)
            if capex > 0:
                validation_results['warnings'].append(
                    "Capital expenditures shows positive value - typically should be negative (cash outflow)"
                )
            
            # Validate financing activities
            financing_activities = cash_flow.get('financing_activities', {})
            
            # Check for missing required fields
            required_fields = ['operating_activities', 'investing_activities', 'financing_activities']
            for field in required_fields:
                if field not in cash_flow or not cash_flow[field]:
                    validation_results['errors'].append(f"Missing required section: {field}")
                    validation_results['is_valid'] = False
            
            # Validate date format
            if 'date' not in cash_flow:
                validation_results['errors'].append("Missing date field in cash flow statement")
                validation_results['is_valid'] = False
            
            # Check for reasonable values
            net_cash_change = cash_flow.get('net_cash_change', 0)
            if abs(net_cash_change) > 1e12:  # $1 trillion threshold
                validation_results['warnings'].append(
                    f"Extremely large net cash change: {net_cash_change:,.2f}. Please verify data accuracy."
                )
            
            validation_results['validation_score'] = (
                100 - (len(validation_results['errors']) * 25) - (len(validation_results['warnings']) * 5)
            )
            validation_results['validation_score'] = max(0, validation_results['validation_score'])
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating cash flow statement: {e}")
            return {
                'is_valid': False,
                'errors': [f"Validation process failed: {str(e)}"],
                'warnings': [],
                'validation_timestamp': datetime.now().isoformat(),
                'validation_score': 0
            }
    
    def _save_cash_flow_to_database(self, cash_flow_statement: Dict[str, Any], company_id: str) -> bool:
        """Save generated cash flow statement to database."""
        try:
            if not self.db_connection:
                logger.warning("No database connection available for saving cash flow data")
                return False
            
            # Prepare data for database insertion
            cash_flow_data = {
                'company_id': company_id,
                'date': cash_flow_statement.get('date'),
                'period': cash_flow_statement.get('period', 'Annual'),
                'generation_method': 'indirect_method',
                'generation_timestamp': datetime.now(),
                
                # Operating activities
                'net_income': cash_flow_statement.get('operating_activities', {}).get('net_income', 0),
                'depreciation_amortization': cash_flow_statement.get('operating_activities', {}).get('depreciation_amortization', 0),
                'stock_based_compensation': cash_flow_statement.get('operating_activities', {}).get('stock_based_compensation', 0),
                'accounts_receivable_change': cash_flow_statement.get('operating_activities', {}).get('accounts_receivable_change', 0),
                'inventory_change': cash_flow_statement.get('operating_activities', {}).get('inventory_change', 0),
                'accounts_payable_change': cash_flow_statement.get('operating_activities', {}).get('accounts_payable_change', 0),
                'accrued_liabilities_change': cash_flow_statement.get('operating_activities', {}).get('accrued_liabilities_change', 0),
                'total_operating_cash_flow': cash_flow_statement.get('operating_activities', {}).get('total_operating_cash_flow', 0),
                
                # Investing activities
                'capital_expenditures': cash_flow_statement.get('investing_activities', {}).get('capital_expenditures', 0),
                'investment_changes': cash_flow_statement.get('investing_activities', {}).get('investment_changes', 0),
                'total_investing_cash_flow': cash_flow_statement.get('investing_activities', {}).get('total_investing_cash_flow', 0),
                
                # Financing activities
                'debt_changes': cash_flow_statement.get('financing_activities', {}).get('debt_changes', 0),
                'equity_changes': cash_flow_statement.get('financing_activities', {}).get('equity_changes', 0),
                'dividends_paid': cash_flow_statement.get('financing_activities', {}).get('dividends_paid', 0),
                'total_financing_cash_flow': cash_flow_statement.get('financing_activities', {}).get('total_financing_cash_flow', 0),
                
                # Summary
                'net_cash_change': cash_flow_statement.get('net_cash_change', 0),
                'cash_beginning': cash_flow_statement.get('cash_beginning', 0),
                'cash_ending': cash_flow_statement.get('cash_ending', 0),
                'cash_reconciliation_difference': cash_flow_statement.get('cash_reconciliation_difference', 0),
                
                # Validation results
                'validation_results': str(cash_flow_statement.get('validation_results', {}))
            }
            
            # Insert into database
            columns = ', '.join(cash_flow_data.keys())
            placeholders = ', '.join(['%s'] * len(cash_flow_data))
            
            insert_query = f"""
                INSERT INTO cash_flow_data ({columns})
                VALUES ({placeholders})
                ON CONFLICT (company_id, date) DO UPDATE SET
                    generation_timestamp = EXCLUDED.generation_timestamp,
                    total_operating_cash_flow = EXCLUDED.total_operating_cash_flow,
                    total_investing_cash_flow = EXCLUDED.total_investing_cash_flow,
                    total_financing_cash_flow = EXCLUDED.total_financing_cash_flow,
                    net_cash_change = EXCLUDED.net_cash_change,
                    validation_results = EXCLUDED.validation_results
            """
            
            cursor = self.db_connection.cursor()
            cursor.execute(insert_query, list(cash_flow_data.values()))
            self.db_connection.commit()
            cursor.close()
            
            logger.info(f"Cash flow statement saved to database for company {company_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving cash flow statement to database: {e}")
            if self.db_connection:
                self.db_connection.rollback()
            return False
    
    def export_cash_flow_statement(self, company_id: str, format_type: str = 'json') -> Dict[str, Any]:
        """
        Export cash flow statement in specified format.
        
        Args:
            company_id (str): Company identifier
            format_type (str): Export format ('json', 'excel', 'csv')
            
        Returns:
            Dict[str, Any]: Export result with file path or data
        """
        try:
            # Fetch latest cash flow data
            cash_flow_data = self.fetch_cash_flow_data(company_id, periods=1)
            
            if cash_flow_data.empty:
                raise ValueError(f"No cash flow data found for company {company_id}")
            
            record = cash_flow_data.iloc[0]
            
            # Prepare export data
            export_data = {
                'company_id': company_id,
                'statement_date': record.get('date'),
                'export_timestamp': datetime.now().isoformat(),
                'cash_flow_statement': {
                    'operating_activities': {
                        'net_income': float(record.get('net_income', 0)),
                        'depreciation_amortization': float(record.get('depreciation_amortization', 0)),
                        'stock_based_compensation': float(record.get('stock_based_compensation', 0)),
                        'accounts_receivable_change': float(record.get('accounts_receivable_change', 0)),
                        'inventory_change': float(record.get('inventory_change', 0)),
                        'accounts_payable_change': float(record.get('accounts_payable_change', 0)),
                        'accrued_liabilities_change': float(record.get('accrued_liabilities_change', 0)),
                        'total_operating_cash_flow': float(record.get('total_operating_cash_flow', 0))
                    },
                    'investing_activities': {
                        'capital_expenditures': float(record.get('capital_expenditures', 0)),
                        'investment_changes': float(record.get('investment_changes', 0)),
                        'total_investing_cash_flow': float(record.get('total_investing_cash_flow', 0))
                    },
                    'financing_activities': {
                        'debt_changes': float(record.get('debt_changes', 0)),
                        'equity_changes': float(record.get('equity_changes', 0)),
                        'dividends_paid': float(record.get('dividends_paid', 0)),
                        'total_financing_cash_flow': float(record.get('total_financing_cash_flow', 0))
                    },
                    'summary': {
                        'net_cash_change': float(record.get('net_cash_change', 0)),
                        'cash_beginning': float(record.get('cash_beginning', 0)),
                        'cash_ending': float(record.get('cash_ending', 0)),
                        'free_cash_flow': float(record.get('total_operating_cash_flow', 0)) - abs(float(record.get('capital_expenditures', 0)))
                    }
                }
            }
            
            if format_type.lower() == 'json':
                return {
                    'success': True,
                    'format': 'json',
                    'data': export_data
                }
            
            elif format_type.lower() == 'excel':
                return self._export_to_excel(export_data, company_id)
            
            elif format_type.lower() == 'csv':
                return self._export_to_csv(export_data, company_id)
            
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
                
        except Exception as e:
            logger.error(f"Error exporting cash flow statement: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _export_to_excel(self, export_data: Dict[str, Any], company_id: str) -> Dict[str, Any]:
        """Export cash flow statement to Excel format."""
        try:
            import pandas as pd
            from datetime import datetime
            
            # Create filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"cash_flow_statement_{company_id}_{timestamp}.xlsx"
            
            # Prepare data for Excel
            cf_statement = export_data['cash_flow_statement']
            
            # Operating Activities
            operating_data = []
            for key, value in cf_statement['operating_activities'].items():
                operating_data.append({
                    'Item': key.replace('_', ' ').title(),
                    'Amount': value
                })
            
            # Investing Activities
            investing_data = []
            for key, value in cf_statement['investing_activities'].items():
                investing_data.append({
                    'Item': key.replace('_', ' ').title(),
                    'Amount': value
                })
            
            # Financing Activities
            financing_data = []
            for key, value in cf_statement['financing_activities'].items():
                financing_data.append({
                    'Item': key.replace('_', ' ').title(),
                    'Amount': value
                })
            
            # Summary
            summary_data = []
            for key, value in cf_statement['summary'].items():
                summary_data.append({
                    'Item': key.replace('_', ' ').title(),
                    'Amount': value
                })
            
            # Create Excel file with multiple sheets
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                pd.DataFrame(operating_data).to_excel(writer, sheet_name='Operating Activities', index=False)
                pd.DataFrame(investing_data).to_excel(writer, sheet_name='Investing Activities', index=False)
                pd.DataFrame(financing_data).to_excel(writer, sheet_name='Financing Activities', index=False)
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            return {
                'success': True,
                'format': 'excel',
                'filename': filename,
                'file_path': os.path.abspath(filename)
            }
            
        except Exception as e:
            logger.error(f"Error exporting to Excel: {e}")
            return {
                'success': False,
                'error': f"Excel export failed: {str(e)}"
            }
    
    def _export_to_csv(self, export_data: Dict[str, Any], company_id: str) -> Dict[str, Any]:
        """Export cash flow statement to CSV format."""
        try:
            import pandas as pd
            from datetime import datetime
            
            # Create filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"cash_flow_statement_{company_id}_{timestamp}.csv"
            
            # Flatten the cash flow statement data
            cf_statement = export_data['cash_flow_statement']
            
            csv_data = []
            
            # Add header information
            csv_data.append({
                'Section': 'Header',
                'Item': 'Company ID',
                'Amount': export_data['company_id']
            })
            csv_data.append({
                'Section': 'Header',
                'Item': 'Statement Date',
                'Amount': export_data['statement_date']
            })
            csv_data.append({
                'Section': 'Header',
                'Item': 'Export Timestamp',
                'Amount': export_data['export_timestamp']
            })
            
            # Operating Activities
            for key, value in cf_statement['operating_activities'].items():
                csv_data.append({
                    'Section': 'Operating Activities',
                    'Item': key.replace('_', ' ').title(),
                    'Amount': value
                })
            
            # Investing Activities
            for key, value in cf_statement['investing_activities'].items():
                csv_data.append({
                    'Section': 'Investing Activities',
                    'Item': key.replace('_', ' ').title(),
                    'Amount': value
                })
            
            # Financing Activities
            for key, value in cf_statement['financing_activities'].items():
                csv_data.append({
                    'Section': 'Financing Activities',
                    'Item': key.replace('_', ' ').title(),
                    'Amount': value
                })
            
            # Summary
            for key, value in cf_statement['summary'].items():
                csv_data.append({
                    'Section': 'Summary',
                    'Item': key.replace('_', ' ').title(),
                    'Amount': value
                })
            
            # Create CSV file
            df = pd.DataFrame(csv_data)
            df.to_csv(filename, index=False)
            
            return {
                'success': True,
                'format': 'csv',
                'filename': filename,
                'file_path': os.path.abspath(filename)
            }
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return {
                'success': False,
                'error': f"CSV export failed: {str(e)}"
            }
    
    def get_cash_flow_summary(self, company_id: str) -> Dict[str, Any]:
        """
        Get a summary of cash flow performance for a company.
        
        Args:
            company_id (str): Company identifier
            
        Returns:
            Dict[str, Any]: Cash flow summary with key metrics
        """
        try:
            # Fetch latest cash flow data
            cash_flow_data = self.fetch_cash_flow_data(company_id, periods=1)
            
            if cash_flow_data.empty:
                return {
                    'success': False,
                    'error': f"No cash flow data found for company {company_id}"
                }
            
            record = cash_flow_data.iloc[0]
            
            # Calculate key metrics
            operating_cf = float(record.get('total_operating_cash_flow', 0))
            investing_cf = float(record.get('total_investing_cash_flow', 0))
            financing_cf = float(record.get('total_financing_cash_flow', 0))
            net_cash_change = float(record.get('net_cash_change', 0))
            capex = abs(float(record.get('capital_expenditures', 0)))
            free_cash_flow = operating_cf - capex
            
            # Determine cash flow health
            health_score = 0
            health_factors = []
            
            if operating_cf > 0:
                health_score += 40
                health_factors.append("Positive operating cash flow")
            else:
                health_factors.append("Negative operating cash flow - concerning")
            
            if free_cash_flow > 0:
                health_score += 30
                health_factors.append("Positive free cash flow")
            else:
                health_factors.append("Negative free cash flow")
            
            if net_cash_change > 0:
                health_score += 20
                health_factors.append("Overall cash increase")
            else:
                health_factors.append("Overall cash decrease")
            
            if operating_cf > capex * 2:  # OCF covers CapEx well
                health_score += 10
                health_factors.append("Strong cash generation relative to investments")
            
            # Determine health level
            if health_score >= 80:
                health_level = "Excellent"
                health_color = "green"
            elif health_score >= 60:
                health_level = "Good"
                health_color = "light-green"
            elif health_score >= 40:
                health_level = "Moderate"
                health_color = "yellow"
            elif health_score >= 20:
                health_level = "Concerning"
                health_color = "orange"
            else:
                health_level = "Critical"
                health_color = "red"
            
            summary = {
                'success': True,
                'company_id': company_id,
                'statement_date': record.get('date'),
                'analysis_timestamp': datetime.now().isoformat(),
                'cash_flow_metrics': {
                    'operating_cash_flow': operating_cf,
                    'investing_cash_flow': investing_cf,
                    'financing_cash_flow': financing_cf,
                    'net_cash_change': net_cash_change,
                    'free_cash_flow': free_cash_flow,
                    'capital_expenditures': -capex  # Show as negative
                },
                'health_assessment': {
                    'health_score': health_score,
                    'health_level': health_level,
                    'health_color': health_color,
                    'health_factors': health_factors
                },
                'key_insights': self._generate_key_insights(record),
                'next_steps': self._suggest_next_steps(operating_cf, free_cash_flow, net_cash_change)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating cash flow summary: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _generate_key_insights(self, record: pd.Series) -> List[str]:
        """Generate key insights from cash flow data."""
        insights = []
        
        try:
            operating_cf = float(record.get('total_operating_cash_flow', 0))
            investing_cf = float(record.get('total_investing_cash_flow', 0))
            financing_cf = float(record.get('total_financing_cash_flow', 0))
            capex = abs(float(record.get('capital_expenditures', 0)))
            
            # Operating cash flow insights
            if operating_cf > 0:
                insights.append(f"💰 Generated ${operating_cf:,.0f} from core operations")
            else:
                insights.append(f"⚠️ Used ${abs(operating_cf):,.0f} in operations - needs attention")
            
            # Investment insights
            if investing_cf < 0:
                insights.append(f"🏗️ Invested ${abs(investing_cf):,.0f} in growth and assets")
            elif investing_cf > 0:
                insights.append(f"📈 Generated ${investing_cf:,.0f} from asset sales/investments")
            
            # Financing insights
            if financing_cf > 0:
                insights.append(f"💳 Raised ${financing_cf:,.0f} through financing activities")
            elif financing_cf < 0:
                insights.append(f"💸 Returned ${abs(financing_cf):,.0f} to investors/creditors")
            
            # CapEx efficiency
            if operating_cf > 0 and capex > 0:
                capex_coverage = operating_cf / capex
                if capex_coverage > 2:
                    insights.append(f"✅ Strong cash generation covers CapEx {capex_coverage:.1f}x over")
                elif capex_coverage > 1:
                    insights.append(f"👍 Operating cash flow covers CapEx with {capex_coverage:.1f}x coverage")
                else:
                    insights.append(f"🤔 CapEx exceeds operating cash flow - monitor sustainability")
            
            return insights[:4]  # Return top 4 insights
            
        except Exception as e:
            logger.warning(f"Error generating insights: {e}")
            return ["Unable to generate detailed insights from current data"]
    
    def _suggest_next_steps(self, operating_cf: float, free_cf: float, net_change: float) -> List[str]:
        """Suggest next steps based on cash flow performance."""
        next_steps = []
        
        try:
            if operating_cf < 0:
                next_steps.extend([
                    "🚨 Immediate action needed: Focus on improving operational cash flow",
                    "📊 Analyze working capital management and collection processes",
                    "💡 Consider cost reduction initiatives to improve cash generation"
                ])
            elif free_cf < 0:
                next_steps.extend([
                    "⚡ Review capital expenditure timing and priorities",
                    "🔍 Evaluate ROI on current investments and projects",
                    "📈 Focus on maximizing operational efficiency"
                ])
            elif net_change < 0:
                next_steps.extend([
                    "📋 Monitor cash position and plan for upcoming periods",
                    "🔄 Consider optimizing financing structure",
                    "📊 Implement regular cash flow forecasting"
                ])
            else:
                next_steps.extend([
                    "✅ Maintain strong cash flow management practices",
                    "🎯 Consider strategic investment opportunities",
                    "📈 Explore growth initiatives with available cash"
                ])
            
            # Always include monitoring
            next_steps.append("📊 Continue monthly cash flow monitoring and analysis")
            
            return next_steps[:3]  # Return top 3 next steps
            
        except Exception as e:
            logger.warning(f"Error generating next steps: {e}")
            return ["Consult with financial advisor for personalized cash flow guidance"]


# Example usage and testing functions
def main():
    """Example usage of the Cash Flow Service."""
    try:
        # Initialize service
        cash_flow_service = CashFlowService(
            forecasting_enabled=True,
            historical_periods=3,
            validation_enabled=True
        )
        
        # Note: In real usage, you would set a database connection
        # cash_flow_service.set_database_connection(your_db_connection)
        
        logger.info("Cash Flow Service initialized successfully")
        logger.info("Service ready for cash flow processing and analysis")
        
        return cash_flow_service
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    service = main()