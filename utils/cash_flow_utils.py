import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)

class CashFlowProcessor:
    """Main class for processing cash flow statement data"""
    
    def __init__(self):
        self.required_fields = [
            'net_income',
            'net_cash_from_operating_activities',
            'net_cash_from_investing_activities',
            'net_cash_from_financing_activities'
        ]
        
    def process_cash_flow_data(self, raw_data: Dict[str, Any], 
                              company_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process raw cash flow data into standardized format
        
        Args:
            raw_data: Raw cash flow data extracted from documents
            company_info: Company metadata
            
        Returns:
            Dict containing processed cash flow data
        """
        try:
            logger.info("Starting cash flow data processing")
            
            # Initialize processed data template
            processed_data = self._initialize_cash_flow_template()
            
            # Map and clean raw data
            processed_data.update(self._map_raw_data_to_schema(raw_data))
            
            # Add company information
            if company_info:
                processed_data.update({
                    'company_name': company_info.get('company_name', 'Unknown'),
                    'industry': company_info.get('industry', 'Technology'),
                    'company_id': company_info.get('company_id')
                })
            
            # Calculate derived metrics
            processed_data = self._calculate_derived_metrics(processed_data)
            
            # Validate and score data
            processed_data = self._validate_and_score_data(processed_data)
            
            # Add metadata
            processed_data.update({
                'generated_at': datetime.now(),
                'year': raw_data.get('year', datetime.now().year)
            })
            
            logger.info("Cash flow processing completed successfully")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing cash flow data: {e}")
            raise
    
    def _initialize_cash_flow_template(self) -> Dict[str, Any]:
        """Initialize cash flow template with all required fields"""
        return {
            # Operating Activities
            'net_income': 0.0,
            'depreciation_and_amortization': 0.0,
            'stock_based_compensation': 0.0,
            'changes_in_working_capital': 0.0,
            'accounts_receivable': 0.0,  # Change in AR
            'inventory': 0.0,  # Change in inventory
            'accounts_payable': 0.0,  # Change in AP
            'net_cash_from_operating_activities': 0.0,
            
            # Investing Activities
            'capital_expenditures': 0.0,
            'acquisitions': 0.0,
            'net_cash_from_investing_activities': 0.0,
            
            # Financing Activities
            'dividends_paid': 0.0,
            'share_repurchases': 0.0,
            'net_cash_from_financing_activities': 0.0,
            
            # Derived Metrics
            'free_cash_flow': 0.0,
            'ocf_to_net_income_ratio': 0.0,
            'liquidation_label': 0,
            'debt_to_equity_ratio': 0.0,
            'interest_coverage_ratio': 0.0
        }
    
    def _map_raw_data_to_schema(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map raw data fields to standardized schema"""
        
        # Field mappings for common variations
        field_mappings = {
            'net_earnings': 'net_income',
            'depreciation': 'depreciation_and_amortization',
            'amortization': 'depreciation_and_amortization',
            'stock_compensation': 'stock_based_compensation',
            'share_based_compensation': 'stock_based_compensation',
            'working_capital_changes': 'changes_in_working_capital',
            'receivables_change': 'accounts_receivable',
            'inventory_change': 'inventory',
            'payables_change': 'accounts_payable',
            'operating_cash_flow': 'net_cash_from_operating_activities',
            'capex': 'capital_expenditures',
            'capital_investments': 'capital_expenditures',
            'investing_cash_flow': 'net_cash_from_investing_activities',
            'financing_cash_flow': 'net_cash_from_financing_activities',
            'dividends': 'dividends_paid',
            'share_buybacks': 'share_repurchases'
        }
        
        mapped_data = {}
        
        for raw_field, value in raw_data.items():
            # Clean and convert values
            if isinstance(value, (int, float, str)):
                try:
                    if isinstance(value, str):
                        value = self._clean_numeric_string(value)
                    mapped_data[raw_field] = float(value) if value != '' else 0.0
                except (ValueError, TypeError):
                    continue
            
            # Apply field mappings
            mapped_field = field_mappings.get(raw_field.lower(), raw_field)
            if mapped_field != raw_field:
                mapped_data[mapped_field] = mapped_data.pop(raw_field, 0.0)
        
        return mapped_data
    
    def _clean_numeric_string(self, value: str) -> float:
        """Clean and convert string to numeric value"""
        if not isinstance(value, str):
            return float(value)
       # Remove common formatting
        cleaned = value.replace('$', '').replace('£', '').replace('€', '')
        cleaned = cleaned.replace(',', '').replace(' ', '')
        
        # Handle parentheses as negative values
        if '(' in cleaned and ')' in cleaned:
            cleaned = '-' + cleaned.replace('(', '').replace(')', '')
        
        return float(cleaned) if cleaned else 0.0
    
    def _calculate_derived_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate derived cash flow metrics"""
        
        # Free Cash Flow = Operating Cash Flow - Capital Expenditures
        operating_cf = data.get('net_cash_from_operating_activities', 0)
        capex = abs(data.get('capital_expenditures', 0))  # CapEx is typically negative
        data['free_cash_flow'] = operating_cf - capex
        
        # OCF to Net Income Ratio
        net_income = data.get('net_income', 1)
        data['ocf_to_net_income_ratio'] = operating_cf / net_income if net_income != 0 else 0
        
        # Calculate working capital changes if not provided
        if data.get('changes_in_working_capital', 0) == 0:
            ar_change = data.get('accounts_receivable', 0)
            inv_change = data.get('inventory', 0)
            ap_change = data.get('accounts_payable', 0)
            data['changes_in_working_capital'] = -(ar_change + inv_change) + ap_change
        
        # Net Change in Cash
        investing_cf = data.get('net_cash_from_investing_activities', 0)
        financing_cf = data.get('net_cash_from_financing_activities', 0)
        data['net_change_in_cash'] = operating_cf + investing_cf + financing_cf
        
        return data
    
    def _validate_and_score_data(self, data: Dict[str, Any]) -> List[str]:
        """Validate cash flow data for consistency"""
        
        errors = []
        
        # Validate free cash flow calculation
        operating_cf = data.get('net_cash_from_operating_activities', 0)
        capex = abs(data.get('capital_expenditures', 0))
        calculated_fcf = operating_cf - capex
        reported_fcf = data.get('free_cash_flow', 0)
        
        if abs(calculated_fcf - reported_fcf) > 1000:
            errors.append("Free cash flow calculation inconsistency")
        
        # Check for unreasonable values
        if abs(operating_cf) > 1e12:  # $1 trillion
            errors.append("Operating cash flow seems unreasonably large")
        
        # Validate working capital components
        ar_change = data.get('accounts_receivable', 0)
        inv_change = data.get('inventory', 0)
        ap_change = data.get('accounts_payable', 0)
        total_wc_change = data.get('changes_in_working_capital', 0)
        
        calculated_wc = -(ar_change + inv_change) + ap_change
        if abs(calculated_wc - total_wc_change) > max(abs(total_wc_change) * 0.2, 10000):
            errors.append("Working capital components don't match total change")
        
        data['validation_errors'] = errors
        return data

class CashFlowAnalyzer:
    """Analyze cash flow data and calculate ratios"""
    
    def __init__(self):
        self.ratio_categories = {
            'operating': ['ocf_margin', 'ocf_to_net_income', 'cash_conversion'],
            'quality': ['accruals_ratio', 'cash_quality_ratio'],
            'coverage': ['debt_coverage', 'interest_coverage', 'dividend_coverage'],
            'efficiency': ['cash_return_on_assets', 'free_cash_flow_yield']
        }
    
    def analyze_cash_flow(self, data: Dict[str, Any], 
                         balance_sheet_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform comprehensive cash flow analysis"""
        
        analysis = {
            'ratios': self.calculate_cash_flow_ratios(data, balance_sheet_data),
            'operating_analysis': self.analyze_operating_cash_flow(data),
            'quality_analysis': self.analyze_cash_flow_quality(data),
            'liquidity_analysis': self.analyze_liquidity_from_cash_flow(data),
            'sustainability_analysis': self.analyze_sustainability(data),
            'risk_assessment': self.assess_cash_flow_risks(data),
            'trends': self.analyze_cash_flow_trends(data)
        }
        
        return analysis
    
    def calculate_cash_flow_ratios(self, data: Dict[str, Any], 
                                  balance_sheet_data: Dict[str, Any] = None) -> Dict[str, float]:
        """Calculate comprehensive cash flow ratios"""
        
        ratios = {}
        
        # Extract key values
        operating_cf = data.get('net_cash_from_operating_activities', 0)
        investing_cf = data.get('net_cash_from_investing_activities', 0)
        financing_cf = data.get('net_cash_from_financing_activities', 0)
        net_income = max(abs(data.get('net_income', 1)), 1)  # Avoid division by zero
        free_cf = data.get('free_cash_flow', 0)
        capex = abs(data.get('capital_expenditures', 0))
        
        # Operating ratios
        ratios['ocf_to_net_income'] = operating_cf / net_income
        ratios['free_cash_flow_conversion'] = free_cf / net_income if net_income > 0 else 0
        
        # Quality ratios
        ratios['cash_quality_ratio'] = operating_cf / (operating_cf + investing_cf + financing_cf) if (operating_cf + investing_cf + financing_cf) != 0 else 0
        
        # Coverage ratios
        if balance_sheet_data:
            total_debt = (balance_sheet_data.get('short_term_debt', 0) + 
                         balance_sheet_data.get('long_term_debt', 0))
            total_assets = max(balance_sheet_data.get('total_assets', 1), 1)
            
            ratios['debt_coverage_ratio'] = operating_cf / total_debt if total_debt > 0 else float('inf')
            ratios['cash_return_on_assets'] = operating_cf / total_assets
            ratios['free_cash_flow_yield'] = free_cf / total_assets
        
        # Efficiency ratios
        ratios['capex_intensity'] = capex / operating_cf if operating_cf > 0 else 0
        ratios['cash_conversion_efficiency'] = abs(operating_cf / net_income) if net_income != 0 else 0
        
        return ratios
    
    def analyze_operating_cash_flow(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze operating cash flow quality and trends"""
        
        operating_cf = data.get('net_cash_from_operating_activities', 0)
        net_income = data.get('net_income', 0)
        
        # Determine operating cash flow status
        if operating_cf > 0 and operating_cf > net_income:
            status = 'excellent'
            quality = 'high'
        elif operating_cf > 0 and operating_cf > net_income * 0.8:
            status = 'good'
            quality = 'good'
        elif operating_cf > 0:
            status = 'adequate'
            quality = 'moderate'
        else:
            status = 'poor'
            quality = 'low'
        
        return {
            'status': status,
            'quality': quality,
            'operating_cash_flow': operating_cf,
            'net_income': net_income,
            'ocf_to_ni_ratio': operating_cf / max(abs(net_income), 1),
            'analysis': self._get_operating_cf_analysis(status, operating_cf, net_income)
        }
    
    def analyze_cash_flow_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the quality of cash flows"""
        
        operating_cf = data.get('net_cash_from_operating_activities', 0)
        investing_cf = data.get('net_cash_from_investing_activities', 0)
        financing_cf = data.get('net_cash_from_financing_activities', 0)
        
        total_cf = operating_cf + investing_cf + financing_cf
        
        # Calculate quality metrics
        operating_percentage = (operating_cf / total_cf * 100) if total_cf != 0 else 0
        
        # Determine quality rating
        if operating_percentage >= 80 and operating_cf > 0:
            quality_rating = 'high'
        elif operating_percentage >= 60 and operating_cf > 0:
            quality_rating = 'good'
        elif operating_percentage >= 40:
            quality_rating = 'moderate'
        else:
            quality_rating = 'low'
        
        return {
            'quality_rating': quality_rating,
            'operating_percentage': operating_percentage,
            'cash_flow_composition': {
                'operating': operating_cf,
                'investing': investing_cf,
                'financing': financing_cf,
                'total': total_cf
            },
            'sustainability_indicators': self._assess_sustainability_indicators(data)
        }
    
    def analyze_liquidity_from_cash_flow(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze liquidity position from cash flow perspective"""
        
        operating_cf = data.get('net_cash_from_operating_activities', 0)
        free_cf = data.get('free_cash_flow', 0)
        capex = abs(data.get('capital_expenditures', 0))
        
        # Liquidity indicators
        cash_adequacy_ratio = operating_cf / capex if capex > 0 else float('inf')
        
        if operating_cf > 0 and free_cf > 0:
            liquidity_status = 'strong'
        elif operating_cf > 0:
            liquidity_status = 'adequate'
        else:
            liquidity_status = 'weak'
        
        return {
            'liquidity_status': liquidity_status,
            'cash_adequacy_ratio': cash_adequacy_ratio,
            'free_cash_flow': free_cf,
            'operating_cash_flow': operating_cf,
            'recommendations': self._get_liquidity_recommendations(liquidity_status)
        }
    
    def analyze_sustainability(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cash flow sustainability"""
        
        operating_cf = data.get('net_cash_from_operating_activities', 0)
        investing_cf = data.get('net_cash_from_investing_activities', 0)
        financing_cf = data.get('net_cash_from_financing_activities', 0)
        free_cf = data.get('free_cash_flow', 0)
        
        # Sustainability metrics
        self_funding_ratio = operating_cf / abs(investing_cf) if investing_cf < 0 else float('inf')
        
        # Determine sustainability
        if free_cf > 0 and operating_cf > 0:
            sustainability = 'sustainable'
        elif operating_cf > 0 and self_funding_ratio > 0.8:
            sustainability = 'moderately_sustainable'
        else:
            sustainability = 'unsustainable'
        
        return {
            'sustainability': sustainability,
            'self_funding_ratio': self_funding_ratio,
            'free_cash_flow_positive': free_cf > 0,
            'operating_cf_positive': operating_cf > 0,
            'sustainability_score': self._calculate_sustainability_score(data)
        }
    
    def assess_cash_flow_risks(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Assess cash flow-related risks"""
        
        risks = []
        
        operating_cf = data.get('net_cash_from_operating_activities', 0)
        free_cf = data.get('free_cash_flow', 0)
        net_income = data.get('net_income', 0)
        
        # Negative operating cash flow risk
        if operating_cf < 0:
            risks.append({
                'type': 'liquidity_risk',
                'severity': 'high',
                'indicator': 'negative_operating_cash_flow',
                'value': operating_cf,
                'description': 'Negative operating cash flow indicates core business is consuming cash'
            })
        
        # Poor cash conversion risk
        if net_income > 0 and operating_cf < net_income * 0.5:
            risks.append({
                'type': 'quality_risk',
                'severity': 'medium',
                'indicator': 'poor_cash_conversion',
                'value': operating_cf / max(net_income, 1),
                'description': 'Low cash conversion from earnings may indicate quality issues'
            })
        
        # Negative free cash flow risk
        if free_cf < 0:
            risks.append({
                'type': 'sustainability_risk',
                'severity': 'medium',
                'indicator': 'negative_free_cash_flow',
                'value': free_cf,
                'description': 'Negative free cash flow limits financial flexibility'
            })
        
        return risks
    
    def analyze_cash_flow_trends(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cash flow trends (placeholder for historical data)"""
        
        # This would require historical data to implement properly
        return {
            'trend_analysis': 'Not available - requires historical data',
            'growth_rates': {},
            'trend_indicators': []
        }
    
    def _get_operating_cf_analysis(self, status: str, operating_cf: float, net_income: float) -> str:
        """Get operating cash flow analysis description"""
        
        if status == 'excellent':
            return f"Strong operating cash flow of ${operating_cf:,.0f} exceeds net income, indicating high-quality earnings."
        elif status == 'good':
            return f"Good operating cash flow of ${operating_cf:,.0f} closely matches earnings quality."
        elif status == 'adequate':
            return f"Adequate operating cash flow of ${operating_cf:,.0f}, but below net income level."
        else:
            return f"Poor operating cash flow of ${operating_cf:,.0f} indicates operational challenges."
    
    def _assess_sustainability_indicators(self, data: Dict[str, Any]) -> Dict[str, bool]:
        """Assess various sustainability indicators"""
        
        return {
            'positive_operating_cf': data.get('net_cash_from_operating_activities', 0) > 0,
            'positive_free_cf': data.get('free_cash_flow', 0) > 0,
            'self_funded_capex': data.get('net_cash_from_operating_activities', 0) > abs(data.get('capital_expenditures', 0)),
            'minimal_external_financing': abs(data.get('net_cash_from_financing_activities', 0)) < data.get('net_cash_from_operating_activities', 0)
        }
    
    def _get_liquidity_recommendations(self, liquidity_status: str) -> List[str]:
        """Get liquidity-specific recommendations"""
        
        recommendations = []
        
        if liquidity_status == 'weak':
            recommendations.extend([
                "Focus on improving operating cash flow through better collections",
                "Consider reducing capital expenditures to preserve cash",
                "Explore additional financing options",
                "Implement strict cash flow management procedures"
            ])
        elif liquidity_status == 'adequate':
            recommendations.extend([
                "Monitor cash flow closely to maintain adequacy",
                "Build cash reserves for unexpected needs",
                "Optimize working capital management"
            ])
        else:  # strong
            recommendations.extend([
                "Consider strategic investments for growth",
                "Evaluate optimal cash levels to avoid excess"
            ])
        
        return recommendations
    
    def _calculate_sustainability_score(self, data: Dict[str, Any]) -> float:
        """Calculate sustainability score out of 100"""
        
        score = 0
        
        # Operating cash flow (40 points)
        operating_cf = data.get('net_cash_from_operating_activities', 0)
        if operating_cf > 0:
            score += 40
        
        # Free cash flow (30 points)
        free_cf = data.get('free_cash_flow', 0)
        if free_cf > 0:
            score += 30
        
        # Self-funding capability (20 points)
        capex = abs(data.get('capital_expenditures', 0))
        if operating_cf > capex:
            score += 20
        
        # Cash quality (10 points)
        net_income = data.get('net_income', 0)
        if net_income > 0 and operating_cf > net_income * 0.8:
            score += 10
        
        return score

class CashFlowGenerator:
    """Generate cash flow statement from balance sheet data"""
    
    def __init__(self):
        self.method = 'indirect'  # Default method
    
    def generate_from_balance_sheet(self, current_bs: Dict[str, Any], 
                                  previous_bs: Dict[str, Any] = None,
                                  income_data: Dict[str, Any] = None,
                                  method: str = 'indirect') -> Dict[str, Any]:
        """
        Generate cash flow statement from balance sheet data
        
        Args:
            current_bs: Current year balance sheet data
            previous_bs: Previous year balance sheet data (optional)
            income_data: Income statement data (optional)
            method: 'indirect' or 'direct'
            
        Returns:
            Dict containing generated cash flow statement
        """
        
        self.method = method
        
        if method == 'indirect':
            return self._generate_indirect_cash_flow(current_bs, previous_bs, income_data)
        else:
            return self._generate_direct_cash_flow(current_bs, previous_bs, income_data)
    
    def _generate_indirect_cash_flow(self, current_bs: Dict[str, Any], 
                                   previous_bs: Dict[str, Any] = None,
                                   income_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate cash flow using indirect method"""
        
        # Initialize cash flow template
        cash_flow = {
            'net_income': 0,
            'depreciation_and_amortization': 0,
            'stock_based_compensation': 0,
            'changes_in_working_capital': 0,
            'accounts_receivable': 0,
            'inventory': 0,
            'accounts_payable': 0,
            'net_cash_from_operating_activities': 0,
            'capital_expenditures': 0,
            'acquisitions': 0,
            'net_cash_from_investing_activities': 0,
            'dividends_paid': 0,
            'share_repurchases': 0,
            'net_cash_from_financing_activities': 0,
            'free_cash_flow': 0,
            'ocf_to_net_income_ratio': 0,
            'liquidation_label': 0,
            'debt_to_equity_ratio': 0,
            'interest_coverage_ratio': 0
        }
        
        # Get net income
        net_income = income_data.get('net_income', 0) if income_data else self._estimate_net_income(current_bs)
        cash_flow['net_income'] = net_income
        
        # Calculate depreciation
        depreciation = self._calculate_depreciation(current_bs, previous_bs)
        cash_flow['depreciation_and_amortization'] = depreciation
        
        # Calculate working capital changes
        if previous_bs:
            wc_changes = self._calculate_working_capital_changes(current_bs, previous_bs)
            cash_flow.update(wc_changes)
        else:
            # Estimate working capital changes
            wc_changes = self._estimate_working_capital_changes(current_bs)
            cash_flow.update(wc_changes)
        
        # Calculate operating cash flow
        operating_cf = (
            net_income + 
            depreciation + 
            cash_flow['stock_based_compensation'] +
            cash_flow['changes_in_working_capital']
        )
        cash_flow['net_cash_from_operating_activities'] = operating_cf
        
        # Calculate investing activities
        capex = self._calculate_capital_expenditure(current_bs, previous_bs)
        cash_flow['capital_expenditures'] = capex
        cash_flow['net_cash_from_investing_activities'] = -capex
        
        # Calculate financing activities
        financing_cf = self._calculate_financing_activities(current_bs, previous_bs)
        cash_flow['net_cash_from_financing_activities'] = financing_cf
        
        # Calculate derived metrics
        cash_flow['free_cash_flow'] = operating_cf - abs(capex)
        cash_flow['ocf_to_net_income_ratio'] = operating_cf / max(abs(net_income), 1)
        
        # Calculate financial ratios
        cash_flow['debt_to_equity_ratio'] = self._calculate_debt_to_equity(current_bs)
        cash_flow['interest_coverage_ratio'] = self._calculate_interest_coverage(current_bs, income_data)
        
        # Assess liquidation risk
        cash_flow['liquidation_label'] = self._assess_liquidation_risk(cash_flow, current_bs)
        
        return cash_flow
    
    def _generate_direct_cash_flow(self, current_bs: Dict[str, Any], 
                                 previous_bs: Dict[str, Any] = None,
                                 income_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate cash flow using direct method (simplified)"""
        
        # Start with indirect method as base
        cash_flow = self._generate_indirect_cash_flow(current_bs, previous_bs, income_data)
        
        # For direct method, we would need detailed transaction data
        # This is a simplified conversion
        
        return cash_flow
    
    def _estimate_net_income(self, balance_sheet: Dict[str, Any]) -> float:
        """Estimate net income if not provided"""
        
        # Conservative estimate based on assets
        total_assets = balance_sheet.get('total_assets', 0)
        return total_assets * 0.05  # 5% ROA assumption
    
    def _calculate_depreciation(self, current_bs: Dict[str, Any], 
                              previous_bs: Dict[str, Any] = None) -> float:
        """Calculate depreciation expense"""
        
        if previous_bs:
            # Calculate from accumulated depreciation change
            current_accum_dep = current_bs.get('accumulated_depreciation', 0)
            previous_accum_dep = previous_bs.get('accumulated_depreciation', 0)
            depreciation = current_accum_dep - previous_accum_dep
            
            # Add any asset disposals (simplified)
            return max(depreciation, 0)
        else:
            # Estimate depreciation as percentage of PPE
            ppe = current_bs.get('property_plant_equipment', 0)
            return ppe * 0.10  # 10% annual depreciation assumption
    
    def _calculate_working_capital_changes(self, current_bs: Dict[str, Any], 
                                         previous_bs: Dict[str, Any]) -> Dict[str, float]:
        """Calculate working capital changes"""
        
        # Calculate changes in working capital components
        ar_change = current_bs.get('accounts_receivable', 0) - previous_bs.get('accounts_receivable', 0)
        inventory_change = current_bs.get('inventory', 0) - previous_bs.get('inventory', 0)
        ap_change = current_bs.get('accounts_payable', 0) - previous_bs.get('accounts_payable', 0)
        
        # Working capital change = -(increase in current assets) + (increase in current liabilities)
        total_wc_change = -(ar_change + inventory_change) + ap_change
        
        return {
            'accounts_receivable': -ar_change,  # Negative because increase reduces cash
            'inventory': -inventory_change,
            'accounts_payable': ap_change,  # Positive because increase adds cash
            'changes_in_working_capital': total_wc_change
        }
    
    def _estimate_working_capital_changes(self, balance_sheet: Dict[str, Any]) -> Dict[str, float]:
        """Estimate working capital changes when previous year data unavailable"""
        
        total_assets = balance_sheet.get('total_assets', 0)
        
        # Estimate changes as percentage of total assets
        return {
            'accounts_receivable': -(total_assets * 0.02),  # 2% increase in AR
            'inventory': -(total_assets * 0.01),  # 1% increase in inventory
            'accounts_payable': total_assets * 0.015,  # 1.5% increase in AP
            'changes_in_working_capital': -(total_assets * 0.015)  # Net working capital increase
        }
    
    def _calculate_capital_expenditure(self, current_bs: Dict[str, Any], 
                                     previous_bs: Dict[str, Any] = None) -> float:
        """Calculate capital expenditures"""
        
        if previous_bs:
            # CapEx = (Current PPE - Previous PPE) + Depreciation
            current_ppe = current_bs.get('property_plant_equipment', 0)
            previous_ppe = previous_bs.get('property_plant_equipment', 0)
            depreciation = self._calculate_depreciation(current_bs, previous_bs)
            
            capex = (current_ppe - previous_ppe) + depreciation
            return max(capex, 0)  # Ensure non-negative
        else:
            # Estimate as percentage of PPE
            ppe = current_bs.get('property_plant_equipment', 0)
            return ppe * 0.08  # 8% of PPE as maintenance capex
    
    def _calculate_financing_activities(self, current_bs: Dict[str, Any], 
                                      previous_bs: Dict[str, Any] = None) -> float:
        """Calculate net cash from financing activities"""
        
        financing_cf = 0
        
        if previous_bs:
            # Calculate debt changes
            current_debt = (current_bs.get('short_term_debt', 0) + 
                           current_bs.get('long_term_debt', 0))
            previous_debt = (previous_bs.get('short_term_debt', 0) + 
                            previous_bs.get('long_term_debt', 0))
            debt_change = current_debt - previous_debt
            
            # Calculate equity changes (simplified)
            current_equity = current_bs.get('total_equity', 0)
            previous_equity = previous_bs.get('total_equity', 0)
            equity_change = current_equity - previous_equity
            
            financing_cf = debt_change + max(equity_change, 0)  # Only count equity increases
        
        return financing_cf
    
    def _calculate_debt_to_equity(self, balance_sheet: Dict[str, Any]) -> float:
        """Calculate debt-to-equity ratio"""
        
        total_debt = (balance_sheet.get('short_term_debt', 0) + 
                     balance_sheet.get('long_term_debt', 0))
        total_equity = max(balance_sheet.get('total_equity', 1), 1)
        
        return total_debt / total_equity
    
    def _calculate_interest_coverage(self, balance_sheet: Dict[str, Any], 
                                   income_data: Dict[str, Any] = None) -> float:
        """Calculate interest coverage ratio"""
        
        if income_data and 'interest_expense' in income_data:
            ebit = income_data.get('ebit', income_data.get('net_income', 0))
            interest_expense = income_data.get('interest_expense', 1)
            return ebit / max(interest_expense, 1)
        else:
            # Estimate interest expense from debt
            total_debt = (balance_sheet.get('short_term_debt', 0) + 
                         balance_sheet.get('long_term_debt', 0))
            estimated_interest = total_debt * 0.05  # 5% interest rate assumption
            estimated_ebit = balance_sheet.get('total_assets', 0) * 0.06  # 6% asset return
            
            return estimated_ebit / max(estimated_interest, 1)
    
    def _assess_liquidation_risk(self, cash_flow: Dict[str, Any], 
                               balance_sheet: Dict[str, Any]) -> int:
        """Assess liquidation risk (0 = low, 1 = high)"""
        
        risk_factors = 0
        
        # Negative net income
        if cash_flow.get('net_income', 0) < 0:
            risk_factors += 1
        
        # Negative operating cash flow
        if cash_flow.get('net_cash_from_operating_activities', 0) < 0:
            risk_factors += 1
        
        # High debt-to-equity ratio
        if cash_flow.get('debt_to_equity_ratio', 0) > 2.0:
            risk_factors += 1
        
        # Low current ratio
        current_ratio = (balance_sheet.get('current_assets', 0) / 
                        max(balance_sheet.get('current_liabilities', 1), 1))
        if current_ratio < 1.0:
            risk_factors += 1
        
        # Return 1 if 2 or more risk factors present
        return 1 if risk_factors >= 2 else 0

# Standalone utility functions
def calculate_cash_flow_ratios(data: Dict[str, Any], 
                              balance_sheet_data: Dict[str, Any] = None) -> Dict[str, float]:
    """Calculate cash flow ratios"""
    analyzer = CashFlowAnalyzer()
    return analyzer.calculate_cash_flow_ratios(data, balance_sheet_data)

def generate_cash_flow_from_balance_sheet(current_bs: Dict[str, Any], 
                                        previous_bs: Dict[str, Any] = None,
                                        method: str = 'indirect') -> Dict[str, Any]:
    """Generate cash flow statement from balance sheet data"""
    generator = CashFlowGenerator()
    return generator.generate_from_balance_sheet(current_bs, previous_bs, method=method)

def validate_cash_flow_data(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate cash flow data integrity"""
    
    errors = []
    
    # Check required fields
    required_fields = ['net_income', 'net_cash_from_operating_activities']
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    # Validate free cash flow calculation
    operating_cf = data.get('net_cash_from_operating_activities', 0)
    capex = data.get('capital_expenditures', 0)
    calculated_fcf = operating_cf - abs(capex)
    reported_fcf = data.get('free_cash_flow', 0)
    
    if abs(calculated_fcf - reported_fcf) > max(abs(reported_fcf) * 0.1, 1000):
        errors.append("Free cash flow calculation inconsistency")
    
    # Check for unreasonable values
    if abs(operating_cf) > 1e12:  # $1 trillion
        errors.append("Operating cash flow seems unreasonably high")
    
    return len(errors) == 0, errors

def generate_cash_flow_insights(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate AI-powered cash flow insights"""
    
    insights = []
    
    # Operating cash flow insights
    operating_cf = data.get('net_cash_from_operating_activities', 0)
    net_income = data.get('net_income', 0)
    
    if operating_cf > net_income and net_income > 0:
        insights.append({
            'type': 'positive',
            'category': 'operating',
            'title': 'Strong Cash Generation',
            'description': f'Operating cash flow (${operating_cf:,.0f}) exceeds net income (${net_income:,.0f}), indicating high-quality earnings.'
        })
    elif operating_cf < 0:
        insights.append({
            'type': 'warning',
            'category': 'operating',
            'title': 'Negative Operating Cash Flow',
            'description': f'Operating cash flow is negative (${operating_cf:,.0f}), indicating operational challenges.'
        })
    
    # Free cash flow insights
    free_cf = data.get('free_cash_flow', 0)
    if free_cf > 0:
        insights.append({
            'type': 'positive',
            'category': 'liquidity',
            'title': 'Positive Free Cash Flow',
            'description': f'Free cash flow of ${free_cf:,.0f} provides financial flexibility for growth and debt service.'
        })
    elif free_cf < 0:
        insights.append({
            'type': 'warning',
            'category': 'liquidity',
            'title': 'Negative Free Cash Flow',
            'description': f'Negative free cash flow (${free_cf:,.0f}) may limit financial flexibility.'
        })
    
    # Debt coverage insights
    debt_to_equity = data.get('debt_to_equity_ratio', 0)
    if debt_to_equity > 2:
        insights.append({
            'type': 'warning',
            'category': 'leverage',
            'title': 'High Financial Leverage',
            'description': f'Debt-to-equity ratio of {debt_to_equity:.2f} indicates high financial leverage and potential risk.'
        })
    elif debt_to_equity < 0.5:
        insights.append({
            'type': 'positive',
            'category': 'leverage',
            'title': 'Conservative Capital Structure',
            'description': f'Low debt-to-equity ratio of {debt_to_equity:.2f} indicates conservative financial management.'
        })
    
    # Liquidation risk insights
    liquidation_risk = data.get('liquidation_label', 0)
    if liquidation_risk == 1:
        insights.append({
            'type': 'critical',
            'category': 'risk',
            'title': 'Elevated Liquidation Risk',
            'description': 'Multiple risk factors indicate elevated probability of financial distress.'
        })
    else:
        insights.append({
            'type': 'positive',
            'category': 'risk',
            'title': 'Low Liquidation Risk',
            'description': 'Financial indicators suggest stable operations with low liquidation risk.'
        })
    
    return insights

# Initialize processors/
cash_flow_processor = CashFlowProcessor()
cash_flow_analyzer = CashFlowAnalyzer()
cash_flow_generator = CashFlowGenerator()