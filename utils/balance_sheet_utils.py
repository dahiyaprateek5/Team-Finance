import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)

class BalanceSheetProcessor:
    """Main class for processing balance sheet data"""
    
    def __init__(self):
        self.validation_tolerance = 0.01  # 1% tolerance for balance equation
        self.required_fields = [
            'total_assets', 'total_liabilities', 'total_equity',
            'current_assets', 'current_liabilities'
        ]
        
    def process_balance_sheet_data(self, raw_data: Dict[str, Any], 
                                 company_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process raw balance sheet data into standardized format
        
        Args:
            raw_data: Raw financial data extracted from documents
            company_info: Company metadata (name, industry, etc.)
            
        Returns:
            Dict containing processed balance sheet data
        """
        try:
            logger.info("Starting balance sheet data processing")
            
            # Initialize processed data with all required fields
            processed_data = self._initialize_balance_sheet_template()
            
            # Map and clean raw data
            processed_data.update(self._map_raw_data_to_schema(raw_data))
            
            # Add company information
            if company_info:
                processed_data.update({
                    'company_name': company_info.get('company_name', 'Unknown'),
                    'industry': company_info.get('industry', 'Technology'),
                    'company_id': company_info.get('company_id')
                })
            
            # Calculate missing values using financial relationships
            processed_data = self._calculate_missing_values(processed_data)
            
            # Validate data integrity
            processed_data = self._validate_and_score_data(processed_data)
            
            # Add metadata
            processed_data.update({
                'generated_at': datetime.now(),
                'data_source': raw_data.get('data_source', 'uploaded_documents'),
                'year': raw_data.get('year', datetime.now().year)
            })
            
            logger.info(f"Balance sheet processing completed. Accuracy: {processed_data.get('accuracy_percentage', 0):.1f}%")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing balance sheet data: {e}")
            raise
    
    def _initialize_balance_sheet_template(self) -> Dict[str, Any]:
        """Initialize balance sheet template with all required fields"""
        return {
            # Current Assets
            'current_assets': 0.0,
            'cash_and_equivalents': 0.0,
            'accounts_receivable': 0.0,
            'inventory': 0.0,
            'prepaid_expenses': 0.0,
            'other_current_assets': 0.0,
            
            # Non-Current Assets
            'non_current_assets': 0.0,
            'property_plant_equipment': 0.0,
            'accumulated_depreciation': 0.0,
            'net_ppe': 0.0,
            'intangible_assets': 0.0,
            'goodwill': 0.0,
            'investments': 0.0,
            'other_non_current_assets': 0.0,
            'total_assets': 0.0,
            
            # Current Liabilities
            'current_liabilities': 0.0,
            'accounts_payable': 0.0,
            'short_term_debt': 0.0,
            'accrued_liabilities': 0.0,
            'deferred_revenue': 0.0,
            'other_current_liabilities': 0.0,
            
            # Non-Current Liabilities
            'non_current_liabilities': 0.0,
            'long_term_debt': 0.0,
            'deferred_tax_liabilities': 0.0,
            'pension_obligations': 0.0,
            'other_non_current_liabilities': 0.0,
            'total_liabilities': 0.0,
            
            # Equity
            'share_capital': 0.0,
            'retained_earnings': 0.0,
            'additional_paid_in_capital': 0.0,
            'treasury_stock': 0.0,
            'accumulated_other_comprehensive_income': 0.0,
            'total_equity': 0.0,
            
            # Validation fields
            'balance_check': False,
            'accuracy_percentage': 0.0,
            'validation_errors': []
        }
    
    def _map_raw_data_to_schema(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map raw data fields to standardized schema"""
        
        # Define field mappings (raw_field -> schema_field)
        field_mappings = {
            # Common variations
            'cash': 'cash_and_equivalents',
            'cash_equivalents': 'cash_and_equivalents',
            'receivables': 'accounts_receivable',
            'trade_receivables': 'accounts_receivable',
            'inventories': 'inventory',
            'ppe': 'property_plant_equipment',
            'fixed_assets': 'property_plant_equipment',
            'payables': 'accounts_payable',
            'trade_payables': 'accounts_payable',
            'debt': 'long_term_debt',
            'equity': 'total_equity',
            'shareholders_equity': 'total_equity',
            'stockholders_equity': 'total_equity'
        }
        
        mapped_data = {}
        
        for raw_field, value in raw_data.items():
            # Convert to float if possible
            if isinstance(value, (int, float, str)):
                try:
                    if isinstance(value, str):
                        # Clean string values
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
        
        # Remove common currency symbols and formatting
        cleaned = value.replace('$', '').replace('£', '').replace('€', '')
        cleaned = cleaned.replace(',', '').replace(' ', '')
        
        # Handle parentheses as negative values
        if '(' in cleaned and ')' in cleaned:
            cleaned = '-' + cleaned.replace('(', '').replace(')', '')
        
        # Handle percentage
        if '%' in cleaned:
            cleaned = cleaned.replace('%', '')
            return float(cleaned) / 100
        
        return float(cleaned) if cleaned else 0.0
    
    def _calculate_missing_values(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate missing values using financial relationships"""
        
        # Calculate current assets total
        if data['current_assets'] == 0:
            data['current_assets'] = (
                data['cash_and_equivalents'] +
                data['accounts_receivable'] +
                data['inventory'] +
                data['prepaid_expenses'] +
                data['other_current_assets']
            )
        
        # Calculate net PPE
        if data['net_ppe'] == 0:
            if data['property_plant_equipment'] > 0:
                data['net_ppe'] = data['property_plant_equipment'] - data['accumulated_depreciation']
            else:
                data['net_ppe'] = data['property_plant_equipment']
        
        # Calculate non-current assets total
        if data['non_current_assets'] == 0:
            data['non_current_assets'] = (
                data['net_ppe'] +
                data['intangible_assets'] +
                data['goodwill'] +
                data['investments'] +
                data['other_non_current_assets']
            )
        
        # Calculate total assets
        if data['total_assets'] == 0:
            data['total_assets'] = data['current_assets'] + data['non_current_assets']
        
        # Calculate current liabilities total
        if data['current_liabilities'] == 0:
            data['current_liabilities'] = (
                data['accounts_payable'] +
                data['short_term_debt'] +
                data['accrued_liabilities'] +
                data['deferred_revenue'] +
                data['other_current_liabilities']
            )
        
        # Calculate non-current liabilities total
        if data['non_current_liabilities'] == 0:
            data['non_current_liabilities'] = (
                data['long_term_debt'] +
                data['deferred_tax_liabilities'] +
                data['pension_obligations'] +
                data['other_non_current_liabilities']
            )
        
        # Calculate total liabilities
        if data['total_liabilities'] == 0:
            data['total_liabilities'] = data['current_liabilities'] + data['non_current_liabilities']
        
        # Calculate total equity
        if data['total_equity'] == 0:
            data['total_equity'] = (
                data['share_capital'] +
                data['retained_earnings'] +
                data['additional_paid_in_capital'] -
                data['treasury_stock'] +
                data['accumulated_other_comprehensive_income']
            )
        
        # Balance equation check - calculate equity from assets and liabilities if needed
        if data['total_equity'] == 0 and data['total_assets'] > 0:
            data['total_equity'] = data['total_assets'] - data['total_liabilities']
        
        return data
    
    def _validate_and_score_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data and calculate accuracy score"""
        
        # Validate balance sheet equation
        balance_valid, balance_diff = self._validate_balance_equation(data)
        data['balance_check'] = balance_valid
        
        # Calculate accuracy percentage
        data['accuracy_percentage'] = self._calculate_accuracy_score(data, balance_valid)
        
        # Collect validation errors
        validation_errors = []
        
        if not balance_valid:
            validation_errors.append(f"Balance equation error: ${balance_diff:,.0f} difference")
        
        # Check for negative values in key fields
        negative_fields = []
        for field in ['total_assets', 'current_assets', 'cash_and_equivalents']:
            if data.get(field, 0) < 0:
                negative_fields.append(field)
        
        if negative_fields:
            validation_errors.append(f"Negative values detected: {', '.join(negative_fields)}")
        
        # Check for missing critical data
        missing_fields = []
        for field in self.required_fields:
            if data.get(field, 0) == 0:
                missing_fields.append(field)
        
        if missing_fields:
            validation_errors.append(f"Missing critical data: {', '.join(missing_fields)}")
        
        data['validation_errors'] = validation_errors
        
        return data
    
    def _validate_balance_equation(self, data: Dict[str, Any]) -> Tuple[bool, float]:
        """Validate Assets = Liabilities + Equity"""
        total_assets = data.get('total_assets', 0)
        total_liabilities = data.get('total_liabilities', 0)
        total_equity = data.get('total_equity', 0)
        
        if total_assets == 0:
            return False, 0
        
        liab_equity_total = total_liabilities + total_equity
        difference = abs(total_assets - liab_equity_total)
        tolerance = max(total_assets * self.validation_tolerance, 1000)
        
        is_valid = difference <= tolerance
        
        return is_valid, difference
    
    def _calculate_accuracy_score(self, data: Dict[str, Any], balance_valid: bool) -> float:
        """Calculate overall data accuracy score"""
        
        # Count non-zero fields
        total_fields = len([k for k in data.keys() if isinstance(data[k], (int, float))])
        complete_fields = len([k for k, v in data.items() if isinstance(v, (int, float)) and v != 0])
        
        # Base completeness score
        completeness_score = (complete_fields / total_fields) * 100 if total_fields > 0 else 0
        
        # Balance equation score
        balance_score = 30 if balance_valid else 0
        
        # Critical fields score
        critical_complete = sum(1 for field in self.required_fields if data.get(field, 0) != 0)
        critical_score = (critical_complete / len(self.required_fields)) * 40
        
        # Final accuracy score
        accuracy = min(completeness_score * 0.3 + balance_score + critical_score, 100)
        
        return max(0, accuracy)

class BalanceSheetAnalyzer:
    """Analyze balance sheet data and calculate financial ratios"""
    
    def __init__(self):
        self.ratio_categories = {
            'liquidity': ['current_ratio', 'quick_ratio', 'cash_ratio'],
            'leverage': ['debt_to_equity', 'debt_to_assets', 'equity_ratio'],
            'efficiency': ['asset_turnover', 'working_capital_ratio'],
            'profitability': ['roa', 'roe', 'profit_margin']
        }
    
    def analyze_balance_sheet(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive balance sheet analysis"""
        
        analysis = {
            'ratios': self.calculate_all_ratios(data),
            'liquidity_analysis': self.analyze_liquidity(data),
            'leverage_analysis': self.analyze_leverage(data),
            'efficiency_analysis': self.analyze_efficiency(data),
            'overall_health': self.assess_financial_health(data),
            'risk_indicators': self.identify_risk_indicators(data),
            'industry_comparison': self.compare_to_industry(data),
            'trends': self.analyze_trends(data)
        }
        
        return analysis
    
    def calculate_all_ratios(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate all financial ratios from balance sheet data"""
        
        ratios = {}
        
        # Extract key values
        current_assets = data.get('current_assets', 0)
        current_liabilities = max(data.get('current_liabilities', 1), 1)
        total_assets = max(data.get('total_assets', 1), 1)
        total_liabilities = data.get('total_liabilities', 0)
        total_equity = max(data.get('total_equity', 1), 1)
        inventory = data.get('inventory', 0)
        cash = data.get('cash_and_equivalents', 0)
        
        # Liquidity Ratios
        ratios['current_ratio'] = current_assets / current_liabilities
        ratios['quick_ratio'] = (current_assets - inventory) / current_liabilities
        ratios['cash_ratio'] = cash / current_liabilities
        
        # Leverage Ratios
        total_debt = (data.get('short_term_debt', 0) + data.get('long_term_debt', 0))
        ratios['debt_to_equity'] = total_debt / total_equity
        ratios['debt_to_assets'] = total_debt / total_assets
        ratios['equity_ratio'] = total_equity / total_assets
        ratios['debt_ratio'] = total_liabilities / total_assets
        
        # Efficiency Ratios
        ratios['asset_utilization'] = current_assets / total_assets
        working_capital = current_assets - current_liabilities
        ratios['working_capital_ratio'] = working_capital / total_assets
        
        return ratios
    
    def analyze_liquidity(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze liquidity position"""
        
        ratios = self.calculate_all_ratios(data)
        current_ratio = ratios.get('current_ratio', 0)
        quick_ratio = ratios.get('quick_ratio', 0)
        cash_ratio = ratios.get('cash_ratio', 0)
        
        # Determine liquidity status
        if current_ratio >= 2.0 and quick_ratio >= 1.0:
            status = 'excellent'
            risk_level = 'low'
        elif current_ratio >= 1.5 and quick_ratio >= 0.8:
            status = 'good'
            risk_level = 'low'
        elif current_ratio >= 1.0 and quick_ratio >= 0.5:
            status = 'adequate'
            risk_level = 'medium'
        else:
            status = 'poor'
            risk_level = 'high'
        
        return {
            'status': status,
            'risk_level': risk_level,
            'current_ratio': current_ratio,
            'quick_ratio': quick_ratio,
            'cash_ratio': cash_ratio,
            'working_capital': data.get('current_assets', 0) - data.get('current_liabilities', 0),
            'recommendations': self._get_liquidity_recommendations(status, ratios)
        }
    
    def analyze_leverage(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze leverage and debt structure"""
        
        ratios = self.calculate_all_ratios(data)
        debt_to_equity = ratios.get('debt_to_equity', 0)
        debt_to_assets = ratios.get('debt_to_assets', 0)
        equity_ratio = ratios.get('equity_ratio', 0)
        
        # Determine leverage status
        if debt_to_equity <= 0.5 and equity_ratio >= 0.6:
            status = 'conservative'
            risk_level = 'low'
        elif debt_to_equity <= 1.0 and equity_ratio >= 0.4:
            status = 'moderate'
            risk_level = 'medium'
        elif debt_to_equity <= 2.0 and equity_ratio >= 0.25:
            status = 'aggressive'
            risk_level = 'medium'
        else:
            status = 'highly_leveraged'
            risk_level = 'high'
        
        return {
            'status': status,
            'risk_level': risk_level,
            'debt_to_equity': debt_to_equity,
            'debt_to_assets': debt_to_assets,
            'equity_ratio': equity_ratio,
            'total_debt': data.get('short_term_debt', 0) + data.get('long_term_debt', 0),
            'recommendations': self._get_leverage_recommendations(status, ratios)
        }
    
    def analyze_efficiency(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze asset efficiency"""
        
        ratios = self.calculate_all_ratios(data)
        asset_utilization = ratios.get('asset_utilization', 0)
        working_capital_ratio = ratios.get('working_capital_ratio', 0)
        
        # Determine efficiency status
        if asset_utilization >= 0.5 and working_capital_ratio >= 0.1:
            status = 'efficient'
        elif asset_utilization >= 0.3 and working_capital_ratio >= 0.05:
            status = 'moderate'
        else:
            status = 'inefficient'
        
        return {
            'status': status,
            'asset_utilization': asset_utilization,
            'working_capital_ratio': working_capital_ratio,
            'recommendations': self._get_efficiency_recommendations(status, ratios)
        }
    
    def assess_financial_health(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall financial health"""
        
        liquidity_analysis = self.analyze_liquidity(data)
        leverage_analysis = self.analyze_leverage(data)
        efficiency_analysis = self.analyze_efficiency(data)
        
        # Calculate health score
        liquidity_score = {'excellent': 30, 'good': 25, 'adequate': 15, 'poor': 5}.get(liquidity_analysis['status'], 0)
        leverage_score = {'conservative': 25, 'moderate': 20, 'aggressive': 15, 'highly_leveraged': 5}.get(leverage_analysis['status'], 0)
        efficiency_score = {'efficient': 20, 'moderate': 15, 'inefficient': 5}.get(efficiency_analysis['status'], 0)
        
        # Add balance sheet quality score
        balance_quality = 25 if data.get('balance_check', False) else 10
        
        total_score = liquidity_score + leverage_score + efficiency_score + balance_quality
        
        # Determine overall health
        if total_score >= 80:
            health_status = 'excellent'
            health_grade = 'A'
        elif total_score >= 65:
            health_status = 'good'
            health_grade = 'B'
        elif total_score >= 50:
            health_status = 'fair'
            health_grade = 'C'
        else:
            health_status = 'poor'
            health_grade = 'D'
        
        return {
            'health_status': health_status,
            'health_grade': health_grade,
            'health_score': total_score,
            'component_scores': {
                'liquidity': liquidity_score,
                'leverage': leverage_score,
                'efficiency': efficiency_score,
                'balance_quality': balance_quality
            }
        }
    
    def identify_risk_indicators(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify financial risk indicators"""
        
        risks = []
        ratios = self.calculate_all_ratios(data)
        
        # Liquidity risks
        if ratios.get('current_ratio', 0) < 1.0:
            risks.append({
                'type': 'liquidity_risk',
                'severity': 'high',
                'indicator': 'current_ratio_low',
                'value': ratios.get('current_ratio', 0),
                'description': 'Current ratio below 1.0 indicates potential difficulty meeting short-term obligations'
            })
        
        # Leverage risks
        if ratios.get('debt_to_equity', 0) > 2.0:
            risks.append({
                'type': 'leverage_risk',
                'severity': 'high',
                'indicator': 'high_debt_to_equity',
                'value': ratios.get('debt_to_equity', 0),
                'description': 'High debt-to-equity ratio indicates excessive financial leverage'
            })
        
        # Equity risks
        if data.get('total_equity', 0) < 0:
            risks.append({
                'type': 'solvency_risk',
                'severity': 'critical',
                'indicator': 'negative_equity',
                'value': data.get('total_equity', 0),
                'description': 'Negative equity indicates the company owes more than it owns'
            })
        
        # Asset risks
        if ratios.get('cash_ratio', 0) < 0.05:
            risks.append({
                'type': 'liquidity_risk',
                'severity': 'medium',
                'indicator': 'low_cash_reserves',
                'value': ratios.get('cash_ratio', 0),
                'description': 'Low cash reserves may impact ability to handle unexpected expenses'
            })
        
        return risks
    
    def compare_to_industry(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare ratios to industry benchmarks"""
        
        industry = data.get('industry', 'Technology')
        ratios = self.calculate_all_ratios(data)
        
        # Industry benchmarks
        industry_benchmarks = {
            'Technology': {
                'current_ratio': 2.5,
                'debt_to_equity': 0.3,
                'equity_ratio': 0.7
            },
            'Manufacturing': {
                'current_ratio': 1.8,
                'debt_to_equity': 0.7,
                'equity_ratio': 0.5
            },
            'Healthcare': {
                'current_ratio': 2.2,
                'debt_to_equity': 0.4,
                'equity_ratio': 0.6
            },
            'Retail': {
                'current_ratio': 1.5,
                'debt_to_equity': 0.8,
                'equity_ratio': 0.45
            }
        }
        
        benchmarks = industry_benchmarks.get(industry, industry_benchmarks['Technology'])
        
        comparison = {}
        for ratio_name, benchmark_value in benchmarks.items():
            actual_value = ratios.get(ratio_name, 0)
            comparison[ratio_name] = {
                'actual': actual_value,
                'benchmark': benchmark_value,
                'variance': actual_value - benchmark_value,
                'variance_percent': ((actual_value - benchmark_value) / benchmark_value * 100) if benchmark_value != 0 else 0,
                'status': 'above' if actual_value > benchmark_value else 'below' if actual_value < benchmark_value else 'at'
            }
        
        return {
            'industry': industry,
            'benchmarks': benchmarks,
            'comparison': comparison
        }
    
    def analyze_trends(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends (placeholder for historical data)"""
        
        # This would require historical data to implement properly
        return {
            'trend_analysis': 'Not available - requires historical data',
            'growth_rates': {},
            'trend_indicators': []
        }
    
    def _get_liquidity_recommendations(self, status: str, ratios: Dict[str, float]) -> List[str]:
        """Get liquidity-specific recommendations"""
        
        recommendations = []
        
        if status == 'poor':
            recommendations.extend([
                "Improve cash flow management by accelerating receivables collection",
                "Consider reducing inventory levels to free up cash",
                "Negotiate extended payment terms with suppliers",
                "Explore short-term financing options if needed"
            ])
        elif status == 'adequate':
            recommendations.extend([
                "Monitor cash flow closely to maintain adequate liquidity",
                "Build cash reserves for unexpected expenses",
                "Optimize working capital management"
            ])
        elif status == 'excellent':
            recommendations.extend([
                "Consider investing excess cash in growth opportunities",
                "Evaluate whether cash levels are optimal or excessive"
            ])
        
        return recommendations
    
    def _get_leverage_recommendations(self, status: str, ratios: Dict[str, float]) -> List[str]:
        """Get leverage-specific recommendations"""
        
        recommendations = []
        
        if status == 'highly_leveraged':
            recommendations.extend([
                "Prioritize debt reduction to improve financial stability",
                "Consider equity financing for future capital needs",
                "Focus on improving profitability to service debt",
                "Review debt covenants and compliance requirements"
            ])
        elif status == 'conservative':
            recommendations.extend([
                "Consider moderate leverage to enhance returns if profitable opportunities exist",
                "Evaluate optimal capital structure for your industry"
            ])
        
        return recommendations
    
    def _get_efficiency_recommendations(self, status: str, ratios: Dict[str, float]) -> List[str]:
        """Get efficiency-specific recommendations"""
        
        recommendations = []
        
        if status == 'inefficient':
            recommendations.extend([
                "Review asset utilization and dispose of underperforming assets",
                "Improve working capital management",
                "Consider operational efficiency improvements"
            ])
        elif status == 'efficient':
            recommendations.extend([
                "Maintain current efficiency levels",
                "Look for opportunities to scale efficiently"
            ])
        
        return recommendations

class BalanceSheetValidator:
    """Validate balance sheet data integrity and consistency"""
    
    def __init__(self):
        self.tolerance = 0.01  # 1% tolerance for calculations
        
    def validate_balance_sheet(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive balance sheet validation"""
        
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'score': 100.0
        }
        
        # Validate balance equation
        balance_valid = self._validate_balance_equation(data)
        if not balance_valid['is_valid']:
            validation_result['errors'].append(balance_valid['error'])
            validation_result['is_valid'] = False
            validation_result['score'] -= 25
        
        # Validate individual components
        component_validation = self._validate_components(data)
        validation_result['errors'].extend(component_validation['errors'])
        validation_result['warnings'].extend(component_validation['warnings'])
        validation_result['score'] -= len(component_validation['errors']) * 10
        validation_result['score'] -= len(component_validation['warnings']) * 5
        
        # Validate reasonableness
        reasonableness_validation = self._validate_reasonableness(data)
        validation_result['warnings'].extend(reasonableness_validation['warnings'])
        validation_result['score'] -= len(reasonableness_validation['warnings']) * 3
        
        # Ensure score doesn't go below 0
        validation_result['score'] = max(0, validation_result['score'])
        
        return validation_result
    
    def _validate_balance_equation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Assets = Liabilities + Equity"""
        
        total_assets = data.get('total_assets', 0)
        total_liabilities = data.get('total_liabilities', 0)
        total_equity = data.get('total_equity', 0)
        
        if total_assets == 0:
            return {
                'is_valid': False,
                'error': 'Total assets cannot be zero'
            }
        
        liab_equity_sum = total_liabilities + total_equity
        difference = abs(total_assets - liab_equity_sum)
        tolerance_amount = total_assets * self.tolerance
        
        if difference > tolerance_amount:
            return {
                'is_valid': False,
                'error': f'Balance equation error: Assets (${total_assets:,.0f}) ≠ Liabilities + Equity (${liab_equity_sum:,.0f}). Difference: ${difference:,.0f}'
            }
        
        return {'is_valid': True}
    
    def _validate_components(self, data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate individual balance sheet components"""
        
        errors = []
        warnings = []
        
        # Validate current assets components
        current_assets_sum = (
            data.get('cash_and_equivalents', 0) +
            data.get('accounts_receivable', 0) +
            data.get('inventory', 0) +
            data.get('prepaid_expenses', 0) +
            data.get('other_current_assets', 0)
        )
        
        current_assets_reported = data.get('current_assets', 0)
        if abs(current_assets_sum - current_assets_reported) > current_assets_reported * self.tolerance:
            errors.append(f'Current assets components (${current_assets_sum:,.0f}) do not sum to total current assets (${current_assets_reported:,.0f})')
        
        # Validate non-current assets components
        non_current_assets_sum = (
            data.get('net_ppe', 0) +
            data.get('intangible_assets', 0) +
            data.get('goodwill', 0) +
            data.get('investments', 0) +
            data.get('other_non_current_assets', 0)
        )
        
        non_current_assets_reported = data.get('non_current_assets', 0)
        if abs(non_current_assets_sum - non_current_assets_reported) > non_current_assets_reported * self.tolerance:
            errors.append(f'Non-current assets components do not sum to total non-current assets')
        
        # Validate total assets
        total_assets_sum = current_assets_reported + non_current_assets_reported
        total_assets_reported = data.get('total_assets', 0)
        if abs(total_assets_sum - total_assets_reported) > total_assets_reported * self.tolerance:
            errors.append(f'Current and non-current assets do not sum to total assets')
        
        # Validate liabilities components
        current_liabilities_sum = (
            data.get('accounts_payable', 0) +
            data.get('short_term_debt', 0) +
            data.get('accrued_liabilities', 0) +
            data.get('deferred_revenue', 0) +
            data.get('other_current_liabilities', 0)
        )
        
        current_liabilities_reported = data.get('current_liabilities', 0)
        if abs(current_liabilities_sum - current_liabilities_reported) > current_liabilities_reported * self.tolerance:
            warnings.append('Current liabilities components may not sum correctly to total current liabilities')
        
        return {'errors': errors, 'warnings': warnings}
    
    def _validate_reasonableness(self, data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate reasonableness of values"""
        
        warnings = []
        
        # Check for negative values that shouldn't be negative
        negative_checks = [
            ('total_assets', 'Total assets'),
            ('current_assets', 'Current assets'),
            ('cash_and_equivalents', 'Cash and equivalents'),
            ('accounts_receivable', 'Accounts receivable'),
            ('inventory', 'Inventory')
        ]
        
        for field, description in negative_checks:
            if data.get(field, 0) < 0:
                warnings.append(f'{description} is negative, which may indicate data issues')
        
        # Check for unreasonable ratios
        current_ratio = data.get('current_assets', 0) / max(data.get('current_liabilities', 1), 1)
        if current_ratio > 10:
            warnings.append(f'Current ratio ({current_ratio:.2f}) is unusually high')
        elif current_ratio < 0.1:
            warnings.append(f'Current ratio ({current_ratio:.2f}) is unusually low')
        
        # Check for excessive debt
        debt_to_equity = (data.get('short_term_debt', 0) + data.get('long_term_debt', 0)) / max(data.get('total_equity', 1), 1)
        if debt_to_equity > 5:
            warnings.append(f'Debt-to-equity ratio ({debt_to_equity:.2f}) is very high')
        
        return {'warnings': warnings}

# Standalone utility functions
def calculate_balance_sheet_ratios(data: Dict[str, Any]) -> Dict[str, float]:
    """Calculate balance sheet financial ratios"""
    analyzer = BalanceSheetAnalyzer()
    return analyzer.calculate_all_ratios(data)

def validate_balance_sheet_equation(data: Dict[str, Any]) -> bool:
    """Validate balance sheet equation"""
    validator = BalanceSheetValidator()
    result = validator._validate_balance_equation(data)
    return result['is_valid']

def generate_balance_sheet_insights(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate AI-powered insights for balance sheet"""
    
    insights = []
    analyzer = BalanceSheetAnalyzer()
    
    # Get analysis results
    liquidity_analysis = analyzer.analyze_liquidity(data)
    leverage_analysis = analyzer.analyze_leverage(data)
    health_assessment = analyzer.assess_financial_health(data)
    risks = analyzer.identify_risk_indicators(data)
    
    # Generate liquidity insights
    if liquidity_analysis['status'] == 'excellent':
        insights.append({
            'type': 'positive',
            'category': 'liquidity',
            'title': 'Strong Liquidity Position',
            'description': f"Current ratio of {liquidity_analysis['current_ratio']:.2f} indicates excellent ability to meet short-term obligations."
        })
    elif liquidity_analysis['status'] == 'poor':
        insights.append({
            'type': 'warning',
            'category': 'liquidity',
            'title': 'Liquidity Concerns',
            'description': f"Current ratio of {liquidity_analysis['current_ratio']:.2f} suggests potential difficulty meeting short-term obligations."
        })
    
    # Generate leverage insights
    if leverage_analysis['status'] == 'conservative':
        insights.append({
            'type': 'positive',
            'category': 'leverage',
            'title': 'Conservative Capital Structure',
            'description': f"Debt-to-equity ratio of {leverage_analysis['debt_to_equity']:.2f} indicates conservative financial management."
        })
    elif leverage_analysis['status'] == 'highly_leveraged':
        insights.append({
            'type': 'warning',
            'category': 'leverage',
            'title': 'High Financial Leverage',
            'description': f"Debt-to-equity ratio of {leverage_analysis['debt_to_equity']:.2f} indicates high financial risk."
        })
    
    # Generate overall health insights
    insights.append({
        'type': 'info',
        'category': 'overall',
        'title': f'Financial Health: {health_assessment["health_status"].title()}',
        'description': f"Overall financial health score of {health_assessment['health_score']:.0f}/100 with grade {health_assessment['health_grade']}."
    })
    
    # Add risk-based insights
    for risk in risks:
        insights.append({
            'type': 'warning',
            'category': 'risk',
            'title': f'Risk Alert: {risk["type"].replace("_", " ").title()}',
            'description': risk['description']
        })
    
    return insights

# Initialize processors
balance_sheet_processor = BalanceSheetProcessor()
balance_sheet_analyzer = BalanceSheetAnalyzer()
balance_sheet_validator = BalanceSheetValidator()