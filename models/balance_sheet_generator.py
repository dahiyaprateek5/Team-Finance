import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import psycopg2
from psycopg2.extras import RealDictCursor
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BalanceSheetGenerator:
    def __init__(self, db_config):
        """
        Initialize Balance Sheet Generator
        
        Args:
            db_config (dict): Database configuration
        """
        self.db_config = db_config
        
        # Industry ratios for estimation
        self.industry_ratios = {
            'technology': {
                'cash_to_assets': 0.15,
                'receivables_to_assets': 0.12,
                'inventory_to_assets': 0.05,
                'current_assets_to_assets': 0.45,
                'ppe_to_assets': 0.25,
                'debt_to_assets': 0.20,
                'equity_to_assets': 0.65
            },
            'manufacturing': {
                'cash_to_assets': 0.08,
                'receivables_to_assets': 0.18,
                'inventory_to_assets': 0.25,
                'current_assets_to_assets': 0.55,
                'ppe_to_assets': 0.40,
                'debt_to_assets': 0.35,
                'equity_to_assets': 0.50
            },
            'retail': {
                'cash_to_assets': 0.06,
                'receivables_to_assets': 0.08,
                'inventory_to_assets': 0.35,
                'current_assets_to_assets': 0.60,
                'ppe_to_assets': 0.30,
                'debt_to_assets': 0.40,
                'equity_to_assets': 0.45
            },
            'healthcare': {
                'cash_to_assets': 0.12,
                'receivables_to_assets': 0.20,
                'inventory_to_assets': 0.15,
                'current_assets_to_assets': 0.50,
                'ppe_to_assets': 0.35,
                'debt_to_assets': 0.25,
                'equity_to_assets': 0.60
            },
            'financial': {
                'cash_to_assets': 0.25,
                'receivables_to_assets': 0.05,
                'inventory_to_assets': 0.01,
                'current_assets_to_assets': 0.35,
                'ppe_to_assets': 0.15,
                'debt_to_assets': 0.70,
                'equity_to_assets': 0.25
            }
        }

    def connect_db(self):
        """Establish database connection"""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except Exception as e:
            print(f"Database connection error: {e}")
            return None

    def clean_financial_value(self, value):
        """Clean and convert financial values to numeric"""
        if pd.isna(value) or value == '' or value is None:
            return np.nan
            
        if isinstance(value, (int, float)):
            return float(value)
            
        value_str = str(value).strip()
        value_str = re.sub(r'[,$\s]', '', value_str)
        value_str = re.sub(r'[()]', '-', value_str)
        
        # Handle multipliers
        multipliers = {'K': 1000, 'M': 1000000, 'B': 1000000000}
        for suffix, multiplier in multipliers.items():
            if value_str.upper().endswith(suffix):
                try:
                    return float(value_str[:-1]) * multiplier
                except:
                    return np.nan
        
        try:
            return float(value_str)
        except:
            return np.nan

    def detect_balance_sheet_structure(self, df):
        """Detect the structure of uploaded balance sheet"""
        structure = {
            'format_type': 'unknown',
            'year_columns': [],
            'item_column': None,
            'data_orientation': 'vertical'  # vertical or horizontal
        }
        
        # Find year columns
        year_pattern = r'20\d{2}'
        for col in df.columns:
            if re.search(year_pattern, str(col)):
                structure['year_columns'].append(col)
        
        # Find item/description column
        text_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if this column contains financial statement items
                sample_values = df[col].dropna().astype(str).str.lower()
                financial_keywords = ['asset', 'liability', 'equity', 'cash', 'receivable', 'inventory', 'debt']
                keyword_count = sum(1 for val in sample_values if any(keyword in val for keyword in financial_keywords))
                
                if keyword_count > len(sample_values) * 0.3:  # 30% threshold
                    text_columns.append((col, keyword_count))
        
        if text_columns:
            structure['item_column'] = max(text_columns, key=lambda x: x[1])[0]
        
        # Determine format type
        if len(structure['year_columns']) > 0 and structure['item_column']:
            structure['format_type'] = 'standard'
        elif len(df.columns) == 2 and df.shape[0] > 10:
            structure['format_type'] = 'simple_two_column'
        
        return structure

    def extract_balance_sheet_data(self, df, structure):
        """Extract balance sheet data based on detected structure"""
        extracted_data = {}
        
        if structure['format_type'] == 'standard':
            # Standard format with item column and year columns
            item_col = structure['item_column']
            
            for year_col in structure['year_columns']:
                year_data = {}
                
                for idx, row in df.iterrows():
                    item_name = str(row[item_col]).lower().strip()
                    value = self.clean_financial_value(row[year_col])
                    
                    # Map to standard balance sheet items
                    mapped_item = self.map_balance_sheet_item(item_name)
                    if mapped_item:
                        year_data[mapped_item] = value
                
                # Extract year from column name
                year_match = re.search(r'20\d{2}', str(year_col))
                year = int(year_match.group()) if year_match else datetime.now().year
                
                extracted_data[year] = year_data
        
        elif structure['format_type'] == 'simple_two_column':
            # Simple two-column format
            item_col = df.columns[0]
            value_col = df.columns[1]
            
            year_data = {}
            for idx, row in df.iterrows():
                item_name = str(row[item_col]).lower().strip()
                value = self.clean_financial_value(row[value_col])
                
                mapped_item = self.map_balance_sheet_item(item_name)
                if mapped_item:
                    year_data[mapped_item] = value
            
            # Use current year as default
            current_year = datetime.now().year
            extracted_data[current_year] = year_data
        
        return extracted_data

    def map_balance_sheet_item(self, item_name):
        """Map item names to standard balance sheet categories"""
        item_mapping = {
            # Assets
            'cash': 'cash_and_equivalents',
            'cash and cash equivalents': 'cash_and_equivalents',
            'cash & equivalents': 'cash_and_equivalents',
            'liquid assets': 'cash_and_equivalents',
            'bank': 'cash_and_equivalents',
            'cash at bank': 'cash_and_equivalents',
            
            'accounts receivable': 'accounts_receivable',
            'receivables': 'accounts_receivable',
            'trade receivables': 'accounts_receivable',
            'debtors': 'accounts_receivable',
            'trade debtors': 'accounts_receivable',
            
            'inventory': 'inventory',
            'stock': 'inventory',
            'inventories': 'inventory',
            'finished goods': 'inventory',
            'raw materials': 'inventory',
            
            'prepaid expenses': 'prepaid_expenses',
            'prepaid': 'prepaid_expenses',
            'prepaids': 'prepaid_expenses',
            
            'current assets': 'current_assets',
            'total current assets': 'current_assets',
            'short term assets': 'current_assets',
            
            'property plant equipment': 'property_plant_equipment',
            'ppe': 'property_plant_equipment',
            'fixed assets': 'property_plant_equipment',
            'plant and equipment': 'property_plant_equipment',
            'property, plant & equipment': 'property_plant_equipment',
            'tangible assets': 'property_plant_equipment',
            
            'accumulated depreciation': 'accumulated_depreciation',
            'depreciation': 'accumulated_depreciation',
            'accumulated amortization': 'accumulated_depreciation',
            
            'intangible assets': 'intangible_assets',
            'intangibles': 'intangible_assets',
            
            'goodwill': 'goodwill',
            
            'investments': 'investments',
            'long term investments': 'investments',
            'securities': 'investments',
            
            'non current assets': 'non_current_assets',
            'non-current assets': 'non_current_assets',
            'long term assets': 'non_current_assets',
            
            'total assets': 'total_assets',
            'assets total': 'total_assets',
            'sum of assets': 'total_assets',
            
            # Liabilities
            'accounts payable': 'accounts_payable',
            'payables': 'accounts_payable',
            'trade payables': 'accounts_payable',
            'creditors': 'accounts_payable',
            'trade creditors': 'accounts_payable',
            
            'short term debt': 'short_term_debt',
            'current debt': 'short_term_debt',
            'short-term borrowings': 'short_term_debt',
            'bank overdraft': 'short_term_debt',
            'current portion of long term debt': 'short_term_debt',
            
            'accrued liabilities': 'accrued_liabilities',
            'accruals': 'accrued_liabilities',
            'accrued expenses': 'accrued_liabilities',
            
            'deferred revenue': 'deferred_revenue',
            'unearned revenue': 'deferred_revenue',
            'advance payments': 'deferred_revenue',
            
            'current liabilities': 'current_liabilities',
            'total current liabilities': 'current_liabilities',
            'short term liabilities': 'current_liabilities',
            
            'long term debt': 'long_term_debt',
            'long-term debt': 'long_term_debt',
            'non-current debt': 'long_term_debt',
            'long term borrowings': 'long_term_debt',
            'term loans': 'long_term_debt',
            
            'deferred tax liabilities': 'deferred_tax_liabilities',
            'deferred tax': 'deferred_tax_liabilities',
            
            'pension obligations': 'pension_obligations',
            'pension liabilities': 'pension_obligations',
            'retirement benefits': 'pension_obligations',
            
            'non current liabilities': 'non_current_liabilities',
            'non-current liabilities': 'non_current_liabilities',
            'long term liabilities': 'non_current_liabilities',
            
            'total liabilities': 'total_liabilities',
            'liabilities total': 'total_liabilities',
            'sum of liabilities': 'total_liabilities',
            
            # Equity
            'share capital': 'share_capital',
            'capital stock': 'share_capital',
            'common stock': 'share_capital',
            'ordinary shares': 'share_capital',
            'issued capital': 'share_capital',
            
            'retained earnings': 'retained_earnings',
            'accumulated profits': 'retained_earnings',
            'reserves': 'retained_earnings',
            'profit and loss account': 'retained_earnings',
            
            'additional paid in capital': 'additional_paid_in_capital',
            'share premium': 'additional_paid_in_capital',
            'capital surplus': 'additional_paid_in_capital',
            
            'treasury stock': 'treasury_stock',
            'treasury shares': 'treasury_stock',
            
            'total equity': 'total_equity',
            'shareholders equity': 'total_equity',
            'stockholders equity': 'total_equity',
            "owner's equity": 'total_equity',
            'equity total': 'total_equity'
        }
        
        # Find best match
        for key, value in item_mapping.items():
            if key in item_name:
                return value
        
        return None

    def estimate_missing_components(self, balance_sheet_data, industry='manufacturing'):
        """Estimate missing balance sheet components using industry ratios"""
        if industry not in self.industry_ratios:
            industry = 'manufacturing'  # Default
        
        ratios = self.industry_ratios[industry]
        
        # If total assets is known, estimate other components
        if 'total_assets' in balance_sheet_data and pd.notna(balance_sheet_data['total_assets']):
            total_assets = balance_sheet_data['total_assets']
            
            # Estimate missing assets
            if pd.isna(balance_sheet_data.get('cash_and_equivalents')):
                balance_sheet_data['cash_and_equivalents'] = total_assets * ratios['cash_to_assets']
            
            if pd.isna(balance_sheet_data.get('accounts_receivable')):
                balance_sheet_data['accounts_receivable'] = total_assets * ratios['receivables_to_assets']
            
            if pd.isna(balance_sheet_data.get('inventory')):
                balance_sheet_data['inventory'] = total_assets * ratios['inventory_to_assets']
            
            if pd.isna(balance_sheet_data.get('property_plant_equipment')):
                balance_sheet_data['property_plant_equipment'] = total_assets * ratios['ppe_to_assets']
            
            # Estimate current assets if missing
            if pd.isna(balance_sheet_data.get('current_assets')):
                current_components = [
                    balance_sheet_data.get('cash_and_equivalents', 0),
                    balance_sheet_data.get('accounts_receivable', 0),
                    balance_sheet_data.get('inventory', 0),
                    balance_sheet_data.get('prepaid_expenses', 0)
                ]
                balance_sheet_data['current_assets'] = sum([x for x in current_components if pd.notna(x)])
        
        # If current assets is known but total assets is missing
        elif 'current_assets' in balance_sheet_data and pd.notna(balance_sheet_data['current_assets']):
            current_assets = balance_sheet_data['current_assets']
            estimated_total_assets = current_assets / ratios['current_assets_to_assets']
            
            if pd.isna(balance_sheet_data.get('total_assets')):
                balance_sheet_data['total_assets'] = estimated_total_assets
            
            if pd.isna(balance_sheet_data.get('property_plant_equipment')):
                balance_sheet_data['property_plant_equipment'] = estimated_total_assets * ratios['ppe_to_assets']
        
        # Estimate liabilities and equity
        total_assets = balance_sheet_data.get('total_assets', 0)
        if total_assets > 0:
            if pd.isna(balance_sheet_data.get('total_liabilities')):
                balance_sheet_data['total_liabilities'] = total_assets * ratios['debt_to_assets']
            
            if pd.isna(balance_sheet_data.get('total_equity')):
                total_liabilities = balance_sheet_data.get('total_liabilities', 0)
                balance_sheet_data['total_equity'] = total_assets - total_liabilities
            
            # Estimate individual liability components
            if pd.isna(balance_sheet_data.get('accounts_payable')):
                balance_sheet_data['accounts_payable'] = total_assets * 0.08  # 8% of assets
            
            if pd.isna(balance_sheet_data.get('short_term_debt')):
                balance_sheet_data['short_term_debt'] = total_assets * 0.05  # 5% of assets
            
            if pd.isna(balance_sheet_data.get('long_term_debt')):
                balance_sheet_data['long_term_debt'] = total_assets * 0.15  # 15% of assets
        
        return balance_sheet_data

    def generate_complete_balance_sheet(self, company_id, year, industry='manufacturing', 
                                      known_data=None, bank_statements=None):
        """
        Generate complete balance sheet with minimal input data
        
        Args:
            company_id: Company identifier
            year: Year for balance sheet
            industry: Company industry
            known_data: Dictionary of known financial data
            bank_statements: Bank statement data for cash analysis
            
        Returns:
            dict: Complete balance sheet data with exact columns from balance_sheet_1 table
        """
        # Initialize balance sheet structure with exact columns from balance_sheet_1 table
        balance_sheet = {
            'company_id': company_id,
            'year': year,
            'generated_at': datetime.now(),
            'current_assets': np.nan,
            'cash_and_equivalents': np.nan,
            'accounts_receivable': np.nan,
            'inventory': np.nan,
            'prepaid_expenses': np.nan,
            'other_current_assets': np.nan,
            'non_current_assets': np.nan,
            'property_plant_equipment': np.nan,
            'accumulated_depreciation': np.nan,
            'net_ppe': np.nan,
            'intangible_assets': np.nan,
            'goodwill': np.nan,
            'investments': np.nan,
            'other_non_current_assets': np.nan,
            'total_assets': np.nan,
            'current_liabilities': np.nan,
            'accounts_payable': np.nan,
            'short_term_debt': np.nan,
            'accrued_liabilities': np.nan,
            'deferred_revenue': np.nan,
            'other_current_liabilities': np.nan,
            'non_current_liabilities': np.nan,
            'long_term_debt': np.nan,
            'deferred_tax_liabilities': np.nan,
            'pension_obligations': np.nan,
            'other_non_current_liabilities': np.nan,
            'total_liabilities': np.nan,
            'share_capital': np.nan,
            'retained_earnings': np.nan,
            'additional_paid_in_capital': np.nan,
            'treasury_stock': np.nan,
            'accumulated_other_comprehensive_income': np.nan,
            'total_equity': np.nan,
            'balance_check': np.nan,
            'accuracy_percentage': np.nan,
            'data_source': 'generated',
            'validation_errors': None
        }
        
        # Update with known data
        if known_data:
            for key, value in known_data.items():
                if key in balance_sheet:
                    balance_sheet[key] = self.clean_financial_value(value)
        
        # Analyze bank statements if provided
        if bank_statements:
            cash_analysis = self.analyze_bank_statements(bank_statements)
            balance_sheet['cash_and_equivalents'] = cash_analysis.get('ending_balance', np.nan)
        
        # Estimate missing components
        balance_sheet = self.estimate_missing_components(balance_sheet, industry)
        
        # Calculate derived values
        balance_sheet = self.calculate_derived_balance_sheet_values(balance_sheet)
        
        # Calculate accuracy score
        balance_sheet = self.calculate_accuracy_score(balance_sheet, known_data)
        
        return balance_sheet

    def analyze_bank_statements(self, bank_statements):
        """Analyze bank statements to extract cash flow patterns"""
        analysis = {
            'ending_balance': 0,
            'average_balance': 0,
            'cash_inflows': 0,
            'cash_outflows': 0,
            'net_cash_flow': 0
        }
        
        if isinstance(bank_statements, pd.DataFrame):
            # Assume bank statements have columns: Date, Description, Amount, Balance
            if 'Balance' in bank_statements.columns:
                analysis['ending_balance'] = bank_statements['Balance'].iloc[-1] if len(bank_statements) > 0 else 0
                analysis['average_balance'] = bank_statements['Balance'].mean()
            
            if 'Amount' in bank_statements.columns:
                positive_amounts = bank_statements[bank_statements['Amount'] > 0]['Amount'].sum()
                negative_amounts = bank_statements[bank_statements['Amount'] < 0]['Amount'].sum()
                
                analysis['cash_inflows'] = positive_amounts
                analysis['cash_outflows'] = abs(negative_amounts)
                analysis['net_cash_flow'] = positive_amounts + negative_amounts
        
        return analysis

    def calculate_derived_balance_sheet_values(self, balance_sheet):
        """Calculate derived values and ensure balance sheet balances"""
        
        # Calculate current assets total
        current_asset_components = [
            balance_sheet.get('cash_and_equivalents', 0) or 0,
            balance_sheet.get('accounts_receivable', 0) or 0,
            balance_sheet.get('inventory', 0) or 0,
            balance_sheet.get('prepaid_expenses', 0) or 0,
            balance_sheet.get('other_current_assets', 0) or 0
        ]
        
        if pd.isna(balance_sheet.get('current_assets')):
            balance_sheet['current_assets'] = sum([x for x in current_asset_components if pd.notna(x)])
        
        # Calculate non-current assets
        non_current_components = [
            balance_sheet.get('property_plant_equipment', 0) or 0,
            balance_sheet.get('intangible_assets', 0) or 0,
            balance_sheet.get('goodwill', 0) or 0,
            balance_sheet.get('investments', 0) or 0,
            balance_sheet.get('other_non_current_assets', 0) or 0
        ]
        
        if pd.isna(balance_sheet.get('non_current_assets')):
            balance_sheet['non_current_assets'] = sum([x for x in non_current_components if pd.notna(x)])
        
        # Calculate net PPE
        ppe = balance_sheet.get('property_plant_equipment', 0) or 0
        acc_dep = balance_sheet.get('accumulated_depreciation', 0) or 0
        balance_sheet['net_ppe'] = ppe - abs(acc_dep)
        
        # Calculate total assets
        if pd.isna(balance_sheet.get('total_assets')):
            balance_sheet['total_assets'] = balance_sheet.get('current_assets', 0) + balance_sheet.get('non_current_assets', 0)
        
        # Calculate current liabilities total
        current_liab_components = [
            balance_sheet.get('accounts_payable', 0) or 0,
            balance_sheet.get('short_term_debt', 0) or 0,
            balance_sheet.get('accrued_liabilities', 0) or 0,
            balance_sheet.get('deferred_revenue', 0) or 0,
            balance_sheet.get('other_current_liabilities', 0) or 0
        ]
        
        if pd.isna(balance_sheet.get('current_liabilities')):
            balance_sheet['current_liabilities'] = sum([x for x in current_liab_components if pd.notna(x)])
        
        # Calculate non-current liabilities
        non_current_liab_components = [
            balance_sheet.get('long_term_debt', 0) or 0,
            balance_sheet.get('deferred_tax_liabilities', 0) or 0,
            balance_sheet.get('pension_obligations', 0) or 0,
            balance_sheet.get('other_non_current_liabilities', 0) or 0
        ]
        
        if pd.isna(balance_sheet.get('non_current_liabilities')):
            balance_sheet['non_current_liabilities'] = sum([x for x in non_current_liab_components if pd.notna(x)])
        
        # Calculate total liabilities
        if pd.isna(balance_sheet.get('total_liabilities')):
            balance_sheet['total_liabilities'] = balance_sheet.get('current_liabilities', 0) + balance_sheet.get('non_current_liabilities', 0)
        
        # Calculate total equity
        if pd.isna(balance_sheet.get('total_equity')):
            balance_sheet['total_equity'] = balance_sheet.get('total_assets', 0) - balance_sheet.get('total_liabilities', 0)
        
        # Balance check
        total_assets = balance_sheet.get('total_assets', 0)
        total_liab_equity = balance_sheet.get('total_liabilities', 0) + balance_sheet.get('total_equity', 0)
        
        if total_assets > 0:
            balance_sheet['balance_check'] = abs(total_assets - total_liab_equity) / total_assets
        else:
            balance_sheet['balance_check'] = 1.0
        
        return balance_sheet

    def calculate_accuracy_score(self, balance_sheet, known_data):
        """Calculate accuracy percentage based on available data"""
        total_fields = 35  # Total number of balance sheet fields
        known_fields = 0
        
        if known_data:
            known_fields = len([v for v in known_data.values() if pd.notna(v)])
        
        # Additional points for calculated fields
        calculated_fields = 0
        if pd.notna(balance_sheet.get('total_assets')):
            calculated_fields += 1
        if pd.notna(balance_sheet.get('total_liabilities')):
            calculated_fields += 1
        if pd.notna(balance_sheet.get('total_equity')):
            calculated_fields += 1
        
        # Balance check penalty
        balance_penalty = 0
        if balance_sheet.get('balance_check', 1) > 0.05:  # 5% threshold
            balance_penalty = 20
        
        # Calculate accuracy percentage
        base_accuracy = ((known_fields + calculated_fields) / total_fields) * 100
        accuracy = max(0, min(100, base_accuracy - balance_penalty))
        
        balance_sheet['accuracy_percentage'] = round(accuracy, 2)
        
        return balance_sheet

    def process_uploaded_balance_sheet(self, df, company_id, company_name, industry=None):
        """
        Process uploaded balance sheet file
        
        Args:
            df: DataFrame with balance sheet data
            company_id: Company identifier
            company_name: Company name
            industry: Company industry
            
        Returns:
            dict: Processed balance sheet data
        """
        # Detect structure
        structure = self.detect_balance_sheet_structure(df)
        
        if structure['format_type'] == 'unknown':
            print("Could not detect balance sheet structure")
            return None
        
        # Extract data
        extracted_data = self.extract_balance_sheet_data(df, structure)
        
        if not extracted_data:
            print("No balance sheet data could be extracted")
            return None
        
        # Get the most recent year's data
        latest_year = max(extracted_data.keys())
        year_data = extracted_data[latest_year]
        
        # Generate complete balance sheet
        complete_balance_sheet = self.generate_complete_balance_sheet(
            company_id=company_id,
            year=latest_year,
            industry=industry,
            known_data=year_data
        )
        
        return complete_balance_sheet

    def save_balance_sheet(self, balance_sheet):
        """Save generated balance sheet to balance_sheet_1 table"""
        conn = self.connect_db()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            # Prepare insert query for balance_sheet_1 table
            columns = list(balance_sheet.keys())
            placeholders = ', '.join(['%s'] * len(columns))
            columns_str = ', '.join(columns)
            
            query = f"""
            INSERT INTO balance_sheet_1 ({columns_str})
            VALUES ({placeholders})
            ON CONFLICT (company_id, year) 
            DO UPDATE SET
                {', '.join([f"{col} = EXCLUDED.{col}" for col in columns if col not in ['id', 'company_id', 'year']])}
            """
            
            values = [balance_sheet[col] for col in columns]
            cursor.execute(query, values)
            
            conn.commit()
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error saving balance sheet to balance_sheet_1 table: {e}")
            if conn:
                conn.rollback()
                conn.close()
            return False

    def validate_balance_sheet(self, balance_sheet):
        """
        Validate balance sheet for consistency and accuracy
        
        Args:
            balance_sheet: Balance sheet data dictionary
            
        Returns:
            dict: Validation results
        """
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check balance equation: Assets = Liabilities + Equity
        total_assets = balance_sheet.get('total_assets', 0) or 0
        total_liabilities = balance_sheet.get('total_liabilities', 0) or 0
        total_equity = balance_sheet.get('total_equity', 0) or 0
        
        balance_difference = abs(total_assets - (total_liabilities + total_equity))
        
        if total_assets > 0:
            balance_percentage = (balance_difference / total_assets) * 100
            
            if balance_percentage > 5:
                validation_results['errors'].append(
                    f"Balance sheet does not balance: {balance_percentage:.1f}% difference"
                )
                validation_results['is_valid'] = False
            elif balance_percentage > 1:
                validation_results['warnings'].append(
                    f"Minor balance difference: {balance_percentage:.1f}%"
                )
        
        # Check for negative values where they shouldn't occur
        negative_checks = [
            'cash_and_equivalents', 'accounts_receivable', 'inventory',
            'total_assets', 'accounts_payable', 'total_liabilities'
        ]
        
        for field in negative_checks:
            value = balance_sheet.get(field, 0) or 0
            if value < 0:
                validation_results['warnings'].append(
                    f"{field} has negative value: {value}"
                )
        
        # Check ratios for reasonableness
        if total_assets > 0:
            current_assets = balance_sheet.get('current_assets', 0) or 0
            current_liabilities = balance_sheet.get('current_liabilities', 0) or 0
            
            # Current ratio check
            if current_liabilities > 0:
                current_ratio = current_assets / current_liabilities
                if current_ratio < 0.5:
                    validation_results['warnings'].append(
                        f"Very low current ratio: {current_ratio:.2f}"
                    )
                elif current_ratio > 10:
                    validation_results['warnings'].append(
                        f"Unusually high current ratio: {current_ratio:.2f}"
                    )
            
            # Debt to equity ratio check
            if total_equity > 0:
                debt_to_equity = total_liabilities / total_equity
                if debt_to_equity > 5:
                    validation_results['warnings'].append(
                        f"Very high debt to equity ratio: {debt_to_equity:.2f}"
                    )
        
        return validation_results

    def get_industry_benchmark_comparison(self, balance_sheet, industry):
        """
        Compare balance sheet against industry benchmarks
        
        Args:
            balance_sheet: Balance sheet data
            industry: Industry category
            
        Returns:
            dict: Benchmark comparison results
        """
        if industry not in self.industry_ratios:
            return {'error': f'No benchmarks available for industry: {industry}'}
        
        benchmarks = self.industry_ratios[industry]
        total_assets = balance_sheet.get('total_assets', 1) or 1
        
        comparison = {}
        
        # Calculate actual ratios
        actual_ratios = {
            'cash_to_assets': (balance_sheet.get('cash_and_equivalents', 0) or 0) / total_assets,
            'receivables_to_assets': (balance_sheet.get('accounts_receivable', 0) or 0) / total_assets,
            'inventory_to_assets': (balance_sheet.get('inventory', 0) or 0) / total_assets,
            'current_assets_to_assets': (balance_sheet.get('current_assets', 0) or 0) / total_assets,
            'ppe_to_assets': (balance_sheet.get('property_plant_equipment', 0) or 0) / total_assets,
            'debt_to_assets': (balance_sheet.get('total_liabilities', 0) or 0) / total_assets,
            'equity_to_assets': (balance_sheet.get('total_equity', 0) or 0) / total_assets
        }
        
        # Compare with benchmarks
        for ratio_name, actual_value in actual_ratios.items():
            benchmark_value = benchmarks.get(ratio_name, 0)
            
            if benchmark_value > 0:
                variance = ((actual_value - benchmark_value) / benchmark_value) * 100
                
                comparison[ratio_name] = {
                    'actual': actual_value,
                    'benchmark': benchmark_value,
                    'variance_percent': variance,
                    'status': self.get_variance_status(variance)
                }
        
        return comparison

    def get_variance_status(self, variance_percent):
        """Determine status based on variance from benchmark"""
        if abs(variance_percent) <= 20:
            return 'Within Range'
        elif variance_percent > 20:
            return 'Above Benchmark'
        else:
            return 'Below Benchmark'

    def bulk_generate_balance_sheets(self, companies_data):
        """
        Generate balance sheets for multiple companies
        
        Args:
            companies_data: List of dictionaries with company information
            
        Returns:
            dict: Results summary
        """
        results = {
            'total_companies': len(companies_data),
            'successful': 0,
            'failed': 0,
            'errors': []
        }
        
        for company_data in companies_data:
            try:
                company_id = company_data.get('company_id')
                year = company_data.get('year', datetime.now().year)
                industry = company_data.get('industry', 'manufacturing')
                known_data = company_data.get('known_data', {})
                
                if not company_id:
                    results['failed'] += 1
                    results['errors'].append('Missing company_id')
                    continue
                
                # Generate balance sheet
                balance_sheet = self.generate_complete_balance_sheet(
                    company_id=company_id,
                    year=year,
                    industry=industry,
                    known_data=known_data
                )
                
                # Save to database
                success = self.save_balance_sheet(balance_sheet)
                
                if success:
                    results['successful'] += 1
                else:
                    results['failed'] += 1
                    results['errors'].append(f'{company_id}: Failed to save to database')
                
            except Exception as e:
                results['failed'] += 1
                results['errors'].append(f'{company_data.get("company_id", "Unknown")}: {str(e)}')
        
        return results

    def export_balance_sheet_to_excel(self, balance_sheet, filepath):
        """
        Export balance sheet to Excel format
        
        Args:
            balance_sheet: Balance sheet data
            filepath: Output file path
            
        Returns:
            bool: Success status
        """
        try:
            # Create formatted DataFrame
            assets_data = [
                ['ASSETS', ''],
                ['Current Assets:', ''],
                ['Cash and Cash Equivalents', balance_sheet.get('cash_and_equivalents', 0)],
                ['Accounts Receivable', balance_sheet.get('accounts_receivable', 0)],
                ['Inventory', balance_sheet.get('inventory', 0)],
                ['Prepaid Expenses', balance_sheet.get('prepaid_expenses', 0)],
                ['Other Current Assets', balance_sheet.get('other_current_assets', 0)],
                ['Total Current Assets', balance_sheet.get('current_assets', 0)],
                ['', ''],
                ['Non-Current Assets:', ''],
                ['Property, Plant & Equipment', balance_sheet.get('property_plant_equipment', 0)],
                ['Accumulated Depreciation', -(balance_sheet.get('accumulated_depreciation', 0) or 0)],
                ['Net PPE', balance_sheet.get('net_ppe', 0)],
                ['Intangible Assets', balance_sheet.get('intangible_assets', 0)],
                ['Goodwill', balance_sheet.get('goodwill', 0)],
                ['Investments', balance_sheet.get('investments', 0)],
                ['Other Non-Current Assets', balance_sheet.get('other_non_current_assets', 0)],
                ['Total Non-Current Assets', balance_sheet.get('non_current_assets', 0)],
                ['', ''],
                ['TOTAL ASSETS', balance_sheet.get('total_assets', 0)],
                ['', ''],
                ['LIABILITIES & EQUITY', ''],
                ['Current Liabilities:', ''],
                ['Accounts Payable', balance_sheet.get('accounts_payable', 0)],
                ['Short-term Debt', balance_sheet.get('short_term_debt', 0)],
                ['Accrued Liabilities', balance_sheet.get('accrued_liabilities', 0)],
                ['Deferred Revenue', balance_sheet.get('deferred_revenue', 0)],
                ['Other Current Liabilities', balance_sheet.get('other_current_liabilities', 0)],
                ['Total Current Liabilities', balance_sheet.get('current_liabilities', 0)],
                ['', ''],
                ['Non-Current Liabilities:', ''],
                ['Long-term Debt', balance_sheet.get('long_term_debt', 0)],
                ['Deferred Tax Liabilities', balance_sheet.get('deferred_tax_liabilities', 0)],
                ['Pension Obligations', balance_sheet.get('pension_obligations', 0)],
                ['Other Non-Current Liabilities', balance_sheet.get('other_non_current_liabilities', 0)],
                ['Total Non-Current Liabilities', balance_sheet.get('non_current_liabilities', 0)],
                ['', ''],
                ['TOTAL LIABILITIES', balance_sheet.get('total_liabilities', 0)],
                ['', ''],
                ['Shareholders Equity:', ''],
                ['Share Capital', balance_sheet.get('share_capital', 0)],
                ['Additional Paid-in Capital', balance_sheet.get('additional_paid_in_capital', 0)],
                ['Retained Earnings', balance_sheet.get('retained_earnings', 0)],
                ['Treasury Stock', -(balance_sheet.get('treasury_stock', 0) or 0)],
                ['Accumulated Other Comprehensive Income', balance_sheet.get('accumulated_other_comprehensive_income', 0)],
                ['TOTAL EQUITY', balance_sheet.get('total_equity', 0)],
                ['', ''],
                ['TOTAL LIABILITIES & EQUITY', balance_sheet.get('total_liabilities', 0) + balance_sheet.get('total_equity', 0)]
            ]
            
            df = pd.DataFrame(assets_data, columns=['Item', f'Year {balance_sheet.get("year", datetime.now().year)}'])
            
            # Save to Excel
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Balance Sheet', index=False)
                
                # Add company information sheet
                info_data = [
                    ['Company ID', balance_sheet.get('company_id', '')],
                    ['Year', balance_sheet.get('year', '')],
                    ['Generated At', balance_sheet.get('generated_at', '')],
                    ['Data Source', balance_sheet.get('data_source', '')],
                    ['Accuracy Percentage', f"{balance_sheet.get('accuracy_percentage', 0):.1f}%"],
                    ['Balance Check', f"{balance_sheet.get('balance_check', 0):.3f}"],
                    ['Validation Errors', balance_sheet.get('validation_errors', 'None')]
                ]
                
                info_df = pd.DataFrame(info_data, columns=['Metric', 'Value'])
                info_df.to_excel(writer, sheet_name='Information', index=False)
            
            print(f"Balance sheet exported to {filepath}")
            return True
            
        except Exception as e:
            print(f"Error exporting balance sheet: {e}")
            return False