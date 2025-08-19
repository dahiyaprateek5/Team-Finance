import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import psycopg2
from psycopg2.extras import RealDictCursor
import pickle
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FinancialHealthModel:
    def __init__(self, db_config):
        """
        Initialize the Financial Health Model
        
        Args:
            db_config (dict): Database configuration containing host, database, user, password, port
        """
        self.db_config = db_config
        self.scaler = StandardScaler()
        self.knn_model = KNeighborsRegressor(n_neighbors=5)
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Column mappings for automatic detection
        self.balance_sheet_mappings = {
            'cash_and_equivalents': ['cash', 'cash and cash equivalents', 'cash & equivalents', 'liquid assets'],
            'accounts_receivable': ['accounts receivable', 'receivables', 'trade receivables', 'debtors'],
            'inventory': ['inventory', 'stock', 'inventories', 'goods'],
            'current_assets': ['current assets', 'total current assets', 'short term assets'],
            'property_plant_equipment': ['property plant equipment', 'ppe', 'fixed assets', 'tangible assets'],
            'total_assets': ['total assets', 'assets total', 'sum of assets'],
            'accounts_payable': ['accounts payable', 'payables', 'trade payables', 'creditors'],
            'short_term_debt': ['short term debt', 'current debt', 'short-term borrowings'],
            'current_liabilities': ['current liabilities', 'total current liabilities', 'short term liabilities'],
            'long_term_debt': ['long term debt', 'long-term debt', 'non-current debt'],
            'total_liabilities': ['total liabilities', 'liabilities total', 'sum of liabilities'],
            'share_capital': ['share capital', 'capital stock', 'common stock'],
            'retained_earnings': ['retained earnings', 'accumulated profits', 'reserves'],
            'total_equity': ['total equity', 'shareholders equity', 'stockholders equity', 'equity total']
        }
        
        self.cash_flow_mappings = {
            'net_income': ['net income', 'profit after tax', 'net profit', 'earnings'],
            'depreciation_and_amortization': ['depreciation', 'amortization', 'depreciation and amortization'],
            'accounts_receivable': ['accounts receivable', 'receivables change', 'trade receivables'],
            'inventory': ['inventory change', 'stock change', 'inventory'],
            'accounts_payable': ['accounts payable', 'payables change', 'trade payables'],
            'capital_expenditures': ['capital expenditures', 'capex', 'capital spending', 'investments in ppe'],
            'dividends_paid': ['dividends paid', 'dividend payments', 'cash dividends'],
            'net_cash_from_operating_activities': ['operating cash flow', 'cash from operations', 'operating activities'],
            'net_cash_from_investing_activities': ['investing cash flow', 'cash from investing', 'investing activities'],
            'net_cash_from_financing_activities': ['financing cash flow', 'cash from financing', 'financing activities']
        }
        
        # Industry benchmarks for missing value imputation
        self.industry_benchmarks = {
            'technology': {
                'current_ratio': 2.5, 'debt_to_equity': 0.3, 'profit_margin': 0.15,
                'asset_turnover': 0.8, 'inventory_turnover': 12
            },
            'manufacturing': {
                'current_ratio': 1.8, 'debt_to_equity': 0.5, 'profit_margin': 0.08,
                'asset_turnover': 1.2, 'inventory_turnover': 6
            },
            'retail': {
                'current_ratio': 1.5, 'debt_to_equity': 0.4, 'profit_margin': 0.05,
                'asset_turnover': 2.0, 'inventory_turnover': 8
            },
            'healthcare': {
                'current_ratio': 2.0, 'debt_to_equity': 0.35, 'profit_margin': 0.12,
                'asset_turnover': 0.9, 'inventory_turnover': 10
            },
            'financial': {
                'current_ratio': 1.2, 'debt_to_equity': 0.8, 'profit_margin': 0.20,
                'asset_turnover': 0.1, 'inventory_turnover': 0
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

    def detect_column_mapping(self, df_columns, mapping_dict):
        """
        Automatically detect column mappings from uploaded file
        
        Args:
            df_columns (list): Column names from uploaded DataFrame
            mapping_dict (dict): Dictionary of standard column mappings
            
        Returns:
            dict: Mapped columns
        """
        detected_mapping = {}
        df_columns_lower = [col.lower().strip() for col in df_columns]
        
        for standard_col, variations in mapping_dict.items():
            for variation in variations:
                for i, col in enumerate(df_columns_lower):
                    if variation.lower() in col or col in variation.lower():
                        detected_mapping[standard_col] = df_columns[i]
                        break
                if standard_col in detected_mapping:
                    break
                    
        return detected_mapping

    def clean_financial_value(self, value):
        """Clean and convert financial values to numeric"""
        if pd.isna(value) or value == '' or value is None:
            return np.nan
            
        if isinstance(value, (int, float)):
            return float(value)
            
        # Convert to string and clean
        value_str = str(value).strip()
        
        # Remove common financial formatting
        value_str = re.sub(r'[,$\s]', '', value_str)
        value_str = re.sub(r'[()]', '-', value_str)  # Convert parentheses to negative
        
        # Handle percentage values
        if '%' in value_str:
            value_str = value_str.replace('%', '')
            try:
                return float(value_str) / 100
            except:
                return np.nan
        
        # Handle multipliers (K, M, B)
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

    def extract_year_from_data(self, df):
        """Extract year from DataFrame columns or data"""
        # Check column names for years
        year_pattern = r'20\d{2}'
        for col in df.columns:
            year_match = re.search(year_pattern, str(col))
            if year_match:
                return int(year_match.group())
        
        # Check first few rows for year information
        for col in df.columns:
            for value in df[col].head(10):
                if pd.notna(value):
                    year_match = re.search(year_pattern, str(value))
                    if year_match:
                        return int(year_match.group())
        
        # Default to current year if not found
        return datetime.now().year

    def impute_missing_values_knn(self, df, target_column, feature_columns):
        """Impute missing values using KNN"""
        try:
            # Prepare data for KNN
            data_for_knn = df[feature_columns + [target_column]].copy()
            data_for_knn = data_for_knn.dropna(subset=feature_columns)
            
            if len(data_for_knn) < 3:  # Not enough data for KNN
                return df[target_column]
            
            # Separate training data (complete cases) and prediction data (missing target)
            complete_cases = data_for_knn.dropna(subset=[target_column])
            missing_cases = data_for_knn[data_for_knn[target_column].isna()]
            
            if len(complete_cases) < 3 or len(missing_cases) == 0:
                return df[target_column]
            
            # Fit KNN model
            X_train = complete_cases[feature_columns]
            y_train = complete_cases[target_column]
            X_predict = missing_cases[feature_columns]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_predict_scaled = scaler.transform(X_predict)
            
            # Train and predict
            knn = KNeighborsRegressor(n_neighbors=min(3, len(complete_cases)))
            knn.fit(X_train_scaled, y_train)
            predictions = knn.predict(X_predict_scaled)
            
            # Fill missing values
            result = df[target_column].copy()
            result.loc[missing_cases.index] = predictions
            
            return result
            
        except Exception as e:
            print(f"KNN imputation error for {target_column}: {e}")
            return df[target_column]

    def impute_missing_values_rf(self, df, target_column, feature_columns):
        """Impute missing values using Random Forest"""
        try:
            # Prepare data for Random Forest
            data_for_rf = df[feature_columns + [target_column]].copy()
            data_for_rf = data_for_rf.dropna(subset=feature_columns)
            
            if len(data_for_rf) < 5:  # Not enough data for RF
                return df[target_column]
            
            # Separate training and prediction data
            complete_cases = data_for_rf.dropna(subset=[target_column])
            missing_cases = data_for_rf[data_for_rf[target_column].isna()]
            
            if len(complete_cases) < 5 or len(missing_cases) == 0:
                return df[target_column]
            
            # Fit Random Forest model
            X_train = complete_cases[feature_columns]
            y_train = complete_cases[target_column]
            X_predict = missing_cases[feature_columns]
            
            # Train and predict
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X_train, y_train)
            predictions = rf.predict(X_predict)
            
            # Fill missing values
            result = df[target_column].copy()
            result.loc[missing_cases.index] = predictions
            
            return result
            
        except Exception as e:
            print(f"Random Forest imputation error for {target_column}: {e}")
            return df[target_column]

    def impute_missing_values_time_series(self, df, target_column, company_id=None):
        """Impute missing values using time series analysis"""
        try:
            if company_id is None:
                return df[target_column]
            
            # Get historical data for the company from balance_sheet_1 table
            conn = self.connect_db()
            if not conn:
                return df[target_column]
            
            cursor = conn.cursor()
            
            # Query historical data from balance_sheet_1 table
            query = f"""
            SELECT year, {target_column}
            FROM balance_sheet_1 
            WHERE company_id = %s AND {target_column} IS NOT NULL
            ORDER BY year
            """
            
            cursor.execute(query, (company_id,))
            historical_data = cursor.fetchall()
            
            if len(historical_data) < 2:
                conn.close()
                return df[target_column]
            
            # Create time series and calculate trends
            years = [row[0] for row in historical_data]
            values = [row[1] for row in historical_data]
            
            # Simple linear trend calculation
            if len(values) >= 2:
                # Calculate average growth rate
                growth_rates = []
                for i in range(1, len(values)):
                    if values[i-1] != 0:
                        growth_rate = (values[i] - values[i-1]) / abs(values[i-1])
                        growth_rates.append(growth_rate)
                
                if growth_rates:
                    avg_growth_rate = np.mean(growth_rates)
                    last_value = values[-1]
                    last_year = years[-1]
                    
                    # Fill missing values based on trend
                    result = df[target_column].copy()
                    for idx in result[result.isna()].index:
                        # Estimate based on years difference and growth rate
                        current_year = df.loc[idx, 'year'] if 'year' in df.columns else datetime.now().year
                        year_diff = current_year - last_year
                        estimated_value = last_value * ((1 + avg_growth_rate) ** year_diff)
                        result.loc[idx] = estimated_value
                    
                    conn.close()
                    return result
            
            conn.close()
            return df[target_column]
            
        except Exception as e:
            print(f"Time series imputation error for {target_column}: {e}")
            return df[target_column]

    def impute_missing_values_peer_analysis(self, df, target_column, industry=None):
        """Impute missing values using peer company analysis"""
        try:
            if industry is None:
                return df[target_column]
            
            conn = self.connect_db()
            if not conn:
                return df[target_column]
            
            cursor = conn.cursor()
            
            # Query peer companies in same industry from cash_flow_statement table
            query = f"""
            SELECT AVG({target_column}) as avg_value, STDDEV({target_column}) as std_value
            FROM cash_flow_statement 
            WHERE industry = %s AND {target_column} IS NOT NULL
            """
            
            try:
                cursor.execute(query, (industry,))
                result_data = cursor.fetchone()
                
                if result_data and result_data[0] is not None:
                    peer_avg = float(result_data[0])
                    peer_std = float(result_data[1]) if result_data[1] is not None else peer_avg * 0.2
                    
                    # Fill missing values with peer average + some random variation
                    result = df[target_column].copy()
                    missing_indices = result[result.isna()].index
                    
                    for idx in missing_indices:
                        # Add some random variation based on standard deviation
                        variation = np.random.normal(0, peer_std * 0.1)
                        estimated_value = peer_avg + variation
                        result.loc[idx] = max(0, estimated_value)  # Ensure non-negative for most financial metrics
                    
                    conn.close()
                    return result
                    
            except Exception as query_error:
                print(f"Peer analysis query error: {query_error}")
            
            conn.close()
            return df[target_column]
            
        except Exception as e:
            print(f"Peer analysis imputation error for {target_column}: {e}")
            return df[target_column]

    def impute_missing_values_industry_benchmark(self, df, target_column, industry=None):
        """Impute missing values using industry benchmarks"""
        try:
            if industry is None or industry.lower() not in self.industry_benchmarks:
                industry = 'manufacturing'  # Default industry
            
            benchmarks = self.industry_benchmarks[industry.lower()]
            result = df[target_column].copy()
            
            # Map target column to benchmark metric
            benchmark_mapping = {
                'current_ratio': 'current_ratio',
                'debt_to_equity_ratio': 'debt_to_equity',
                'profit_margin': 'profit_margin'
            }
            
            if target_column in benchmark_mapping:
                benchmark_value = benchmarks.get(benchmark_mapping[target_column])
                if benchmark_value is not None:
                    # Fill missing values with benchmark value
                    missing_indices = result[result.isna()].index
                    result.loc[missing_indices] = benchmark_value
            else:
                # For other columns, use statistical relationships
                if 'total_assets' in df.columns and not df['total_assets'].isna().all():
                    median_assets = df['total_assets'].median()
                    
                    # Simple heuristics based on typical financial relationships
                    if target_column == 'cash_and_equivalents':
                        result.fillna(median_assets * 0.1, inplace=True)  # ~10% of assets
                    elif target_column == 'accounts_receivable':
                        result.fillna(median_assets * 0.15, inplace=True)  # ~15% of assets
                    elif target_column == 'inventory':
                        result.fillna(median_assets * 0.2, inplace=True)   # ~20% of assets
                    elif target_column == 'accounts_payable':
                        result.fillna(median_assets * 0.12, inplace=True)  # ~12% of assets
            
            return result
            
        except Exception as e:
            print(f"Industry benchmark imputation error for {target_column}: {e}")
            return df[target_column]

    def weighted_ensemble_imputation(self, df, target_column, feature_columns, company_id=None, industry=None):
        """
        Weighted ensemble method combining all imputation techniques
        
        Args:
            df: DataFrame with data
            target_column: Column to impute
            feature_columns: Related columns for ML methods
            company_id: Company identifier for time series
            industry: Industry for peer analysis and benchmarks
            
        Returns:
            Series with imputed values
        """
        # Get results from each method
        knn_result = self.impute_missing_values_knn(df, target_column, feature_columns)
        rf_result = self.impute_missing_values_rf(df, target_column, feature_columns)
        ts_result = self.impute_missing_values_time_series(df, target_column, company_id)
        peer_result = self.impute_missing_values_peer_analysis(df, target_column, industry)
        benchmark_result = self.impute_missing_values_industry_benchmark(df, target_column, industry)
        
        # Weights for ensemble (can be adjusted based on data quality and availability)
        weights = {
            'knn': 0.25,
            'rf': 0.25,
            'ts': 0.20,
            'peer': 0.15,
            'benchmark': 0.15
        }
        
        # Create weighted average for missing values
        result = df[target_column].copy()
        missing_indices = result[result.isna()].index
        
        for idx in missing_indices:
            values = []
            valid_weights = []
            
            # Collect valid predictions
            if not pd.isna(knn_result.loc[idx]):
                values.append(knn_result.loc[idx])
                valid_weights.append(weights['knn'])
                
            if not pd.isna(rf_result.loc[idx]):
                values.append(rf_result.loc[idx])
                valid_weights.append(weights['rf'])
                
            if not pd.isna(ts_result.loc[idx]):
                values.append(ts_result.loc[idx])
                valid_weights.append(weights['ts'])
                
            if not pd.isna(peer_result.loc[idx]):
                values.append(peer_result.loc[idx])
                valid_weights.append(weights['peer'])
                
            if not pd.isna(benchmark_result.loc[idx]):
                values.append(benchmark_result.loc[idx])
                valid_weights.append(weights['benchmark'])
            
            # Calculate weighted average
            if values:
                # Normalize weights
                total_weight = sum(valid_weights)
                normalized_weights = [w/total_weight for w in valid_weights]
                
                # Calculate weighted average
                weighted_value = sum(v*w for v, w in zip(values, normalized_weights))
                result.loc[idx] = weighted_value
            else:
                # Fallback to simple mean of the column if available
                if not result.dropna().empty:
                    result.loc[idx] = result.dropna().mean()
                else:
                    result.loc[idx] = 0  # Last resort
        
        return result

    def process_balance_sheet(self, df, company_id, company_name, industry=None):
        """
        Process uploaded balance sheet data and save to balance_sheet_1 table
        
        Args:
            df: DataFrame with balance sheet data
            company_id: Company identifier
            company_name: Company name
            industry: Company industry
            
        Returns:
            dict: Processed balance sheet data
        """
        # Detect column mappings
        detected_mapping = self.detect_column_mapping(df.columns, self.balance_sheet_mappings)
        
        # Extract year
        year = self.extract_year_from_data(df)
        
        # Create standard balance sheet structure with exact columns from balance_sheet_1 table
        balance_sheet_data = {
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
            'data_source': 'uploaded_file',
            'validation_errors': None
        }
        
        # Map detected values
        for standard_col, source_col in detected_mapping.items():
            if source_col in df.columns:
                # Take the first non-null value from the column
                value = df[source_col].dropna().iloc[0] if not df[source_col].dropna().empty else np.nan
                balance_sheet_data[standard_col] = self.clean_financial_value(value)
        
        # Convert to DataFrame for processing
        df_processed = pd.DataFrame([balance_sheet_data])
        
        # Define feature groups for imputation
        asset_features = ['cash_and_equivalents', 'accounts_receivable', 'inventory', 'property_plant_equipment']
        liability_features = ['accounts_payable', 'short_term_debt', 'long_term_debt']
        equity_features = ['share_capital', 'retained_earnings']
        
        # Impute missing values using ensemble method
        for col in balance_sheet_data.keys():
            if col in asset_features + liability_features + equity_features:
                if col in asset_features:
                    features = [f for f in asset_features if f != col and f in df_processed.columns]
                elif col in liability_features:
                    features = [f for f in liability_features if f != col and f in df_processed.columns]
                else:
                    features = [f for f in equity_features if f != col and f in df_processed.columns]
                
                if features:
                    df_processed[col] = self.weighted_ensemble_imputation(
                        df_processed, col, features, company_id, industry
                    )
        
        # Calculate derived values
        processed_data = df_processed.iloc[0].to_dict()
        
        # Calculate totals if missing
        if pd.isna(processed_data['current_assets']):
            current_asset_components = [
                processed_data.get('cash_and_equivalents', 0),
                processed_data.get('accounts_receivable', 0),
                processed_data.get('inventory', 0),
                processed_data.get('prepaid_expenses', 0),
                processed_data.get('other_current_assets', 0)
            ]
            processed_data['current_assets'] = sum([x for x in current_asset_components if not pd.isna(x)])
        
        if pd.isna(processed_data['total_assets']):
            processed_data['total_assets'] = (
                processed_data.get('current_assets', 0) + 
                processed_data.get('non_current_assets', 0)
            )
        
        if pd.isna(processed_data['current_liabilities']):
            current_liab_components = [
                processed_data.get('accounts_payable', 0),
                processed_data.get('short_term_debt', 0),
                processed_data.get('accrued_liabilities', 0),
                processed_data.get('deferred_revenue', 0),
                processed_data.get('other_current_liabilities', 0)
            ]
            processed_data['current_liabilities'] = sum([x for x in current_liab_components if not pd.isna(x)])
        
        if pd.isna(processed_data['total_liabilities']):
            processed_data['total_liabilities'] = (
                processed_data.get('current_liabilities', 0) + 
                processed_data.get('non_current_liabilities', 0)
            )
        
        if pd.isna(processed_data['total_equity']):
            processed_data['total_equity'] = (
                processed_data.get('total_assets', 0) - 
                processed_data.get('total_liabilities', 0)
            )
        
        # Balance check
        total_assets = processed_data.get('total_assets', 0)
        total_liab_equity = processed_data.get('total_liabilities', 0) + processed_data.get('total_equity', 0)
        
        if total_assets > 0:
            processed_data['balance_check'] = abs(total_assets - total_liab_equity) / total_assets
            processed_data['accuracy_percentage'] = max(0, 100 - (processed_data['balance_check'] * 100))
        else:
            processed_data['balance_check'] = 1.0
            processed_data['accuracy_percentage'] = 0.0
        
        return processed_data

    def calculate_cash_flow_metrics(self, balance_sheet_current, balance_sheet_previous, income_statement=None):
        """
        Calculate cash flow statement from balance sheet data using indirect method
        
        Args:
            balance_sheet_current: Current year balance sheet data
            balance_sheet_previous: Previous year balance sheet data
            income_statement: Income statement data (optional)
            
        Returns:
            dict: Cash flow statement data with exact columns for cash_flow_statement table
        """
        cash_flow_data = {
            'company_id': balance_sheet_current.get('company_id'),
            'year': balance_sheet_current.get('year'),
            'generated_at': datetime.now(),
            'company_name': '',
            'industry': '',
            'net_income': np.nan,
            'depreciation_and_amortization': np.nan,
            'stock_based_compensation': np.nan,
            'changes_in_working_capital': np.nan,
            'accounts_receivable': np.nan,
            'inventory': np.nan,
            'accounts_payable': np.nan,
            'net_cash_from_operating_activities': np.nan,
            'capital_expenditures': np.nan,
            'acquisitions': np.nan,
            'net_cash_from_investing_activities': np.nan,
            'dividends_paid': np.nan,
            'share_repurchases': np.nan,
            'net_cash_from_financing_activities': np.nan,
            'free_cash_flow': np.nan,
            'ocf_to_net_income_ratio': np.nan,
            'liquidation_label': 0,
            'debt_to_equity_ratio': np.nan,
            'interest_coverage_ratio': np.nan
        }
        
        # Calculate changes in working capital components
        if balance_sheet_previous:
            # Accounts Receivable change
            ar_current = balance_sheet_current.get('accounts_receivable', 0)
            ar_previous = balance_sheet_previous.get('accounts_receivable', 0)
            cash_flow_data['accounts_receivable'] = ar_current - ar_previous
            
            # Inventory change
            inv_current = balance_sheet_current.get('inventory', 0)
            inv_previous = balance_sheet_previous.get('inventory', 0)
            cash_flow_data['inventory'] = inv_current - inv_previous
            
            # Accounts Payable change
            ap_current = balance_sheet_current.get('accounts_payable', 0)
            ap_previous = balance_sheet_previous.get('accounts_payable', 0)
            cash_flow_data['accounts_payable'] = ap_current - ap_previous
            
            # Changes in Working Capital (negative of increases in current assets, positive for increases in current liabilities)
            cash_flow_data['changes_in_working_capital'] = (
                -(cash_flow_data['accounts_receivable']) +
                -(cash_flow_data['inventory']) +
                cash_flow_data['accounts_payable']
            )
            
            # Capital Expenditures calculation
            ppe_current = balance_sheet_current.get('property_plant_equipment', 0)
            ppe_previous = balance_sheet_previous.get('property_plant_equipment', 0)
            depreciation = balance_sheet_current.get('accumulated_depreciation', 0) - balance_sheet_previous.get('accumulated_depreciation', 0)
            cash_flow_data['capital_expenditures'] = ppe_current - ppe_previous + abs(depreciation)
            
            # Debt changes for financing activities
            ltd_current = balance_sheet_current.get('long_term_debt', 0)
            ltd_previous = balance_sheet_previous.get('long_term_debt', 0)
            std_current = balance_sheet_current.get('short_term_debt', 0)
            std_previous = balance_sheet_previous.get('short_term_debt', 0)
            
            debt_change = (ltd_current - ltd_previous) + (std_current - std_previous)
            
            # Retained earnings change for dividend calculation
            re_current = balance_sheet_current.get('retained_earnings', 0)
            re_previous = balance_sheet_previous.get('retained_earnings', 0)
            re_change = re_current - re_previous
        
        # Use income statement data if available
        if income_statement:
            cash_flow_data['net_income'] = income_statement.get('net_income', 0)
            cash_flow_data['depreciation_and_amortization'] = income_statement.get('depreciation', 0)
        else:
            # Estimate net income from retained earnings change
            if balance_sheet_previous:
                cash_flow_data['net_income'] = re_change + cash_flow_data.get('dividends_paid', 0)
        
        # Calculate operating cash flow using indirect method
        net_income = cash_flow_data.get('net_income', 0)
        depreciation = cash_flow_data.get('depreciation_and_amortization', 0)
        wc_change = cash_flow_data.get('changes_in_working_capital', 0)
        
        cash_flow_data['net_cash_from_operating_activities'] = net_income + depreciation - wc_change
        
        # Calculate investing cash flow
        capex = cash_flow_data.get('capital_expenditures', 0)
        acquisitions = cash_flow_data.get('acquisitions', 0)
        cash_flow_data['net_cash_from_investing_activities'] = -capex - acquisitions
        
        # Calculate financing cash flow
        dividends = cash_flow_data.get('dividends_paid', 0)
        share_repurchases = cash_flow_data.get('share_repurchases', 0)
        if balance_sheet_previous:
            cash_flow_data['net_cash_from_financing_activities'] = debt_change - dividends - share_repurchases
        
        # Calculate derived metrics
        ocf = cash_flow_data.get('net_cash_from_operating_activities', 0)
        cash_flow_data['free_cash_flow'] = ocf - capex
        
        if net_income != 0:
            cash_flow_data['ocf_to_net_income_ratio'] = ocf / net_income
        
        # Calculate financial ratios
        total_debt = balance_sheet_current.get('long_term_debt', 0) + balance_sheet_current.get('short_term_debt', 0)
        total_equity = balance_sheet_current.get('total_equity', 1)
        
        if total_equity != 0:
            cash_flow_data['debt_to_equity_ratio'] = total_debt / total_equity
        
        # Liquidation label (1 if high risk, 0 if low risk)
        if net_income < 0 and ocf < 0:
            cash_flow_data['liquidation_label'] = 1
        
        return cash_flow_data

    def process_cash_flow_statement(self, df, company_id, company_name, industry=None):
        """
        Process uploaded cash flow statement data and save to cash_flow_statement table
        
        Args:
            df: DataFrame with cash flow data
            company_id: Company identifier  
            company_name: Company name
            industry: Company industry
            
        Returns:
            dict: Processed cash flow statement data
        """
        # Detect column mappings
        detected_mapping = self.detect_column_mapping(df.columns, self.cash_flow_mappings)
        
        # Extract year
        year = self.extract_year_from_data(df)
        
        # Create standard cash flow structure with exact columns from cash_flow_statement table
        cash_flow_data = {
            'company_id': company_id,
            'year': year,
            'generated_at': datetime.now(),
            'company_name': company_name,
            'industry': industry or 'unknown',
            'net_income': np.nan,
            'depreciation_and_amortization': np.nan,
            'stock_based_compensation': np.nan,
            'changes_in_working_capital': np.nan,
            'accounts_receivable': np.nan,
            'inventory': np.nan,
            'accounts_payable': np.nan,
            'net_cash_from_operating_activities': np.nan,
            'capital_expenditures': np.nan,
            'acquisitions': np.nan,
            'net_cash_from_investing_activities': np.nan,
            'dividends_paid': np.nan,
            'share_repurchases': np.nan,
            'net_cash_from_financing_activities': np.nan,
            'free_cash_flow': np.nan,
            'ocf_to_net_income_ratio': np.nan,
            'liquidation_label': 0,
            'debt_to_equity_ratio': np.nan,
            'interest_coverage_ratio': np.nan
        }
        
        # Map detected values
        for standard_col, source_col in detected_mapping.items():
            if source_col in df.columns:
                value = df[source_col].dropna().iloc[0] if not df[source_col].dropna().empty else np.nan
                cash_flow_data[standard_col] = self.clean_financial_value(value)
        
        # Convert to DataFrame for processing
        df_processed = pd.DataFrame([cash_flow_data])
        
        # Define feature groups for imputation
        operating_features = ['net_income', 'depreciation_and_amortization', 'changes_in_working_capital']
        investing_features = ['capital_expenditures', 'acquisitions']
        financing_features = ['dividends_paid', 'share_repurchases']
        
        # Impute missing values using ensemble method
        all_features = operating_features + investing_features + financing_features
        for col in all_features:
            if pd.isna(df_processed[col].iloc[0]):
                features = [f for f in all_features if f != col and not pd.isna(df_processed[f].iloc[0])]
                if features:
                    df_processed[col] = self.weighted_ensemble_imputation(
                        df_processed, col, features, company_id, industry
                    )
        
        # Calculate derived metrics
        processed_data = df_processed.iloc[0].to_dict()
        
        # Calculate operating cash flow if missing
        if pd.isna(processed_data['net_cash_from_operating_activities']):
            net_income = processed_data.get('net_income', 0)
            depreciation = processed_data.get('depreciation_and_amortization', 0)
            wc_change = processed_data.get('changes_in_working_capital', 0)
            processed_data['net_cash_from_operating_activities'] = net_income + depreciation - wc_change
        
        # Calculate free cash flow
        ocf = processed_data.get('net_cash_from_operating_activities', 0)
        capex = processed_data.get('capital_expenditures', 0)
        processed_data['free_cash_flow'] = ocf - capex
        
        # Calculate OCF to Net Income ratio
        net_income = processed_data.get('net_income', 0)
        if net_income != 0:
            processed_data['ocf_to_net_income_ratio'] = ocf / net_income
        
        # Liquidation label
        if net_income < 0 and ocf < 0:
            processed_data['liquidation_label'] = 1
        
        return processed_data

    def save_balance_sheet_to_db(self, balance_sheet_data):
        """Save balance sheet data to balance_sheet_1 table"""
        conn = self.connect_db()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            # Prepare insert query for balance_sheet_1 table
            columns = list(balance_sheet_data.keys())
            placeholders = ', '.join(['%s'] * len(columns))
            columns_str = ', '.join(columns)
            
            query = f"""
            INSERT INTO balance_sheet_1 ({columns_str})
            VALUES ({placeholders})
            ON CONFLICT (company_id, year) 
            DO UPDATE SET
                {', '.join([f"{col} = EXCLUDED.{col}" for col in columns if col not in ['id', 'company_id', 'year']])}
            """
            
            values = [balance_sheet_data[col] for col in columns]
            cursor.execute(query, values)
            
            conn.commit()
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error saving balance sheet to balance_sheet_1 table: {e}")
            conn.rollback()
            conn.close()
            return False

    def save_cash_flow_to_db(self, cash_flow_data):
        """Save cash flow data to cash_flow_statement table"""
        conn = self.connect_db()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            # Prepare insert query for cash_flow_statement table
            columns = list(cash_flow_data.keys())
            placeholders = ', '.join(['%s'] * len(columns))
            columns_str = ', '.join(columns)
            
            query = f"""
            INSERT INTO cash_flow_statement ({columns_str})
            VALUES ({placeholders})
            ON CONFLICT (company_id, year)
            DO UPDATE SET
                {', '.join([f"{col} = EXCLUDED.{col}" for col in columns if col not in ['id', 'company_id', 'year']])}
            """
            
            values = [cash_flow_data[col] for col in columns]
            cursor.execute(query, values)
            
            conn.commit()
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error saving cash flow to cash_flow_statement table: {e}")
            conn.rollback()
            conn.close()
            return False

    def generate_financial_report(self, company_id, years=None):
        """
        Generate comprehensive financial report for a company
        
        Args:
            company_id: Company identifier
            years: List of years to include (default: last 3 years)
            
        Returns:
            dict: Financial report with analysis
        """
        if years is None:
            years = [datetime.now().year - i for i in range(3)]
        
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
            
            # Generate analysis
            report = {
                'company_id': company_id,
                'generated_at': datetime.now(),
                'balance_sheets': [dict(bs) for bs in balance_sheets],
                'cash_flows': [dict(cf) for cf in cash_flows],
                'financial_health_score': self.calculate_financial_health_score(balance_sheets, cash_flows),
                'recommendations': self.generate_recommendations(balance_sheets, cash_flows),
                'risk_assessment': self.assess_financial_risk(balance_sheets, cash_flows)
            }
            
            return report
            
        except Exception as e:
            print(f"Error generating financial report: {e}")
            if conn:
                conn.close()
            return None

    def calculate_financial_health_score(self, balance_sheets, cash_flows):
        """Calculate overall financial health score (0-100)"""
        if not balance_sheets or not cash_flows:
            return 0
        
        latest_bs = balance_sheets[0]
        latest_cf = cash_flows[0]
        
        score_components = {}
        
        # Liquidity Score (25 points)
        current_ratio = latest_bs['current_assets'] / max(latest_bs['current_liabilities'], 1)
        score_components['liquidity'] = min(25, current_ratio * 12.5)  # Max at ratio of 2.0
        
        # Profitability Score (25 points)
        if latest_cf['net_income'] > 0:
            score_components['profitability'] = 25
        elif latest_cf['net_income'] > -0.1 * latest_bs['total_assets']:
            score_components['profitability'] = 15
        else:
            score_components['profitability'] = 0
        
        # Debt Management Score (25 points)
        debt_ratio = latest_cf['debt_to_equity_ratio'] or 0
        if debt_ratio <= 0.3:
            score_components['debt'] = 25
        elif debt_ratio <= 0.6:
            score_components['debt'] = 15
        else:
            score_components['debt'] = 5
        
        # Cash Flow Score (25 points)
        if latest_cf['net_cash_from_operating_activities'] > 0:
            score_components['cash_flow'] = 25
        elif latest_cf['free_cash_flow'] > 0:
            score_components['cash_flow'] = 15
        else:
            score_components['cash_flow'] = 0
        
        total_score = sum(score_components.values())
        
        return {
            'total_score': round(total_score, 1),
            'components': score_components,
            'rating': self.get_rating_from_score(total_score)
        }

    def get_rating_from_score(self, score):
        """Convert numerical score to rating"""
        if score >= 80:
            return 'Excellent'
        elif score >= 60:
            return 'Good'
        elif score >= 40:
            return 'Fair'
        elif score >= 20:
            return 'Poor'
        else:
            return 'Critical'

    def generate_recommendations(self, balance_sheets, cash_flows):
        """Generate financial recommendations based on analysis"""
        if not balance_sheets or not cash_flows:
            return []
        
        latest_bs = balance_sheets[0]
        latest_cf = cash_flows[0]
        recommendations = []
        
        # Liquidity recommendations
        current_ratio = latest_bs['current_assets'] / max(latest_bs['current_liabilities'], 1)
        if current_ratio < 1.5:
            recommendations.append({
                'type': 'liquidity',
                'priority': 'high',
                'message': 'Improve liquidity by reducing current liabilities or increasing current assets',
                'impact': 'Reduces short-term financial risk'
            })
        
        # Cash flow recommendations
        if latest_cf['net_cash_from_operating_activities'] < 0:
            recommendations.append({
                'type': 'cash_flow',
                'priority': 'high', 
                'message': 'Focus on improving operating cash flow through better collections and expense management',
                'impact': 'Improves operational efficiency'
            })
        
        # Debt recommendations
        debt_ratio = latest_cf['debt_to_equity_ratio'] or 0
        if debt_ratio > 0.6:
            recommendations.append({
                'type': 'debt',
                'priority': 'medium',
                'message': 'Consider reducing debt levels to improve financial stability',
                'impact': 'Reduces financial leverage risk'
            })
        
        return recommendations

    def assess_financial_risk(self, balance_sheets, cash_flows):
        """Assess overall financial risk"""
        if not balance_sheets or not cash_flows:
            return {'level': 'unknown', 'factors': []}
        
        latest_cf = cash_flows[0]
        risk_factors = []
        
        # Check liquidation indicators
        if latest_cf['liquidation_label'] == 1:
            risk_factors.append('Negative profitability and cash flow')
        
        # Determine risk level
        if len(risk_factors) >= 3:
            risk_level = 'high'
        elif len(risk_factors) >= 2:
            risk_level = 'medium'
        elif len(risk_factors) >= 1:
            risk_level = 'low'
        else:
            risk_level = 'minimal'
        
        return {
            'level': risk_level,
            'factors': risk_factors,
            'liquidation_probability': latest_cf['liquidation_label']
        }