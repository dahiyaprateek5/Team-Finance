import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import psycopg2
from psycopg2.extras import RealDictCursor
import pickle
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CashFlowPredictionModels:
    def __init__(self, db_config):
        """
        Initialize Cash Flow Prediction Models
        
        Args:
            db_config (dict): Database configuration
        """
        self.db_config = db_config
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'model': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
                'name': 'Random Forest'
            },
            'knn': {
                'model': KNeighborsRegressor(n_neighbors=5),
                'name': 'K-Nearest Neighbors'
            },
            'linear_regression': {
                'model': LinearRegression(),
                'name': 'Linear Regression'
            }
        }
        
        # Cash flow components for prediction - exact columns from cash_flow_statement table
        self.target_components = [
            'net_income',
            'depreciation_and_amortization', 
            'changes_in_working_capital',
            'net_cash_from_operating_activities',
            'capital_expenditures',
            'net_cash_from_investing_activities',
            'net_cash_from_financing_activities',
            'free_cash_flow'
        ]

    def connect_db(self):
        """Establish database connection"""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except Exception as e:
            print(f"Database connection error: {e}")
            return None

    def load_training_data(self, min_records=50):
        """
        Load cash flow data from cash_flow_statement table for model training
        
        Args:
            min_records: Minimum number of records required for training
            
        Returns:
            pd.DataFrame: Training data
        """
        conn = self.connect_db()
        if not conn:
            return None
        
        try:
            # Query data from cash_flow_statement table
            query = """
            SELECT cf.*, bs.total_assets, bs.current_assets, bs.current_liabilities, bs.total_equity
            FROM cash_flow_statement cf
            LEFT JOIN balance_sheet_1 bs ON cf.company_id = bs.company_id AND cf.year = bs.year
            WHERE cf.net_income IS NOT NULL 
            AND bs.total_assets IS NOT NULL
            AND bs.total_assets > 0
            ORDER BY cf.generated_at DESC
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            print(f"Loaded {len(df)} records from cash_flow_statement table")
            
            if len(df) < min_records:
                print(f"Warning: Only {len(df)} records available, minimum {min_records} recommended")
            
            return df
            
        except Exception as e:
            print(f"Error loading training data: {e}")
            if conn:
                conn.close()
            return None

    def prepare_features(self, df):
        """
        Prepare features for cash flow model training
        
        Args:
            df: DataFrame with cash flow and balance sheet data
            
        Returns:
            dict: Features and targets for each component
        """
        prepared_data = {}
        
        # Base features from balance sheet that can predict cash flow components
        base_features = [
            'total_assets',
            'current_assets', 
            'current_liabilities',
            'total_equity',
            'year'
        ]
        
        # Add industry encoding if available
        if 'industry' in df.columns:
            industry_dummies = pd.get_dummies(df['industry'], prefix='industry')
            feature_df = pd.concat([df[base_features], industry_dummies], axis=1)
        else:
            feature_df = df[base_features]
        
        # Add derived ratios as features
        feature_df['current_ratio'] = df['current_assets'] / df['current_liabilities'].replace(0, 1)
        feature_df['equity_ratio'] = df['total_equity'] / df['total_assets'].replace(0, 1)
        feature_df['asset_size_log'] = np.log(df['total_assets'].replace(0, 1))
        
        # Prepare data for each target component
        for target in self.target_components:
            if target in df.columns:
                # Create feature set excluding the target
                available_features = [col for col in feature_df.columns if col != target]
                
                # Filter out rows where target or key features are missing
                valid_rows = (
                    df[target].notna() & 
                    df['total_assets'].notna() & 
                    (df['total_assets'] > 0)
                )
                
                if valid_rows.sum() > 10:  # Minimum 10 valid samples
                    X = feature_df.loc[valid_rows, available_features].fillna(0)
                    y = df.loc[valid_rows, target]
                    
                    prepared_data[target] = {
                        'X': X,
                        'y': y,
                        'features': available_features
                    }
        
        return prepared_data

    def train_component_models(self, training_data):
        """
        Train prediction models for each cash flow component
        
        Args:
            training_data: Dictionary of prepared training data
            
        Returns:
            dict: Training results and metrics
        """
        training_results = {}
        
        for component, data in training_data.items():
            print(f"\nTraining models for {component}...")
            
            X, y = data['X'], data['y']
            features = data['features']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            component_models = {}
            component_results = {}
            
            # Train each model type
            for model_name, config in self.model_configs.items():
                try:
                    model = config['model']
                    
                    # Scale features for KNN and Linear Regression
                    if model_name in ['knn', 'linear_regression']:
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        # Train model
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        
                        # Store scaler
                        self.scalers[f"{component}_{model_name}"] = scaler
                        
                    else:
                        # Random Forest doesn't need scaling
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Cross-validation score
                    if model_name in ['knn', 'linear_regression']:
                        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
                    else:
                        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
                    
                    component_models[model_name] = model
                    component_results[model_name] = {
                        'mae': mae,
                        'r2': r2,
                        'cv_score': -cv_scores.mean(),
                        'cv_std': cv_scores.std()
                    }
                    
                    # Feature importance for Random Forest
                    if model_name == 'random_forest':
                        feature_importance = dict(zip(features, model.feature_importances_))
                        self.feature_importance[component] = feature_importance
                    
                    print(f"  {config['name']}: MAE={mae:.2f}, R²={r2:.3f}, CV={-cv_scores.mean():.2f}±{cv_scores.std():.2f}")
                    
                except Exception as e:
                    print(f"  Error training {config['name']}: {e}")
                    continue
            
            # Store best model for this component
            if component_results:
                best_model_name = min(component_results.keys(), key=lambda x: component_results[x]['mae'])
                self.models[component] = {
                    'model': component_models[best_model_name],
                    'model_type': best_model_name,
                    'features': features,
                    'metrics': component_results[best_model_name]
                }
                
                training_results[component] = {
                    'best_model': best_model_name,
                    'all_results': component_results,
                    'feature_count': len(features)
                }
        
        return training_results

    def predict_cash_flow_component(self, component, input_data):
        """
        Predict a specific cash flow component
        
        Args:
            component: Component name to predict
            input_data: Input features as DataFrame or dict
            
        Returns:
            float: Predicted value
        """
        if component not in self.models:
            print(f"No trained model found for {component}")
            return None
        
        model_info = self.models[component]
        model = model_info['model']
        model_type = model_info['model_type']
        features = model_info['features']
        
        try:
            # Prepare input data
            if isinstance(input_data, dict):
                input_df = pd.DataFrame([input_data])
            else:
                input_df = input_data.copy()
            
            # Ensure all required features are present
            for feature in features:
                if feature not in input_df.columns:
                    input_df[feature] = 0
            
            # Select and order features
            X = input_df[features].fillna(0)
            
            # Apply scaling if needed
            if model_type in ['knn', 'linear_regression']:
                scaler_key = f"{component}_{model_type}"
                if scaler_key in self.scalers:
                    X = self.scalers[scaler_key].transform(X)
            
            # Make prediction
            prediction = model.predict(X)
            
            return prediction[0] if len(prediction) == 1 else prediction
            
        except Exception as e:
            print(f"Error predicting {component}: {e}")
            return None

    def predict_complete_cash_flow(self, balance_sheet_data, company_info=None):
        """
        Predict complete cash flow statement from balance sheet data
        
        Args:
            balance_sheet_data: Balance sheet data as dict
            company_info: Additional company information (industry, etc.)
            
        Returns:
            dict: Complete predicted cash flow statement
        """
        # Prepare input features
        input_data = balance_sheet_data.copy()
        
        # Add company info if provided
        if company_info:
            input_data.update(company_info)
        
        # Add derived ratios
        total_assets = input_data.get('total_assets', 1)
        current_assets = input_data.get('current_assets', 0)
        current_liabilities = input_data.get('current_liabilities', 1)
        total_equity = input_data.get('total_equity', 1)
        
        input_data['current_ratio'] = current_assets / max(current_liabilities, 1)
        input_data['equity_ratio'] = total_equity / max(total_assets, 1)
        input_data['asset_size_log'] = np.log(max(total_assets, 1))
        
        # Initialize cash flow statement with exact columns from cash_flow_statement table
        predicted_cash_flow = {
            'company_id': input_data.get('company_id', 'UNKNOWN'),
            'year': input_data.get('year', datetime.now().year),
            'generated_at': datetime.now(),
            'company_name': company_info.get('company_name', '') if company_info else '',
            'industry': company_info.get('industry', '') if company_info else '',
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
        
        # Predict each component
        for component in self.target_components:
            prediction = self.predict_cash_flow_component(component, input_data)
            if prediction is not None:
                predicted_cash_flow[component] = prediction
        
        # Calculate derived metrics
        predicted_cash_flow = self.calculate_derived_metrics(predicted_cash_flow, balance_sheet_data)
        
        return predicted_cash_flow

    def calculate_derived_metrics(self, cash_flow_data, balance_sheet_data):
        """Calculate derived cash flow metrics"""
        
        # OCF to Net Income Ratio
        net_income = cash_flow_data.get('net_income', 0)
        ocf = cash_flow_data.get('net_cash_from_operating_activities', 0)
        
        if net_income != 0:
            cash_flow_data['ocf_to_net_income_ratio'] = ocf / net_income
        
        # Debt to Equity Ratio (from balance sheet)
        total_debt = (balance_sheet_data.get('long_term_debt', 0) or 0) + (balance_sheet_data.get('short_term_debt', 0) or 0)
        total_equity = balance_sheet_data.get('total_equity', 1) or 1
        
        if total_equity != 0:
            cash_flow_data['debt_to_equity_ratio'] = total_debt / total_equity
        
        # Liquidation Label
        if net_income < 0 and ocf < 0:
            cash_flow_data['liquidation_label'] = 1
        else:
            cash_flow_data['liquidation_label'] = 0
        
        return cash_flow_data

    def save_models(self, filepath):
        """Save trained models to file"""
        try:
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'feature_importance': self.feature_importance,
                'model_configs': self.model_configs,
                'target_components': self.target_components,
                'trained_at': datetime.now()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"Models saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"Error saving models: {e}")
            return False

    def load_models(self, filepath):
        """Load trained models from file"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.feature_importance = model_data['feature_importance']
            
            print(f"Models loaded from {filepath}")
            print(f"Trained at: {model_data.get('trained_at', 'Unknown')}")
            print(f"Available components: {list(self.models.keys())}")
            
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    def evaluate_models(self, test_data=None):
        """
        Evaluate model performance
        
        Args:
            test_data: Test dataset (if None, will load from database)
            
        Returns:
            dict: Evaluation results
        """
        if test_data is None:
            test_data = self.load_training_data()
        
        if test_data is None or len(test_data) == 0:
            return {'error': 'No test data available'}
        
        evaluation_results = {}
        
        # Prepare features
        prepared_data = self.prepare_features(test_data)
        
        for component in self.target_components:
            if component not in self.models or component not in prepared_data:
                continue
            
            model_info = self.models[component]
            data = prepared_data[component]
            
            X, y = data['X'], data['y']
            
            try:
                # Make predictions
                predictions = []
                for idx in range(len(X)):
                    pred = self.predict_cash_flow_component(component, X.iloc[[idx]])
                    predictions.append(pred if pred is not None else 0)
                
                predictions = np.array(predictions)
                
                # Calculate metrics
                mae = mean_absolute_error(y, predictions)
                r2 = r2_score(y, predictions)
                
                # Calculate percentage error
                y_nonzero = y[y != 0]
                pred_nonzero = predictions[y != 0]
                
                if len(y_nonzero) > 0:
                    mape = np.mean(np.abs((y_nonzero - pred_nonzero) / y_nonzero)) * 100
                else:
                    mape = np.inf
                
                evaluation_results[component] = {
                    'mae': mae,
                    'r2': r2,
                    'mape': mape,
                    'model_type': model_info['model_type'],
                    'sample_size': len(y)
                }
                
            except Exception as e:
                evaluation_results[component] = {'error': str(e)}
        
        return evaluation_results

    def get_feature_importance(self, component):
        """Get feature importance for a specific component"""
        if component in self.feature_importance:
            return self.feature_importance[component]
        else:
            return {}

    def retrain_models(self, min_records=50):
        """
        Retrain all models with latest data
        
        Args:
            min_records: Minimum records required for training
            
        Returns:
            dict: Retraining results
        """
        print("Loading latest training data...")
        training_data_df = self.load_training_data(min_records)
        
        if training_data_df is None or len(training_data_df) < min_records:
            return {'success': False, 'message': 'Insufficient training data'}
        
        print("Preparing features...")
        prepared_data = self.prepare_features(training_data_df)
        
        if not prepared_data:
            return {'success': False, 'message': 'No valid features prepared'}
        
        print("Training models...")
        training_results = self.train_component_models(prepared_data)
        
        return {
            'success': True,
            'components_trained': list(training_results.keys()),
            'training_results': training_results,
            'data_size': len(training_data_df)
        }

    def predict_cash_flow_trends(self, company_id, years_ahead=3):
        """
        Predict cash flow trends for future years
        
        Args:
            company_id: Company identifier
            years_ahead: Number of years to predict
            
        Returns:
            list: Predicted cash flows for future years
        """
        conn = self.connect_db()
        if not conn:
            return None
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get latest balance sheet data
            query = """
            SELECT * FROM balance_sheet_1 
            WHERE company_id = %s 
            ORDER BY year DESC 
            LIMIT 1
            """
            
            cursor.execute(query, (company_id,))
            latest_bs = cursor.fetchone()
            
            if not latest_bs:
                cursor.close()
                conn.close()
                return None
            
            cursor.close()
            conn.close()
            
            predictions = []
            current_year = latest_bs['year']
            
            # Simple growth assumptions for balance sheet evolution
            growth_rate = 0.05  # 5% annual growth assumption
            
            for year_offset in range(1, years_ahead + 1):
                future_year = current_year + year_offset
                
                # Project future balance sheet (simplified)
                future_bs = dict(latest_bs)
                for key in ['total_assets', 'current_assets', 'total_equity']:
                    if future_bs.get(key):
                        future_bs[key] *= (1 + growth_rate) ** year_offset
                
                future_bs['year'] = future_year
                
                # Predict cash flow
                predicted_cf = self.predict_complete_cash_flow(future_bs)
                predictions.append(predicted_cf)
            
            return predictions
            
        except Exception as e:
            print(f"Error predicting trends for {company_id}: {e}")
            if conn:
                conn.close()
            return None

    def benchmark_against_industry(self, company_id, industry=None):
        """
        Benchmark company's predicted cash flow against industry averages
        
        Args:
            company_id: Company identifier
            industry: Industry category (if None, will try to determine from data)
            
        Returns:
            dict: Benchmark comparison
        """
        conn = self.connect_db()
        if not conn:
            return None
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get company's latest cash flow
            query = """
            SELECT * FROM cash_flow_statement 
            WHERE company_id = %s 
            ORDER BY year DESC 
            LIMIT 1
            """
            
            cursor.execute(query, (company_id,))
            company_cf = cursor.fetchone()
            
            if not company_cf:
                cursor.close()
                conn.close()
                return None
            
            # Get industry if not provided
            if industry is None:
                industry = company_cf.get('industry', 'unknown')
            
            # Get industry averages
            industry_query = """
            SELECT 
                AVG(net_income) as avg_net_income,
                AVG(net_cash_from_operating_activities) as avg_ocf,
                AVG(free_cash_flow) as avg_fcf,
                AVG(debt_to_equity_ratio) as avg_debt_ratio,
                COUNT(*) as sample_size
            FROM cash_flow_statement 
            WHERE industry = %s 
            AND year >= %s
            """
            
            cursor.execute(industry_query, (industry, company_cf['year'] - 2))
            industry_avg = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            if not industry_avg or industry_avg['sample_size'] < 5:
                return {'error': 'Insufficient industry data for benchmarking'}
            
            # Calculate benchmarks
            benchmarks = {}
            
            metrics = ['net_income', 'net_cash_from_operating_activities', 'free_cash_flow', 'debt_to_equity_ratio']
            
            for metric in metrics:
                company_value = company_cf.get(metric, 0) or 0
                industry_value = industry_avg.get(f'avg_{metric.replace("net_cash_from_", "").replace("_activities", "")}', 0) or 0
                
                if industry_value != 0:
                    ratio = company_value / industry_value
                    performance = 'Above Average' if ratio > 1.1 else 'Below Average' if ratio < 0.9 else 'Average'
                else:
                    ratio = None
                    performance = 'Unknown'
                
                benchmarks[metric] = {
                    'company_value': company_value,
                    'industry_average': industry_value,
                    'ratio': ratio,
                    'performance': performance
                }
            
            return {
                'company_id': company_id,
                'industry': industry,
                'year': company_cf['year'],
                'industry_sample_size': industry_avg['sample_size'],
                'benchmarks': benchmarks
            }
            
        except Exception as e:
            print(f"Error benchmarking {company_id}: {e}")
            if conn:
                conn.close()
            return None