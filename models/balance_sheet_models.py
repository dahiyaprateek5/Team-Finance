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

class BalanceSheetPredictionModels:
    def __init__(self, db_config):
        """
        Initialize Balance Sheet Prediction Models
        
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
        
        # Balance sheet components for prediction - exact columns from balance_sheet_1 table
        self.target_components = [
            'cash_and_equivalents',
            'accounts_receivable', 
            'inventory',
            'property_plant_equipment',
            'accounts_payable',
            'short_term_debt',
            'long_term_debt',
            'retained_earnings'
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
        Load balance sheet data from balance_sheet_1 table for model training
        
        Args:
            min_records: Minimum number of records required for training
            
        Returns:
            pd.DataFrame: Training data
        """
        conn = self.connect_db()
        if not conn:
            return None
        
        try:
            # Query data from balance_sheet_1 table
            query = """
            SELECT * FROM balance_sheet_1 
            WHERE total_assets IS NOT NULL 
            AND total_assets > 0
            AND accuracy_percentage > 70
            ORDER BY generated_at DESC
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            print(f"Loaded {len(df)} records from balance_sheet_1 table")
            
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
        Prepare features for balance sheet model training
        
        Args:
            df: DataFrame with balance sheet data
            
        Returns:
            dict: Features and targets for each component
        """
        prepared_data = {}
        
        # Base features that can be used to predict other components
        base_features = [
            'total_assets',
            'current_assets', 
            'total_liabilities',
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
        Train prediction models for each balance sheet component
        
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

    def predict_balance_sheet_component(self, component, input_data):
        """
        Predict a specific balance sheet component
        
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

    def predict_complete_balance_sheet(self, base_data, company_info=None):
        """
        Predict complete balance sheet from partial data
        
        Args:
            base_data: Known balance sheet data as dict
            company_info: Additional company information
            
        Returns:
            dict: Complete predicted balance sheet with exact columns from balance_sheet_1 table
        """
        # Initialize balance sheet structure with exact columns from balance_sheet_1 table
        predicted_bs = {
            'company_id': base_data.get('company_id', 'UNKNOWN'),
            'year': base_data.get('year', datetime.now().year),
            'generated_at': datetime.now(),
            'current_assets': 0,
            'cash_and_equivalents': 0,
            'accounts_receivable': 0,
            'inventory': 0,
            'prepaid_expenses': 0,
            'other_current_assets': 0,
            'non_current_assets': 0,
            'property_plant_equipment': 0,
            'accumulated_depreciation': 0,
            'net_ppe': 0,
            'intangible_assets': 0,
            'goodwill': 0,
            'investments': 0,
            'other_non_current_assets': 0,
            'total_assets': 0,
            'current_liabilities': 0,
            'accounts_payable': 0,
            'short_term_debt': 0,
            'accrued_liabilities': 0,
            'deferred_revenue': 0,
            'other_current_liabilities': 0,
            'non_current_liabilities': 0,
            'long_term_debt': 0,
            'deferred_tax_liabilities': 0,
            'pension_obligations': 0,
            'other_non_current_liabilities': 0,
            'total_liabilities': 0,
            'share_capital': 0,
            'retained_earnings': 0,
            'additional_paid_in_capital': 0,
            'treasury_stock': 0,
            'accumulated_other_comprehensive_income': 0,
            'total_equity': 0,
            'balance_check': 0,
            'accuracy_percentage': 0,
            'data_source': 'predicted',
            'validation_errors': None
        }
        
        # Copy known values
        for key, value in base_data.items():
            if key in predicted_bs and value is not None:
                predicted_bs[key] = value
        
        # Add company info if provided
        if company_info:
            # Add derived features
            total_assets = predicted_bs.get('total_assets', 1) or 1
            current_assets = predicted_bs.get('current_assets', 0) or 0
            current_liabilities = predicted_bs.get('current_liabilities', 1) or 1
            total_equity = predicted_bs.get('total_equity', 1) or 1
            
            base_data['current_ratio'] = current_assets / max(current_liabilities, 1)
            base_data['equity_ratio'] = total_equity / max(total_assets, 1)
            base_data['asset_size_log'] = np.log(max(total_assets, 1))
        
        # Predict missing components
        for component in self.target_components:
            if pd.isna(predicted_bs.get(component)) or predicted_bs.get(component) == 0:
                prediction = self.predict_balance_sheet_component(component, base_data)
                if prediction is not None:
                    predicted_bs[component] = max(0, prediction)  # Ensure non-negative values
        
        # Calculate derived totals
        predicted_bs = self.calculate_balance_sheet_totals(predicted_bs)
        
        # Validate balance equation
        predicted_bs = self.validate_balance_equation(predicted_bs)
        
        return predicted_bs

    def calculate_balance_sheet_totals(self, balance_sheet):
        """Calculate total values for balance sheet"""
        
        # Current Assets Total
        current_asset_components = [
            balance_sheet.get('cash_and_equivalents', 0) or 0,
            balance_sheet.get('accounts_receivable', 0) or 0,
            balance_sheet.get('inventory', 0) or 0,
            balance_sheet.get('prepaid_expenses', 0) or 0,
            balance_sheet.get('other_current_assets', 0) or 0
        ]
        balance_sheet['current_assets'] = sum([x for x in current_asset_components if x is not None])
        
        # Non-Current Assets Total
        non_current_components = [
            balance_sheet.get('property_plant_equipment', 0) or 0,
            balance_sheet.get('intangible_assets', 0) or 0,
            balance_sheet.get('goodwill', 0) or 0,
            balance_sheet.get('investments', 0) or 0,
            balance_sheet.get('other_non_current_assets', 0) or 0
        ]
        balance_sheet['non_current_assets'] = sum([x for x in non_current_components if x is not None])
        
        # Net PPE calculation
        ppe = balance_sheet.get('property_plant_equipment', 0) or 0
        acc_dep = balance_sheet.get('accumulated_depreciation', 0) or 0
        balance_sheet['net_ppe'] = ppe - abs(acc_dep)
        
        # Total Assets
        balance_sheet['total_assets'] = balance_sheet['current_assets'] + balance_sheet['non_current_assets']
        
        # Current Liabilities Total
        current_liab_components = [
            balance_sheet.get('accounts_payable', 0) or 0,
            balance_sheet.get('short_term_debt', 0) or 0,
            balance_sheet.get('accrued_liabilities', 0) or 0,
            balance_sheet.get('deferred_revenue', 0) or 0,
            balance_sheet.get('other_current_liabilities', 0) or 0
        ]
        balance_sheet['current_liabilities'] = sum([x for x in current_liab_components if x is not None])
        
        # Non-Current Liabilities Total
        non_current_liab_components = [
            balance_sheet.get('long_term_debt', 0) or 0,
            balance_sheet.get('deferred_tax_liabilities', 0) or 0,
            balance_sheet.get('pension_obligations', 0) or 0,
            balance_sheet.get('other_non_current_liabilities', 0) or 0
        ]
        balance_sheet['non_current_liabilities'] = sum([x for x in non_current_liab_components if x is not None])
        
        # Total Liabilities
        balance_sheet['total_liabilities'] = balance_sheet['current_liabilities'] + balance_sheet['non_current_liabilities']
        
        # Total Equity (from accounting equation)
        balance_sheet['total_equity'] = balance_sheet['total_assets'] - balance_sheet['total_liabilities']
        
        return balance_sheet

    def validate_balance_equation(self, balance_sheet):
        """Validate and adjust balance sheet equation"""
        total_assets = balance_sheet.get('total_assets', 0) or 0
        total_liabilities = balance_sheet.get('total_liabilities', 0) or 0
        total_equity = balance_sheet.get('total_equity', 0) or 0
        
        # Check balance equation: Assets = Liabilities + Equity
        total_liab_equity = total_liabilities + total_equity
        
        if total_assets > 0:
            balance_difference = abs(total_assets - total_liab_equity)
            balance_sheet['balance_check'] = balance_difference / total_assets
            
            # Calculate accuracy percentage
            accuracy = max(0, 100 - (balance_sheet['balance_check'] * 100))
            balance_sheet['accuracy_percentage'] = accuracy
            
            # If balance is significantly off, adjust equity
            if balance_sheet['balance_check'] > 0.05:  # 5% threshold
                balance_sheet['total_equity'] = total_assets - total_liabilities
                balance_sheet['validation_errors'] = f"Balance adjusted: difference was {balance_difference:.2f}"
        else:
            balance_sheet['balance_check'] = 1.0
            balance_sheet['accuracy_percentage'] = 0.0
            balance_sheet['validation_errors'] = "Total assets is zero or negative"
        
        return balance_sheet

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
            
            print(f"Balance sheet models saved to {filepath}")
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
            
            print(f"Balance sheet models loaded from {filepath}")
            print(f"Trained at: {model_data.get('trained_at', 'Unknown')}")
            print(f"Available components: {list(self.models.keys())}")
            
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    def save_prediction_to_db(self, balance_sheet_data):
        """Save predicted balance sheet to balance_sheet_1 table"""
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
            print(f"Error saving predicted balance sheet to balance_sheet_1 table: {e}")
            if conn:
                conn.rollback()
                conn.close()
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
                    pred = self.predict_balance_sheet_component(component, X.iloc[[idx]])
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
        Retrain all models with latest data from balance_sheet_1 table
        
        Args:
            min_records: Minimum records required for training
            
        Returns:
            dict: Retraining results
        """
        print("Loading latest training data from balance_sheet_1 table...")
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

    def predict_future_balance_sheets(self, company_id, years_ahead=3, growth_assumptions=None):
        """
        Predict balance sheets for future years
        
        Args:
            company_id: Company identifier
            years_ahead: Number of years to predict
            growth_assumptions: Dictionary of growth rates for different components
            
        Returns:
            list: Predicted balance sheets for future years
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
            
            # Default growth assumptions
            if growth_assumptions is None:
                growth_assumptions = {
                    'revenue_growth': 0.10,  # 10% annual growth
                    'asset_growth': 0.08,    # 8% asset growth
                    'debt_growth': 0.05      # 5% debt growth
                }
            
            for year_offset in range(1, years_ahead + 1):
                future_year = current_year + year_offset
                
                # Project future balance sheet components
                future_bs = dict(latest_bs)
                
                # Apply growth rates
                for key in ['total_assets', 'current_assets', 'property_plant_equipment']:
                    if future_bs.get(key):
                        future_bs[key] *= (1 + growth_assumptions['asset_growth']) ** year_offset
                
                for key in ['long_term_debt', 'short_term_debt']:
                    if future_bs.get(key):
                        future_bs[key] *= (1 + growth_assumptions['debt_growth']) ** year_offset
                
                future_bs['year'] = future_year
                future_bs['data_source'] = 'predicted_future'
                
                # Use ML model to refine predictions
                predicted_bs = self.predict_complete_balance_sheet(future_bs)
                predictions.append(predicted_bs)
            
            return predictions
            
        except Exception as e:
            print(f"Error predicting future balance sheets for {company_id}: {e}")
            if conn:
                conn.close()
            return None

    def generate_balance_sheet_scenarios(self, base_data, scenarios):
        """
        Generate balance sheet predictions for different scenarios
        
        Args:
            base_data: Base balance sheet data
            scenarios: List of scenario dictionaries with assumptions
            
        Returns:
            dict: Predictions for each scenario
        """
        scenario_predictions = {}
        
        for scenario_name, scenario_assumptions in scenarios.items():
            # Apply scenario assumptions to base data
            scenario_data = base_data.copy()
            
            for key, multiplier in scenario_assumptions.items():
                if key in scenario_data and scenario_data[key] is not None:
                    scenario_data[key] *= multiplier
            
            # Generate prediction for this scenario
            prediction = self.predict_complete_balance_sheet(scenario_data)
            scenario_predictions[scenario_name] = prediction
        
        return scenario_predictions