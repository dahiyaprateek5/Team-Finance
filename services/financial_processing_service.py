import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import logging
from typing import Dict, List, Optional, Tuple, Any
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import warnings
import json
warnings.filterwarnings('ignore')

class FinancialProcessingService:
    """
    Financial Processing Service for comprehensive financial analysis
    Fetches data from cash_flow_statement and balance_sheet_1 tables
    Provides risk assessment, financial health scoring, and predictive analytics
    """
    
    def __init__(self, db_config: Dict[str, str] = None):
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        
        # Initialize ML models
        self.risk_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.cash_flow_model = RandomForestRegressor(n_estimators=150, random_state=42)
        self.liquidation_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Database connection
        if db_config is None:
            db_config = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'database': os.getenv('DB_NAME', 'financial_risk_db'),
                'user': os.getenv('DB_USER', 'postgres'),
                'password': os.getenv('DB_PASSWORD', 'password'),
                'port': os.getenv('DB_PORT', '5432')
            }
        
        self.db_config = db_config
        self.engine = create_engine(
            f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        
        self._initialize_models()
    
    def get_db_connection(self):
        """Get database connection"""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except Exception as e:
            self.logger.error(f"Database connection failed: {e}")
            raise
    
    def _initialize_models(self):
        """Initialize ML models with data from database"""
        try:
            # Fetch training data
            training_data = self._fetch_training_data()
            if not training_data.empty:
                self._train_all_models(training_data)
                self.logger.info("Financial processing models initialized successfully")
            else:
                self.logger.warning("No training data available for financial models")
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
    
    def _fetch_training_data(self) -> pd.DataFrame:
        """Fetch comprehensive training data from cash_flow_statement and balance_sheet_1"""
        query = """
        SELECT 
            cf.company_id,
            cf.year,
            cf.net_income,
            cf.depreciation_and_amortization,
            cf.stock_based_compensation,
            cf.changes_in_working_capital,
            cf.accounts_receivable,
            cf.inventory,
            cf.accounts_payable,
            cf.net_cash_from_operating_activities,
            cf.capital_expenditures,
            cf.acquisitions,
            cf.net_cash_from_investing_activities,
            cf.dividends_paid,
            cf.share_repurchases,
            cf.net_cash_from_financing_activities,
            cf.free_cash_flow,
            cf.ocf_to_net_income_ratio,
            cf.liquidation_label,
            cf.debt_to_equity_ratio,
            cf.interest_coverage_ratio,
            bs.total_assets,
            bs.total_liabilities,
            bs.total_equity,
            bs.current_assets,
            bs.current_liabilities,
            bs.cash_and_equivalents,
            bs.property_plant_equipment,
            c.industry,
            c.sector,
            c.company_size
        FROM cash_flow_statement cf
        JOIN balance_sheet_1 bs ON cf.company_id = bs.company_id AND cf.year = bs.year
        JOIN companies c ON cf.company_id = c.id
        WHERE cf.year >= 2020 
        AND cf.net_income IS NOT NULL 
        AND bs.total_assets IS NOT NULL
        ORDER BY cf.company_id, cf.year
        """
        
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn)
            return df
        except Exception as e:
            self.logger.error(f"Failed to fetch training data: {e}")
            return pd.DataFrame()
    
    def _train_all_models(self, data: pd.DataFrame):
        """Train all ML models with fetched data"""
        try:
            # Prepare data for training
            data_encoded = self._prepare_training_features(data)
            
            if len(data_encoded) < 10:
                self.logger.warning("Insufficient data for model training")
                return
            
            # Train liquidation risk model
            self._train_liquidation_model(data_encoded)
            
            # Train cash flow prediction model
            self._train_cash_flow_model(data_encoded)
            
            # Train financial risk model
            self._train_risk_model(data_encoded)
            
            self.logger.info("All financial models trained successfully")
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
    
    def _prepare_training_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model training"""
        data_encoded = data.copy()
        
        # Encode categorical variables
        data_encoded['industry_encoded'] = pd.Categorical(data['industry']).codes
        data_encoded['sector_encoded'] = pd.Categorical(data['sector']).codes
        data_encoded['company_size_encoded'] = pd.Categorical(data['company_size']).codes
        
        # Create additional financial ratios
        data_encoded['current_ratio'] = data_encoded['current_assets'] / (data_encoded['current_liabilities'] + 0.001)
        data_encoded['roa'] = data_encoded['net_income'] / (data_encoded['total_assets'] + 0.001)
        data_encoded['cash_to_assets'] = data_encoded['net_cash_from_operating_activities'] / (data_encoded['total_assets'] + 0.001)
        
        # Fill missing values
        numeric_columns = data_encoded.select_dtypes(include=[np.number]).columns
        data_encoded[numeric_columns] = data_encoded[numeric_columns].fillna(0)
        
        return data_encoded
    
    def _train_liquidation_model(self, data: pd.DataFrame):
        """Train liquidation prediction model"""
        try:
            features = [
                'net_income', 'net_cash_from_operating_activities', 
                'debt_to_equity_ratio', 'free_cash_flow', 'total_assets',
                'ocf_to_net_income_ratio', 'industry_encoded', 'sector_encoded',
                'current_assets', 'current_liabilities', 'current_ratio'
            ]
            
            available_features = [f for f in features if f in data.columns]
            
            if len(available_features) < 5 or 'liquidation_label' not in data.columns:
                self.logger.warning("Insufficient features for liquidation model training")
                return
            
            X = data[available_features].fillna(0)
            y = data['liquidation_label'].fillna(0)
            
            if len(X) > 10:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Scale features
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                
                # Train model
                self.liquidation_model.fit(X_train_scaled, y_train)
                
                # Evaluate
                y_pred = self.liquidation_model.predict(X_test_scaled)
                accuracy = r2_score(y_test, y_pred)
                self.logger.info(f"Liquidation model accuracy: {accuracy:.3f}")
                
        except Exception as e:
            self.logger.error(f"Liquidation model training failed: {e}")
    
    def _train_cash_flow_model(self, data: pd.DataFrame):
        """Train cash flow prediction model"""
        try:
            features = [
                'net_income', 'depreciation_and_amortization', 'changes_in_working_capital',
                'total_assets', 'current_assets', 'current_liabilities',
                'accounts_receivable', 'inventory', 'accounts_payable',
                'industry_encoded', 'sector_encoded'
            ]
            
            available_features = [f for f in features if f in data.columns]
            
            if len(available_features) < 5 or 'net_cash_from_operating_activities' not in data.columns:
                self.logger.warning("Insufficient features for cash flow model training")
                return
            
            X = data[available_features].fillna(0)
            y = data['net_cash_from_operating_activities'].fillna(0)
            
            if len(X) > 10:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train model
                self.cash_flow_model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = self.cash_flow_model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                self.logger.info(f"Cash flow model MAE: {mae:.2f}")
                
        except Exception as e:
            self.logger.error(f"Cash flow model training failed: {e}")
    
    def _train_risk_model(self, data: pd.DataFrame):
        """Train financial risk assessment model"""
        try:
            features = [
                'debt_to_equity_ratio', 'interest_coverage_ratio',
                'free_cash_flow', 'net_income', 'total_assets',
                'ocf_to_net_income_ratio', 'industry_encoded', 'sector_encoded',
                'current_assets', 'current_liabilities', 'total_liabilities', 'current_ratio'
            ]
            
            available_features = [f for f in features if f in data.columns]
            
            if len(available_features) < 6:
                self.logger.warning("Insufficient features for risk model training")
                return
            
            # Create risk score target (composite score)
            data['risk_score'] = self._calculate_risk_score(data)
            
            X = data[available_features].fillna(0)
            y = data['risk_score'].fillna(50)  # Default medium risk
            
            if len(X) > 10:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train model
                self.risk_model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = self.risk_model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                self.logger.info(f"Risk model R² score: {r2:.3f}")
                
        except Exception as e:
            self.logger.error(f"Risk model training failed: {e}")
    
    def _calculate_risk_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate composite risk score for training"""
        risk_score = pd.Series(50, index=data.index)  # Start with medium risk (50)
        
        # Adjust based on financial ratios
        # Calculate current ratio from balance sheet data
        if 'current_assets' in data.columns and 'current_liabilities' in data.columns:
            current_ratio = data['current_assets'] / (data['current_liabilities'] + 0.001)  # Avoid division by zero
            risk_score -= (current_ratio - 1.5) * 10  # Good liquidity reduces risk
        
        if 'debt_to_equity_ratio' in data.columns:
            risk_score += data['debt_to_equity_ratio'] * 20  # High debt increases risk
        
        if 'net_income' in data.columns:
            risk_score -= (data['net_income'] > 0) * 15  # Profitability reduces risk
        
        if 'free_cash_flow' in data.columns:
            risk_score -= (data['free_cash_flow'] > 0) * 10  # Positive FCF reduces risk
        
        # Clamp to 0-100 range
        risk_score = np.clip(risk_score, 0, 100)
        
        return risk_score
    
    def get_company_financial_data(self, company_id: int, years: List[int] = None) -> Dict[str, Any]:
        """Fetch comprehensive financial data for a company"""
        if years is None:
            years = [2022, 2023, 2024]
        
        year_placeholders = ','.join(['%s'] * len(years))
        
        query = f"""
        SELECT 
            cf.*,
            bs.total_assets,
            bs.total_liabilities,
            bs.total_equity,
            bs.current_assets,
            bs.current_liabilities,
            bs.cash_and_equivalents as bs_cash,
            c.company_name,
            c.industry,
            c.sector
        FROM cash_flow_statement cf
        JOIN balance_sheet_1 bs ON cf.company_id = bs.company_id AND cf.year = bs.year
        JOIN companies c ON cf.company_id = c.id
        WHERE cf.company_id = %s 
        AND cf.year IN ({year_placeholders})
        ORDER BY cf.year DESC
        """
        
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query, [company_id] + years)
            results = cursor.fetchall()
            conn.close()
            
            return {
                'success': True,
                'data': [dict(row) for row in results],
                'company_id': company_id,
                'years_available': [row['year'] for row in results]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to fetch company financial data: {e}")
            return {'success': False, 'error': str(e)}
    
    def analyze_financial_health(self, company_id: int) -> Dict[str, Any]:
        """Comprehensive financial health analysis"""
        try:
            # Fetch company data
            financial_data = self.get_company_financial_data(company_id)
            
            if not financial_data['success'] or not financial_data['data']:
                return {
                    'success': False,
                    'error': 'No financial data available for analysis'
                }
            
            data = financial_data['data']
            latest_data = data[0]  # Most recent year
            
            # Calculate financial health metrics
            health_metrics = self._calculate_health_metrics(data)
            
            # Predict liquidation risk
            liquidation_risk = self._predict_liquidation_risk(latest_data)
            
            # Generate cash flow forecast
            cash_flow_forecast = self._forecast_cash_flow(data)
            
            # Calculate overall financial score
            financial_score = self._calculate_financial_score(health_metrics, liquidation_risk)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(health_metrics, latest_data)
            
            # Save analysis to database
            self._save_analysis_to_db(company_id, {
                'health_metrics': health_metrics,
                'liquidation_risk': liquidation_risk,
                'financial_score': financial_score,
                'analysis_date': datetime.now()
            })
            
            return {
                'success': True,
                'company_id': company_id,
                'company_name': latest_data.get('company_name'),
                'analysis_date': datetime.now().isoformat(),
                'financial_score': financial_score,
                'health_metrics': health_metrics,
                'liquidation_risk': liquidation_risk,
                'cash_flow_forecast': cash_flow_forecast,
                'recommendations': recommendations,
                'data_years': financial_data['years_available']
            }
            
        except Exception as e:
            self.logger.error(f"Financial health analysis failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _calculate_health_metrics(self, data: List[Dict]) -> Dict[str, float]:
        """Calculate key financial health metrics"""
        metrics = {}
        
        try:
            latest = data[0]
            
            # Liquidity ratios
            if latest.get('current_assets') and latest.get('current_liabilities'):
                metrics['current_ratio'] = latest['current_assets'] / latest['current_liabilities']
            
            # Profitability ratios (need revenue from external source or calculate from cash flow)
            if latest.get('net_income') and latest.get('total_assets'):
                metrics['roa'] = latest['net_income'] / latest['total_assets']
            
            # Cash flow metrics
            if latest.get('net_cash_from_operating_activities') and latest.get('net_income'):
                metrics['ocf_to_net_income'] = latest['net_cash_from_operating_activities'] / latest['net_income']
            
            # Leverage ratios
            if latest.get('debt_to_equity_ratio'):
                metrics['debt_to_equity'] = latest['debt_to_equity_ratio']
            
            if latest.get('interest_coverage_ratio'):
                metrics['interest_coverage'] = latest['interest_coverage_ratio']
            
            # Cash flow efficiency
            if latest.get('free_cash_flow') and latest.get('net_cash_from_operating_activities'):
                metrics['fcf_margin'] = latest['free_cash_flow'] / latest['net_cash_from_operating_activities']
            
            # Growth metrics (if multiple years available)
            if len(data) >= 2:
                current_assets = latest.get('total_assets', 0)
                previous_assets = data[1].get('total_assets', 0)
                if previous_assets > 0:
                    metrics['asset_growth'] = (current_assets - previous_assets) / previous_assets
            
        except Exception as e:
            self.logger.error(f"Health metrics calculation failed: {e}")
        
        return metrics
    
    def _predict_liquidation_risk(self, data: Dict) -> Dict[str, float]:
        """Predict liquidation risk using trained model"""
        try:
            features = [
                'net_income', 'net_cash_from_operating_activities', 
                'debt_to_equity_ratio', 'free_cash_flow', 'total_assets',
                'ocf_to_net_income_ratio', 'current_assets', 'current_liabilities'
            ]
            
            # Prepare feature vector
            feature_vector = []
            for feature in features:
                if feature in data and data[feature] is not None:
                    feature_vector.append(float(data[feature]))
                else:
                    feature_vector.append(0.0)
            
            # Add current ratio
            if data.get('current_assets') and data.get('current_liabilities'):
                current_ratio = data['current_assets'] / data['current_liabilities']
                feature_vector.append(current_ratio)
            else:
                feature_vector.append(1.0)
            
            # Add encoded categorical features (placeholder values)
            feature_vector.extend([0, 0])  # industry_encoded, sector_encoded
            
            # Make prediction
            if hasattr(self.liquidation_model, 'predict'):
                feature_array = np.array(feature_vector).reshape(1, -1)
                feature_array_scaled = self.scaler.transform(feature_array)
                risk_score = self.liquidation_model.predict(feature_array_scaled)[0]
                confidence = min(max(abs(risk_score), 0.6), 0.95)  # Confidence between 60-95%
            else:
                # Fallback calculation
                risk_score = self._calculate_fallback_liquidation_risk(data)
                confidence = 0.7
            
            return {
                'risk_score': float(np.clip(risk_score, 0, 1)),
                'risk_level': 'High' if risk_score > 0.7 else 'Medium' if risk_score > 0.3 else 'Low',
                'confidence': float(confidence)
            }
            
        except Exception as e:
            self.logger.error(f"Liquidation risk prediction failed: {e}")
            return {'risk_score': 0.5, 'risk_level': 'Medium', 'confidence': 0.6}
    
    def _calculate_fallback_liquidation_risk(self, data: Dict) -> float:
        """Fallback liquidation risk calculation"""
        risk_factors = []
        
        # Negative income
        if data.get('net_income', 0) < 0:
            risk_factors.append(0.3)
        
        # Negative cash flow
        if data.get('net_cash_from_operating_activities', 0) < 0:
            risk_factors.append(0.4)
        
        # Low liquidity
        if data.get('current_assets', 0) > 0 and data.get('current_liabilities', 0) > 0:
            current_ratio = data['current_assets'] / data['current_liabilities']
            if current_ratio < 1.0:
                risk_factors.append(0.3)
        
        # High leverage
        debt_to_equity = data.get('debt_to_equity_ratio', 0.5)
        if debt_to_equity > 2.0:
            risk_factors.append(0.2)
        
        return min(sum(risk_factors), 1.0)
    
    def _calculate_financial_score(self, health_metrics: Dict[str, float], liquidation_risk: Dict[str, float]) -> Dict[str, Any]:
        """Calculate overall financial health score (0-100)"""
        try:
            score = 50  # Start with neutral score
            
            # Liquidity component (25% weight)
            current_ratio = health_metrics.get('current_ratio', 1.5)
            if current_ratio >= 2.0:
                score += 20
            elif current_ratio >= 1.5:
                score += 15
            elif current_ratio >= 1.0:
                score += 10
            else:
                score -= 10
            
            # Asset utilization component (25% weight)
            roa = health_metrics.get('roa', 0)
            if roa > 0.1:
                score += 20
            elif roa > 0.05:
                score += 15
            elif roa > 0:
                score += 10
            else:
                score -= 15
            
            # Cash flow component (25% weight)
            ocf_ratio = health_metrics.get('ocf_to_net_income', 1.0)
            if ocf_ratio > 1.2:
                score += 20
            elif ocf_ratio > 1.0:
                score += 15
            elif ocf_ratio > 0.8:
                score += 10
            else:
                score -= 10
            
            # Risk component (25% weight)
            risk_score = liquidation_risk.get('risk_score', 0.5)
            score -= (risk_score * 25)  # Higher risk reduces score
            
            # Clamp score to 0-100
            final_score = max(0, min(100, score))
            
            # Determine rating
            if final_score >= 80:
                rating = 'Excellent'
            elif final_score >= 60:
                rating = 'Good'
            elif final_score >= 40:
                rating = 'Fair'
            else:
                rating = 'Poor'
            
            return {
                'score': float(final_score),
                'rating': rating,
                'components': {
                    'liquidity': min(25, max(0, 15 + (current_ratio - 1.5) * 10)),
                    'profitability': min(25, max(0, 15 + roa * 100)),
                    'cash_flow': min(25, max(0, 15 + (ocf_ratio - 1.0) * 10)),
                    'risk': min(25, max(0, 25 - risk_score * 25))
                }
            }
            
        except Exception as e:
            self.logger.error(f"Financial score calculation failed: {e}")
            return {'score': 50.0, 'rating': 'Fair'}
    
    def _generate_recommendations(self, health_metrics: Dict[str, float], latest_data: Dict) -> List[Dict[str, Any]]:
        """Generate actionable financial recommendations"""
        recommendations = []
        
        try:
            # Liquidity recommendations
            current_ratio = health_metrics.get('current_ratio', 1.5)
            if current_ratio < 1.2:
                recommendations.append({
                    'category': 'Liquidity',
                    'priority': 'High',
                    'issue': 'Low current ratio indicates potential liquidity problems',
                    'recommendation': 'Improve working capital management, accelerate receivables collection, or secure short-term financing',
                    'impact': 'Improved cash flow and reduced financial stress',
                    'timeline': '1-3 months'
                })
            
            # Cash flow recommendations
            ocf_ratio = health_metrics.get('ocf_to_net_income', 1.0)
            if ocf_ratio < 0.8:
                recommendations.append({
                    'category': 'Cash Flow',
                    'priority': 'Medium',
                    'issue': 'Operating cash flow is low relative to net income',
                    'recommendation': 'Focus on working capital optimization and cash conversion cycle improvement',
                    'impact': 'Better cash generation from operations',
                    'timeline': '2-4 months'
                })
            
            # Debt recommendations
            debt_to_equity = health_metrics.get('debt_to_equity', 0.5)
            if debt_to_equity > 1.5:
                recommendations.append({
                    'category': 'Leverage',
                    'priority': 'Medium',
                    'issue': 'High debt-to-equity ratio indicates excessive leverage',
                    'recommendation': 'Consider debt reduction strategies or equity financing',
                    'impact': 'Reduced financial risk and interest expenses',
                    'timeline': '6-12 months'
                })
            
            # Asset efficiency recommendations
            roa = health_metrics.get('roa', 0)
            if roa < 0.02:  # Less than 2% ROA
                recommendations.append({
                    'category': 'Asset Efficiency',
                    'priority': 'Low',
                    'issue': 'Low return on assets indicates inefficient asset utilization',
                    'recommendation': 'Review asset utilization, consider asset optimization or divestment of underperforming assets',
                    'impact': 'Improved profitability and asset efficiency',
                    'timeline': '6-18 months'
                })
            
        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {e}")
        
        return recommendations
    
    def _save_analysis_to_db(self, company_id: int, analysis_data: Dict):
        """Save financial analysis results to database"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            insert_query = """
            INSERT INTO financial_analysis (
                company_id, analysis_date, financial_score, 
                liquidation_risk_score, health_metrics, 
                created_at, updated_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (company_id, analysis_date) 
            DO UPDATE SET
                financial_score = EXCLUDED.financial_score,
                liquidation_risk_score = EXCLUDED.liquidation_risk_score,
                health_metrics = EXCLUDED.health_metrics,
                updated_at = EXCLUDED.updated_at
            """
            
            cursor.execute(insert_query, (
                company_id,
                analysis_data['analysis_date'].date(),
                analysis_data['financial_score'],
                analysis_data['liquidation_risk']['risk_score'],
                json.dumps(analysis_data['health_metrics']),
                datetime.now(),
                datetime.now()
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            self.logger.info(f"Financial analysis saved for company {company_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to save analysis: {e}")
            if 'conn' in locals():
                conn.rollback()
                conn.close()
    
    def _forecast_cash_flow(self, data: List[Dict]) -> Dict[str, Any]:
        """Forecast future cash flow"""
        try:
            if len(data) < 2:
                return {'error': 'Insufficient data for forecasting'}
            
            # Extract cash flow trends
            cash_flows = [d.get('net_cash_from_operating_activities', 0) for d in reversed(data)]
            years = [d.get('year') for d in reversed(data)]
            
            # Simple linear trend
            if len(cash_flows) >= 2:
                growth_rate = (cash_flows[-1] - cash_flows[0]) / len(cash_flows)
                next_year_forecast = cash_flows[-1] + growth_rate
                
                return {
                    'next_year_forecast': float(next_year_forecast),
                    'growth_rate': float(growth_rate),
                    'trend': 'Improving' if growth_rate > 0 else 'Declining',
                    'historical_data': list(zip(years, cash_flows))
                }
            
        except Exception as e:
            self.logger.error(f"Cash flow forecasting failed: {e}")
        
        return {'error': 'Forecasting calculation failed'}
    
    def get_industry_comparison(self, company_id: int) -> Dict[str, Any]:
        """Get industry comparison and benchmarking"""
        try:
            # Get company info
            company_query = """
            SELECT industry, sector, company_size 
            FROM companies 
            WHERE id = %s
            """
            
            conn = self.get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(company_query, (company_id,))
            company_info = cursor.fetchone()
            
            if not company_info:
                return {'success': False, 'error': 'Company not found'}
            
            # Get industry peers data
            peers_query = """
            SELECT 
                cf.net_income,
                cf.net_cash_from_operating_activities,
                cf.debt_to_equity_ratio,
                cf.free_cash_flow,
                bs.total_assets,
                bs.current_assets,
                bs.current_liabilities
            FROM cash_flow_statement cf
            JOIN balance_sheet_1 bs ON cf.company_id = bs.company_id AND cf.year = bs.year
            JOIN companies c ON cf.company_id = c.id
            WHERE c.industry = %s 
            AND c.sector = %s
            AND cf.year = (SELECT MAX(year) FROM cash_flow_statement)
            AND cf.company_id != %s
            """
            
            cursor.execute(peers_query, (
                company_info['industry'], 
                company_info['sector'], 
                company_id
            ))
            peers_data = cursor.fetchall()
            conn.close()
            
            if not peers_data:
                return {'success': False, 'error': 'No peer data available'}
            
            # Calculate industry benchmarks
            peers_df = pd.DataFrame([dict(row) for row in peers_data])
            
            # Calculate current ratio for peers
            peers_df['current_ratio'] = peers_df['current_assets'] / (peers_df['current_liabilities'] + 0.001)
            
            benchmarks = {}
            for column in peers_df.select_dtypes(include=[np.number]).columns:
                benchmarks[column] = {
                    'median': float(peers_df[column].median()),
                    'mean': float(peers_df[column].mean()),
                    'percentile_25': float(peers_df[column].quantile(0.25)),
                    'percentile_75': float(peers_df[column].quantile(0.75))
                }
            
            # Get company's current metrics
            company_data = self.get_company_financial_data(company_id, [2024])
            
            return {
                'success': True,
                'company_id': company_id,
                'industry': company_info['industry'],
                'sector': company_info['sector'],
                'benchmarks': benchmarks,
                'peer_count': len(peers_data),
                'company_metrics': company_data['data'][0] if company_data['success'] and company_data['data'] else None
            }
            
        except Exception as e:
            self.logger.error(f"Industry comparison failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def generate_cash_flow_statement(self, company_id: int, source_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate cash flow statement using indirect method"""
        try:
            self.logger.info(f"Generating cash flow statement for company {company_id}")
            
            # Fetch balance sheet data for 2 years
            balance_sheet_data = self._fetch_balance_sheet_data(company_id, [2023, 2024])
            
            if len(balance_sheet_data) < 2:
                return {
                    'success': False,
                    'error': 'Insufficient balance sheet data for cash flow generation'
                }
            
            current_year = balance_sheet_data[0]
            previous_year = balance_sheet_data[1]
            
            # Calculate cash flow statement using indirect method
            cash_flow_statement = self._calculate_cash_flow_indirect_method(
                current_year, previous_year, source_data
            )
            
            # Save to cash_flow_statement table
            self._save_cash_flow_to_db(company_id, cash_flow_statement)
            
            return {
                'success': True,
                'company_id': company_id,
                'cash_flow_statement': cash_flow_statement,
                'method': 'Indirect Method',
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Cash flow statement generation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _fetch_balance_sheet_data(self, company_id: int, years: List[int]) -> List[Dict]:
        """Fetch balance sheet data for specified years"""
        year_placeholders = ','.join(['%s'] * len(years))
        
        query = f"""
        SELECT * FROM balance_sheet_1 
        WHERE company_id = %s 
        AND year IN ({year_placeholders})
        ORDER BY year DESC
        """
        
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query, [company_id] + years)
            results = cursor.fetchall()
            conn.close()
            
            return [dict(row) for row in results]
            
        except Exception as e:
            self.logger.error(f"Failed to fetch balance sheet data: {e}")
            return []
    
    def _calculate_cash_flow_indirect_method(self, current: Dict, previous: Dict, additional_data: Dict) -> Dict[str, float]:
        """Calculate cash flow statement using indirect method"""
        cash_flow = {}
        
        try:
            # Get company info
            company_query = """
            SELECT company_name, industry FROM companies WHERE id = %s
            """
            conn = self.get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(company_query, (current.get('company_id'),))
            company_info = cursor.fetchone()
            conn.close()
            
            if company_info:
                cash_flow['company_name'] = company_info['company_name']
                cash_flow['industry'] = company_info['industry']
            
            # Operating Activities
            cash_flow['net_income'] = additional_data.get('net_income', 0)
            
            # Add back non-cash expenses
            cash_flow['depreciation_and_amortization'] = additional_data.get('depreciation', 
                abs(current.get('accumulated_depreciation', 0) - previous.get('accumulated_depreciation', 0)))
            cash_flow['stock_based_compensation'] = additional_data.get('stock_compensation', 0)
            
            # Changes in working capital
            ar_change = current.get('accounts_receivable', 0) - previous.get('accounts_receivable', 0)
            inventory_change = current.get('inventory', 0) - previous.get('inventory', 0)
            ap_change = current.get('accounts_payable', 0) - previous.get('accounts_payable', 0)
            
            cash_flow['accounts_receivable'] = ar_change
            cash_flow['inventory'] = inventory_change
            cash_flow['accounts_payable'] = ap_change
            
            cash_flow['changes_in_working_capital'] = -(ar_change + inventory_change) + ap_change
            
            # Net Cash from Operating Activities
            cash_flow['net_cash_from_operating_activities'] = (
                cash_flow['net_income'] +
                cash_flow['depreciation_and_amortization'] +
                cash_flow['stock_based_compensation'] +
                cash_flow['changes_in_working_capital']
            )
            
            # Investing Activities
            ppe_change = current.get('property_plant_equipment', 0) - previous.get('property_plant_equipment', 0)
            cash_flow['capital_expenditures'] = -(ppe_change + cash_flow['depreciation_and_amortization'])
            cash_flow['acquisitions'] = additional_data.get('acquisitions', 0)
            
            cash_flow['net_cash_from_investing_activities'] = (
                cash_flow['capital_expenditures'] - cash_flow['acquisitions']
            )
            
            # Financing Activities
            long_term_debt_change = current.get('long_term_debt', 0) - previous.get('long_term_debt', 0)
            short_term_debt_change = current.get('short_term_debt', 0) - previous.get('short_term_debt', 0)
            
            cash_flow['dividends_paid'] = additional_data.get('dividends', 0)
            cash_flow['share_repurchases'] = additional_data.get('share_repurchases', 0)
            
            cash_flow['net_cash_from_financing_activities'] = (
                long_term_debt_change + short_term_debt_change - 
                cash_flow['dividends_paid'] - cash_flow['share_repurchases']
            )
            
            # Additional metrics
            cash_flow['free_cash_flow'] = (
                cash_flow['net_cash_from_operating_activities'] - 
                abs(cash_flow['capital_expenditures'])
            )
            
            if cash_flow['net_income'] != 0:
                cash_flow['ocf_to_net_income_ratio'] = (
                    cash_flow['net_cash_from_operating_activities'] / cash_flow['net_income']
                )
            else:
                cash_flow['ocf_to_net_income_ratio'] = 0
            
            # Calculate financial ratios
            total_debt = current.get('long_term_debt', 0) + current.get('short_term_debt', 0)
            total_equity = current.get('total_equity', 1)
            cash_flow['debt_to_equity_ratio'] = total_debt / total_equity if total_equity > 0 else 0
            
            # Interest coverage ratio (requires EBIT calculation)
            ebit = cash_flow['net_income'] + additional_data.get('interest_expense', 0) + additional_data.get('taxes', 0)
            interest_expense = additional_data.get('interest_expense', 1)
            cash_flow['interest_coverage_ratio'] = ebit / interest_expense if interest_expense > 0 else 0
            
            # Liquidation risk assessment
            cash_flow['liquidation_label'] = self._calculate_liquidation_label([previous, current])
            
            # Add year
            cash_flow['year'] = current.get('year', datetime.now().year)
            
        except Exception as e:
            self.logger.error(f"Cash flow calculation failed: {e}")
        
        return cash_flow
    
    def _calculate_liquidation_label(self, historical_data: List[Dict]) -> int:
        """Calculate liquidation label based on 2+ years of negative performance"""
        negative_years = 0
        
        for year_data in historical_data:
            net_income = year_data.get('net_income', 0)
            operating_cf = year_data.get('net_cash_from_operating_activities', 0)
            
            if net_income < 0 and operating_cf < 0:
                negative_years += 1
        
        return 1 if negative_years >= 2 else 0
    
    def _save_cash_flow_to_db(self, company_id: int, cash_flow_data: Dict[str, float]):
        """Save generated cash flow statement to cash_flow_statement table"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            insert_query = """
            INSERT INTO cash_flow_statement (
                company_id, year, generated_at, company_name, industry,
                net_income, depreciation_and_amortization, stock_based_compensation,
                changes_in_working_capital, accounts_receivable, inventory,
                accounts_payable, net_cash_from_operating_activities,
                capital_expenditures, acquisitions, net_cash_from_investing_activities,
                dividends_paid, share_repurchases, net_cash_from_financing_activities,
                free_cash_flow, ocf_to_net_income_ratio, liquidation_label,
                debt_to_equity_ratio, interest_coverage_ratio
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (company_id, year) 
            DO UPDATE SET
                net_income = EXCLUDED.net_income,
                depreciation_and_amortization = EXCLUDED.depreciation_and_amortization,
                stock_based_compensation = EXCLUDED.stock_based_compensation,
                changes_in_working_capital = EXCLUDED.changes_in_working_capital,
                accounts_receivable = EXCLUDED.accounts_receivable,
                inventory = EXCLUDED.inventory,
                accounts_payable = EXCLUDED.accounts_payable,
                net_cash_from_operating_activities = EXCLUDED.net_cash_from_operating_activities,
                capital_expenditures = EXCLUDED.capital_expenditures,
                acquisitions = EXCLUDED.acquisitions,
                net_cash_from_investing_activities = EXCLUDED.net_cash_from_investing_activities,
                dividends_paid = EXCLUDED.dividends_paid,
                share_repurchases = EXCLUDED.share_repurchases,
                net_cash_from_financing_activities = EXCLUDED.net_cash_from_financing_activities,
                free_cash_flow = EXCLUDED.free_cash_flow,
                ocf_to_net_income_ratio = EXCLUDED.ocf_to_net_income_ratio,
                liquidation_label = EXCLUDED.liquidation_label,
                debt_to_equity_ratio = EXCLUDED.debt_to_equity_ratio,
                interest_coverage_ratio = EXCLUDED.interest_coverage_ratio,
                generated_at = EXCLUDED.generated_at
            """
            
            cursor.execute(insert_query, (
                company_id,
                cash_flow_data.get('year', datetime.now().year),
                datetime.now(),
                cash_flow_data.get('company_name', ''),
                cash_flow_data.get('industry', ''),
                cash_flow_data.get('net_income', 0),
                cash_flow_data.get('depreciation_and_amortization', 0),
                cash_flow_data.get('stock_based_compensation', 0),
                cash_flow_data.get('changes_in_working_capital', 0),
                cash_flow_data.get('accounts_receivable', 0),
                cash_flow_data.get('inventory', 0),
                cash_flow_data.get('accounts_payable', 0),
                cash_flow_data.get('net_cash_from_operating_activities', 0),
                cash_flow_data.get('capital_expenditures', 0),
                cash_flow_data.get('acquisitions', 0),
                cash_flow_data.get('net_cash_from_investing_activities', 0),
                cash_flow_data.get('dividends_paid', 0),
                cash_flow_data.get('share_repurchases', 0),
                cash_flow_data.get('net_cash_from_financing_activities', 0),
                cash_flow_data.get('free_cash_flow', 0),
                cash_flow_data.get('ocf_to_net_income_ratio', 0),
                cash_flow_data.get('liquidation_label', 0),
                cash_flow_data.get('debt_to_equity_ratio', 0),
                cash_flow_data.get('interest_coverage_ratio', 0)
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            self.logger.info(f"Cash flow statement saved for company {company_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to save cash flow statement: {e}")
            if 'conn' in locals():
                conn.rollback()
                conn.close()
    
    def get_financial_trends(self, company_id: int, years: int = 5) -> Dict[str, Any]:
        """Get financial trends and analysis over multiple years"""
        try:
            start_year = datetime.now().year - years
            
            query = """
            SELECT 
                cf.year,
                cf.net_income,
                cf.net_cash_from_operating_activities,
                cf.free_cash_flow,
                cf.debt_to_equity_ratio,
                bs.current_assets,
                bs.current_liabilities,
                bs.total_assets
            FROM cash_flow_statement cf
            JOIN balance_sheet_1 bs ON cf.company_id = bs.company_id AND cf.year = bs.year
            WHERE cf.company_id = %s 
            AND cf.year >= %s
            ORDER BY cf.year
            """
            
            conn = self.get_db_connection()
            df = pd.read_sql(query, conn, params=(company_id, start_year))
            conn.close()
            
            if df.empty:
                return {'success': False, 'error': 'No trend data available'}
            
            # Calculate current ratio
            df['current_ratio'] = df['current_assets'] / (df['current_liabilities'] + 0.001)
            
            # Calculate trends
            trends = {}
            for column in ['net_income', 'net_cash_from_operating_activities', 'free_cash_flow', 'total_assets']:
                if column in df.columns:
                    values = df[column].dropna()
                    if len(values) >= 2:
                        # Calculate CAGR (Compound Annual Growth Rate)
                        first_value = values.iloc[0]
                        last_value = values.iloc[-1]
                        num_years = len(values) - 1
                        
                        if first_value > 0:
                            cagr = (pow(last_value / first_value, 1/num_years) - 1) * 100
                        else:
                            cagr = 0
                        
                        trends[column] = {
                            'cagr': float(cagr),
                            'direction': 'Improving' if cagr > 0 else 'Declining',
                            'volatility': float(values.std() / values.mean() * 100) if values.mean() != 0 else 0
                        }
            
            return {
                'success': True,
                'company_id': company_id,
                'trends': trends,
                'historical_data': df.to_dict('records'),
                'years_analyzed': len(df)
            }
            
        except Exception as e:
            self.logger.error(f"Financial trends analysis failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_cash_flow_by_id(self, company_id: int, year: int = None) -> Dict[str, Any]:
        """Retrieve cash flow statement for a specific company and year"""
        if year is None:
            year = datetime.now().year
        
        query = """
        SELECT * FROM cash_flow_statement 
        WHERE company_id = %s AND year = %s
        """
        
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query, (company_id, year))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'success': True,
                    'data': dict(result),
                    'company_id': company_id,
                    'year': year
                }
            else:
                return {
                    'success': False,
                    'error': f'No cash flow statement found for company {company_id} in {year}'
                }
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve cash flow statement: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_all_cash_flows(self, company_id: int, start_year: int = None, end_year: int = None) -> Dict[str, Any]:
        """Get all cash flow statements for a company within a date range"""
        if start_year is None:
            start_year = datetime.now().year - 5
        if end_year is None:
            end_year = datetime.now().year
        
        query = """
        SELECT * FROM cash_flow_statement 
        WHERE company_id = %s 
        AND year BETWEEN %s AND %s
        ORDER BY year DESC
        """
        
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query, (company_id, start_year, end_year))
            results = cursor.fetchall()
            conn.close()
            
            return {
                'success': True,
                'data': [dict(row) for row in results],
                'company_id': company_id,
                'years_count': len(results),
                'date_range': f"{start_year}-{end_year}"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve cash flow statements: {e}")
            return {'success': False, 'error': str(e)}
    
    def compare_cash_flows(self, company_id: int, year1: int, year2: int) -> Dict[str, Any]:
        """Compare cash flow statements between two years"""
        try:
            cf1 = self.get_cash_flow_by_id(company_id, year1)
            cf2 = self.get_cash_flow_by_id(company_id, year2)
            
            if not cf1['success'] or not cf2['success']:
                return {
                    'success': False,
                    'error': 'Could not retrieve cash flow statements for comparison'
                }
            
            data1 = cf1['data']
            data2 = cf2['data']
            
            # Calculate changes
            changes = {}
            financial_fields = [
                'net_income', 'net_cash_from_operating_activities', 'free_cash_flow',
                'net_cash_from_investing_activities', 'net_cash_from_financing_activities',
                'capital_expenditures', 'dividends_paid'
            ]
            
            for field in financial_fields:
                val1 = data1.get(field, 0)
                val2 = data2.get(field, 0)
                
                if val1 != 0:
                    percent_change = ((val2 - val1) / val1) * 100
                else:
                    percent_change = 0 if val2 == 0 else 100
                
                changes[field] = {
                    f'{year1}_value': val1,
                    f'{year2}_value': val2,
                    'absolute_change': val2 - val1,
                    'percent_change': percent_change
                }
            
            return {
                'success': True,
                'company_id': company_id,
                'comparison_years': [year1, year2],
                'changes': changes,
                'summary': {
                    'operating_cf_growth': changes['net_cash_from_operating_activities']['percent_change'],
                    'free_cf_growth': changes['free_cash_flow']['percent_change'],
                    'net_income_growth': changes['net_income']['percent_change']
                }
            }
            
        except Exception as e:
            self.logger.error(f"Cash flow comparison failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def calculate_cash_flow_ratios(self, company_id: int, year: int = None) -> Dict[str, Any]:
        """Calculate comprehensive cash flow ratios and metrics"""
        try:
            cash_flow_data = self.get_cash_flow_by_id(company_id, year)
            if not cash_flow_data['success']:
                return cash_flow_data
            
            cf_data = cash_flow_data['data']
            
            # Get balance sheet data for additional calculations
            balance_sheet_data = self._fetch_balance_sheet_data(company_id, [year or datetime.now().year])
            bs_data = balance_sheet_data[0] if balance_sheet_data else {}
            
            ratios = {}
            
            # Cash flow ratios
            if cf_data.get('net_cash_from_operating_activities') and cf_data.get('net_income'):
                ratios['operating_cf_to_net_income'] = cf_data['net_cash_from_operating_activities'] / cf_data['net_income']
            
            if cf_data.get('free_cash_flow') and cf_data.get('net_cash_from_operating_activities'):
                ratios['free_cf_to_operating_cf'] = cf_data['free_cash_flow'] / cf_data['net_cash_from_operating_activities']
            
            # Cash coverage ratios
            if cf_data.get('net_cash_from_operating_activities') and bs_data.get('current_liabilities'):
                ratios['operating_cf_to_current_liabilities'] = cf_data['net_cash_from_operating_activities'] / bs_data['current_liabilities']
            
            if cf_data.get('net_cash_from_operating_activities') and bs_data.get('total_liabilities'):
                ratios['operating_cf_to_total_debt'] = cf_data['net_cash_from_operating_activities'] / bs_data['total_liabilities']
            
            # Investment ratios
            if cf_data.get('capital_expenditures') and cf_data.get('net_cash_from_operating_activities'):
                ratios['capex_to_operating_cf'] = abs(cf_data['capital_expenditures']) / cf_data['net_cash_from_operating_activities']
            
            # Quality ratios
            if cf_data.get('net_cash_from_operating_activities') and bs_data.get('total_assets'):
                ratios['operating_cf_to_assets'] = cf_data['net_cash_from_operating_activities'] / bs_data['total_assets']
            
            return {
                'success': True,
                'company_id': company_id,
                'year': year or datetime.now().year,
                'cash_flow_ratios': ratios,
                'cash_flow_data': cf_data,
                'balance_sheet_data': bs_data
            }
            
        except Exception as e:
            self.logger.error(f"Cash flow ratios calculation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def generate_financial_dashboard_data(self, company_id: int) -> Dict[str, Any]:
        """Generate comprehensive data for financial dashboard"""
        try:
            # Get latest financial analysis
            financial_analysis = self.analyze_financial_health(company_id)
            
            # Get financial trends
            trends = self.get_financial_trends(company_id, 3)
            
            # Get industry comparison
            industry_comparison = self.get_industry_comparison(company_id)
            
            # Get cash flow ratios
            cf_ratios = self.calculate_cash_flow_ratios(company_id)
            
            # Get latest cash flow and balance sheet data
            latest_cf = self.get_cash_flow_by_id(company_id)
            latest_bs_query = """
            SELECT * FROM balance_sheet_1 
            WHERE company_id = %s 
            ORDER BY year DESC 
            LIMIT 1
            """
            
            conn = self.get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(latest_bs_query, (company_id,))
            latest_bs_result = cursor.fetchone()
            conn.close()
            
            latest_bs = dict(latest_bs_result) if latest_bs_result else {}
            
            # Prepare dashboard data
            dashboard_data = {
                'company_id': company_id,
                'generated_at': datetime.now().isoformat(),
                'financial_health': financial_analysis if financial_analysis.get('success') else None,
                'trends': trends if trends.get('success') else None,
                'industry_comparison': industry_comparison if industry_comparison.get('success') else None,
                'cash_flow_ratios': cf_ratios if cf_ratios.get('success') else None,
                'latest_financial_data': {
                    'cash_flow': latest_cf['data'] if latest_cf.get('success') else None,
                    'balance_sheet': latest_bs
                },
                'key_metrics': self._extract_key_metrics(financial_analysis, latest_cf, latest_bs),
                'alerts': self._generate_financial_alerts(financial_analysis, latest_cf, latest_bs)
            }
            
            return {
                'success': True,
                'dashboard_data': dashboard_data
            }
            
        except Exception as e:
            self.logger.error(f"Dashboard data generation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _extract_key_metrics(self, financial_analysis: Dict, cash_flow: Dict, balance_sheet: Dict) -> Dict[str, Any]:
        """Extract key metrics for dashboard display"""
        metrics = {}
        
        try:
            # Financial health score
            if financial_analysis.get('success') and 'financial_score' in financial_analysis:
                metrics['financial_score'] = financial_analysis['financial_score']
            
            # Liquidation risk
            if financial_analysis.get('success') and 'liquidation_risk' in financial_analysis:
                metrics['liquidation_risk'] = financial_analysis['liquidation_risk']
            
            # Key cash flow metrics
            if cash_flow.get('success') and cash_flow.get('data'):
                cf_data = cash_flow['data']
                metrics['operating_cash_flow'] = cf_data.get('net_cash_from_operating_activities', 0)
                metrics['free_cash_flow'] = cf_data.get('free_cash_flow', 0)
                metrics['net_income'] = cf_data.get('net_income', 0)
            
            # Key balance sheet metrics
            if balance_sheet:
                metrics['total_assets'] = balance_sheet.get('total_assets', 0)
                metrics['total_liabilities'] = balance_sheet.get('total_liabilities', 0)
                metrics['total_equity'] = balance_sheet.get('total_equity', 0)
                
                # Calculate current ratio
                if balance_sheet.get('current_assets') and balance_sheet.get('current_liabilities'):
                    metrics['current_ratio'] = balance_sheet['current_assets'] / balance_sheet['current_liabilities']
            
        except Exception as e:
            self.logger.error(f"Key metrics extraction failed: {e}")
        
        return metrics
    
    def _generate_financial_alerts(self, financial_analysis: Dict, cash_flow: Dict, balance_sheet: Dict) -> List[Dict[str, str]]:
        """Generate financial alerts and warnings"""
        alerts = []
        
        try:
            # Liquidation risk alert
            if financial_analysis.get('success') and 'liquidation_risk' in financial_analysis:
                risk_data = financial_analysis['liquidation_risk']
                if risk_data.get('risk_level') == 'High':
                    alerts.append({
                        'type': 'danger',
                        'title': 'High Liquidation Risk',
                        'message': f"Company shows high risk of liquidation ({risk_data.get('risk_score', 0):.1%})",
                        'priority': 'high'
                    })
            
            # Cash flow alerts
            if cash_flow.get('success') and cash_flow.get('data'):
                cf_data = cash_flow['data']
                
                if cf_data.get('net_cash_from_operating_activities', 0) < 0:
                    alerts.append({
                        'type': 'warning',
                        'title': 'Negative Operating Cash Flow',
                        'message': 'Company has negative cash flow from operations',
                        'priority': 'medium'
                    })
                
                if cf_data.get('free_cash_flow', 0) < 0:
                    alerts.append({
                        'type': 'warning',
                        'title': 'Negative Free Cash Flow',
                        'message': 'Company has negative free cash flow',
                        'priority': 'medium'
                    })
            
            # Liquidity alerts
            if balance_sheet:
                current_assets = balance_sheet.get('current_assets', 0)
                current_liabilities = balance_sheet.get('current_liabilities', 1)
                current_ratio = current_assets / current_liabilities
                
                if current_ratio < 1.0:
                    alerts.append({
                        'type': 'danger',
                        'title': 'Liquidity Crisis',
                        'message': f'Current ratio is {current_ratio:.2f}, indicating potential liquidity problems',
                        'priority': 'high'
                    })
                elif current_ratio < 1.2:
                    alerts.append({
                        'type': 'warning',
                        'title': 'Low Liquidity',
                        'message': f'Current ratio is {current_ratio:.2f}, monitor liquidity closely',
                        'priority': 'medium'
                    })
            
        except Exception as e:
            self.logger.error(f"Financial alerts generation failed: {e}")
        
        return alerts
    
    def export_financial_data(self, company_id: int, format: str = 'csv') -> Dict[str, Any]:
        """Export comprehensive financial data in specified format"""
        try:
            # Get all financial data
            cash_flows = self.get_all_cash_flows(company_id)
            
            balance_sheets_query = """
            SELECT * FROM balance_sheet_1 
            WHERE company_id = %s 
            ORDER BY year DESC
            """
            
            conn = self.get_db_connection()
            balance_sheets_df = pd.read_sql(balance_sheets_query, conn, params=(company_id,))
            cash_flows_df = pd.DataFrame(cash_flows['data']) if cash_flows.get('success') else pd.DataFrame()
            conn.close()
            
            # Create export filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format.lower() == 'csv':
                bs_filename = f"balance_sheets_company_{company_id}_{timestamp}.csv"
                cf_filename = f"cash_flows_company_{company_id}_{timestamp}.csv"
                
                balance_sheets_df.to_csv(bs_filename, index=False)
                if not cash_flows_df.empty:
                    cash_flows_df.to_csv(cf_filename, index=False)
                
                return {
                    'success': True,
                    'files': {
                        'balance_sheets': bs_filename,
                        'cash_flows': cf_filename if not cash_flows_df.empty else None
                    },
                    'format': 'CSV',
                    'records_count': {
                        'balance_sheets': len(balance_sheets_df),
                        'cash_flows': len(cash_flows_df)
                    }
                }
            
            elif format.lower() == 'excel':
                filename = f"financial_data_company_{company_id}_{timestamp}.xlsx"
                
                with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                    balance_sheets_df.to_excel(writer, sheet_name='Balance Sheets', index=False)
                    if not cash_flows_df.empty:
                        cash_flows_df.to_excel(writer, sheet_name='Cash Flows', index=False)
                
                return {
                    'success': True,
                    'file': filename,
                    'format': 'Excel',
                    'records_count': {
                        'balance_sheets': len(balance_sheets_df),
                        'cash_flows': len(cash_flows_df)
                    }
                }
            
            else:
                return {
                    'success': False,
                    'error': f'Unsupported format: {format}. Supported formats: csv, excel'
                }
            
        except Exception as e:
            self.logger.error(f"Financial data export failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def delete_financial_data(self, company_id: int, data_type: str = 'all', year: int = None) -> Dict[str, Any]:
        """Delete financial data for a company"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            deleted_records = 0
            
            if data_type in ['all', 'cash_flow']:
                if year:
                    cursor.execute("DELETE FROM cash_flow_statement WHERE company_id = %s AND year = %s", (company_id, year))
                else:
                    cursor.execute("DELETE FROM cash_flow_statement WHERE company_id = %s", (company_id,))
                deleted_records += cursor.rowcount
            
            if data_type in ['all', 'balance_sheet']:
                if year:
                    cursor.execute("DELETE FROM balance_sheet_1 WHERE company_id = %s AND year = %s", (company_id, year))
                else:
                    cursor.execute("DELETE FROM balance_sheet_1 WHERE company_id = %s", (company_id,))
                deleted_records += cursor.rowcount
            
            if data_type in ['all', 'analysis']:
                cursor.execute("DELETE FROM financial_analysis WHERE company_id = %s", (company_id,))
                deleted_records += cursor.rowcount
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return {
                'success': True,
                'message': f'Deleted {deleted_records} records for company {company_id}',
                'data_type': data_type,
                'year': year
            }
            
        except Exception as e:
            self.logger.error(f"Financial data deletion failed: {e}")
            return {'success': False, 'error': str(e)}


# Example usage and testing functions
def main():
    """Main function to demonstrate the Financial Processing Service"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize the service
        print("Initializing Financial Processing Service...")
        service = FinancialProcessingService()
        
        # Test company ID (replace with actual company ID from your database)
        test_company_id = 1
        
        print(f"\n=== Testing Financial Analysis for Company {test_company_id} ===")
        
        # 1. Analyze financial health
        print("\n1. Analyzing Financial Health...")
        health_analysis = service.analyze_financial_health(test_company_id)
        if health_analysis['success']:
            print(f"Financial Score: {health_analysis['financial_score']['score']:.1f}/100")
            print(f"Rating: {health_analysis['financial_score']['rating']}")
            print(f"Liquidation Risk: {health_analysis['liquidation_risk']['risk_level']}")
        else:
            print(f"Error: {health_analysis['error']}")
        
        # 2. Get financial trends
        print("\n2. Getting Financial Trends...")
        trends = service.get_financial_trends(test_company_id)
        if trends['success']:
            print(f"Years analyzed: {trends['years_analyzed']}")
            for metric, trend_data in trends['trends'].items():
                print(f"{metric}: {trend_data['direction']} ({trend_data['cagr']:.2f}% CAGR)")
        else:
            print(f"Error: {trends['error']}")
        
        # 3. Get industry comparison
        print("\n3. Getting Industry Comparison...")
        industry_comp = service.get_industry_comparison(test_company_id)
        if industry_comp['success']:
            print(f"Industry: {industry_comp['industry']}")
            print(f"Sector: {industry_comp['sector']}")
            print(f"Peer companies: {industry_comp['peer_count']}")
        else:
            print(f"Error: {industry_comp['error']}")
        
        # 4. Generate cash flow statement
        print("\n4. Generating Cash Flow Statement...")
        source_data = {
            'net_income': 1000000,
            'depreciation': 50000,
            'stock_compensation': 25000,
            'acquisitions': 0,
            'dividends': 30000,
            'share_repurchases': 0,
            'interest_expense': 5000,
            'taxes': 200000
        }
        
        cash_flow_gen = service.generate_cash_flow_statement(test_company_id, source_data)
        if cash_flow_gen['success']:
            print("Cash flow statement generated successfully")
            print(f"Method: {cash_flow_gen['method']}")
        else:
            print(f"Error: {cash_flow_gen['error']}")
        
        # 5. Calculate cash flow ratios
        print("\n5. Calculating Cash Flow Ratios...")
        ratios = service.calculate_cash_flow_ratios(test_company_id)
        if ratios['success']:
            print("Cash flow ratios calculated:")
            for ratio_name, ratio_value in ratios['cash_flow_ratios'].items():
                print(f"  {ratio_name}: {ratio_value:.3f}")
        else:
            print(f"Error: {ratios['error']}")
        
        # 6. Generate dashboard data
        print("\n6. Generating Dashboard Data...")
        dashboard = service.generate_financial_dashboard_data(test_company_id)
        if dashboard['success']:
            print("Dashboard data generated successfully")
            alerts = dashboard['dashboard_data']['alerts']
            print(f"Financial alerts: {len(alerts)}")
            for alert in alerts[:3]:  # Show first 3 alerts
                print(f"  - {alert['title']}: {alert['message']}")
        else:
            print(f"Error: {dashboard['error']}")
        
        # 7. Export financial data
        print("\n7. Exporting Financial Data...")
        export_result = service.export_financial_data(test_company_id, 'csv')
        if export_result['success']:
            print(f"Data exported to CSV files:")
            print(f"  Balance sheets: {export_result['files']['balance_sheets']}")
            if export_result['files']['cash_flows']:
                print(f"  Cash flows: {export_result['files']['cash_flows']}")
        else:
            print(f"Error: {export_result['error']}")
        
        print("\n=== Financial Processing Service Test Complete ===")
        
    except Exception as e:
        print(f"Error during testing: {e}")


if __name__ == "__main__":
    main()