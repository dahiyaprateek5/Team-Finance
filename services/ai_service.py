import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pickle
import psycopg2
from psycopg2.extras import RealDictCursor
import sqlite3

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_algorithms.base_model import BaseFinancialModel, PredictionResult, RiskLevel
from ml_algorithms.time_series_analyzer import FinancialTimeSeriesAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AnalysisRequest:
    """Request object for AI analysis."""
    company_id: str
    financial_data: Optional[pd.DataFrame] = None  # Made optional for DB fetching
    analysis_types: List[str] = None
    historical_periods: int = 3
    confidence_threshold: float = 0.75
    include_explanations: bool = True
    include_recommendations: bool = True
    use_database: bool = True  # New parameter
    
    def __post_init__(self):
        if self.analysis_types is None:
            self.analysis_types = ['liquidation_risk', 'financial_health', 'cash_flow_forecast']
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert DataFrame to dict for JSON serialization
        if self.financial_data is not None:
            data['financial_data'] = self.financial_data.to_dict('records')
        return data

@dataclass
class AnalysisResponse:
    """Response object for AI analysis."""
    company_id: str
    analysis_timestamp: datetime
    risk_assessment: Dict[str, Any]
    financial_health_score: float
    predictions: List[Dict[str, Any]]
    insights: List[str]
    recommendations: List[str]
    confidence_scores: Dict[str, float]
    data_quality_score: float
    processing_time: float
    data_source: str = "database"  # Track data source
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['analysis_timestamp'] = self.analysis_timestamp.isoformat()
        return data

class DatabaseManager:
    """Database connection and query manager."""
    
    def __init__(self, 
                 db_type: str = "postgresql",
                 db_config: Dict[str, Any] = None):
        """
        Initialize database manager.
        
        Args:
            db_type (str): Database type ('postgresql' or 'sqlite')
            db_config (dict): Database configuration parameters
        """
        self.db_type = db_type
        self.db_config = db_config or {}
        self.connection = None
        
        # Default configurations
        if db_type == "postgresql" and not db_config:
            self.db_config = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': os.getenv('DB_PORT', 5432),
                'database': os.getenv('DB_NAME', 'financial_risk_db'),
                'user': os.getenv('DB_USER', 'postgres'),
                'password': os.getenv('DB_PASSWORD', 'password')
            }
        elif db_type == "sqlite" and not db_config:
            self.db_config = {
                'database': os.getenv('SQLITE_DB', './data/financial_data.db')
            }
    
    def connect(self):
        """Establish database connection."""
        try:
            if self.db_type == "postgresql":
                self.connection = psycopg2.connect(**self.db_config)
                logger.info("Connected to PostgreSQL database")
            elif self.db_type == "sqlite":
                self.connection = sqlite3.connect(self.db_config['database'])
                self.connection.row_factory = sqlite3.Row  # For dict-like access
                logger.info("Connected to SQLite database")
            else:
                raise ValueError(f"Unsupported database type: {self.db_type}")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def disconnect(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Database connection closed")
    
    def execute_query(self, query: str, params: List = None) -> pd.DataFrame:
        """Execute query and return results as DataFrame."""
        try:
            if not self.connection:
                self.connect()
            
            df = pd.read_sql_query(query, self.connection, params=params)
            return df
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def fetch_company_financial_data(self, 
                                   company_id: str, 
                                   periods: int = 3) -> pd.DataFrame:
        """Fetch comprehensive financial data for a company."""
        try:
            # Combined query to get all financial data
            query = """
            SELECT 
                f.company_id,
                f.date,
                f.period,
                f.fiscal_year,
                
                -- Income Statement Data
                i.revenue,
                i.gross_profit,
                i.operating_income,
                i.net_income,
                i.ebitda,
                i.depreciation_amortization,
                i.interest_expense,
                i.tax_expense,
                
                -- Balance Sheet Data
                b.total_assets,
                b.current_assets,
                b.cash_and_equivalents,
                b.accounts_receivable,
                b.inventory,
                b.property_plant_equipment,
                b.intangible_assets,
                b.current_liabilities,
                b.accounts_payable,
                b.short_term_debt,
                b.long_term_debt,
                b.total_debt,
                b.total_equity,
                b.retained_earnings,
                
                -- Cash Flow Data
                c.operating_cash_flow,
                c.investing_cash_flow,
                c.financing_cash_flow,
                c.net_cash_change,
                c.free_cash_flow,
                c.capital_expenditures,
                c.dividends_paid,
                
                -- Additional calculated fields
                CASE 
                    WHEN b.current_liabilities > 0 
                    THEN b.current_assets / b.current_liabilities 
                    ELSE NULL 
                END as current_ratio,
                
                CASE 
                    WHEN b.total_assets > 0 
                    THEN i.net_income / b.total_assets 
                    ELSE NULL 
                END as roa,
                
                CASE 
                    WHEN b.total_equity > 0 
                    THEN i.net_income / b.total_equity 
                    ELSE NULL 
                END as roe,
                
                CASE 
                    WHEN i.revenue > 0 
                    THEN i.net_income / i.revenue 
                    ELSE NULL 
                END as profit_margin,
                
                CASE 
                    WHEN b.total_equity > 0 
                    THEN b.total_debt / b.total_equity 
                    ELSE NULL 
                END as debt_to_equity,
                
                CASE 
                    WHEN b.total_assets > 0 
                    THEN b.total_debt / b.total_assets 
                    ELSE NULL 
                END as debt_to_assets,
                
                CASE 
                    WHEN b.total_assets > 0 
                    THEN i.revenue / b.total_assets 
                    ELSE NULL 
                END as asset_turnover
                
            FROM financial_data f
            LEFT JOIN income_statement_data i ON f.company_id = i.company_id AND f.date = i.date
            LEFT JOIN balance_sheet_data b ON f.company_id = b.company_id AND f.date = b.date
            LEFT JOIN cash_flow_data c ON f.company_id = c.company_id AND f.date = c.date
            
            WHERE f.company_id = %s
            ORDER BY f.date DESC
            LIMIT %s
            """
            
            if self.db_type == "sqlite":
                query = query.replace("%s", "?")
            
            params = [company_id, periods]
            df = self.execute_query(query, params)
            
            if df.empty:
                logger.warning(f"No financial data found for company {company_id}")
                # Try alternative table structure
                return self._fetch_company_data_alternative(company_id, periods)
            
            # Convert date column to datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            # Sort by date descending (most recent first)
            df = df.sort_values('date', ascending=False)
            
            logger.info(f"Fetched {len(df)} periods of financial data for company {company_id}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching company financial data: {e}")
            # Return empty DataFrame with expected columns
            return self._get_empty_financial_dataframe()
    
    def _fetch_company_data_alternative(self, 
                                      company_id: str, 
                                      periods: int = 3) -> pd.DataFrame:
        """Alternative method to fetch data if main query fails."""
        try:
            # Try simpler queries for individual tables
            tables_queries = {
                'balance_sheet_data': """
                    SELECT *, 'balance_sheet' as source_table 
                    FROM balance_sheet_data 
                    WHERE company_id = %s 
                    ORDER BY date DESC LIMIT %s
                """,
                'income_statement_data': """
                    SELECT *, 'income_statement' as source_table 
                    FROM income_statement_data 
                    WHERE company_id = %s 
                    ORDER BY date DESC LIMIT %s
                """,
                'cash_flow_data': """
                    SELECT *, 'cash_flow' as source_table 
                    FROM cash_flow_data 
                    WHERE company_id = %s 
                    ORDER BY date DESC LIMIT %s
                """
            }
            
            combined_data = []
            
            for table_name, query in tables_queries.items():
                try:
                    if self.db_type == "sqlite":
                        query = query.replace("%s", "?")
                    
                    df = self.execute_query(query, [company_id, periods])
                    if not df.empty:
                        combined_data.append(df)
                        logger.info(f"Fetched data from {table_name}")
                except Exception as e:
                    logger.warning(f"Could not fetch from {table_name}: {e}")
                    continue
            
            if combined_data:
                # Merge DataFrames on company_id and date
                result_df = combined_data[0]
                for df in combined_data[1:]:
                    result_df = pd.merge(result_df, df, on=['company_id', 'date'], how='outer', suffixes=('', '_y'))
                
                # Remove duplicate columns
                result_df = result_df.loc[:, ~result_df.columns.str.endswith('_y')]
                
                # Calculate ratios
                result_df = self._calculate_financial_ratios(result_df)
                
                return result_df
            
            return self._get_empty_financial_dataframe()
            
        except Exception as e:
            logger.error(f"Alternative data fetch failed: {e}")
            return self._get_empty_financial_dataframe()
    
    def _calculate_financial_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate financial ratios for the DataFrame."""
        try:
            # Current Ratio
            if 'current_assets' in df.columns and 'current_liabilities' in df.columns:
                df['current_ratio'] = df['current_assets'] / df['current_liabilities']
            
            # ROA
            if 'net_income' in df.columns and 'total_assets' in df.columns:
                df['roa'] = df['net_income'] / df['total_assets']
            
            # ROE
            if 'net_income' in df.columns and 'total_equity' in df.columns:
                df['roe'] = df['net_income'] / df['total_equity']
            
            # Profit Margin
            if 'net_income' in df.columns and 'revenue' in df.columns:
                df['profit_margin'] = df['net_income'] / df['revenue']
            
            # Debt to Equity
            if 'total_debt' in df.columns and 'total_equity' in df.columns:
                df['debt_to_equity'] = df['total_debt'] / df['total_equity']
            
            # Debt to Assets
            if 'total_debt' in df.columns and 'total_assets' in df.columns:
                df['debt_to_assets'] = df['total_debt'] / df['total_assets']
            
            # Asset Turnover
            if 'revenue' in df.columns and 'total_assets' in df.columns:
                df['asset_turnover'] = df['revenue'] / df['total_assets']
            
            # Quick Ratio (approximation)
            if 'current_assets' in df.columns and 'inventory' in df.columns and 'current_liabilities' in df.columns:
                df['quick_ratio'] = (df['current_assets'] - df['inventory']) / df['current_liabilities']
            
            return df
            
        except Exception as e:
            logger.warning(f"Error calculating financial ratios: {e}")
            return df
    
    def _get_empty_financial_dataframe(self) -> pd.DataFrame:
        """Return empty DataFrame with expected financial columns."""
        columns = [
            'company_id', 'date', 'period', 'fiscal_year',
            'revenue', 'gross_profit', 'operating_income', 'net_income', 'ebitda',
            'total_assets', 'current_assets', 'cash_and_equivalents', 'accounts_receivable',
            'inventory', 'current_liabilities', 'total_debt', 'total_equity',
            'operating_cash_flow', 'free_cash_flow', 'current_ratio', 'roa', 'roe',
            'profit_margin', 'debt_to_equity', 'asset_turnover'
        ]
        
        return pd.DataFrame(columns=columns)
    
    def get_company_list(self, limit: int = 100) -> List[str]:
        """Get list of available companies in database."""
        try:
            query = """
            SELECT DISTINCT company_id 
            FROM financial_data 
            ORDER BY company_id 
            LIMIT %s
            """
            
            if self.db_type == "sqlite":
                query = query.replace("%s", "?")
            
            df = self.execute_query(query, [limit])
            return df['company_id'].tolist()
            
        except Exception as e:
            logger.error(f"Error fetching company list: {e}")
            return []
    
    def get_company_info(self, company_id: str) -> Dict[str, Any]:
        """Get basic information about a company."""
        try:
            query = """
            SELECT 
                company_id,
                MAX(date) as latest_data_date,
                MIN(date) as earliest_data_date,
                COUNT(*) as total_records,
                industry,
                sector,
                company_name
            FROM financial_data 
            WHERE company_id = %s
            GROUP BY company_id, industry, sector, company_name
            """
            
            if self.db_type == "sqlite":
                query = query.replace("%s", "?")
            
            df = self.execute_query(query, [company_id])
            
            if not df.empty:
                return df.iloc[0].to_dict()
            else:
                return {'company_id': company_id, 'status': 'not_found'}
                
        except Exception as e:
            logger.error(f"Error fetching company info: {e}")
            return {'company_id': company_id, 'error': str(e)}

class AIAnalysisService:
    """
    AI-powered financial analysis service that orchestrates various ML models
    and algorithms to provide comprehensive financial risk assessment.
    """
    
    def __init__(self, 
                 model_directory: str = "./models/",
                 confidence_threshold: float = 0.75,
                 enable_async: bool = True,
                 db_type: str = "postgresql",
                 db_config: Dict[str, Any] = None):
        """
        Initialize AI Analysis Service.
        
        Args:
            model_directory (str): Directory containing trained models
            confidence_threshold (float): Minimum confidence for predictions
            enable_async (bool): Enable asynchronous processing
            db_type (str): Database type ('postgresql' or 'sqlite')
            db_config (dict): Database configuration
        """
        self.model_directory = model_directory
        self.confidence_threshold = confidence_threshold
        self.enable_async = enable_async
        self.models = {}
        self.time_series_analyzer = None
        self.executor = ThreadPoolExecutor(max_workers=4) if enable_async else None
        
        # Initialize database manager
        self.db_manager = DatabaseManager(db_type=db_type, db_config=db_config)
        
        # Initialize components
        self._initialize_models()
        self._initialize_analyzers()
        
        # Analysis configuration
        self.analysis_config = {
            'liquidation_risk': {
                'enabled': True,
                'model_name': 'liquidation_risk_model',
                'weight': 0.4
            },
            'financial_health': {
                'enabled': True,
                'model_name': 'financial_health_model',
                'weight': 0.3
            },
            'cash_flow_forecast': {
                'enabled': True,
                'use_time_series': True,
                'weight': 0.3
            }
        }
        
        logger.info("AI Analysis Service with Database Integration initialized successfully")
    
    def _initialize_models(self):
        """Initialize and load trained ML models."""
        try:
            # Load liquidation risk model
            liquidation_model_path = os.path.join(self.model_directory, "liquidation_risk_model.pkl")
            if os.path.exists(liquidation_model_path):
                self.models['liquidation_risk'] = self._load_model(liquidation_model_path)
                logger.info("Liquidation risk model loaded")
            
            # Load financial health model
            health_model_path = os.path.join(self.model_directory, "financial_health_model.pkl")
            if os.path.exists(health_model_path):
                self.models['financial_health'] = self._load_model(health_model_path)
                logger.info("Financial health model loaded")
            
            # Load ensemble model
            ensemble_model_path = os.path.join(self.model_directory, "ensemble_model.pkl")
            if os.path.exists(ensemble_model_path):
                self.models['ensemble'] = self._load_model(ensemble_model_path)
                logger.info("Ensemble model loaded")
        
        except Exception as e:
            logger.warning(f"Some models could not be loaded: {e}")
    
    def _initialize_analyzers(self):
        """Initialize specialized analyzers."""
        try:
            self.time_series_analyzer = FinancialTimeSeriesAnalyzer(
                frequency='quarterly',
                confidence_level=0.95
            )
            logger.info("Time series analyzer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize time series analyzer: {e}")
    
    def _load_model(self, model_path: str) -> Any:
        """Load a trained model from file."""
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            return None
    
    async def analyze_async(self, request: AnalysisRequest) -> AnalysisResponse:
        """
        Perform asynchronous AI analysis.
        
        Args:
            request (AnalysisRequest): Analysis request
            
        Returns:
            AnalysisResponse: Analysis results
        """
        if not self.enable_async:
            return self.analyze(request)
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.analyze, request)
    
    def analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        """
        Perform comprehensive AI-powered financial analysis.
        
        Args:
            request (AnalysisRequest): Analysis request
            
        Returns:
            AnalysisResponse: Analysis results
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting AI analysis for company {request.company_id}")
            
            # Fetch financial data from database if not provided
            if request.use_database and request.financial_data is None:
                logger.info(f"Fetching financial data from database for {request.company_id}")
                financial_data = self.db_manager.fetch_company_financial_data(
                    request.company_id, 
                    request.historical_periods
                )
                
                if financial_data.empty:
                    raise ValueError(f"No financial data found for company {request.company_id}")
                
                request.financial_data = financial_data
                data_source = "database"
                
            elif request.financial_data is not None:
                data_source = "provided"
                logger.info(f"Using provided financial data for {request.company_id}")
            
            else:
                raise ValueError("No financial data available - either provide data or enable database usage")
            
            # Validate input data
            data_quality_score = self._assess_data_quality(request.financial_data)
            
            # Prepare features
            features_df = self._prepare_features(request.financial_data)
            
            # Perform different types of analysis
            risk_assessment = {}
            predictions = []
            confidence_scores = {}
            
            for analysis_type in request.analysis_types:
                if analysis_type in self.analysis_config and self.analysis_config[analysis_type]['enabled']:
                    result = self._perform_analysis(analysis_type, features_df, request)
                    risk_assessment[analysis_type] = result['assessment']
                    predictions.extend(result['predictions'])
                    confidence_scores[analysis_type] = result['confidence']
            
            # Calculate overall financial health score
            financial_health_score = self._calculate_health_score(risk_assessment, confidence_scores)
            
            # Generate insights and recommendations
            insights = self._generate_insights(risk_assessment, features_df, request)
            recommendations = self._generate_recommendations(risk_assessment, insights, request)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            response = AnalysisResponse(
                company_id=request.company_id,
                analysis_timestamp=datetime.now(),
                risk_assessment=risk_assessment,
                financial_health_score=financial_health_score,
                predictions=predictions,
                insights=insights,
                recommendations=recommendations,
                confidence_scores=confidence_scores,
                data_quality_score=data_quality_score,
                processing_time=processing_time,
                data_source=data_source
            )
            
            logger.info(f"AI analysis completed for {request.company_id} in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error during AI analysis: {e}")
            raise
    
    def get_available_companies(self, limit: int = 100) -> List[str]:
        """Get list of companies available for analysis."""
        return self.db_manager.get_company_list(limit)
    
    def get_company_info(self, company_id: str) -> Dict[str, Any]:
        """Get information about a specific company."""
        return self.db_manager.get_company_info(company_id)
    
    def analyze_company_by_id(self, 
                            company_id: str,
                            analysis_types: List[str] = None,
                            historical_periods: int = 3) -> AnalysisResponse:
        """
        Convenient method to analyze a company by ID using database data.
        
        Args:
            company_id (str): Company identifier
            analysis_types (list): Types of analysis to perform
            historical_periods (int): Number of historical periods to analyze
            
        Returns:
            AnalysisResponse: Analysis results
        """
        if analysis_types is None:
            analysis_types = ['liquidation_risk', 'financial_health', 'cash_flow_forecast']
        
        request = AnalysisRequest(
            company_id=company_id,
            analysis_types=analysis_types,
            historical_periods=historical_periods,
            use_database=True
        )
        
        return self.analyze(request)
    
    # Keep all the existing analysis methods unchanged
    def _assess_data_quality(self, data: pd.DataFrame) -> float:
        """Assess the quality of input financial data."""
        try:
            total_cells = data.size
            missing_cells = data.isnull().sum().sum()
            completeness = 1 - (missing_cells / total_cells) if total_cells > 0 else 0
            
            # Check for required columns
            required_columns = ['revenue', 'total_assets', 'current_assets', 'current_liabilities']
            available_required = sum(1 for col in required_columns if col in data.columns)
            column_completeness = available_required / len(required_columns)
            
            # Check for data consistency
            consistency_score = 1.0  # Start with perfect score
            
            # Check for negative values where they shouldn't be
            positive_columns = ['revenue', 'total_assets', 'current_assets']
            for col in positive_columns:
                if col in data.columns:
                    negative_count = (data[col] < 0).sum()
                    if negative_count > 0:
                        consistency_score -= 0.1
            
            # Overall quality score
            quality_score = (completeness * 0.4 + column_completeness * 0.4 + consistency_score * 0.2)
            return min(1.0, max(0.0, quality_score))
            
        except Exception as e:
            logger.warning(f"Error assessing data quality: {e}")
            return 0.5  # Default medium quality
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML models."""
        try:
            features_df = data.copy()
            
            # Calculate financial ratios if not already present
            if 'current_ratio' not in features_df.columns and 'current_assets' in data.columns and 'current_liabilities' in data.columns:
                features_df['current_ratio'] = data['current_assets'] / data['current_liabilities']
            
            if 'roa' not in features_df.columns and 'net_income' in data.columns and 'total_assets' in data.columns:
                features_df['roa'] = data['net_income'] / data['total_assets']
            
            if 'debt_to_equity' not in features_df.columns and 'total_debt' in data.columns and 'total_equity' in data.columns:
                features_df['debt_to_equity'] = data['total_debt'] / data['total_equity']
            
            if 'asset_turnover' not in features_df.columns and 'revenue' in data.columns and 'total_assets' in data.columns:
                features_df['asset_turnover'] = data['revenue'] / data['total_assets']
            
            # Handle missing values
            numeric_columns = features_df.select_dtypes(include=[np.number]).columns
            features_df[numeric_columns] = features_df[numeric_columns].fillna(features_df[numeric_columns].median())
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return data
    
    def _perform_analysis(self, analysis_type: str, features_df: pd.DataFrame, request: AnalysisRequest) -> Dict[str, Any]:
        """Perform specific type of analysis."""
        try:
            if analysis_type == 'liquidation_risk':
                return self._analyze_liquidation_risk(features_df, request)
            elif analysis_type == 'financial_health':
                return self._analyze_financial_health(features_df, request)
            elif analysis_type == 'cash_flow_forecast':
                return self._analyze_cash_flow_forecast(features_df, request)
            else:
                logger.warning(f"Unknown analysis type: {analysis_type}")
                return {'assessment': {}, 'predictions': [], 'confidence': 0.0}
                
        except Exception as e:
            logger.error(f"Error in {analysis_type} analysis: {e}")
            return {'assessment': {}, 'predictions': [], 'confidence': 0.0}
    
    def _analyze_liquidation_risk(self, features_df: pd.DataFrame, request: AnalysisRequest) -> Dict[str, Any]:
        """Analyze liquidation risk using ML models."""
        try:
            model = self.models.get('liquidation_risk')
            if not model:
                # Fallback to rule-based analysis
                return self._rule_based_liquidation_analysis(features_df)
            
            # Use the trained model for prediction
            latest_data = features_df.iloc[-1:] if len(features_df) > 0 else features_df
            
            # Make prediction (placeholder - replace with actual model prediction)
            prediction_prob = 0.3  # This should be: model.predict_proba(latest_data)[0][1]
            risk_level = RiskLevel.LOW if prediction_prob < 0.3 else RiskLevel.MEDIUM if prediction_prob < 0.7 else RiskLevel.HIGH
            
            assessment = {
                'liquidation_probability': prediction_prob,
                'risk_level': risk_level.value,
                'key_factors': self._identify_liquidation_factors(features_df),
                'time_horizon': '12 months'
            }
            
            predictions = [{
                'type': 'liquidation_risk',
                'value': prediction_prob,
                'confidence': 0.85,
                'risk_level': risk_level.value
            }]
            
            return {
                'assessment': assessment,
                'predictions': predictions,
                'confidence': 0.85
            }
            
        except Exception as e:
            logger.error(f"Error in liquidation risk analysis: {e}")
            return self._rule_based_liquidation_analysis(features_df)
    
    def _rule_based_liquidation_analysis(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Rule-based liquidation risk analysis as fallback."""
        try:
            risk_score = 0.0
            factors = []
            
            latest_data = features_df.iloc[-1] if len(features_df) > 0 else pd.Series()
            
            # Check current ratio
            current_ratio = latest_data.get('current_ratio', 1.0)
            if current_ratio < 1.0:
                risk_score += 0.3
                factors.append('Low current ratio indicating liquidity issues')
            
            # Check profitability
            net_income = latest_data.get('net_income', 0)
            if net_income < 0:
                risk_score += 0.2
                factors.append('Negative net income')
            
            # Check debt levels
            debt_to_equity = latest_data.get('debt_to_equity', 0)
            if debt_to_equity > 2.0:
                risk_score += 0.2
                factors.append('High debt-to-equity ratio')
            
            # Check cash flow
            operating_cash_flow = latest_data.get('operating_cash_flow', 0)
            if operating_cash_flow < 0:
                risk_score += 0.3
                factors.append('Negative operating cash flow')
            
            risk_level = RiskLevel.LOW if risk_score < 0.3 else RiskLevel.MEDIUM if risk_score < 0.7 else RiskLevel.HIGH
            
            assessment = {
                'liquidation_probability': min(risk_score, 1.0),
                'risk_level': risk_level.value,
                'key_factors': factors,
                'time_horizon': '12 months'
            }
            
            return {
                'assessment': assessment,
                'predictions': [],
                'confidence': 0.70
            }
            
        except Exception as e:
            logger.error(f"Error in rule-based analysis: {e}")
            return {'assessment': {}, 'predictions': [], 'confidence': 0.0}
    
    def _analyze_financial_health(self, features_df: pd.DataFrame, request: AnalysisRequest) -> Dict[str, Any]:
        """Analyze overall financial health."""
        try:
            latest_data = features_df.iloc[-1] if len(features_df) > 0 else pd.Series()
            
            # Calculate health components
            liquidity_score = self._calculate_liquidity_score(latest_data)
            profitability_score = self._calculate_profitability_score(latest_data)
            leverage_score = self._calculate_leverage_score(latest_data)
            efficiency_score = self._calculate_efficiency_score(latest_data)
            
            # Overall health score (weighted average)
            health_score = (
                liquidity_score * 0.3 +
                profitability_score * 0.3 +
                leverage_score * 0.2 +
                efficiency_score * 0.2
            )
            
            assessment = {
                'overall_score': health_score,
                'liquidity_score': liquidity_score,
                'profitability_score': profitability_score,
                'leverage_score': leverage_score,
                'efficiency_score': efficiency_score,
                'grade': self._get_health_grade(health_score)
            }
            
            predictions = [{
                'type': 'financial_health',
                'value': health_score,
                'confidence': 0.80,
                'grade': assessment['grade']
            }]
            
            return {
                'assessment': assessment,
                'predictions': predictions,
                'confidence': 0.80
            }
            
        except Exception as e:
            logger.error(f"Error in financial health analysis: {e}")
            return {'assessment': {}, 'predictions': [], 'confidence': 0.0}
    
    def _analyze_cash_flow_forecast(self, features_df: pd.DataFrame, request: AnalysisRequest) -> Dict[str, Any]:
        """Analyze and forecast cash flow patterns."""
        try:
            if not self.time_series_analyzer:
                return {'assessment': {}, 'predictions': [], 'confidence': 0.0}
            
            # Prepare time series data
            if 'date' in features_df.columns and 'operating_cash_flow' in features_df.columns:
                ts_data = features_df[['date', 'operating_cash_flow']].copy()
                ts_data['date'] = pd.to_datetime(ts_data['date'])
                
                # Analyze time series
                analysis_result = self.time_series_analyzer.analyze_time_series(
                    ts_data, 'date', 'operating_cash_flow', request.company_id
                )
                
                if 'forecasts' in analysis_result:
                    forecasts = analysis_result['forecasts']
                    trend_analysis = analysis_result.get('trend_analysis', {})
                    
                    assessment = {
                        'forecast_horizon': '4 quarters',
                        'trend_direction': trend_analysis.get('linear_trend', {}).get('slope', 0),
                        'volatility': analysis_result.get('volatility_analysis', {}).get('basic_measures', {}).get('volatility', 0),
                        'seasonality_detected': analysis_result.get('seasonality_analysis', {}).get('seasonality_type', 'none') != 'none'
                    }
                    
                    predictions = []
                    for method, forecast_data in forecasts.items():
                        if 'forecast' in forecast_data:
                            forecast_values = list(forecast_data['forecast'].values())
                            predictions.append({
                                'type': 'cash_flow_forecast',
                                'method': method,
                                'values': forecast_values[:4],  # Next 4 quarters
                                'confidence': 0.75
                            })
                    
                    return {
                        'assessment': assessment,
                        'predictions': predictions,
                        'confidence': 0.75
                    }
            
            return {'assessment': {}, 'predictions': [], 'confidence': 0.0}
            
        except Exception as e:
            logger.error(f"Error in cash flow forecast analysis: {e}")
            return {'assessment': {}, 'predictions': [], 'confidence': 0.0}
    
    def _calculate_liquidity_score(self, data: pd.Series) -> float:
        """Calculate liquidity score (0-100)."""
        try:
            current_ratio = data.get('current_ratio', 1.0)
            quick_ratio = data.get('quick_ratio', current_ratio * 0.8)  # Approximation
            
            # Score based on ratios
            current_score = min(100, current_ratio * 50) if current_ratio >= 1 else current_ratio * 50
            quick_score = min(100, quick_ratio * 60) if quick_ratio >= 1 else quick_ratio * 60
            
            return (current_score + quick_score) / 2
            
        except Exception:
            return 50.0  # Default medium score
    
    def _calculate_profitability_score(self, data: pd.Series) -> float:
        """Calculate profitability score (0-100)."""
        try:
            roa = data.get('roa', 0) * 100  # Convert to percentage
            roe = data.get('roe', 0) * 100
            profit_margin = data.get('profit_margin', 0) * 100
            
            # Score based on profitability metrics
            roa_score = min(100, max(0, (roa + 5) * 10))  # -5% to 10% mapped to 0-100
            roe_score = min(100, max(0, (roe + 5) * 8))
            margin_score = min(100, max(0, (profit_margin + 5) * 12))
            
            return (roa_score + roe_score + margin_score) / 3
            
        except Exception:
            return 50.0
    
    def _calculate_leverage_score(self, data: pd.Series) -> float:
        """Calculate leverage score (0-100), higher is better (less leverage)."""
        try:
            debt_to_equity = data.get('debt_to_equity', 1.0)
            debt_to_assets = data.get('debt_to_assets', 0.5)
            
            # Lower debt ratios get higher scores
            de_score = max(0, 100 - debt_to_equity * 30)
            da_score = max(0, 100 - debt_to_assets * 100)
            
            return (de_score + da_score) / 2
            
        except Exception:
            return 50.0
    
    def _calculate_efficiency_score(self, data: pd.Series) -> float:
        """Calculate efficiency score (0-100)."""
        try:
            asset_turnover = data.get('asset_turnover', 1.0)
            inventory_turnover = data.get('inventory_turnover', 5.0)
            
            # Score based on turnover ratios
            asset_score = min(100, asset_turnover * 50)
            inventory_score = min(100, inventory_turnover * 10)
            
            return (asset_score + inventory_score) / 2
            
        except Exception:
            return 50.0
    
    def _get_health_grade(self, score: float) -> str:
        """Convert health score to letter grade."""
        if score >= 90:
            return 'A+'
        elif score >= 80:
            return 'A'
        elif score >= 70:
            return 'B'
        elif score >= 60:
            return 'C'
        elif score >= 50:
            return 'D'
        else:
            return 'F'
    
    def _identify_liquidation_factors(self, features_df: pd.DataFrame) -> List[str]:
        """Identify key factors contributing to liquidation risk."""
        factors = []
        try:
            latest_data = features_df.iloc[-1] if len(features_df) > 0 else pd.Series()
            
            # Check various risk factors
            if latest_data.get('current_ratio', 1.0) < 1.0:
                factors.append('Current ratio below 1.0 indicates liquidity problems')
            
            if latest_data.get('net_income', 0) < 0:
                factors.append('Negative net income indicates unprofitability')
            
            if latest_data.get('operating_cash_flow', 0) < 0:
                factors.append('Negative operating cash flow')
            
            if latest_data.get('debt_to_equity', 0) > 2.0:
                factors.append('High debt-to-equity ratio indicates overleveraging')
            
            if latest_data.get('quick_ratio', 1.0) < 0.5:
                factors.append('Low quick ratio indicates poor short-term liquidity')
            
            # Check trends if multiple periods available
            if len(features_df) > 1:
                revenue_trend = features_df['revenue'].pct_change().mean() if 'revenue' in features_df.columns else 0
                if revenue_trend < -0.1:
                    factors.append('Declining revenue trend')
                
                if 'total_assets' in features_df.columns:
                    asset_trend = features_df['total_assets'].pct_change().mean()
                    if asset_trend < -0.05:
                        factors.append('Declining asset base')
            
            return factors[:5]  # Return top 5 factors
            
        except Exception as e:
            logger.warning(f"Error identifying liquidation factors: {e}")
            return ['Unable to identify specific risk factors']
    
    def _calculate_health_score(self, risk_assessment: Dict[str, Any], confidence_scores: Dict[str, float]) -> float:
        """Calculate overall financial health score."""
        try:
            scores = []
            weights = []
            
            # Liquidation risk (inverse scoring)
            if 'liquidation_risk' in risk_assessment:
                liquidation_prob = risk_assessment['liquidation_risk'].get('liquidation_probability', 0.5)
                liquidation_score = (1 - liquidation_prob) * 100
                scores.append(liquidation_score)
                weights.append(self.analysis_config['liquidation_risk']['weight'])
            
            # Financial health
            if 'financial_health' in risk_assessment:
                health_score = risk_assessment['financial_health'].get('overall_score', 50)
                scores.append(health_score)
                weights.append(self.analysis_config['financial_health']['weight'])
            
            # Cash flow (simplified scoring)
            if 'cash_flow_forecast' in risk_assessment:
                cf_assessment = risk_assessment['cash_flow_forecast']
                trend = cf_assessment.get('trend_direction', 0)
                volatility = cf_assessment.get('volatility', 0.5)
                
                # Score based on positive trend and low volatility
                cf_score = max(0, min(100, 50 + trend * 1000 - volatility * 50))
                scores.append(cf_score)
                weights.append(self.analysis_config['cash_flow_forecast']['weight'])
            
            # Calculate weighted average
            if scores and weights:
                weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
                return min(100, max(0, weighted_score))
            
            return 50.0  # Default neutral score
            
        except Exception as e:
            logger.error(f"Error calculating health score: {e}")
            return 50.0
    
    def _generate_insights(self, risk_assessment: Dict[str, Any], features_df: pd.DataFrame, request: AnalysisRequest) -> List[str]:
        """Generate AI-powered insights based on analysis results."""
        insights = []
        
        try:
            # Liquidation risk insights
            if 'liquidation_risk' in risk_assessment:
                liquidation_data = risk_assessment['liquidation_risk']
                risk_level = liquidation_data.get('risk_level', 'medium')
                
                if risk_level == 'high':
                    insights.append("⚠️ High liquidation risk detected. Immediate attention to cash flow management is critical.")
                elif risk_level == 'medium':
                    insights.append("⚡ Moderate liquidation risk identified. Consider improving liquidity position.")
                else:
                    insights.append("✅ Low liquidation risk indicates stable financial position.")
            
            # Financial health insights
            if 'financial_health' in risk_assessment:
                health_data = risk_assessment['financial_health']
                grade = health_data.get('grade', 'C')
                overall_score = health_data.get('overall_score', 50)
                
                if overall_score >= 80:
                    insights.append(f"💪 Excellent financial health (Grade {grade}). Company shows strong fundamentals.")
                elif overall_score >= 60:
                    insights.append(f"📈 Good financial health (Grade {grade}) with room for improvement.")
                else:
                    insights.append(f"🔴 Poor financial health (Grade {grade}). Urgent improvements needed.")
                
                # Specific component insights
                liquidity_score = health_data.get('liquidity_score', 50)
                if liquidity_score < 40:
                    insights.append("💧 Liquidity concerns detected. Focus on improving working capital management.")
                
                profitability_score = health_data.get('profitability_score', 50)
                if profitability_score < 40:
                    insights.append("📉 Profitability issues identified. Review cost structure and revenue optimization.")
            
            # Cash flow insights
            if 'cash_flow_forecast' in risk_assessment:
                cf_data = risk_assessment['cash_flow_forecast']
                trend_direction = cf_data.get('trend_direction', 0)
                
                if trend_direction > 0:
                    insights.append("📊 Positive cash flow trend detected, indicating improving operational efficiency.")
                elif trend_direction < -0.1:
                    insights.append("⬇️ Declining cash flow trend requires immediate attention to operations.")
                
                if cf_data.get('seasonality_detected', False):
                    insights.append("🗓️ Seasonal patterns identified in cash flow. Plan for cyclical variations.")
            
            # Database-specific insights
            data_periods = len(features_df)
            latest_date = features_df['date'].max() if 'date' in features_df.columns else 'Unknown'
            insights.append(f"📅 Analysis based on {data_periods} periods of data. Latest data: {latest_date}")
            
            # Industry comparison insights (if available)
            latest_data = features_df.iloc[-1] if len(features_df) > 0 else pd.Series()
            current_ratio = latest_data.get('current_ratio', 1.0)
            
            if current_ratio > 2.0:
                insights.append("💰 Strong liquidity position provides good buffer against financial stress.")
            elif current_ratio < 1.0:
                insights.append("⚠️ Current ratio below 1.0 suggests potential short-term payment difficulties.")
            
            # Growth insights
            if len(features_df) > 1 and 'revenue' in features_df.columns:
                revenue_growth = features_df['revenue'].pct_change().mean()
                if revenue_growth > 0.1:
                    insights.append("🚀 Strong revenue growth indicates expanding business operations.")
                elif revenue_growth < -0.05:
                    insights.append("📉 Revenue decline trend requires strategic review and corrective action.")
            
            return insights[:8]  # Return top 8 insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return ["Analysis completed but detailed insights could not be generated."]
    
    def _generate_recommendations(self, risk_assessment: Dict[str, Any], insights: List[str], request: AnalysisRequest) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        try:
            # Liquidation risk recommendations
            if 'liquidation_risk' in risk_assessment:
                liquidation_data = risk_assessment['liquidation_risk']
                risk_level = liquidation_data.get('risk_level', 'medium')
                key_factors = liquidation_data.get('key_factors', [])
                
                if risk_level == 'high':
                    recommendations.extend([
                        "🚨 URGENT: Implement immediate cash conservation measures",
                        "💼 Consider emergency funding options or asset liquidation",
                        "📞 Engage with creditors for payment restructuring discussions"
                    ])
                elif risk_level == 'medium':
                    recommendations.extend([
                        "📋 Develop 12-month cash flow forecast and monitoring system",
                        "💡 Identify opportunities to improve working capital efficiency",
                        "🤝 Establish contingency funding arrangements"
                    ])
                
                # Specific factor-based recommendations
                for factor in key_factors:
                    if 'liquidity' in factor.lower():
                        recommendations.append("💧 Focus on accelerating receivables collection and optimizing inventory levels")
                    elif 'debt' in factor.lower():
                        recommendations.append("📊 Consider debt restructuring or refinancing options")
                    elif 'cash flow' in factor.lower():
                        recommendations.append("⚡ Implement strict cash flow monitoring and control procedures")
            
            # Financial health recommendations
            if 'financial_health' in risk_assessment:
                health_data = risk_assessment['financial_health']
                overall_score = health_data.get('overall_score', 50)
                
                if overall_score < 60:
                    recommendations.extend([
                        "📈 Develop comprehensive financial improvement plan",
                        "🔍 Conduct detailed cost analysis and reduction program",
                        "💼 Consider strategic partnerships or operational restructuring"
                    ])
                
                # Component-specific recommendations
                liquidity_score = health_data.get('liquidity_score', 50)
                if liquidity_score < 50:
                    recommendations.append("💧 Implement working capital optimization initiatives")
                
                profitability_score = health_data.get('profitability_score', 50)
                if profitability_score < 50:
                    recommendations.extend([
                        "💰 Review pricing strategy and cost structure optimization",
                        "🎯 Focus on high-margin products/services and customer segments"
                    ])
                
                leverage_score = health_data.get('leverage_score', 50)
                if leverage_score < 50:
                    recommendations.append("⚖️ Develop debt reduction strategy and improve capital structure")
            
            # Cash flow recommendations
            if 'cash_flow_forecast' in risk_assessment:
                cf_data = risk_assessment['cash_flow_forecast']
                trend_direction = cf_data.get('trend_direction', 0)
                volatility = cf_data.get('volatility', 0)
                
                if trend_direction < 0:
                    recommendations.extend([
                        "📊 Analyze cash flow drivers and implement improvement measures",
                        "🔄 Review operational efficiency and cost management practices"
                    ])
                
                if volatility > 0.3:
                    recommendations.append("📈 Implement cash flow smoothing strategies to reduce volatility")
                
                if cf_data.get('seasonality_detected', False):
                    recommendations.append("🗓️ Develop seasonal cash flow management and planning processes")
            
            # Database-specific recommendations
            recommendations.extend([
                "📊 Establish automated monthly financial data collection and analysis",
                "🔄 Implement regular data quality checks and validation procedures",
                "📈 Set up automated alerts for key financial metric thresholds"
            ])
            
            # General business recommendations
            recommendations.extend([
                "📊 Establish monthly financial reporting and KPI monitoring",
                "🎯 Set specific financial targets and milestones for next 12 months",
                "👥 Consider engaging financial advisory services for strategic guidance"
            ])
            
            return recommendations[:10]  # Return top 10 recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Seek professional financial advisory services for detailed recommendations."]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models and service status."""
        return {
            'service_version': '1.0.0',
            'models_loaded': list(self.models.keys()),
            'time_series_analyzer_available': self.time_series_analyzer is not None,
            'analysis_types_supported': list(self.analysis_config.keys()),
            'confidence_threshold': self.confidence_threshold,
            'async_enabled': self.enable_async,
            'database_type': self.db_manager.db_type,
            'database_connected': self.db_manager.connection is not None
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform service health check."""
        try:
            status = {
                'service_status': 'healthy',
                'models_available': len(self.models),
                'analyzers_available': 1 if self.time_series_analyzer else 0,
                'database_status': 'connected' if self.db_manager.connection else 'disconnected',
                'last_check': datetime.now().isoformat()
            }
            
            # Test database connection
            try:
                companies = self.db_manager.get_company_list(limit=1)
                status['database_test'] = 'success'
                status['sample_companies_available'] = len(companies)
            except Exception as e:
                status['database_test'] = f'failed: {str(e)}'
                status['sample_companies_available'] = 0
            
            # Test basic functionality with real database data
            try:
                companies = self.db_manager.get_company_list(limit=1)
                if companies:
                    test_company = companies[0]
                    test_request = AnalysisRequest(
                        company_id=test_company,
                        analysis_types=['financial_health'],
                        confidence_threshold=0.5,
                        historical_periods=1
                    )
                    
                    test_result = self.analyze(test_request)
                    status['test_analysis_successful'] = test_result.financial_health_score > 0
                    status['test_company'] = test_company
                else:
                    # No companies found in database
                    status['test_analysis_successful'] = False
                    status['test_company'] = None
                    status['message'] = 'No companies found in database for testing'
                    status['suggestion'] = 'Please load financial data into the database first'
                    
            except Exception as e:
                status['test_analysis_successful'] = False
                status['test_error'] = str(e)
                status['message'] = 'Database test failed'
            
            return status
            
        except Exception as e:
            return {
                'service_status': 'unhealthy',
                'error': str(e),
                'last_check': datetime.now().isoformat()
            }
    
    def __del__(self):
        """Cleanup resources."""
        if self.executor:
            self.executor.shutdown(wait=True)
        if hasattr(self, 'db_manager'):
            self.db_manager.disconnect()


# Example usage and testing functions
def main():
    """Example usage of the AI Analysis Service with Database Integration."""
    try:
        # Initialize service with database connection
        db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'financial_risk_db',
            'user': 'postgres',
            'password': 'password'
        }
        
        ai_service = AIAnalysisService(
            model_directory="./models/",
            confidence_threshold=0.75,
            enable_async=True,
            db_type="postgresql",
            db_config=db_config
        )
        
        # Get available companies
        companies = ai_service.get_available_companies(limit=10)
        print(f"Available companies: {companies}")
        
        # Analyze a specific company
        if companies:
            company_id = companies[0]
            print(f"Analyzing company: {company_id}")
            
            # Get company info
            company_info = ai_service.get_company_info(company_id)
            print(f"Company info: {company_info}")
            
            # Perform analysis
            result = ai_service.analyze_company_by_id(
                company_id=company_id,
                analysis_types=['liquidation_risk', 'financial_health', 'cash_flow_forecast'],
                historical_periods=3
            )
            
            print(f"Analysis completed for {company_id}")
            print(f"Financial Health Score: {result.financial_health_score:.2f}")
            print(f"Data Quality Score: {result.data_quality_score:.2f}")
            print(f"Processing Time: {result.processing_time:.2f}s")
            print(f"Key Insights: {result.insights[:3]}")
            print(f"Top Recommendations: {result.recommendations[:3]}")
        
        # Perform health check
        health_status = ai_service.health_check()
        print(f"Service Health: {health_status}")
        
        logger.info("AI Analysis Service with Database Integration initialized and tested successfully")
        return ai_service
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    service = main()