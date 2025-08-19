"""
services/data_imputation.py
===========================

Data Imputation Service - Complete Implementation
================================================

This service handles missing financial data imputation using multiple advanced methods.
It implements the 5-method imputation system described in the project proposal:
1. K-Nearest Neighbours (KNN)
2. Random Forest
3. Time Series Analysis
4. Peer Company Analysis
5. Industry Benchmark

Author: Prateek Dahiya
Project: Financial Risk Assessment Model for Small Companies and Startups
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
import logging
import warnings

# Machine Learning imports
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# Time series imports
from scipy import stats
from scipy.interpolate import interp1d
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ImputationResult:
    """Results from data imputation process."""
    company_id: str
    imputation_timestamp: datetime
    original_missing_count: int
    imputed_values_count: int
    methods_used: List[str]
    confidence_scores: Dict[str, float]
    quality_metrics: Dict[str, float]
    imputed_fields: List[str]
    validation_results: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['imputation_timestamp'] = self.imputation_timestamp.isoformat()
        return data

class DataImputationService:
    """
    Advanced data imputation service for financial data.
    Implements multiple imputation methods with intelligent fallback chains.
    """
    
    def __init__(self, 
                 enable_validation: bool = True,
                 use_ensemble: bool = True,
                 confidence_threshold: float = 0.6):
        """
        Initialize Data Imputation Service.
        
        Args:
            enable_validation (bool): Enable imputation validation
            use_ensemble (bool): Use ensemble methods for better accuracy
            confidence_threshold (float): Minimum confidence for accepting imputed values
        """
        self.enable_validation = enable_validation
        self.use_ensemble = use_ensemble
        self.confidence_threshold = confidence_threshold
        
        # Database connection (to be injected)
        self.db_connection = None
        
        # Industry benchmarks and ratios
        self.industry_benchmarks = {
            'technology': {
                'asset_to_revenue_ratio': (0.1, 0.2),
                'current_ratio': (1.5, 3.0),
                'debt_to_equity_ratio': (0.1, 0.4),
                'profit_margin': (0.08, 0.25),
                'inventory_turnover': (10, 20),
                'collection_period_days': (30, 45)
            },
            'manufacturing': {
                'asset_to_revenue_ratio': (0.3, 0.5),
                'current_ratio': (1.2, 2.5),
                'debt_to_equity_ratio': (0.2, 0.6),
                'profit_margin': (0.05, 0.15),
                'inventory_turnover': (4, 8),
                'collection_period_days': (45, 60)
            },
            'retail': {
                'asset_to_revenue_ratio': (0.1, 0.2),
                'current_ratio': (1.0, 2.0),
                'debt_to_equity_ratio': (0.3, 0.7),
                'profit_margin': (0.02, 0.08),
                'inventory_turnover': (6, 12),
                'collection_period_days': (15, 30)
            },
            'healthcare': {
                'asset_to_revenue_ratio': (0.4, 0.8),
                'current_ratio': (1.5, 3.0),
                'debt_to_equity_ratio': (0.2, 0.5),
                'profit_margin': (0.10, 0.20),
                'inventory_turnover': (8, 15),
                'collection_period_days': (60, 90)
            },
            'financial': {
                'asset_to_revenue_ratio': (8.0, 12.0),
                'current_ratio': (1.0, 1.5),
                'debt_to_equity_ratio': (4.0, 8.0),
                'profit_margin': (0.15, 0.30),
                'inventory_turnover': (0, 0),  # No inventory
                'collection_period_days': (30, 60)
            }
        }
        
        # Imputation method weights for ensemble
        self.method_weights = {
            'knn': 0.25,
            'random_forest': 0.25,
            'time_series': 0.20,
            'peer_company': 0.20,
            'industry_benchmark': 0.10
        }
        
        # Field-specific imputation strategies
        self.field_strategies = {
            'cash_and_equivalents': ['peer_company', 'industry_benchmark', 'time_series'],
            'accounts_receivable': ['peer_company', 'industry_benchmark', 'knn'],
            'inventory': ['industry_benchmark', 'peer_company', 'time_series'],
            'property_plant_equipment': ['peer_company', 'time_series', 'industry_benchmark'],
            'accounts_payable': ['peer_company', 'industry_benchmark', 'knn'],
            'short_term_debt': ['peer_company', 'random_forest', 'industry_benchmark'],
            'long_term_debt': ['peer_company', 'random_forest', 'industry_benchmark'],
            'total_equity': ['peer_company', 'time_series', 'knn'],
            'revenue': ['time_series', 'peer_company', 'industry_benchmark'],
            'net_income': ['time_series', 'peer_company', 'random_forest'],
            'depreciation_amortization': ['industry_benchmark', 'peer_company', 'time_series']
        }
        
        logger.info("Data Imputation Service initialized")
    
    def set_database_connection(self, db_connection):
        """Set database connection for data fetching."""
        self.db_connection = db_connection
        logger.info("Database connection set for Data Imputation Service")
    
    def impute_missing_data(self, 
                           company_id: str, 
                           data: pd.DataFrame,
                           company_industry: str = 'technology') -> ImputationResult:
        """
        Main method to impute missing financial data.
        
        Args:
            company_id (str): Company identifier
            data (pd.DataFrame): Financial data with missing values
            company_industry (str): Company's industry for benchmarking
            
        Returns:
            ImputationResult: Comprehensive imputation results
        """
        try:
            logger.info(f"Starting data imputation for company {company_id}")
            
            # Analyze missing data
            missing_analysis = self._analyze_missing_data(data)
            original_missing_count = missing_analysis['total_missing']
            
            if original_missing_count == 0:
                logger.info("No missing data found - returning original dataset")
                return ImputationResult(
                    company_id=company_id,
                    imputation_timestamp=datetime.now(),
                    original_missing_count=0,
                    imputed_values_count=0,
                    methods_used=[],
                    confidence_scores={},
                    quality_metrics={'data_completeness': 100.0},
                    imputed_fields=[],
                    validation_results={'status': 'no_imputation_needed'}
                )
            
            # Create a copy for imputation
            imputed_data = data.copy()
            methods_used = []
            confidence_scores = {}
            imputed_fields = []
            
            # Get peer company data
            peer_data = self._fetch_peer_company_data(company_industry)
            
            # Process each field with missing data
            for field in missing_analysis['missing_fields']:
                logger.info(f"Imputing field: {field}")
                
                # Get field-specific strategy
                strategy = self.field_strategies.get(field, ['knn', 'industry_benchmark'])
                
                imputation_success = False
                best_method = None
                best_confidence = 0
                best_values = None
                
                # Try each method in strategy order
                for method in strategy:
                    try:
                        result = self._impute_field(
                            imputed_data, field, method, 
                            company_industry, peer_data
                        )
                        
                        if result['success'] and result['confidence'] > best_confidence:
                            best_method = method
                            best_confidence = result['confidence']
                            best_values = result['imputed_values']
                            
                        if result['confidence'] >= self.confidence_threshold:
                            imputation_success = True
                            break
                            
                    except Exception as e:
                        logger.warning(f"Method {method} failed for field {field}: {e}")
                        continue
                
                # Apply best imputation if found
                if best_values is not None and best_confidence > 0.3:  # Minimum threshold
                    missing_mask = imputed_data[field].isna()
                    imputed_data.loc[missing_mask, field] = best_values[missing_mask]
                    
                    methods_used.append(best_method)
                    confidence_scores[field] = best_confidence
                    imputed_fields.append(field)
                    
                    logger.info(f"Field {field} imputed using {best_method} "
                              f"with confidence {best_confidence:.3f}")
                else:
                    logger.warning(f"Failed to impute field {field} with sufficient confidence")
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(data, imputed_data)
            
            # Validate imputed data
            validation_results = {}
            if self.enable_validation:
                validation_results = self._validate_imputed_data(data, imputed_data)
            
            # Update the original dataframe with imputed values
            data.update(imputed_data)
            
            result = ImputationResult(
                company_id=company_id,
                imputation_timestamp=datetime.now(),
                original_missing_count=original_missing_count,
                imputed_values_count=len(imputed_fields),
                methods_used=list(set(methods_used)),
                confidence_scores=confidence_scores,
                quality_metrics=quality_metrics,
                imputed_fields=imputed_fields,
                validation_results=validation_results
            )
            
            logger.info(f"Imputation completed for {company_id}. "
                       f"Filled {len(imputed_fields)} fields using {len(set(methods_used))} methods")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in data imputation: {e}")
            raise
    
    def _analyze_missing_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns."""
        try:
            missing_info = {}
            total_missing = 0
            missing_fields = []
            
            for column in data.columns:
                missing_count = data[column].isna().sum()
                if missing_count > 0:
                    missing_fields.append(column)
                    total_missing += missing_count
                    missing_info[column] = {
                        'missing_count': missing_count,
                        'missing_percentage': (missing_count / len(data)) * 100
                    }
            
            return {
                'total_missing': total_missing,
                'missing_fields': missing_fields,
                'missing_info': missing_info,
                'data_completeness': ((data.size - total_missing) / data.size) * 100
            }
            
        except Exception as e:
            logger.error(f"Error analyzing missing data: {e}")
            return {'total_missing': 0, 'missing_fields': []}
    
    def _impute_field(self, 
                     data: pd.DataFrame, 
                     field: str, 
                     method: str,
                     industry: str,
                     peer_data: pd.DataFrame) -> Dict[str, Any]:
        """Impute a specific field using the specified method."""
        try:
            if method == 'knn':
                return self._impute_knn(data, field)
            elif method == 'random_forest':
                return self._impute_random_forest(data, field)
            elif method == 'time_series':
                return self._impute_time_series(data, field)
            elif method == 'peer_company':
                return self._impute_peer_company(data, field, peer_data)
            elif method == 'industry_benchmark':
                return self._impute_industry_benchmark(data, field, industry)
            else:
                raise ValueError(f"Unknown imputation method: {method}")
                
        except Exception as e:
            logger.error(f"Error in {method} imputation for field {field}: {e}")
            return {
                'success': False,
                'confidence': 0.0,
                'imputed_values': None,
                'error': str(e)
            }
    
    def _impute_knn(self, data: pd.DataFrame, field: str) -> Dict[str, Any]:
        """Impute using K-Nearest Neighbors."""
        try:
            # Select numeric columns for KNN
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 3:  # Need sufficient features
                return {
                    'success': False,
                    'confidence': 0.0,
                    'imputed_values': None,
                    'error': 'Insufficient numeric features for KNN'
                }
            
            # Prepare data for KNN
            knn_data = data[numeric_cols].copy()
            
            # Use KNN imputation
            imputer = KNNImputer(n_neighbors=min(5, len(knn_data) - 1))
            imputed_array = imputer.fit_transform(knn_data)
            
            # Get imputed values for the specific field
            field_idx = numeric_cols.index(field)
            imputed_values = imputed_array[:, field_idx]
            
            # Calculate confidence based on variance of nearest neighbors
            missing_mask = data[field].isna()
            if missing_mask.sum() == 0:
                confidence = 1.0
            else:
                # Calculate confidence based on consistency of neighbors
                non_missing_values = data[field].dropna()
                if len(non_missing_values) > 0:
                    cv = non_missing_values.std() / abs(non_missing_values.mean()) if non_missing_values.mean() != 0 else 1.0
                    confidence = max(0.1, 1.0 - min(cv, 1.0))
                else:
                    confidence = 0.5
            
            return {
                'success': True,
                'confidence': confidence,
                'imputed_values': pd.Series(imputed_values, index=data.index),
                'method_details': {
                    'n_neighbors': min(5, len(knn_data) - 1),
                    'features_used': len(numeric_cols)
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'confidence': 0.0,
                'imputed_values': None,
                'error': str(e)
            }
    
    def _impute_random_forest(self, data: pd.DataFrame, field: str) -> Dict[str, Any]:
        """Impute using Random Forest regression."""
        try:
            # Select numeric columns for features
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_cols if col != field]
            
            if len(feature_cols) < 2:
                return {
                    'success': False,
                    'confidence': 0.0,
                    'imputed_values': None,
                    'error': 'Insufficient features for Random Forest'
                }
            
            # Prepare training data (non-missing values)
            complete_mask = data[field].notna()
            missing_mask = data[field].isna()
            
            if complete_mask.sum() < 3:  # Need minimum training samples
                return {
                    'success': False,
                    'confidence': 0.0,
                    'imputed_values': None,
                    'error': 'Insufficient training samples'
                }
            
            # Handle missing values in features using simple imputation
            feature_data = data[feature_cols].copy()
            for col in feature_cols:
                if feature_data[col].isna().any():
                    feature_data[col].fillna(feature_data[col].median(), inplace=True)
            
            # Train Random Forest
            X_train = feature_data[complete_mask]
            y_train = data.loc[complete_mask, field]
            
            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            rf.fit(X_train, y_train)
            
            # Predict missing values
            X_predict = feature_data[missing_mask]
            if len(X_predict) > 0:
                predicted_values = rf.predict(X_predict)
            else:
                predicted_values = []
            
            # Create full series with original values and predictions
            imputed_series = data[field].copy()
            if len(predicted_values) > 0:
                imputed_series.loc[missing_mask] = predicted_values
            
            # Calculate confidence based on model performance
            if len(X_train) > 2:
                # Cross-validation score approximation
                train_pred = rf.predict(X_train)
                r2_score = 1 - (np.sum((y_train - train_pred) ** 2) / 
                              np.sum((y_train - y_train.mean()) ** 2))
                confidence = max(0.1, min(0.95, r2_score))
            else:
                confidence = 0.5
            
            return {
                'success': True,
                'confidence': confidence,
                'imputed_values': imputed_series,
                'method_details': {
                    'n_estimators': 100,
                    'features_used': len(feature_cols),
                    'training_samples': len(X_train),
                    'feature_importance': dict(zip(feature_cols, rf.feature_importances_))
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'confidence': 0.0,
                'imputed_values': None,
                'error': str(e)
            }
    
    def _impute_time_series(self, data: pd.DataFrame, field: str) -> Dict[str, Any]:
        """Impute using time series analysis."""
        try:
            # Check if we have date column
            date_columns = []
            for col in data.columns:
                if 'date' in col.lower() or 'year' in col.lower():
                    date_columns.append(col)
            
            if not date_columns:
                # Use index as time if no date column
                time_series = data[field].copy()
            else:
                # Sort by date column
                date_col = date_columns[0]
                sorted_data = data.sort_values(date_col)
                time_series = sorted_data[field].copy()
            
            # Remove missing values for analysis
            clean_series = time_series.dropna()
            
            if len(clean_series) < 3:
                return {
                    'success': False,
                    'confidence': 0.0,
                    'imputed_values': None,
                    'error': 'Insufficient time series data'
                }
            
            # Interpolate missing values
            if len(clean_series) >= 3:
                # Linear interpolation for simple cases
                imputed_series = time_series.interpolate(method='linear')
                
                # Fill remaining NaN values (at edges) with forward/backward fill
                imputed_series = imputed_series.fillna(method='ffill').fillna(method='bfill')
                
                # For remaining NaN, use mean
                imputed_series = imputed_series.fillna(clean_series.mean())
                
                # Calculate confidence based on trend stability
                if len(clean_series) > 2:
                    # Calculate trend consistency
                    trend_changes = np.diff(clean_series.values)
                    if len(trend_changes) > 1:
                        trend_cv = np.std(trend_changes) / (abs(np.mean(trend_changes)) + 1e-8)
                        confidence = max(0.2, 1.0 - min(trend_cv / 2, 0.8))
                    else:
                        confidence = 0.6
                else:
                    confidence = 0.4
                
                return {
                    'success': True,
                    'confidence': confidence,
                    'imputed_values': imputed_series,
                    'method_details': {
                        'interpolation_method': 'linear',
                        'data_points_used': len(clean_series)
                    }
                }
            
            return {
                'success': False,
                'confidence': 0.0,
                'imputed_values': None,
                'error': 'Unable to perform time series imputation'
            }
            
        except Exception as e:
            return {
                'success': False,
                'confidence': 0.0,
                'imputed_values': None,
                'error': str(e)
            }
    
    def _impute_peer_company(self, 
                           data: pd.DataFrame, 
                           field: str, 
                           peer_data: pd.DataFrame) -> Dict[str, Any]:
        """Impute using peer company analysis."""
        try:
            if peer_data.empty or field not in peer_data.columns:
                return {
                    'success': False,
                    'confidence': 0.0,
                    'imputed_values': None,
                    'error': 'No peer data available for field'
                }
            
            # Get peer values for the field
            peer_values = peer_data[field].dropna()
            
            if len(peer_values) == 0:
                return {
                    'success': False,
                    'confidence': 0.0,
                    'imputed_values': None,
                    'error': 'No valid peer values found'
                }
            
            # Calculate statistics from peer companies
            peer_median = peer_values.median()
            peer_mean = peer_values.mean()
            peer_std = peer_values.std()
            
            # Use median as imputation value (more robust)
            imputation_value = peer_median
            
            # Create imputed series
            imputed_series = data[field].copy()
            missing_mask = imputed_series.isna()
            imputed_series.loc[missing_mask] = imputation_value
            
            # Calculate confidence based on peer data consistency
            if len(peer_values) > 1 and peer_mean != 0:
                cv = peer_std / abs(peer_mean)
                confidence = max(0.3, 1.0 - min(cv, 0.7))
            else:
                confidence = 0.5
            
            # Adjust confidence based on number of peers
            peer_count_factor = min(1.0, len(peer_values) / 5)  # Max confidence with 5+ peers
            confidence *= peer_count_factor
            
            return {
                'success': True,
                'confidence': confidence,
                'imputed_values': imputed_series,
                'method_details': {
                    'peer_count': len(peer_values),
                    'peer_median': peer_median,
                    'peer_mean': peer_mean,
                    'peer_std': peer_std,
                    'imputation_value': imputation_value
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'confidence': 0.0,
                'imputed_values': None,
                'error': str(e)
            }
    
    def _impute_industry_benchmark(self, 
                                 data: pd.DataFrame, 
                                 field: str, 
                                 industry: str) -> Dict[str, Any]:
        """Impute using industry benchmarks."""
        try:
            # Get industry benchmarks
            benchmarks = self.industry_benchmarks.get(industry, {})
            
            # Try to derive value using industry ratios
            imputation_value = None
            confidence = 0.0
            method_used = None
            
            # Field-specific industry benchmark calculations
            if field == 'accounts_receivable' and 'revenue' in data.columns:
                revenue = data['revenue'].median()
                if pd.notna(revenue) and revenue > 0:
                    collection_days = benchmarks.get('collection_period_days', (45, 60))
                    avg_days = np.mean(collection_days)
                    imputation_value = (revenue * avg_days) / 365
                    confidence = 0.7
                    method_used = 'revenue_based_calculation'
            
            elif field == 'inventory' and 'revenue' in data.columns:
                revenue = data['revenue'].median()
                if pd.notna(revenue) and revenue > 0:
                    turnover_range = benchmarks.get('inventory_turnover', (6, 12))
                    avg_turnover = np.mean(turnover_range)
                    # Assuming COGS is 70% of revenue
                    cogs = revenue * 0.7
                    imputation_value = cogs / avg_turnover
                    confidence = 0.6
                    method_used = 'inventory_turnover_calculation'
            
            elif field == 'current_assets' and 'current_liabilities' in data.columns:
                current_liabilities = data['current_liabilities'].median()
                if pd.notna(current_liabilities) and current_liabilities > 0:
                    current_ratio_range = benchmarks.get('current_ratio', (1.5, 2.5))
                    avg_ratio = np.mean(current_ratio_range)
                    imputation_value = current_liabilities * avg_ratio
                    confidence = 0.65
                    method_used = 'current_ratio_calculation'
            
            elif field == 'total_debt' and 'total_equity' in data.columns:
                total_equity = data['total_equity'].median()
                if pd.notna(total_equity) and total_equity > 0:
                    debt_equity_range = benchmarks.get('debt_to_equity_ratio', (0.3, 0.6))
                    avg_ratio = np.mean(debt_equity_range)
                    imputation_value = total_equity * avg_ratio
                    confidence = 0.6
                    method_used = 'debt_equity_calculation'
            
            elif field == 'net_income' and 'revenue' in data.columns:
                revenue = data['revenue'].median()
                if pd.notna(revenue) and revenue > 0:
                    margin_range = benchmarks.get('profit_margin', (0.05, 0.15))
                    avg_margin = np.mean(margin_range)
                    imputation_value = revenue * avg_margin
                    confidence = 0.5
                    method_used = 'profit_margin_calculation'
            
            # Fallback: use general industry average if available
            if imputation_value is None:
                # Use a generic multiplier based on industry
                if 'revenue' in data.columns:
                    revenue = data['revenue'].median()
                    if pd.notna(revenue) and revenue > 0:
                        # Generic field multipliers by industry
                        multipliers = {
                            'technology': {'cash_and_equivalents': 0.1, 'accounts_receivable': 0.12},
                            'manufacturing': {'inventory': 0.15, 'property_plant_equipment': 0.4},
                            'retail': {'inventory': 0.25, 'accounts_receivable': 0.05},
                            'healthcare': {'accounts_receivable': 0.2, 'property_plant_equipment': 0.6},
                            'financial': {'cash_and_equivalents': 0.05}
                        }
                        
                        field_multiplier = multipliers.get(industry, {}).get(field)
                        if field_multiplier:
                            imputation_value = revenue * field_multiplier
                            confidence = 0.4
                            method_used = 'industry_multiplier'
            
            if imputation_value is not None:
                # Create imputed series
                imputed_series = data[field].copy()
                missing_mask = imputed_series.isna()
                imputed_series.loc[missing_mask] = imputation_value
                
                return {
                    'success': True,
                    'confidence': confidence,
                    'imputed_values': imputed_series,
                    'method_details': {
                        'industry': industry,
                        'method_used': method_used,
                        'imputation_value': imputation_value,
                        'benchmarks_used': benchmarks
                    }
                }
            
            return {
                'success': False,
                'confidence': 0.0,
                'imputed_values': None,
                'error': 'No suitable industry benchmark found for field'
            }
            
        except Exception as e:
            return {
                'success': False,
                'confidence': 0.0,
                'imputed_values': None,
                'error': str(e)
            }
    
    def _fetch_peer_company_data(self, industry: str) -> pd.DataFrame:
        """Fetch peer company data from database."""
        try:
            if not self.db_connection:
                logger.warning("No database connection available for peer data")
                return pd.DataFrame()
            
            # Query for peer companies in the same industry
            query = """
                SELECT company_id, industry, 
                       cash_and_equivalents, accounts_receivable, inventory,
                       property_plant_equipment, current_assets, total_assets,
                       accounts_payable, current_liabilities, total_debt,
                       total_equity, revenue, net_income
                FROM financial_data 
                WHERE industry = %s 
                AND date >= %s
                ORDER BY date DESC
                LIMIT 100
            """
            
            # Get data from last 2 years
            cutoff_date = datetime.now() - timedelta(days=730)
            
            df = pd.read_sql_query(query, self.db_connection, 
                                 params=[industry, cutoff_date])
            
            if df.empty:
                logger.warning(f"No peer company data found for industry: {industry}")
            else:
                logger.info(f"Found {len(df)} peer company records for {industry}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching peer company data: {e}")
            return pd.DataFrame()
    
    def _calculate_quality_metrics(self, 
                                 original_data: pd.DataFrame, 
                                 imputed_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate quality metrics for imputed data."""
        try:
            metrics = {}
            
            # Data completeness improvement
            original_completeness = ((original_data.size - original_data.isna().sum().sum()) / original_data.size) * 100
            imputed_completeness = ((imputed_data.size - imputed_data.isna().sum().sum()) / imputed_data.size) * 100
            
            metrics['original_completeness'] = original_completeness
            metrics['imputed_completeness'] = imputed_completeness
            metrics['completeness_improvement'] = imputed_completeness - original_completeness
            
            # Statistical consistency checks
            numeric_cols = original_data.select_dtypes(include=[np.number]).columns
            
            correlation_changes = []
            distribution_changes = []
            
            for col in numeric_cols:
                if original_data[col].notna().sum() > 1:
                    # Check correlation preservation with other fields
                    for other_col in numeric_cols:
                        if col != other_col and original_data[other_col].notna().sum() > 1:
                            # Calculate correlations
                            orig_corr = original_data[col].corr(original_data[other_col])
                            imp_corr = imputed_data[col].corr(imputed_data[other_col])
                            
                            if pd.notna(orig_corr) and pd.notna(imp_corr):
                                correlation_changes.append(abs(orig_corr - imp_corr))
                    
                    # Check distribution preservation
                    orig_values = original_data[col].dropna()
                    imp_values = imputed_data[col].dropna()
                    
                    if len(orig_values) > 0 and len(imp_values) > 0:
                        # Compare means and standard deviations
                        mean_change = abs((imp_values.mean() - orig_values.mean()) / (orig_values.mean() + 1e-8))
                        std_change = abs((imp_values.std() - orig_values.std()) / (orig_values.std() + 1e-8))
                        
                        distribution_changes.extend([mean_change, std_change])
            
            # Calculate average changes
            if correlation_changes:
                metrics['avg_correlation_change'] = np.mean(correlation_changes)
                metrics['max_correlation_change'] = np.max(correlation_changes)
            else:
                metrics['avg_correlation_change'] = 0.0
                metrics['max_correlation_change'] = 0.0
            
            if distribution_changes:
                metrics['avg_distribution_change'] = np.mean(distribution_changes)
                metrics['max_distribution_change'] = np.max(distribution_changes)
            else:
                metrics['avg_distribution_change'] = 0.0
                metrics['max_distribution_change'] = 0.0
            
            # Overall quality score (0-100)
            quality_score = 100
            quality_score -= min(30, metrics['avg_correlation_change'] * 100)  # Penalize correlation changes
            quality_score -= min(20, metrics['avg_distribution_change'] * 50)   # Penalize distribution changes
            quality_score = max(0, quality_score)
            
            metrics['overall_quality_score'] = quality_score
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error calculating quality metrics: {e}")
            return {
                'overall_quality_score': 50.0,
                'error': str(e)
            }
    
    def _validate_imputed_data(self, 
                             original_data: pd.DataFrame, 
                             imputed_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate imputed data for consistency and reasonableness."""
        try:
            validation_results = {
                'is_valid': True,
                'warnings': [],
                'errors': [],
                'validation_timestamp': datetime.now().isoformat()
            }
            
            numeric_cols = original_data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                orig_values = original_data[col].dropna()
                imp_values = imputed_data[col].dropna()
                
                if len(orig_values) == 0:
                    continue
                
                # Check for negative values where they shouldn't exist
                non_negative_fields = [
                    'cash_and_equivalents', 'accounts_receivable', 'inventory',
                    'current_assets', 'total_assets', 'revenue'
                ]
                
                if col in non_negative_fields:
                    negative_count = (imp_values < 0).sum()
                    if negative_count > 0:
                        validation_results['warnings'].append(
                            f"Field {col} has {negative_count} negative values after imputation"
                        )
                
                # Check for extreme outliers
                if len(orig_values) > 2:
                    orig_q1, orig_q3 = orig_values.quantile([0.25, 0.75])
                    orig_iqr = orig_q3 - orig_q1
                    
                    if orig_iqr > 0:
                        lower_bound = orig_q1 - 3 * orig_iqr
                        upper_bound = orig_q3 + 3 * orig_iqr
                        
                        outliers = ((imp_values < lower_bound) | (imp_values > upper_bound)).sum()
                        if outliers > 0:
                            validation_results['warnings'].append(
                                f"Field {col} has {outliers} potential outliers after imputation"
                            )
                
                # Check for unrealistic ratios
                if col == 'accounts_receivable' and 'revenue' in imp_values.index:
                    revenue = imputed_data['revenue'].median()
                    ar_median = imp_values.median()
                    if revenue > 0 and ar_median > revenue:
                        validation_results['warnings'].append(
                            "Accounts receivable exceeds annual revenue - may be unrealistic"
                        )
                
                if col == 'inventory' and 'revenue' in imp_values.index:
                    revenue = imputed_data['revenue'].median()
                    inventory_median = imp_values.median()
                    if revenue > 0 and inventory_median > revenue * 0.5:
                        validation_results['warnings'].append(
                            "Inventory is very high relative to revenue"
                        )
            
            # Check balance sheet equation
            if all(field in imputed_data.columns for field in ['total_assets', 'total_liabilities', 'total_equity']):
                assets = imputed_data['total_assets'].median()
                liabilities = imputed_data['total_liabilities'].median()
                equity = imputed_data['total_equity'].median()
                
                if pd.notna(assets) and pd.notna(liabilities) and pd.notna(equity):
                    balance_diff = abs(assets - (liabilities + equity))
                    if balance_diff > assets * 0.05:  # 5% tolerance
                        validation_results['errors'].append(
                            f"Balance sheet equation not satisfied: Assets ({assets:,.0f}) != "
                            f"Liabilities + Equity ({liabilities + equity:,.0f})"
                        )
                        validation_results['is_valid'] = False
            
            # Set overall validation status
            if len(validation_results['errors']) > 0:
                validation_results['is_valid'] = False
            elif len(validation_results['warnings']) > 5:
                validation_results['is_valid'] = False
                validation_results['errors'].append("Too many validation warnings - data quality concerns")
            
            validation_results['validation_score'] = max(0, 100 - len(validation_results['errors']) * 20 - len(validation_results['warnings']) * 5)
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating imputed data: {e}")
            return {
                'is_valid': False,
                'errors': [f"Validation failed: {str(e)}"],
                'warnings': [],
                'validation_score': 0,
                'validation_timestamp': datetime.now().isoformat()
            }
    
    def save_imputation_results(self, result: ImputationResult) -> bool:
        """Save imputation results to database."""
        try:
            if not self.db_connection:
                logger.warning("No database connection available for saving results")
                return False
            
            # Prepare data for database insertion
            imputation_data = {
                'company_id': result.company_id,
                'imputation_timestamp': result.imputation_timestamp,
                'original_missing_count': result.original_missing_count,
                'imputed_values_count': result.imputed_values_count,
                'methods_used': ','.join(result.methods_used),
                'confidence_scores': str(result.confidence_scores),
                'quality_metrics': str(result.quality_metrics),
                'imputed_fields': ','.join(result.imputed_fields),
                'validation_results': str(result.validation_results)
            }
            
            # Insert into database
            columns = ', '.join(imputation_data.keys())
            placeholders = ', '.join(['%s'] * len(imputation_data))
            
            insert_query = f"""
                INSERT INTO imputation_log ({columns})
                VALUES ({placeholders})
            """
            
            cursor = self.db_connection.cursor()
            cursor.execute(insert_query, list(imputation_data.values()))
            self.db_connection.commit()
            cursor.close()
            
            logger.info(f"Imputation results saved for company {result.company_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving imputation results: {e}")
            if self.db_connection:
                self.db_connection.rollback()
            return False
    
    def get_imputation_history(self, company_id: str) -> List[Dict[str, Any]]:
        """Get imputation history for a company."""
        try:
            if not self.db_connection:
                return []
            
            query = """
                SELECT * FROM imputation_log 
                WHERE company_id = %s 
                ORDER BY imputation_timestamp DESC 
                LIMIT 10
            """
            
            df = pd.read_sql_query(query, self.db_connection, params=[company_id])
            
            if df.empty:
                return []
            
            return df.to_dict('records')
            
        except Exception as e:
            logger.error(f"Error fetching imputation history: {e}")
            return []
    
    def bulk_impute_companies(self, 
                            company_ids: List[str], 
                            industry_mapping: Dict[str, str] = None) -> Dict[str, ImputationResult]:
        """Perform bulk imputation for multiple companies."""
        try:
            if industry_mapping is None:
                industry_mapping = {}
            
            results = {}
            
            for company_id in company_ids:
                try:
                    logger.info(f"Processing bulk imputation for company {company_id}")
                    
                    # Fetch company data
                    company_data = self._fetch_company_data(company_id)
                    
                    if company_data.empty:
                        logger.warning(f"No data found for company {company_id}")
                        continue
                    
                    # Get industry
                    industry = industry_mapping.get(company_id, 'technology')
                    
                    # Perform imputation
                    result = self.impute_missing_data(company_id, company_data, industry)
                    
                    # Save results
                    self.save_imputation_results(result)
                    
                    results[company_id] = result
                    
                    logger.info(f"Completed imputation for {company_id}")
                    
                except Exception as e:
                    logger.error(f"Error processing company {company_id}: {e}")
                    continue
            
            logger.info(f"Bulk imputation completed for {len(results)} companies")
            return results
            
        except Exception as e:
            logger.error(f"Error in bulk imputation: {e}")
            return {}
    
    def _fetch_company_data(self, company_id: str) -> pd.DataFrame:
        """Fetch company financial data from database."""
        try:
            if not self.db_connection:
                return pd.DataFrame()
            
            query = """
                SELECT * FROM financial_data 
                WHERE company_id = %s 
                ORDER BY date DESC 
                LIMIT 5
            """
            
            df = pd.read_sql_query(query, self.db_connection, params=[company_id])
            return df
            
        except Exception as e:
            logger.error(f"Error fetching company data: {e}")
            return pd.DataFrame()
    
    def generate_imputation_report(self, result: ImputationResult) -> Dict[str, Any]:
        """Generate comprehensive imputation report."""
        try:
            report = {
                'executive_summary': {
                    'company_id': result.company_id,
                    'imputation_date': result.imputation_timestamp.strftime('%Y-%m-%d'),
                    'data_completeness_improvement': f"{result.imputed_values_count} fields imputed",
                    'overall_confidence': np.mean(list(result.confidence_scores.values())) if result.confidence_scores else 0,
                    'quality_score': result.quality_metrics.get('overall_quality_score', 0)
                },
                'detailed_results': {
                    'original_missing_count': result.original_missing_count,
                    'imputed_fields': result.imputed_fields,
                    'methods_used': result.methods_used,
                    'field_confidence_scores': result.confidence_scores
                },
                'quality_assessment': result.quality_metrics,
                'validation_results': result.validation_results,
                'recommendations': self._generate_imputation_recommendations(result),
                'next_steps': self._suggest_data_improvement_steps(result)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating imputation report: {e}")
            return {'error': str(e)}
    
    def _generate_imputation_recommendations(self, result: ImputationResult) -> List[str]:
        """Generate recommendations based on imputation results."""
        recommendations = []
        
        try:
            avg_confidence = np.mean(list(result.confidence_scores.values())) if result.confidence_scores else 0
            
            if avg_confidence < 0.5:
                recommendations.append("🔍 Low confidence imputation - consider collecting additional data")
            
            if len(result.imputed_fields) > 5:
                recommendations.append("📊 Many fields required imputation - improve data collection processes")
            
            if 'validation_results' in result.validation_results:
                if not result.validation_results.get('is_valid', True):
                    recommendations.append("⚠️ Validation issues detected - review imputed values manually")
            
            high_confidence_methods = [method for method, conf in result.confidence_scores.items() if conf > 0.8]
            if high_confidence_methods:
                recommendations.append(f"✅ High confidence achieved for: {', '.join(high_confidence_methods)}")
            
            if 'peer_company' in result.methods_used:
                recommendations.append("👥 Peer company analysis used - results reflect industry patterns")
            
            if 'time_series' in result.methods_used:
                recommendations.append("📈 Time series analysis applied - consider seasonal patterns")
            
            recommendations.append("📋 Regular data quality audits recommended")
            recommendations.append("🔄 Re-run imputation when new data becomes available")
            
            return recommendations[:6]  # Return top 6 recommendations
            
        except Exception as e:
            logger.warning(f"Error generating recommendations: {e}")
            return ["Consult with data analyst for detailed imputation review"]
    
    def _suggest_data_improvement_steps(self, result: ImputationResult) -> List[str]:
        """Suggest steps to improve data quality."""
        steps = []
        
        try:
            if len(result.imputed_fields) > 3:
                steps.extend([
                    "🎯 Prioritize collection of frequently missing fields",
                    "📋 Implement data validation at point of entry",
                    "🔄 Establish regular data quality monitoring"
                ])
            
            if any(conf < 0.4 for conf in result.confidence_scores.values()):
                steps.append("📊 Focus on improving data sources for low-confidence fields")
            
            steps.extend([
                "🤝 Consider partnering with industry data providers",
                "💼 Train staff on proper financial data recording",
                "🔍 Implement automated data quality checks"
            ])
            
            return steps[:5]  # Return top 5 steps
            
        except Exception as e:
            logger.warning(f"Error generating improvement steps: {e}")
            return ["Consult with data management specialist"]


# Utility functions and example usage
def create_imputation_tables(db_connection):
    """Create database tables for imputation logging."""
    try:
        cursor = db_connection.cursor()
        
        # Create imputation log table
        create_table_query = """
        CREATE TABLE IF NOT EXISTS imputation_log (
            id SERIAL PRIMARY KEY,
            company_id VARCHAR(50) NOT NULL,
            imputation_timestamp TIMESTAMP NOT NULL,
            original_missing_count INTEGER,
            imputed_values_count INTEGER,
            methods_used TEXT,
            confidence_scores TEXT,
            quality_metrics TEXT,
            imputed_fields TEXT,
            validation_results TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_imputation_company_id ON imputation_log(company_id);
        CREATE INDEX IF NOT EXISTS idx_imputation_timestamp ON imputation_log(imputation_timestamp);
        """
        
        cursor.execute(create_table_query)
        db_connection.commit()
        cursor.close()
        
        logger.info("Imputation tables created successfully")
        
    except Exception as e:
        logger.error(f"Error creating imputation tables: {e}")
        raise

def main():
    """Example usage of the Data Imputation Service."""
    try:
        # Initialize service
        imputation_service = DataImputationService(
            enable_validation=True,
            use_ensemble=True,
            confidence_threshold=0.6
        )
        
        # Note: In real usage, you would set a database connection
        # imputation_service.set_database_connection(your_db_connection)
        
        logger.info("Data Imputation Service initialized successfully")
        logger.info("Service ready for missing data imputation")
        
        return imputation_service
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    service = main()