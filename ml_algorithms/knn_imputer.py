"""
KNN Imputer Module
Advanced K-Nearest Neighbors imputation for missing financial data
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ImputationResult:
    """Data class for imputation results"""
    imputed_data: pd.DataFrame
    missing_mask: pd.DataFrame
    imputation_quality: Dict[str, float]
    method_used: str
    confidence_scores: Dict[str, float]

class AdvancedKNNImputer:
    """
    Advanced KNN Imputer for financial data with multiple strategies
    """
    
    def __init__(self, 
                 n_neighbors: int = 5,
                 weights: str = 'distance',
                 metric: str = 'euclidean',
                 industry_aware: bool = True,
                 financial_context: bool = True):
        """
        Initialize Advanced KNN Imputer
        
        Args:
            n_neighbors (int): Number of neighbors to use
            weights (str): Weight function for neighbors ('uniform', 'distance')
            metric (str): Distance metric ('euclidean', 'cosine', 'manhattan')
            industry_aware (bool): Use industry-specific imputation
            financial_context (bool): Apply financial logic to imputation
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.industry_aware = industry_aware
        self.financial_context = financial_context
        
        # Financial data type mapping
        self.FINANCIAL_DATA_TYPES = {
            'ratio': ['current_ratio', 'quick_ratio', 'debt_to_equity', 'roa', 'roe', 'profit_margin'],
            'monetary': ['revenue', 'net_income', 'total_assets', 'current_assets', 'total_debt'],
            'percentage': ['growth_rate', 'margin', 'return'],
            'count': ['employees', 'branches', 'locations'],
            'categorical': ['industry', 'sector', 'performance_category', 'risk_level']
        }
        
        # Industry-specific typical ranges
        self.INDUSTRY_RANGES = {
            'technology': {
                'current_ratio': (1.5, 3.0),
                'debt_to_equity': (0.1, 0.8),
                'roa': (0.05, 0.25),
                'profit_margin': (0.10, 0.30)
            },
            'healthcare': {
                'current_ratio': (1.2, 2.5),
                'debt_to_equity': (0.2, 1.0),
                'roa': (0.03, 0.20),
                'profit_margin': (0.08, 0.25)
            },
            'industrial': {
                'current_ratio': (1.0, 2.0),
                'debt_to_equity': (0.3, 1.5),
                'roa': (0.02, 0.15),
                'profit_margin': (0.05, 0.20)
            },
            'financial': {
                'current_ratio': (1.0, 1.5),
                'debt_to_equity': (0.5, 3.0),
                'roa': (0.01, 0.12),
                'profit_margin': (0.15, 0.35)
            }
        }
        
        # Scaler for numerical features
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # Imputation statistics
        self.imputation_stats = {}
    
    def fit_transform(self, 
                     data: pd.DataFrame, 
                     industry_column: str = 'industry',
                     company_id_column: str = 'company_id') -> ImputationResult:
        """
        Fit and transform data with KNN imputation
        
        Args:
            data (pd.DataFrame): Input data with missing values
            industry_column (str): Column name for industry information
            company_id_column (str): Column name for company identifier
            
        Returns:
            ImputationResult: Complete imputation results
        """
        try:
            logger.info(f"Starting KNN imputation for {len(data)} records")
            
            # Prepare data
            prepared_data = self._prepare_data(data.copy())
            
            # Create missing value mask
            missing_mask = prepared_data.isnull()
            
            # Perform imputation based on strategy
            if self.industry_aware and industry_column in data.columns:
                imputed_data = self._industry_aware_imputation(prepared_data, industry_column)
            else:
                imputed_data = self._standard_knn_imputation(prepared_data)
            
            # Apply financial logic constraints
            if self.financial_context:
                imputed_data = self._apply_financial_constraints(imputed_data, data.get(industry_column))
            
            # Calculate imputation quality
            quality_metrics = self._calculate_imputation_quality(data, imputed_data, missing_mask)
            
            # Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(data, imputed_data, missing_mask)
            
            result = ImputationResult(
                imputed_data=imputed_data,
                missing_mask=missing_mask,
                imputation_quality=quality_metrics,
                method_used='industry_aware_knn' if self.industry_aware else 'standard_knn',
                confidence_scores=confidence_scores
            )
            
            logger.info("KNN imputation completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in KNN imputation: {e}")
            raise
    
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for KNN imputation"""
        try:
            # Handle categorical variables
            categorical_columns = data.select_dtypes(include=['object']).columns
            
            for col in categorical_columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                
                # Handle missing categorical values
                data[col] = data[col].fillna('Unknown')
                data[col] = self.label_encoders[col].fit_transform(data[col])
            
            # Convert to numeric
            for col in data.columns:
                if data[col].dtype == 'object':
                    try:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                    except:
                        pass
            
            return data
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return data
    
    def _industry_aware_imputation(self, data: pd.DataFrame, industry_column: str) -> pd.DataFrame:
        """Perform industry-aware KNN imputation"""
        try:
            imputed_data = data.copy()
            industries = data[industry_column].unique()
            
            for industry in industries:
                if pd.isna(industry):
                    continue
                
                # Get industry subset
                industry_mask = data[industry_column] == industry
                industry_data = data[industry_mask].copy()
                
                if len(industry_data) < self.n_neighbors:
                    # Not enough data in industry, use global imputation
                    logger.warning(f"Insufficient data for industry {industry}, using global imputation")
                    continue
                
                # Remove industry column for imputation
                imputation_data = industry_data.drop(columns=[industry_column], errors='ignore')
                
                # Perform KNN imputation for this industry
                industry_imputed = self._perform_knn_imputation(imputation_data)
                
                # Update the main dataset
                imputed_data.loc[industry_mask, industry_imputed.columns] = industry_imputed
            
            return imputed_data
            
        except Exception as e:
            logger.error(f"Error in industry-aware imputation: {e}")
            return self._standard_knn_imputation(data)
    
    def _standard_knn_imputation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Perform standard KNN imputation"""
        try:
            return self._perform_knn_imputation(data)
            
        except Exception as e:
            logger.error(f"Error in standard KNN imputation: {e}")
            return data
    
    def _perform_knn_imputation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Core KNN imputation logic"""
        try:
            # Separate numerical and categorical data
            numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numerical_cols:
                logger.warning("No numerical columns found for KNN imputation")
                return data
            
            numerical_data = data[numerical_cols].copy()
            
            # Check if there's enough complete data
            complete_rows = numerical_data.dropna()
            if len(complete_rows) < self.n_neighbors:
                logger.warning("Insufficient complete data for KNN imputation")
                return self._fallback_imputation(data)
            
            # Use appropriate KNN strategy based on data characteristics
            if self._should_use_weighted_knn(numerical_data):
                imputed_data = self._weighted_knn_imputation(numerical_data)
            else:
                imputed_data = self._simple_knn_imputation(numerical_data)
            
            # Combine with original non-numerical data
            result = data.copy()
            result[numerical_cols] = imputed_data
            
            return result
            
        except Exception as e:
            logger.error(f"Error performing KNN imputation: {e}")
            return self._fallback_imputation(data)
    
    def _should_use_weighted_knn(self, data: pd.DataFrame) -> bool:
        """Determine if weighted KNN should be used"""
        try:
            # Use weighted KNN if data has high variance or outliers
            for col in data.select_dtypes(include=[np.number]).columns:
                col_data = data[col].dropna()
                if len(col_data) > 10:
                    cv = col_data.std() / col_data.mean() if col_data.mean() != 0 else 0
                    if cv > 1.0:  # High coefficient of variation
                        return True
            return False
            
        except Exception as e:
            logger.error(f"Error checking for weighted KNN: {e}")
            return False
    
    def _weighted_knn_imputation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Perform weighted KNN imputation with custom weights"""
        try:
            imputed_data = data.copy()
            
            for col in data.columns:
                if data[col].isnull().any():
                    imputed_data[col] = self._impute_column_weighted(data, col)
            
            return imputed_data
            
        except Exception as e:
            logger.error(f"Error in weighted KNN imputation: {e}")
            return self._simple_knn_imputation(data)
    
    def _impute_column_weighted(self, data: pd.DataFrame, target_column: str) -> pd.Series:
        """Impute a single column using weighted KNN"""
        try:
            result = data[target_column].copy()
            missing_indices = data[target_column].isnull()
            
            if not missing_indices.any():
                return result
            
            # Get complete data for similarity calculation
            complete_data = data.dropna()
            if len(complete_data) < self.n_neighbors:
                # Fallback to mean imputation
                result.fillna(result.mean(), inplace=True)
                return result
            
            # For each missing value
            for idx in data[missing_indices].index:
                row_data = data.loc[idx].drop(target_column)
                
                # Calculate distances to complete rows
                distances = []
                for comp_idx in complete_data.index:
                    comp_row = complete_data.loc[comp_idx].drop(target_column)
                    
                    # Only use columns that are not missing in the target row
                    valid_cols = row_data.notna() & comp_row.notna()
                    
                    if valid_cols.any():
                        if self.metric == 'euclidean':
                            dist = np.sqrt(np.sum((row_data[valid_cols] - comp_row[valid_cols]) ** 2))
                        elif self.metric == 'manhattan':
                            dist = np.sum(np.abs(row_data[valid_cols] - comp_row[valid_cols]))
                        else:  # cosine
                            dist = 1 - cosine_similarity([row_data[valid_cols]], [comp_row[valid_cols]])[0][0]
                    else:
                        dist = np.inf
                    
                    distances.append((comp_idx, dist))
                
                # Get k nearest neighbors
                distances.sort(key=lambda x: x[1])
                neighbors = distances[:self.n_neighbors]
                
                # Calculate weighted average
                if self.weights == 'distance':
                    weights = [1 / (dist + 1e-8) for _, dist in neighbors]  # Add small epsilon to avoid division by zero
                else:
                    weights = [1] * len(neighbors)
                
                neighbor_values = [complete_data.loc[idx, target_column] for idx, _ in neighbors]
                
                # Weighted average
                if sum(weights) > 0:
                    imputed_value = np.average(neighbor_values, weights=weights)
                else:
                    imputed_value = np.mean(neighbor_values)
                
                result.iloc[data.index.get_loc(idx)] = imputed_value
            
            return result
            
        except Exception as e:
            logger.error(f"Error imputing column {target_column}: {e}")
            return data[target_column].fillna(data[target_column].mean())
    
    def _simple_knn_imputation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Perform simple KNN imputation using sklearn"""
        try:
            # Use sklearn's KNNImputer
            imputer = KNNImputer(
                n_neighbors=min(self.n_neighbors, len(data) - 1),
                weights=self.weights,
                metric='nan_euclidean'  # sklearn's KNN imputer uses nan_euclidean
            )
            
            imputed_array = imputer.fit_transform(data)
            imputed_data = pd.DataFrame(imputed_array, columns=data.columns, index=data.index)
            
            return imputed_data
            
        except Exception as e:
            logger.error(f"Error in simple KNN imputation: {e}")
            return self._fallback_imputation(data)
    
    def _fallback_imputation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fallback imputation method (mean/mode)"""
        try:
            imputed_data = data.copy()
            
            for col in data.columns:
                if data[col].isnull().any():
                    if data[col].dtype in ['int64', 'float64']:
                        # Use median for numerical data (more robust than mean)
                        imputed_data[col].fillna(data[col].median(), inplace=True)
                    else:
                        # Use mode for categorical data
                        mode_value = data[col].mode()
                        if not mode_value.empty:
                            imputed_data[col].fillna(mode_value[0], inplace=True)
            
            return imputed_data
            
        except Exception as e:
            logger.error(f"Error in fallback imputation: {e}")
            return data
    
    def _apply_financial_constraints(self, data: pd.DataFrame, industry: Optional[pd.Series] = None) -> pd.DataFrame:
        """Apply financial logic constraints to imputed values"""
        try:
            constrained_data = data.copy()
            
            # Apply ratio constraints
            constrained_data = self._apply_ratio_constraints(constrained_data)
            
            # Apply industry-specific constraints
            if industry is not None and self.industry_aware:
                constrained_data = self._apply_industry_constraints(constrained_data, industry)
            
            # Apply monetary value constraints
            constrained_data = self._apply_monetary_constraints(constrained_data)
            
            # Apply logical relationships
            constrained_data = self._apply_logical_relationships(constrained_data)
            
            return constrained_data
            
        except Exception as e:
            logger.error(f"Error applying financial constraints: {e}")
            return data
    
    def _apply_ratio_constraints(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply constraints to financial ratios"""
        try:
            # Current ratio constraints (typically 0.1 to 10)
            if 'current_ratio' in data.columns:
                data['current_ratio'] = data['current_ratio'].clip(lower=0.1, upper=10.0)
            
            # Debt-to-equity constraints (typically 0 to 5)
            if 'debt_to_equity' in data.columns:
                data['debt_to_equity'] = data['debt_to_equity'].clip(lower=0.0, upper=5.0)
            
            # ROA constraints (typically -0.5 to 0.5)
            if 'roa' in data.columns:
                data['roa'] = data['roa'].clip(lower=-0.5, upper=0.5)
            
            # ROE constraints (typically -1.0 to 1.0)
            if 'roe' in data.columns:
                data['roe'] = data['roe'].clip(lower=-1.0, upper=1.0)
            
            # Profit margin constraints (typically -1.0 to 1.0)
            if 'profit_margin' in data.columns:
                data['profit_margin'] = data['profit_margin'].clip(lower=-1.0, upper=1.0)
            
            return data
            
        except Exception as e:
            logger.error(f"Error applying ratio constraints: {e}")
            return data
    
    def _apply_industry_constraints(self, data: pd.DataFrame, industry: pd.Series) -> pd.DataFrame:
        """Apply industry-specific constraints"""
        try:
            constrained_data = data.copy()
            
            for idx, ind in industry.items():
                if pd.isna(ind) or ind not in self.INDUSTRY_RANGES:
                    continue
                
                industry_ranges = self.INDUSTRY_RANGES[ind]
                
                for metric, (min_val, max_val) in industry_ranges.items():
                    if metric in constrained_data.columns:
                        # Apply industry-specific constraints with some tolerance
                        tolerance = (max_val - min_val) * 0.2  # 20% tolerance
                        lower_bound = max(0, min_val - tolerance)
                        upper_bound = max_val + tolerance
                        
                        constrained_data.loc[idx, metric] = np.clip(
                            constrained_data.loc[idx, metric],
                            lower_bound,
                            upper_bound
                        )
            
            return constrained_data
            
        except Exception as e:
            logger.error(f"Error applying industry constraints: {e}")
            return data
    
    def _apply_monetary_constraints(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply constraints to monetary values"""
        try:
            monetary_columns = ['revenue', 'net_income', 'total_assets', 'current_assets', 
                              'current_liabilities', 'total_debt', 'shareholders_equity']
            
            for col in monetary_columns:
                if col in data.columns:
                    # Ensure non-negative values for most monetary items
                    if col != 'net_income':  # Net income can be negative
                        data[col] = data[col].clip(lower=0)
                    
                    # Apply reasonable upper bounds based on data distribution
                    if data[col].quantile(0.95) > 0:
                        upper_bound = data[col].quantile(0.95) * 10  # 10x the 95th percentile
                        data[col] = data[col].clip(upper=upper_bound)
            
            return data
            
        except Exception as e:
            logger.error(f"Error applying monetary constraints: {e}")
            return data
    
    def _apply_logical_relationships(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply logical relationships between financial metrics"""
        try:
            # Current assets should be <= Total assets
            if 'current_assets' in data.columns and 'total_assets' in data.columns:
                data['current_assets'] = np.minimum(data['current_assets'], data['total_assets'])
            
            # Current liabilities should be <= Total liabilities (if available)
            if 'current_liabilities' in data.columns and 'total_debt' in data.columns:
                # Assume total debt is a proxy for total liabilities
                data['current_liabilities'] = np.minimum(data['current_liabilities'], data['total_debt'])
            
            # Shareholders equity = Total assets - Total debt (approximately)
            if all(col in data.columns for col in ['total_assets', 'total_debt', 'shareholders_equity']):
                # Only apply if the relationship is severely violated
                calculated_equity = data['total_assets'] - data['total_debt']
                equity_diff = np.abs(data['shareholders_equity'] - calculated_equity)
                severe_violations = equity_diff > data['total_assets'] * 0.3  # 30% of total assets
                
                data.loc[severe_violations, 'shareholders_equity'] = calculated_equity[severe_violations]
            
            return data
            
        except Exception as e:
            logger.error(f"Error applying logical relationships: {e}")
            return data
    
    def _calculate_imputation_quality(self, 
                                    original_data: pd.DataFrame, 
                                    imputed_data: pd.DataFrame, 
                                    missing_mask: pd.DataFrame) -> Dict[str, float]:
        """Calculate quality metrics for imputation"""
        try:
            quality_metrics = {}
            
            # Overall missing data percentage
            total_missing = missing_mask.sum().sum()
            total_cells = missing_mask.size
            quality_metrics['missing_percentage'] = (total_missing / total_cells) * 100
            
            # Per-column missing percentages
            column_missing = {}
            for col in missing_mask.columns:
                col_missing = missing_mask[col].sum()
                col_total = len(missing_mask[col])
                column_missing[col] = (col_missing / col_total) * 100
            
            quality_metrics['column_missing_percentages'] = column_missing
            
            # Imputation consistency (for numerical columns)
            numerical_cols = imputed_data.select_dtypes(include=[np.number]).columns
            consistency_scores = {}
            
            for col in numerical_cols:
                if col in original_data.columns and missing_mask[col].any():
                    # Compare distribution of imputed values vs original values
                    original_values = original_data[col].dropna()
                    imputed_values = imputed_data.loc[missing_mask[col], col]
                    
                    if len(original_values) > 0 and len(imputed_values) > 0:
                        # Use statistical tests for consistency
                        orig_mean = original_values.mean()
                        orig_std = original_values.std()
                        imp_mean = imputed_values.mean()
                        imp_std = imputed_values.std()
                        
                        # Normalize differences
                        mean_diff = abs(orig_mean - imp_mean) / (orig_std + 1e-8)
                        std_ratio = min(orig_std, imp_std) / (max(orig_std, imp_std) + 1e-8)
                        
                        # Consistency score (higher is better)
                        consistency = max(0, 1 - mean_diff) * std_ratio
                        consistency_scores[col] = consistency
            
            quality_metrics['consistency_scores'] = consistency_scores
            quality_metrics['average_consistency'] = np.mean(list(consistency_scores.values())) if consistency_scores else 0
            
            # Imputation coverage
            columns_with_missing = missing_mask.any(axis=0).sum()
            columns_imputed = len([col for col in missing_mask.columns if missing_mask[col].any()])
            quality_metrics['imputation_coverage'] = (columns_imputed / max(columns_with_missing, 1)) * 100
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error calculating imputation quality: {e}")
            return {}
    
    def _calculate_confidence_scores(self, 
                                   original_data: pd.DataFrame, 
                                   imputed_data: pd.DataFrame, 
                                   missing_mask: pd.DataFrame) -> Dict[str, float]:
        """Calculate confidence scores for imputed values"""
        try:
            confidence_scores = {}
            
            for col in missing_mask.columns:
                if missing_mask[col].any():
                    # Calculate confidence based on:
                    # 1. Number of neighbors used
                    # 2. Variance of neighbor values
                    # 3. Distance to neighbors
                    # 4. Data completeness
                    
                    missing_count = missing_mask[col].sum()
                    total_count = len(missing_mask[col])
                    completeness = 1 - (missing_count / total_count)
                    
                    # Base confidence on data completeness
                    base_confidence = completeness
                    
                    # Adjust based on imputation method effectiveness
                    if col in imputed_data.select_dtypes(include=[np.number]).columns:
                        # For numerical data, check variance consistency
                        if col in original_data.columns:
                            orig_values = original_data[col].dropna()
                            imputed_values = imputed_data.loc[missing_mask[col], col]
                            
                            if len(orig_values) > 1 and len(imputed_values) > 0:
                                orig_var = orig_values.var()
                                imp_var = imputed_values.var()
                                
                                # Variance similarity boosts confidence
                                var_similarity = min(orig_var, imp_var) / (max(orig_var, imp_var) + 1e-8)
                                adjusted_confidence = base_confidence * (0.5 + 0.5 * var_similarity)
                            else:
                                adjusted_confidence = base_confidence * 0.7  # Lower confidence for insufficient data
                        else:
                            adjusted_confidence = base_confidence * 0.8
                    else:
                        # For categorical data, use base confidence
                        adjusted_confidence = base_confidence * 0.9
                    
                    confidence_scores[col] = min(1.0, max(0.0, adjusted_confidence))
            
            return confidence_scores
            
        except Exception as e:
            logger.error(f"Error calculating confidence scores: {e}")
            return {}
    
    def cross_validate_imputation(self, 
                                data: pd.DataFrame, 
                                test_fraction: float = 0.1,
                                n_trials: int = 5) -> Dict[str, float]:
        """Cross-validate imputation performance"""
        try:
            logger.info(f"Starting cross-validation with {n_trials} trials")
            
            validation_scores = []
            
            for trial in range(n_trials):
                # Create artificial missing data
                test_data = data.copy()
                
                # Randomly remove some existing values for testing
                for col in data.select_dtypes(include=[np.number]).columns:
                    available_indices = data[col].dropna().index
                    if len(available_indices) > 10:  # Need sufficient data
                        n_to_remove = max(1, int(len(available_indices) * test_fraction))
                        remove_indices = np.random.choice(available_indices, n_to_remove, replace=False)
                        test_data.loc[remove_indices, col] = np.nan
                
                # Perform imputation
                result = self.fit_transform(test_data)
                imputed = result.imputed_data
                
                # Calculate accuracy on removed values
                trial_scores = []
                for col in data.select_dtypes(include=[np.number]).columns:
                    original_values = data[col].dropna()
                    if len(original_values) > 0:
                        # Find artificially missing values that were imputed
                        artificial_missing = test_data[col].isna() & data[col].notna()
                        
                        if artificial_missing.any():
                            true_values = data.loc[artificial_missing, col]
                            predicted_values = imputed.loc[artificial_missing, col]
                            
                            # Calculate relative error
                            relative_errors = np.abs(true_values - predicted_values) / (np.abs(true_values) + 1e-8)
                            mean_relative_error = np.mean(relative_errors)
                            accuracy = max(0, 1 - mean_relative_error)
                            trial_scores.append(accuracy)
                
                if trial_scores:
                    validation_scores.append(np.mean(trial_scores))
            
            cv_results = {
                'mean_accuracy': np.mean(validation_scores) if validation_scores else 0,
                'std_accuracy': np.std(validation_scores) if validation_scores else 0,
                'min_accuracy': np.min(validation_scores) if validation_scores else 0,
                'max_accuracy': np.max(validation_scores) if validation_scores else 0,
                'n_trials': len(validation_scores)
            }
            
            logger.info(f"Cross-validation completed. Mean accuracy: {cv_results['mean_accuracy']:.3f}")
            return cv_results
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            return {}
    
    def get_imputation_report(self, result: ImputationResult) -> str:
        """Generate a comprehensive imputation report"""
        try:
            report_lines = []
            report_lines.append("=== KNN Imputation Report ===")
            report_lines.append(f"Method Used: {result.method_used}")
            report_lines.append(f"Number of neighbors: {self.n_neighbors}")
            report_lines.append(f"Weight function: {self.weights}")
            report_lines.append(f"Distance metric: {self.metric}")
            report_lines.append("")
            
            # Missing data summary
            quality = result.imputation_quality
            report_lines.append("Missing Data Summary:")
            report_lines.append(f"  Overall missing percentage: {quality.get('missing_percentage', 0):.2f}%")
            report_lines.append(f"  Imputation coverage: {quality.get('imputation_coverage', 0):.2f}%")
            report_lines.append(f"  Average consistency score: {quality.get('average_consistency', 0):.3f}")
            report_lines.append("")
            
            # Per-column analysis
            if 'column_missing_percentages' in quality:
                report_lines.append("Per-Column Missing Data:")
                for col, pct in quality['column_missing_percentages'].items():
                    if pct > 0:
                        confidence = result.confidence_scores.get(col, 0)
                        report_lines.append(f"  {col}: {pct:.1f}% missing, {confidence:.3f} confidence")
                report_lines.append("")
            
            # Consistency scores
            if 'consistency_scores' in quality:
                report_lines.append("Consistency Scores (higher is better):")
                for col, score in quality['consistency_scores'].items():
                    report_lines.append(f"  {col}: {score:.3f}")
                report_lines.append("")
            
            # Recommendations
            report_lines.append("Recommendations:")
            avg_confidence = np.mean(list(result.confidence_scores.values())) if result.confidence_scores else 0
            
            if avg_confidence >= 0.8:
                report_lines.append("  ✓ High confidence imputation - results are reliable")
            elif avg_confidence >= 0.6:
                report_lines.append("  ⚠ Moderate confidence imputation - validate critical calculations")
            else:
                report_lines.append("  ⚠ Low confidence imputation - consider additional data sources")
            
            if quality.get('missing_percentage', 0) > 30:
                report_lines.append("  ⚠ High percentage of missing data - results may be unreliable")
            
            if quality.get('average_consistency', 0) < 0.7:
                report_lines.append("  ⚠ Low consistency scores - imputed values may not match data distribution")
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"Error generating imputation report: {e}")
            return "Error generating report"
    
    def save_imputation_results(self, 
                              result: ImputationResult, 
                              filepath: str, 
                              include_metadata: bool = True):
        """Save imputation results to file"""
        try:
            # Save imputed data
            result.imputed_data.to_csv(f"{filepath}_imputed.csv", index=False)
            
            if include_metadata:
                # Save missing mask
                result.missing_mask.to_csv(f"{filepath}_missing_mask.csv", index=False)
                
                # Save quality metrics
                with open(f"{filepath}_quality_report.txt", 'w') as f:
                    f.write(self.get_imputation_report(result))
                
                # Save confidence scores
                confidence_df = pd.DataFrame.from_dict(result.confidence_scores, orient='index', columns=['confidence'])
                confidence_df.to_csv(f"{filepath}_confidence_scores.csv")
            
            logger.info(f"Imputation results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving imputation results: {e}")


class MultiStrategyKNNImputer:
    """
    Multi-strategy KNN imputer that combines different approaches
    """
    
    def __init__(self):
        """Initialize multi-strategy imputer"""
        self.strategies = {
            'conservative': AdvancedKNNImputer(n_neighbors=3, weights='distance', metric='euclidean'),
            'balanced': AdvancedKNNImputer(n_neighbors=5, weights='distance', metric='euclidean'),
            'aggressive': AdvancedKNNImputer(n_neighbors=7, weights='uniform', metric='manhattan')
        }
        
        self.ensemble_weights = {
            'conservative': 0.4,
            'balanced': 0.4,
            'aggressive': 0.2
        }
    
    def fit_transform(self, data: pd.DataFrame) -> ImputationResult:
        """Fit and transform using ensemble of strategies"""
        try:
            results = {}
            
            # Apply each strategy
            for strategy_name, imputer in self.strategies.items():
                logger.info(f"Applying {strategy_name} strategy")
                results[strategy_name] = imputer.fit_transform(data)
            
            # Combine results using weighted ensemble
            final_imputed = self._ensemble_combine(data, results)
            
            # Calculate ensemble quality metrics
            missing_mask = data.isnull()
            quality_metrics = self._calculate_ensemble_quality(data, final_imputed, missing_mask, results)
            confidence_scores = self._calculate_ensemble_confidence(results)
            
            return ImputationResult(
                imputed_data=final_imputed,
                missing_mask=missing_mask,
                imputation_quality=quality_metrics,
                method_used='ensemble_knn',
                confidence_scores=confidence_scores
            )
            
        except Exception as e:
            logger.error(f"Error in multi-strategy imputation: {e}")
            # Fallback to balanced strategy
            return self.strategies['balanced'].fit_transform(data)
    
    def _ensemble_combine(self, original_data: pd.DataFrame, results: Dict[str, ImputationResult]) -> pd.DataFrame:
        """Combine multiple imputation results using weighted ensemble"""
        try:
            combined_data = original_data.copy()
            missing_mask = original_data.isnull()
            
            for col in original_data.columns:
                if missing_mask[col].any():
                    # Get imputed values from each strategy
                    imputed_values = {}
                    for strategy_name, result in results.items():
                        if col in result.imputed_data.columns:
                            imputed_values[strategy_name] = result.imputed_data.loc[missing_mask[col], col]
                    
                    if imputed_values:
                        # For numerical columns, use weighted average
                        if original_data[col].dtype in ['int64', 'float64']:
                            weighted_values = pd.Series(index=missing_mask[missing_mask[col]].index, dtype=float)
                            
                            for idx in weighted_values.index:
                                values = []
                                weights = []
                                
                                for strategy_name, values_series in imputed_values.items():
                                    if idx in values_series.index:
                                        values.append(values_series[idx])
                                        weights.append(self.ensemble_weights[strategy_name])
                                
                                if values:
                                    weighted_values[idx] = np.average(values, weights=weights)
                            
                            combined_data.loc[missing_mask[col], col] = weighted_values
                        
                        else:
                            # For categorical columns, use majority voting
                            for idx in missing_mask[missing_mask[col]].index:
                                votes = {}
                                for strategy_name, values_series in imputed_values.items():
                                    if idx in values_series.index:
                                        value = values_series[idx]
                                        weight = self.ensemble_weights[strategy_name]
                                        votes[value] = votes.get(value, 0) + weight
                                
                                if votes:
                                    best_value = max(votes.items(), key=lambda x: x[1])[0]
                                    combined_data.loc[idx, col] = best_value
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Error in ensemble combination: {e}")
            # Return the balanced strategy result as fallback
            return results['balanced'].imputed_data if 'balanced' in results else original_data
    
    def _calculate_ensemble_quality(self, 
                                  original_data: pd.DataFrame,
                                  final_imputed: pd.DataFrame,
                                  missing_mask: pd.DataFrame,
                                  results: Dict[str, ImputationResult]) -> Dict[str, float]:
        """Calculate quality metrics for ensemble imputation"""
        try:
            # Start with basic quality metrics
            total_missing = missing_mask.sum().sum()
            total_cells = missing_mask.size
            
            quality_metrics = {
                'missing_percentage': (total_missing / total_cells) * 100,
                'imputation_coverage': 100.0,  # Ensemble should cover all missing values
                'strategy_agreement': self._calculate_strategy_agreement(results),
                'ensemble_consistency': self._calculate_ensemble_consistency(original_data, final_imputed, missing_mask)
            }
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error calculating ensemble quality: {e}")
            return {}
    
    def _calculate_strategy_agreement(self, results: Dict[str, ImputationResult]) -> float:
        """Calculate agreement between different strategies"""
        try:
            if len(results) < 2:
                return 1.0
            
            agreements = []
            strategy_names = list(results.keys())
            
            for i in range(len(strategy_names)):
                for j in range(i + 1, len(strategy_names)):
                    strategy1 = results[strategy_names[i]]
                    strategy2 = results[strategy_names[j]]
                    
                    # Compare imputed values
                    agreement = self._compare_imputation_results(strategy1, strategy2)
                    agreements.append(agreement)
            
            return np.mean(agreements) if agreements else 1.0
            
        except Exception as e:
            logger.error(f"Error calculating strategy agreement: {e}")
            return 0.5
    
    def _compare_imputation_results(self, result1: ImputationResult, result2: ImputationResult) -> float:
        """Compare two imputation results for agreement"""
        try:
            agreements = []
            
            common_columns = set(result1.imputed_data.columns) & set(result2.imputed_data.columns)
            
            for col in common_columns:
                missing1 = result1.missing_mask[col]
                missing2 = result2.missing_mask[col]
                common_missing = missing1 & missing2
                
                if common_missing.any():
                    values1 = result1.imputed_data.loc[common_missing, col]
                    values2 = result2.imputed_data.loc[common_missing, col]
                    
                    if result1.imputed_data[col].dtype in ['int64', 'float64']:
                        # For numerical values, calculate relative agreement
                        rel_diff = np.abs(values1 - values2) / (np.abs(values1) + np.abs(values2) + 1e-8)
                        agreement = 1 - np.mean(rel_diff)
                    else:
                        # For categorical values, calculate exact agreement
                        agreement = (values1 == values2).mean()
                    
                    agreements.append(max(0, agreement))
            
            return np.mean(agreements) if agreements else 1.0
            
        except Exception as e:
            logger.error(f"Error comparing imputation results: {e}")
            return 0.5
    
    def _calculate_ensemble_consistency(self, 
                                      original_data: pd.DataFrame,
                                      final_imputed: pd.DataFrame,
                                      missing_mask: pd.DataFrame) -> float:
        """Calculate consistency of ensemble imputation"""
        try:
            consistencies = []
            
            for col in original_data.select_dtypes(include=[np.number]).columns:
                if missing_mask[col].any():
                    original_values = original_data[col].dropna()
                    imputed_values = final_imputed.loc[missing_mask[col], col]
                    
                    if len(original_values) > 1 and len(imputed_values) > 0:
                        # Compare statistical properties
                        orig_mean = original_values.mean()
                        orig_std = original_values.std()
                        imp_mean = imputed_values.mean()
                        imp_std = imputed_values.std()
                        
                        # Calculate consistency scores
                        mean_consistency = 1 - abs(orig_mean - imp_mean) / (orig_std + 1e-8)
                        std_consistency = min(orig_std, imp_std) / (max(orig_std, imp_std) + 1e-8)
                        
                        overall_consistency = (mean_consistency + std_consistency) / 2
                        consistencies.append(max(0, overall_consistency))
            
            return np.mean(consistencies) if consistencies else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating ensemble consistency: {e}")
            return 0.5
    
    def _calculate_ensemble_confidence(self, results: Dict[str, ImputationResult]) -> Dict[str, float]:
        """Calculate confidence scores for ensemble imputation"""
        try:
            ensemble_confidence = {}
            
            # Get all columns that were imputed
            all_columns = set()
            for result in results.values():
                all_columns.update(result.confidence_scores.keys())
            
            for col in all_columns:
                confidences = []
                for result in results.values():
                    if col in result.confidence_scores:
                        confidences.append(result.confidence_scores[col])
                
                if confidences:
                    # Use weighted average of confidences
                    weighted_conf = 0
                    total_weight = 0
                    
                    for i, conf in enumerate(confidences):
                        strategy_name = list(results.keys())[i]
                        weight = self.ensemble_weights.get(strategy_name, 1.0)
                        weighted_conf += conf * weight
                        total_weight += weight
                    
                    ensemble_confidence[col] = weighted_conf / total_weight if total_weight > 0 else 0.5
            
            return ensemble_confidence
            
        except Exception as e:
            logger.error(f"Error calculating ensemble confidence: {e}")
            return {}


# Example usage and testing
if __name__ == "__main__":
    # Create sample financial data with missing values
    np.random.seed(42)
    n_companies = 100
    
    sample_data = pd.DataFrame({
        'company_id': [f'COMP_{i:03d}' for i in range(n_companies)],
        'industry': np.random.choice(['technology', 'healthcare', 'industrial', 'financial'], n_companies),
        'revenue': np.random.lognormal(15, 1, n_companies),
        'net_income': np.random.normal(0.1, 0.05, n_companies) * np.random.lognormal(15, 1, n_companies),
        'total_assets': np.random.lognormal(16, 1, n_companies),
        'current_assets': np.random.uniform(0.3, 0.7, n_companies) * np.random.lognormal(16, 1, n_companies),
        'current_liabilities': np.random.uniform(0.1, 0.4, n_companies) * np.random.lognormal(16, 1, n_companies),
        'total_debt': np.random.uniform(0.2, 0.8, n_companies) * np.random.lognormal(16, 1, n_companies),
        'shareholders_equity': np.random.uniform(0.2, 0.6, n_companies) * np.random.lognormal(16, 1, n_companies)
    })
    
    # Calculate some ratios
    sample_data['current_ratio'] = sample_data['current_assets'] / sample_data['current_liabilities']
    sample_data['debt_to_equity'] = sample_data['total_debt'] / sample_data['shareholders_equity']
    sample_data['roa'] = sample_data['net_income'] / sample_data['total_assets']
    
    # Introduce missing values randomly
    missing_cols = ['revenue', 'net_income', 'current_ratio', 'debt_to_equity', 'roa']
    for col in missing_cols:
        missing_indices = np.random.choice(sample_data.index, size=int(0.15 * len(sample_data)), replace=False)
        sample_data.loc[missing_indices, col] = np.nan
    
    print("=== Testing KNN Imputer ===")
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Missing values per column:")
    for col in sample_data.columns:
        missing_count = sample_data[col].isnull().sum()
        if missing_count > 0:
            print(f"  {col}: {missing_count} ({missing_count/len(sample_data)*100:.1f}%)")
    
    # Test Advanced KNN Imputer
    print("\n--- Testing Advanced KNN Imputer ---")
    advanced_imputer = AdvancedKNNImputer(
        n_neighbors=5,
        weights='distance',
        metric='euclidean',
        industry_aware=True,
        financial_context=True
    )
    
    result = advanced_imputer.fit_transform(sample_data, industry_column='industry')
    
    print(f"Imputation method: {result.method_used}")
    print(f"Missing percentage: {result.imputation_quality.get('missing_percentage', 0):.2f}%")
    print(f"Average consistency: {result.imputation_quality.get('average_consistency', 0):.3f}")
    
    # Print confidence scores
    print("\nConfidence scores:")
    for col, score in result.confidence_scores.items():
        print(f"  {col}: {score:.3f}")
    
    # Test Multi-Strategy Imputer
    print("\n--- Testing Multi-Strategy KNN Imputer ---")
    multi_imputer = MultiStrategyKNNImputer()
    ensemble_result = multi_imputer.fit_transform(sample_data)
    
    print(f"Ensemble method: {ensemble_result.method_used}")
    print(f"Strategy agreement: {ensemble_result.imputation_quality.get('strategy_agreement', 0):.3f}")
    print(f"Ensemble consistency: {ensemble_result.imputation_quality.get('ensemble_consistency', 0):.3f}")
    
    # Cross-validation test
    print("\n--- Testing Cross-Validation ---")
    cv_results = advanced_imputer.cross_validate_imputation(sample_data, test_fraction=0.1, n_trials=3)
    print(f"Mean accuracy: {cv_results.get('mean_accuracy', 0):.3f}")
    print(f"Std accuracy: {cv_results.get('std_accuracy', 0):.3f}")
    
    # Generate and print report
    print("\n--- Imputation Report ---")
    report = advanced_imputer.get_imputation_report(result)
    print(report)
    
    print("\n=== KNN Imputer testing completed! ===")