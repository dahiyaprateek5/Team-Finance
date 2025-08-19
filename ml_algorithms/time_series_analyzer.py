"""
Time Series Analyzer Module
Advanced time series analysis for financial data with forecasting capabilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Statistical and ML imports
from scipy import stats
from scipy.fft import fft, fftfreq
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrendType(Enum):
    """Types of trends in time series"""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"
    CYCLICAL = "cyclical"

class SeasonalityType(Enum):
    """Types of seasonality patterns"""
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    MONTHLY = "monthly"
    NONE = "none"

@dataclass
class TimeSeriesComponents:
    """Components of time series decomposition"""
    trend: pd.Series
    seasonal: pd.Series
    residual: pd.Series
    observed: pd.Series
    trend_type: TrendType
    seasonality_type: SeasonalityType
    strength_of_trend: float
    strength_of_seasonality: float

@dataclass
class ForecastResult:
    """Time series forecast results"""
    forecast: pd.Series
    confidence_interval_lower: pd.Series
    confidence_interval_upper: pd.Series
    model_type: str
    accuracy_metrics: Dict[str, float]
    forecast_horizon: int
    model_parameters: Dict[str, Any]

@dataclass
class AnomalyDetectionResult:
    """Anomaly detection results"""
    anomalies: pd.Series
    anomaly_scores: pd.Series
    threshold: float
    method_used: str
    anomaly_periods: List[Tuple[pd.Timestamp, pd.Timestamp]]

class FinancialTimeSeriesAnalyzer:
    """
    Comprehensive time series analyzer for financial data
    """
    
    def __init__(self, 
                 frequency: str = 'quarterly',
                 confidence_level: float = 0.95):
        """
        Initialize Time Series Analyzer
        
        Args:
            frequency (str): Data frequency ('quarterly', 'annual', 'monthly')
            confidence_level (float): Confidence level for forecasts
        """
        self.frequency = frequency
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
        # Financial time series specific parameters
        self.FINANCIAL_METRICS = [
            'revenue', 'net_income', 'total_assets', 'current_ratio',
            'debt_to_equity', 'roa', 'roe', 'profit_margin'
        ]
        
        # Seasonality patterns for financial data
        self.SEASONALITY_PATTERNS = {
            'quarterly': 4,
            'monthly': 12,
            'annual': 1
        }
        
        # Model cache
        self.fitted_models = {}
        
        # Analysis results cache
        self.analysis_cache = {}
    
    def analyze_time_series(self, 
                           data: pd.DataFrame,
                           date_column: str,
                           value_column: str,
                           company_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Comprehensive time series analysis
        
        Args:
            data (pd.DataFrame): Time series data
            date_column (str): Column name for dates
            value_column (str): Column name for values
            company_id (str): Optional company identifier
            
        Returns:
            Dict: Complete analysis results
        """
        try:
            logger.info(f"Starting time series analysis for {value_column}")
            
            # Prepare time series
            ts_data = self._prepare_time_series(data, date_column, value_column)
            if ts_data.empty:
                return {'error': 'Insufficient data for time series analysis'}
            
            # Basic statistics
            basic_stats = self._calculate_basic_statistics(ts_data)
            
            # Stationarity tests
            stationarity = self._test_stationarity(ts_data)
            
            # Decomposition
            decomposition = self._decompose_time_series(ts_data)
            
            # Trend analysis
            trend_analysis = self._analyze_trend(ts_data, decomposition)
            
            # Seasonality analysis
            seasonality_analysis = self._analyze_seasonality(ts_data, decomposition)
            
            # Volatility analysis
            volatility_analysis = self._analyze_volatility(ts_data)
            
            # Anomaly detection
            anomalies = self._detect_anomalies(ts_data)
            
            # Correlation analysis (if multiple series)
            correlation_analysis = self._analyze_correlations(data, date_column, value_column)
            
            # Forecasting
            forecast_results = self._generate_forecasts(ts_data)
            
            # Financial insights
            financial_insights = self._generate_financial_insights(
                ts_data, value_column, trend_analysis, volatility_analysis
            )
            
            analysis_result = {
                'company_id': company_id,
                'metric': value_column,
                'data_period': {
                    'start': ts_data.index.min(),
                    'end': ts_data.index.max(),
                    'periods': len(ts_data)
                },
                'basic_statistics': basic_stats,
                'stationarity': stationarity,
                'decomposition': decomposition,
                'trend_analysis': trend_analysis,
                'seasonality_analysis': seasonality_analysis,
                'volatility_analysis': volatility_analysis,
                'anomalies': anomalies,
                'correlation_analysis': correlation_analysis,
                'forecasts': forecast_results,
                'financial_insights': financial_insights,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Cache results
            cache_key = f"{company_id}_{value_column}" if company_id else value_column
            self.analysis_cache[cache_key] = analysis_result
            
            logger.info(f"Time series analysis completed for {value_column}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in time series analysis: {e}")
            return {'error': str(e)}
    
    def _prepare_time_series(self, 
                           data: pd.DataFrame, 
                           date_column: str, 
                           value_column: str) -> pd.Series:
        """Prepare and clean time series data"""
        try:
            # Copy data
            df = data.copy()
            
            # Convert date column
            df[date_column] = pd.to_datetime(df[date_column])
            
            # Set date as index
            df.set_index(date_column, inplace=True)
            
            # Sort by date
            df.sort_index(inplace=True)
            
            # Get the value series
            ts_series = df[value_column].copy()
            
            # Remove missing values
            ts_series = ts_series.dropna()
            
            # Convert to numeric
            ts_series = pd.to_numeric(ts_series, errors='coerce')
            ts_series = ts_series.dropna()
            
            # Remove duplicates (keep last)
            ts_series = ts_series[~ts_series.index.duplicated(keep='last')]
            
            # Check for minimum data points
            if len(ts_series) < 3:
                logger.warning(f"Insufficient data points: {len(ts_series)}")
                return pd.Series()
            
            return ts_series
            
        except Exception as e:
            logger.error(f"Error preparing time series: {e}")
            return pd.Series()
    
    def _calculate_basic_statistics(self, ts_data: pd.Series) -> Dict[str, float]:
        """Calculate basic statistical measures"""
        try:
            stats_dict = {
                'count': len(ts_data),
                'mean': float(ts_data.mean()),
                'median': float(ts_data.median()),
                'std': float(ts_data.std()),
                'min': float(ts_data.min()),
                'max': float(ts_data.max()),
                'skewness': float(ts_data.skew()),
                'kurtosis': float(ts_data.kurtosis()),
                'coefficient_of_variation': float(ts_data.std() / ts_data.mean()) if ts_data.mean() != 0 else 0
            }
            
            # Percentiles
            for p in [25, 75, 95]:
                stats_dict[f'percentile_{p}'] = float(ts_data.quantile(p/100))
            
            # Growth statistics
            if len(ts_data) > 1:
                pct_change = ts_data.pct_change().dropna()
                stats_dict['mean_growth_rate'] = float(pct_change.mean())
                stats_dict['volatility'] = float(pct_change.std())
                
                # Compound annual growth rate (CAGR)
                if len(ts_data) > 1:
                    years = (ts_data.index[-1] - ts_data.index[0]).days / 365.25
                    if years > 0 and ts_data.iloc[0] > 0:
                        cagr = (ts_data.iloc[-1] / ts_data.iloc[0]) ** (1/years) - 1
                        stats_dict['cagr'] = float(cagr)
            
            return stats_dict
            
        except Exception as e:
            logger.error(f"Error calculating basic statistics: {e}")
            return {}
    
    def _test_stationarity(self, ts_data: pd.Series) -> Dict[str, Any]:
        """Test time series stationarity"""
        try:
            stationarity_results = {}
            
            # Augmented Dickey-Fuller test
            adf_result = adfuller(ts_data.dropna())
            stationarity_results['adf_test'] = {
                'statistic': float(adf_result[0]),
                'p_value': float(adf_result[1]),
                'critical_values': {k: float(v) for k, v in adf_result[4].items()},
                'is_stationary': adf_result[1] < 0.05
            }
            
            # KPSS test
            try:
                kpss_result = kpss(ts_data.dropna())
                stationarity_results['kpss_test'] = {
                    'statistic': float(kpss_result[0]),
                    'p_value': float(kpss_result[1]),
                    'critical_values': {k: float(v) for k, v in kpss_result[3].items()},
                    'is_stationary': kpss_result[1] > 0.05
                }
            except Exception as e:
                logger.warning(f"KPSS test failed: {e}")
                stationarity_results['kpss_test'] = {'error': str(e)}
            
            # Overall assessment
            adf_stationary = stationarity_results['adf_test']['is_stationary']
            kpss_stationary = stationarity_results.get('kpss_test', {}).get('is_stationary', True)
            
            stationarity_results['overall_assessment'] = {
                'is_stationary': adf_stationary and kpss_stationary,
                'needs_differencing': not (adf_stationary and kpss_stationary),
                'recommendation': 'stationary' if (adf_stationary and kpss_stationary) else 'apply_differencing'
            }
            
            return stationarity_results
            
        except Exception as e:
            logger.error(f"Error testing stationarity: {e}")
            return {}
    
    def _decompose_time_series(self, ts_data: pd.Series) -> Dict[str, Any]:
        """Decompose time series into trend, seasonal, and residual components"""
        try:
            if len(ts_data) < 8:  # Need minimum periods for decomposition
                return {'error': 'Insufficient data for decomposition'}
            
            # Determine period for decomposition
            period = self._determine_seasonality_period(ts_data)
            
            if period is None or period >= len(ts_data):
                # No seasonality or insufficient data
                return {'error': 'Cannot determine seasonality or insufficient data'}
            
            # Perform decomposition
            decomposition = seasonal_decompose(
                ts_data, 
                model='additive',  # Can also use 'multiplicative'
                period=period,
                extrapolate_trend='freq'
            )
            
            # Calculate strength of components
            trend_strength = self._calculate_trend_strength(decomposition)
            seasonal_strength = self._calculate_seasonal_strength(decomposition)
            
            # Determine trend type
            trend_type = self._classify_trend(decomposition.trend)
            
            # Determine seasonality type
            seasonality_type = self._classify_seasonality(period)
            
            result = {
                'period': period,
                'trend_strength': trend_strength,
                'seasonal_strength': seasonal_strength,
                'trend_type': trend_type.value,
                'seasonality_type': seasonality_type.value,
                'components': {
                    'trend': decomposition.trend.dropna().to_dict(),
                    'seasonal': decomposition.seasonal.to_dict(),
                    'residual': decomposition.resid.dropna().to_dict(),
                    'observed': decomposition.observed.to_dict()
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in time series decomposition: {e}")
            return {'error': str(e)}
    
    def _determine_seasonality_period(self, ts_data: pd.Series) -> Optional[int]:
        """Determine the seasonality period using autocorrelation"""
        try:
            if len(ts_data) < 8:
                return None
            
            # Try different periods based on frequency
            if self.frequency == 'quarterly':
                candidate_periods = [4]
            elif self.frequency == 'monthly':
                candidate_periods = [12, 6, 4, 3]
            elif self.frequency == 'annual':
                return None  # No seasonality for annual data
            else:
                candidate_periods = [4, 12]
            
            # Test each candidate period
            best_period = None
            best_score = 0
            
            for period in candidate_periods:
                if period < len(ts_data):
                    try:
                        # Calculate autocorrelation at lag = period
                        autocorr = ts_data.autocorr(lag=period)
                        if not np.isnan(autocorr) and abs(autocorr) > best_score:
                            best_score = abs(autocorr)
                            best_period = period
                    except:
                        continue
            
            return best_period if best_score > 0.3 else None  # Threshold for significant seasonality
            
        except Exception as e:
            logger.error(f"Error determining seasonality period: {e}")
            return None
    
    def _calculate_trend_strength(self, decomposition) -> float:
        """Calculate the strength of the trend component"""
        try:
            trend_var = np.var(decomposition.trend.dropna())
            residual_var = np.var(decomposition.resid.dropna())
            
            if trend_var + residual_var == 0:
                return 0.0
            
            trend_strength = trend_var / (trend_var + residual_var)
            return min(1.0, max(0.0, trend_strength))
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return 0.0
    
    def _calculate_seasonal_strength(self, decomposition) -> float:
        """Calculate the strength of the seasonal component"""
        try:
            seasonal_var = np.var(decomposition.seasonal.dropna())
            residual_var = np.var(decomposition.resid.dropna())
            
            if seasonal_var + residual_var == 0:
                return 0.0
            
            seasonal_strength = seasonal_var / (seasonal_var + residual_var)
            return min(1.0, max(0.0, seasonal_strength))
            
        except Exception as e:
            logger.error(f"Error calculating seasonal strength: {e}")
            return 0.0
    
    def _classify_trend(self, trend_series: pd.Series) -> TrendType:
        """Classify the type of trend"""
        try:
            trend_clean = trend_series.dropna()
            if len(trend_clean) < 2:
                return TrendType.STABLE
            
            # Calculate overall slope
            x = np.arange(len(trend_clean))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, trend_clean.values)
            
            # Calculate coefficient of variation
            cv = trend_clean.std() / trend_clean.mean() if trend_clean.mean() != 0 else 0
            
            # Classify based on slope and significance
            if abs(r_value) < 0.3:  # Low correlation with linear trend
                if cv > 0.2:
                    return TrendType.VOLATILE
                else:
                    return TrendType.STABLE
            elif slope > 0 and p_value < 0.05:
                return TrendType.INCREASING
            elif slope < 0 and p_value < 0.05:
                return TrendType.DECREASING
            else:
                return TrendType.CYCLICAL if cv > 0.15 else TrendType.STABLE
            
        except Exception as e:
            logger.error(f"Error classifying trend: {e}")
            return TrendType.STABLE
    
    def _classify_seasonality(self, period: Optional[int]) -> SeasonalityType:
        """Classify the type of seasonality"""
        if period is None:
            return SeasonalityType.NONE
        elif period == 4:
            return SeasonalityType.QUARTERLY
        elif period == 12:
            return SeasonalityType.MONTHLY
        elif period == 1:
            return SeasonalityType.ANNUAL
        else:
            return SeasonalityType.NONE
    
    def _analyze_trend(self, ts_data: pd.Series, decomposition: Dict) -> Dict[str, Any]:
        """Detailed trend analysis"""
        try:
            trend_analysis = {}
            
            # Linear trend
            x = np.arange(len(ts_data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, ts_data.values)
            
            trend_analysis['linear_trend'] = {
                'slope': float(slope),
                'intercept': float(intercept),
                'r_squared': float(r_value ** 2),
                'p_value': float(p_value),
                'is_significant': p_value < 0.05
            }
            
            # Growth rate analysis
            pct_change = ts_data.pct_change().dropna()
            if len(pct_change) > 0:
                trend_analysis['growth_analysis'] = {
                    'mean_growth_rate': float(pct_change.mean()),
                    'median_growth_rate': float(pct_change.median()),
                    'growth_volatility': float(pct_change.std()),
                    'positive_periods': int((pct_change > 0).sum()),
                    'negative_periods': int((pct_change < 0).sum()),
                    'growth_consistency': float((pct_change > 0).mean())
                }
            
            # Turning points
            turning_points = self._identify_turning_points(ts_data)
            trend_analysis['turning_points'] = turning_points
            
            # Trend classification from decomposition
            if 'trend_type' in decomposition:
                trend_analysis['trend_classification'] = decomposition['trend_type']
                trend_analysis['trend_strength'] = decomposition.get('trend_strength', 0)
            
            return trend_analysis
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return {}
    
    def _identify_turning_points(self, ts_data: pd.Series) -> Dict[str, Any]:
        """Identify significant turning points in the time series"""
        try:
            # Calculate first and second derivatives
            first_diff = ts_data.diff()
            second_diff = first_diff.diff()
            
            # Find peaks and troughs
            peaks = []
            troughs = []
            
            for i in range(1, len(ts_data) - 1):
                if (first_diff.iloc[i-1] > 0 and first_diff.iloc[i+1] < 0):  # Peak
                    peaks.append({
                        'date': ts_data.index[i],
                        'value': ts_data.iloc[i],
                        'type': 'peak'
                    })
                elif (first_diff.iloc[i-1] < 0 and first_diff.iloc[i+1] > 0):  # Trough
                    troughs.append({
                        'date': ts_data.index[i],
                        'value': ts_data.iloc[i],
                        'type': 'trough'
                    })
            
            return {
                'peaks': peaks,
                'troughs': troughs,
                'total_turning_points': len(peaks) + len(troughs)
            }
            
        except Exception as e:
            logger.error(f"Error identifying turning points: {e}")
            return {}
    
    def _analyze_seasonality(self, ts_data: pd.Series, decomposition: Dict) -> Dict[str, Any]:
        """Detailed seasonality analysis"""
        try:
            seasonality_analysis = {}
            
            # From decomposition
            if 'seasonality_type' in decomposition:
                seasonality_analysis['seasonality_type'] = decomposition['seasonality_type']
                seasonality_analysis['seasonal_strength'] = decomposition.get('seasonal_strength', 0)
                seasonality_analysis['period'] = decomposition.get('period')
            
            # Seasonal patterns
            if decomposition.get('period'):
                period = decomposition['period']
                seasonal_patterns = self._extract_seasonal_patterns(ts_data, period)
                seasonality_analysis.update(seasonal_patterns)
            
            # Fourier analysis for frequency detection
            fourier_analysis = self._fourier_analysis(ts_data)
            seasonality_analysis['fourier_analysis'] = fourier_analysis
            
            return seasonality_analysis
            
        except Exception as e:
            logger.error(f"Error in seasonality analysis: {e}")
            return {}
    
    def _extract_seasonal_patterns(self, ts_data: pd.Series, period: int) -> Dict[str, Any]:
        """Extract detailed seasonal patterns"""
        try:
            # Group by seasonal periods
            seasonal_data = []
            for i in range(len(ts_data)):
                seasonal_data.append({
                    'period_position': i % period,
                    'value': ts_data.iloc[i],
                    'date': ts_data.index[i]
                })
            
            seasonal_df = pd.DataFrame(seasonal_data)
            
            # Calculate seasonal indices
            seasonal_indices = seasonal_df.groupby('period_position')['value'].agg(['mean', 'std', 'count'])
            
            # Identify seasonal peaks and troughs
            seasonal_means = seasonal_indices['mean']
            peak_season = seasonal_means.idxmax()
            trough_season = seasonal_means.idxmin()
            
            patterns = {
                'seasonal_indices': seasonal_indices.to_dict(),
                'peak_season': int(peak_season),
                'trough_season': int(trough_season),
                'seasonal_amplitude': float(seasonal_means.max() - seasonal_means.min()),
                'seasonal_cv': float(seasonal_means.std() / seasonal_means.mean()) if seasonal_means.mean() != 0 else 0
            }
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error extracting seasonal patterns: {e}")
            return {}
    
    def _fourier_analysis(self, ts_data: pd.Series) -> Dict[str, Any]:
        """Perform Fourier analysis to detect frequency components"""
        try:
            # Remove trend (detrend)
            detrended = ts_data - ts_data.rolling(window=min(4, len(ts_data)//2), center=True).mean()
            detrended = detrended.dropna()
            
            if len(detrended) < 4:
                return {}
            
            # Perform FFT
            fft_values = fft(detrended.values)
            frequencies = fftfreq(len(detrended))
            
            # Get magnitude spectrum
            magnitude = np.abs(fft_values)
            
            # Find dominant frequencies
            # Exclude DC component (frequency = 0)
            non_dc_indices = frequencies != 0
            dominant_freq_idx = np.argmax(magnitude[non_dc_indices])
            dominant_frequency = frequencies[non_dc_indices][dominant_freq_idx]
            
            # Convert to period
            if dominant_frequency != 0:
                dominant_period = 1 / abs(dominant_frequency)
            else:
                dominant_period = None
            
            fourier_result = {
                'dominant_frequency': float(dominant_frequency),
                'dominant_period': float(dominant_period) if dominant_period else None,
                'spectral_energy': float(np.sum(magnitude[non_dc_indices] ** 2)),
                'frequency_bands': self._analyze_frequency_bands(frequencies, magnitude)
            }
            
            return fourier_result
            
        except Exception as e:
            logger.error(f"Error in Fourier analysis: {e}")
            return {}
    
    def _analyze_frequency_bands(self, frequencies: np.ndarray, magnitude: np.ndarray) -> Dict[str, float]:
        """Analyze energy in different frequency bands"""
        try:
            # Define frequency bands for financial data
            bands = {
                'low_frequency': (0, 0.1),      # Long-term trends
                'medium_frequency': (0.1, 0.3), # Business cycles
                'high_frequency': (0.3, 0.5)    # Short-term fluctuations
            }
            
            band_energy = {}
            total_energy = np.sum(magnitude ** 2)
            
            for band_name, (low_freq, high_freq) in bands.items():
                band_mask = (np.abs(frequencies) >= low_freq) & (np.abs(frequencies) < high_freq)
                band_energy[band_name] = float(np.sum(magnitude[band_mask] ** 2) / total_energy)
            
            return band_energy
            
        except Exception as e:
            logger.error(f"Error analyzing frequency bands: {e}")
            return {}
    
    def _analyze_volatility(self, ts_data: pd.Series) -> Dict[str, Any]:
        """Comprehensive volatility analysis"""
        try:
            volatility_analysis = {}
            
            # Returns calculation
            returns = ts_data.pct_change().dropna()
            
            if len(returns) == 0:
                return {'error': 'Insufficient data for volatility analysis'}
            
            # Basic volatility measures
            volatility_analysis['basic_measures'] = {
                'volatility': float(returns.std()),
                'annualized_volatility': float(returns.std() * np.sqrt(4)),  # Assuming quarterly data
                'mean_return': float(returns.mean()),
                'return_skewness': float(returns.skew()),
                'return_kurtosis': float(returns.kurtosis())
            }
            
            # Rolling volatility
            if len(returns) >= 4:
                rolling_vol = returns.rolling(window=4).std()
                volatility_analysis['rolling_volatility'] = {
                    'mean': float(rolling_vol.mean()),
                    'min': float(rolling_vol.min()),
                    'max': float(rolling_vol.max()),
                    'current': float(rolling_vol.iloc[-1]) if not rolling_vol.empty else 0
                }
            
            # Volatility clustering
            volatility_clustering = self._detect_volatility_clustering(returns)
            volatility_analysis['clustering'] = volatility_clustering
            
            # Extreme value analysis
            extreme_analysis = self._analyze_extreme_values(returns)
            volatility_analysis['extreme_values'] = extreme_analysis
            
            # Risk metrics
            risk_metrics = self._calculate_risk_metrics(returns)
            volatility_analysis['risk_metrics'] = risk_metrics
            
            return volatility_analysis
            
        except Exception as e:
            logger.error(f"Error in volatility analysis: {e}")
            return {}
    
    def _detect_volatility_clustering(self, returns: pd.Series) -> Dict[str, Any]:
        """Detect volatility clustering patterns"""
        try:
            # Calculate squared returns as proxy for volatility
            squared_returns = returns ** 2
            
            # Autocorrelation of squared returns
            autocorr_lags = min(10, len(squared_returns) // 4)
            autocorrelations = []
            
            for lag in range(1, autocorr_lags + 1):
                try:
                    autocorr = squared_returns.autocorr(lag=lag)
                    if not np.isnan(autocorr):
                        autocorrelations.append(autocorr)
                except:
                    continue
            
            clustering_result = {
                'average_autocorr': float(np.mean(autocorrelations)) if autocorrelations else 0,
                'max_autocorr': float(np.max(autocorrelations)) if autocorrelations else 0,
                'clustering_detected': (np.mean(autocorrelations) > 0.1) if autocorrelations else False
            }
            
            return clustering_result
            
        except Exception as e:
            logger.error(f"Error detecting volatility clustering: {e}")
            return {}
    
    def _analyze_extreme_values(self, returns: pd.Series) -> Dict[str, Any]:
        """Analyze extreme values and tail behavior"""
        try:
            # Calculate percentiles
            percentiles = [1, 5, 95, 99]
            extreme_values = {}
            
            for p in percentiles:
                extreme_values[f'percentile_{p}'] = float(returns.quantile(p/100))
            
            # Tail ratios
            extreme_values['tail_ratio'] = float(
                abs(extreme_values['percentile_1']) / abs(extreme_values['percentile_99'])
            ) if extreme_values['percentile_99'] != 0 else 0
            
            # Count of extreme values (beyond 2 standard deviations)
            std_dev = returns.std()
            mean_return = returns.mean()
            
            extreme_positive = (returns > mean_return + 2 * std_dev).sum()
            extreme_negative = (returns < mean_return - 2 * std_dev).sum()
            
            extreme_values['extreme_events'] = {
                'positive_extremes': int(extreme_positive),
                'negative_extremes': int(extreme_negative),
                'total_extremes': int(extreme_positive + extreme_negative),
                'extreme_frequency': float((extreme_positive + extreme_negative) / len(returns))
            }
            
            return extreme_values
            
        except Exception as e:
            logger.error(f"Error analyzing extreme values: {e}")
            return {}
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate various risk metrics"""
        try:
            risk_metrics = {}
            
            # Value at Risk (VaR)
            confidence_levels = [0.95, 0.99]
            for conf in confidence_levels:
                var_level = (1 - conf) * 100
                risk_metrics[f'var_{int(conf*100)}'] = float(returns.quantile(1 - conf))
            
            # Expected Shortfall (Conditional VaR)
            for conf in confidence_levels:
                var_threshold = returns.quantile(1 - conf)
                tail_returns = returns[returns <= var_threshold]
                if len(tail_returns) > 0:
                    risk_metrics[f'expected_shortfall_{int(conf*100)}'] = float(tail_returns.mean())
            
            # Sharpe ratio (assuming risk-free rate = 0)
            if returns.std() != 0:
                risk_metrics['sharpe_ratio'] = float(returns.mean() / returns.std())
            
            # Maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            risk_metrics['max_drawdown'] = float(drawdown.min())
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def _detect_anomalies(self, ts_data: pd.Series) -> Dict[str, Any]:
        """Detect anomalies in time series"""
        try:
            anomaly_results = {}
            
            # Statistical anomaly detection
            statistical_anomalies = self._statistical_anomaly_detection(ts_data)
            anomaly_results['statistical'] = statistical_anomalies
            
            # Seasonal anomaly detection
            seasonal_anomalies = self._seasonal_anomaly_detection(ts_data)
            anomaly_results['seasonal'] = seasonal_anomalies
            
            # Combine anomalies
            all_anomalies = set()
            if statistical_anomalies.get('anomaly_indices'):
                all_anomalies.update(statistical_anomalies['anomaly_indices'])
            if seasonal_anomalies.get('anomaly_indices'):
                all_anomalies.update(seasonal_anomalies['anomaly_indices'])
            
            anomaly_results['combined'] = {
                'total_anomalies': len(all_anomalies),
                'anomaly_dates': [ts_data.index[i].isoformat() for i in all_anomalies if i < len(ts_data)],
                'anomaly_rate': len(all_anomalies) / len(ts_data) if len(ts_data) > 0 else 0
            }
            
            return anomaly_results
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return {}
    
    def _statistical_anomaly_detection(self, ts_data: pd.Series) -> Dict[str, Any]:
        """Statistical anomaly detection using z-score and IQR methods"""
        try:
            # Z-score method
            z_scores = np.abs(stats.zscore(ts_data))
            z_threshold = 2.5
            z_anomalies = np.where(z_scores > z_threshold)[0]
            
            # IQR method
            Q1 = ts_data.quantile(0.25)
            Q3 = ts_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_anomalies = np.where((ts_data < lower_bound) | (ts_data > upper_bound))[0]
            
            # Combine methods
            combined_anomalies = list(set(z_anomalies) | set(iqr_anomalies))
            
            return {
                'z_score_anomalies': z_anomalies.tolist(),
                'iqr_anomalies': iqr_anomalies.tolist(),
                'anomaly_indices': combined_anomalies,
                'anomaly_count': len(combined_anomalies)
            }
            
        except Exception as e:
            logger.error(f"Error in statistical anomaly detection: {e}")
            return {}
    
    def _seasonal_anomaly_detection(self, ts_data: pd.Series) -> Dict[str, Any]:
        """Seasonal anomaly detection"""
        try:
            if len(ts_data) < 8:
                return {}
            
            # Try to decompose the series
            period = self._determine_seasonality_period(ts_data)
            if period is None:
                return {}
            
            try:
                decomposition = seasonal_decompose(ts_data, model='additive', period=period, extrapolate_trend='freq')
                
                # Anomalies in residual component
                residuals = decomposition.resid.dropna()
                if len(residuals) > 0:
                    residual_std = residuals.std()
                    residual_mean = residuals.mean()
                    
                    # Find outliers in residuals
                    threshold = 2 * residual_std
                    anomaly_mask = np.abs(residuals - residual_mean) > threshold
                    
                    anomaly_indices = residuals[anomaly_mask].index
                    original_indices = [ts_data.index.get_loc(idx) for idx in anomaly_indices if idx in ts_data.index]
                    
                    return {
                        'seasonal_anomalies': original_indices,
                        'anomaly_indices': original_indices,
                        'anomaly_count': len(original_indices)
                    }
                
            except Exception as e:
                logger.warning(f"Seasonal decomposition failed: {e}")
                return {}
            
            return {}
            
        except Exception as e:
            logger.error(f"Error in seasonal anomaly detection: {e}")
            return {}
    
    def _analyze_correlations(self, data: pd.DataFrame, date_column: str, value_column: str) -> Dict[str, Any]:
        """Analyze correlations with other time series"""
        try:
            correlation_analysis = {}
            
            # Identify other numerical columns
            numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if value_column in numerical_cols:
                numerical_cols.remove(value_column)
            
            if not numerical_cols:
                return {'note': 'No other numerical columns found for correlation analysis'}
            
            # Calculate correlations
            correlations = {}
            target_series = data.set_index(date_column)[value_column].dropna()
            
            for col in numerical_cols:
                other_series = data.set_index(date_column)[col].dropna()
                
                # Align series
                aligned_data = pd.concat([target_series, other_series], axis=1, join='inner').dropna()
                if len(aligned_data) > 3:  # Need minimum data points
                    correlation = aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1])
                    if not np.isnan(correlation):
                        correlations[col] = float(correlation)
            
            # Sort by absolute correlation
            sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            
            correlation_analysis['correlations'] = dict(sorted_correlations)
            correlation_analysis['strongest_positive'] = max(correlations.items(), key=lambda x: x[1]) if correlations else None
            correlation_analysis['strongest_negative'] = min(correlations.items(), key=lambda x: x[1]) if correlations else None
            correlation_analysis['highest_absolute'] = max(correlations.items(), key=lambda x: abs(x[1])) if correlations else None
            
            return correlation_analysis
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            return {}
    
    def _generate_forecasts(self, ts_data: pd.Series, forecast_horizon: int = 4) -> Dict[str, Any]:
        """Generate forecasts using multiple methods"""
        try:
            forecast_results = {}
            
            if len(ts_data) < 4:
                return {'error': 'Insufficient data for forecasting'}
            
            # ARIMA forecast
            arima_forecast = self._arima_forecast(ts_data, forecast_horizon)
            if arima_forecast:
                forecast_results['arima'] = arima_forecast
            
            # Exponential smoothing forecast
            exp_smoothing_forecast = self._exponential_smoothing_forecast(ts_data, forecast_horizon)
            if exp_smoothing_forecast:
                forecast_results['exponential_smoothing'] = exp_smoothing_forecast
            
            # Simple trend forecast
            trend_forecast = self._trend_forecast(ts_data, forecast_horizon)
            if trend_forecast:
                forecast_results['trend'] = trend_forecast
            
            # Ensemble forecast
            if len(forecast_results) > 1:
                ensemble_forecast = self._ensemble_forecast(forecast_results, forecast_horizon)
                forecast_results['ensemble'] = ensemble_forecast
            
            return forecast_results
            
        except Exception as e:
            logger.error(f"Error generating forecasts: {e}")
            return {}
    
    def _arima_forecast(self, ts_data: pd.Series, forecast_horizon: int) -> Optional[Dict[str, Any]]:
        """ARIMA model forecast"""
        try:
            # Auto-select ARIMA parameters
            best_aic = np.inf
            best_order = None
            best_model = None
            
            # Try different parameter combinations
            for p in range(3):
                for d in range(2):
                    for q in range(3):
                        try:
                            model = ARIMA(ts_data, order=(p, d, q))
                            fitted_model = model.fit()
                            
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_order = (p, d, q)
                                best_model = fitted_model
                        except:
                            continue
            
            if best_model is None:
                return None
            
            # Generate forecast
            forecast = best_model.forecast(steps=forecast_horizon)
            conf_int = best_model.get_forecast(steps=forecast_horizon).conf_int()
            
            # Create future dates
            last_date = ts_data.index[-1]
            if self.frequency == 'quarterly':
                future_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq='Q')[1:]
            elif self.frequency == 'annual':
                future_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq='Y')[1:]
            elif self.frequency == 'monthly':
                future_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq='M')[1:]
            else:
                future_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1)[1:]
            
            return {
                'forecast': dict(zip(future_dates, forecast)),
                'confidence_interval_lower': dict(zip(future_dates, conf_int.iloc[:, 0])),
                'confidence_interval_upper': dict(zip(future_dates, conf_int.iloc[:, 1])),
                'model_order': best_order,
                'aic': float(best_aic),
                'model_type': 'ARIMA'
            }
            
        except Exception as e:
            logger.error(f"Error in ARIMA forecast: {e}")
            return None
    
    def _exponential_smoothing_forecast(self, ts_data: pd.Series, forecast_horizon: int) -> Optional[Dict[str, Any]]:
        """Exponential smoothing forecast"""
        try:
            # Determine seasonality
            period = self._determine_seasonality_period(ts_data)
            
            # Fit exponential smoothing model
            if period and period < len(ts_data):
                model = ExponentialSmoothing(
                    ts_data, 
                    trend='add', 
                    seasonal='add', 
                    seasonal_periods=period
                )
            else:
                model = ExponentialSmoothing(ts_data, trend='add')
            
            fitted_model = model.fit()
            
            # Generate forecast
            forecast = fitted_model.forecast(forecast_horizon)
            
            # Simple confidence intervals (±2 std errors)
            forecast_std = ts_data.std()
            lower_bound = forecast - 1.96 * forecast_std
            upper_bound = forecast + 1.96 * forecast_std
            
            # Create future dates
            last_date = ts_data.index[-1]
            if self.frequency == 'quarterly':
                future_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq='Q')[1:]
            elif self.frequency == 'annual':
                future_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq='Y')[1:]
            elif self.frequency == 'monthly':
                future_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq='M')[1:]
            else:
                future_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1)[1:]
            
            return {
                'forecast': dict(zip(future_dates, forecast)),
                'confidence_interval_lower': dict(zip(future_dates, lower_bound)),
                'confidence_interval_upper': dict(zip(future_dates, upper_bound)),
                'model_type': 'Exponential Smoothing',
                'has_seasonality': period is not None
            }
            
        except Exception as e:
            logger.error(f"Error in exponential smoothing forecast: {e}")
            return None
    
    def _trend_forecast(self, ts_data: pd.Series, forecast_horizon: int) -> Dict[str, Any]:
        """Simple linear trend forecast"""
        try:
            # Fit linear trend
            x = np.arange(len(ts_data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, ts_data.values)
            
            # Generate forecast
            future_x = np.arange(len(ts_data), len(ts_data) + forecast_horizon)
            forecast_values = slope * future_x + intercept
            
            # Simple confidence intervals
            residuals = ts_data.values - (slope * x + intercept)
            residual_std = np.std(residuals)
            margin_of_error = 1.96 * residual_std
            
            lower_bound = forecast_values - margin_of_error
            upper_bound = forecast_values + margin_of_error
            
            # Create future dates
            last_date = ts_data.index[-1]
            if self.frequency == 'quarterly':
                future_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq='Q')[1:]
            elif self.frequency == 'annual':
                future_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq='Y')[1:]
            elif self.frequency == 'monthly':
                future_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq='M')[1:]
            else:
                future_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1)[1:]
            
            return {
                'forecast': dict(zip(future_dates, forecast_values)),
                'confidence_interval_lower': dict(zip(future_dates, lower_bound)),
                'confidence_interval_upper': dict(zip(future_dates, upper_bound)),
                'slope': float(slope),
                'r_squared': float(r_value ** 2),
                'model_type': 'Linear Trend'
            }
            
        except Exception as e:
            logger.error(f"Error in trend forecast: {e}")
            return {}
    
    def _ensemble_forecast(self, forecast_results: Dict, forecast_horizon: int) -> Dict[str, Any]:
        """Combine multiple forecasts into ensemble"""
        try:
            # Define weights for different methods
            weights = {
                'arima': 0.4,
                'exponential_smoothing': 0.4,
                'trend': 0.2
            }
            
            # Collect all forecasts
            all_forecasts = {}
            for method, result in forecast_results.items():
                if 'forecast' in result and method in weights:
                    all_forecasts[method] = result['forecast']
            
            if not all_forecasts:
                return {}
            
            # Get common dates
            common_dates = None
            for forecast in all_forecasts.values():
                dates = set(forecast.keys())
                if common_dates is None:
                    common_dates = dates
                else:
                    common_dates = common_dates.intersection(dates)
            
            # Calculate weighted ensemble
            ensemble_forecast = {}
            ensemble_lower = {}
            ensemble_upper = {}
            
            for date in common_dates:
                weighted_values = []
                method_weights = []
                
                for method, forecast in all_forecasts.items():
                    if date in forecast:
                        weighted_values.append(forecast[date])
                        method_weights.append(weights[method])
                
                if weighted_values:
                    ensemble_value = np.average(weighted_values, weights=method_weights)
                    ensemble_forecast[date] = float(ensemble_value)
                    
                    # Simple confidence intervals based on variance
                    forecast_variance = np.var(weighted_values)
                    margin_of_error = 1.96 * np.sqrt(forecast_variance)
                    ensemble_lower[date] = float(ensemble_value - margin_of_error)
                    ensemble_upper[date] = float(ensemble_value + margin_of_error)
            
            return {
                'forecast': ensemble_forecast,
                'confidence_interval_lower': ensemble_lower,
                'confidence_interval_upper': ensemble_upper,
                'model_type': 'Ensemble',
                'component_weights': weights,
                'methods_used': list(all_forecasts.keys())
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble forecast: {e}")
            return {}
    
    def _generate_financial_insights(self, 
                                   ts_data: pd.Series, 
                                   metric_name: str, 
                                   trend_analysis: Dict, 
                                   volatility_analysis: Dict) -> Dict[str, Any]:
        """Generate financial insights specific to the metric"""
        try:
            insights = {
                'metric_name': metric_name,
                'insights': [],
                'recommendations': [],
                'risk_assessment': {},
                'performance_summary': {}
            }
            
            # Performance summary
            latest_value = ts_data.iloc[-1]
            mean_value = ts_data.mean()
            
            insights['performance_summary'] = {
                'latest_value': float(latest_value),
                'historical_average': float(mean_value),
                'vs_average': float((latest_value - mean_value) / mean_value) if mean_value != 0 else 0,
                'percentile_rank': float(stats.percentileofscore(ts_data, latest_value))
            }
            
            # Trend insights
            if trend_analysis.get('linear_trend', {}).get('is_significant'):
                slope = trend_analysis['linear_trend']['slope']
                if slope > 0:
                    insights['insights'].append(f"{metric_name} shows a significant upward trend")
                    insights['recommendations'].append("Monitor for sustainability of positive trend")
                else:
                    insights['insights'].append(f"{metric_name} shows a significant downward trend")
                    insights['recommendations'].append("Investigate causes of declining trend")
            
            # Volatility insights
            volatility = volatility_analysis.get('basic_measures', {}).get('volatility', 0)
            if volatility > 0.3:  # High volatility threshold
                insights['insights'].append(f"{metric_name} exhibits high volatility")
                insights['recommendations'].append("Consider risk management strategies")
            elif volatility < 0.1:  # Low volatility
                insights['insights'].append(f"{metric_name} shows stable performance")
            
            # Risk assessment
            extreme_events = volatility_analysis.get('extreme_values', {}).get('extreme_events', {})
            total_extremes = extreme_events.get('total_extremes', 0)
            
            insights['risk_assessment'] = {
                'volatility_level': 'high' if volatility > 0.3 else 'medium' if volatility > 0.1 else 'low',
                'extreme_events_count': total_extremes,
                'risk_level': self._assess_risk_level(volatility, total_extremes, len(ts_data))
            }
            
            # Metric-specific insights
            metric_insights = self._get_metric_specific_insights(metric_name, ts_data, trend_analysis, volatility_analysis)
            insights['metric_specific'] = metric_insights
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating financial insights: {e}")
            return {}
    
    def _assess_risk_level(self, volatility: float, extreme_events: int, data_length: int) -> str:
        """Assess overall risk level"""
        try:
            risk_score = 0
            
            # Volatility component
            if volatility > 0.4:
                risk_score += 3
            elif volatility > 0.2:
                risk_score += 2
            elif volatility > 0.1:
                risk_score += 1
            
            # Extreme events component
            extreme_rate = extreme_events / data_length if data_length > 0 else 0
            if extreme_rate > 0.2:
                risk_score += 3
            elif extreme_rate > 0.1:
                risk_score += 2
            elif extreme_rate > 0.05:
                risk_score += 1
            
            # Risk level mapping
            if risk_score >= 5:
                return 'high'
            elif risk_score >= 3:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            logger.error(f"Error assessing risk level: {e}")
            return 'unknown'
    
    def _get_metric_specific_insights(self, 
                                    metric_name: str, 
                                    ts_data: pd.Series, 
                                    trend_analysis: Dict, 
                                    volatility_analysis: Dict) -> Dict[str, Any]:
        """Generate insights specific to financial metrics"""
        try:
            latest_value = ts_data.iloc[-1]
            insights = {}
            
            if 'revenue' in metric_name.lower():
                growth_rate = trend_analysis.get('growth_analysis', {}).get('mean_growth_rate', 0)
                insights['growth_assessment'] = 'strong' if growth_rate > 0.1 else 'moderate' if growth_rate > 0.05 else 'weak'
                insights['interpretation'] = f"Revenue growth rate: {growth_rate:.2%}"
                
            elif 'ratio' in metric_name.lower():
                if 'current' in metric_name.lower():
                    if latest_value > 2.0:
                        insights['interpretation'] = "Strong liquidity position"
                    elif latest_value > 1.0:
                        insights['interpretation'] = "Adequate liquidity"
                    else:
                        insights['interpretation'] = "Liquidity concerns"
                        
                elif 'debt' in metric_name.lower():
                    if latest_value > 2.0:
                        insights['interpretation'] = "High leverage - potential risk"
                    elif latest_value > 1.0:
                        insights['interpretation'] = "Moderate leverage"
                    else:
                        insights['interpretation'] = "Conservative leverage"
            
            elif 'roa' in metric_name.lower() or 'roe' in metric_name.lower():
                if latest_value > 0.15:
                    insights['interpretation'] = "Excellent returns"
                elif latest_value > 0.1:
                    insights['interpretation'] = "Good returns"
                elif latest_value > 0.05:
                    insights['interpretation'] = "Moderate returns"
                else:
                    insights['interpretation'] = "Poor returns"
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting metric-specific insights: {e}")
            return {}


class TimeSeriesBatchAnalyzer:
    """
    Batch analyzer for multiple time series
    """
    
    def __init__(self, analyzer: FinancialTimeSeriesAnalyzer):
        """Initialize batch analyzer"""
        self.analyzer = analyzer
        self.batch_results = {}
    
    def analyze_multiple_series(self, 
                              data: pd.DataFrame,
                              date_column: str,
                              value_columns: List[str],
                              company_id_column: Optional[str] = None) -> Dict[str, Any]:
        """Analyze multiple time series"""
        try:
            batch_results = {}
            
            for value_col in value_columns:
                logger.info(f"Analyzing time series for {value_col}")
                
                company_id = data[company_id_column].iloc[0] if company_id_column and company_id_column in data.columns else None
                
                analysis_result = self.analyzer.analyze_time_series(
                    data, date_column, value_col, company_id
                )
                
                batch_results[value_col] = analysis_result
            
            # Cross-series analysis
            cross_analysis = self._cross_series_analysis(batch_results)
            
            return {
                'individual_analyses': batch_results,
                'cross_series_analysis': cross_analysis,
                'summary_statistics': self._generate_batch_summary(batch_results)
            }
            
        except Exception as e:
            logger.error(f"Error in batch analysis: {e}")
            return {'error': str(e)}
    
    def _cross_series_analysis(self, batch_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze relationships between multiple time series"""
        try:
            cross_analysis = {}
            
            # Extract correlation information
            correlations = {}
            for metric, result in batch_results.items():
                if 'correlation_analysis' in result:
                    correlations[metric] = result['correlation_analysis'].get('correlations', {})
            
            cross_analysis['correlations'] = correlations
            
            # Find common patterns
            patterns = self._identify_common_patterns(batch_results)
            cross_analysis['common_patterns'] = patterns
            
            # Risk correlation
            risk_correlation = self._analyze_risk_correlation(batch_results)
            cross_analysis['risk_correlation'] = risk_correlation
            
            return cross_analysis
            
        except Exception as e:
            logger.error(f"Error in cross-series analysis: {e}")
            return {}
    
    def _identify_common_patterns(self, batch_results: Dict[str, Any]) -> Dict[str, Any]:
        """Identify common patterns across time series"""
        try:
            patterns = {}
            
            # Trend patterns
            trend_directions = {}
            for metric, result in batch_results.items():
                trend_analysis = result.get('trend_analysis', {})
                linear_trend = trend_analysis.get('linear_trend', {})
                if linear_trend.get('is_significant'):
                    direction = 'increasing' if linear_trend.get('slope', 0) > 0 else 'decreasing'
                    trend_directions[metric] = direction
            
            patterns['trend_directions'] = trend_directions
            
            # Common trend direction
            if trend_directions:
                direction_counts = {}
                for direction in trend_directions.values():
                    direction_counts[direction] = direction_counts.get(direction, 0) + 1
                
                most_common = max(direction_counts.items(), key=lambda x: x[1])
                patterns['dominant_trend'] = most_common[0]
                patterns['trend_consensus'] = most_common[1] / len(trend_directions)
            
            # Volatility patterns
            volatility_levels = {}
            for metric, result in batch_results.items():
                risk_level = result.get('financial_insights', {}).get('risk_assessment', {}).get('risk_level')
                if risk_level:
                    volatility_levels[metric] = risk_level
            
            patterns['volatility_distribution'] = volatility_levels
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error identifying common patterns: {e}")
            return {}
    
    def _analyze_risk_correlation(self, batch_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze risk correlation across metrics"""
        try:
            risk_metrics = {}
            
            for metric, result in batch_results.items():
                risk_assessment = result.get('financial_insights', {}).get('risk_assessment', {})
                if risk_assessment:
                    volatility_level = risk_assessment.get('volatility_level')
                    risk_level = risk_assessment.get('risk_level')
                    
                    risk_metrics[metric] = {
                        'volatility_level': volatility_level,
                        'risk_level': risk_level
                    }
            
            # Count risk levels
            risk_counts = {'low': 0, 'medium': 0, 'high': 0}
            volatility_counts = {'low': 0, 'medium': 0, 'high': 0}
            
            for metrics in risk_metrics.values():
                risk_level = metrics.get('risk_level')
                volatility_level = metrics.get('volatility_level')
                
                if risk_level in risk_counts:
                    risk_counts[risk_level] += 1
                if volatility_level in volatility_counts:
                    volatility_counts[volatility_level] += 1
            
            return {
                'risk_distribution': risk_counts,
                'volatility_distribution': volatility_counts,
                'overall_risk_profile': self._assess_overall_risk(risk_counts),
                'individual_risk_metrics': risk_metrics
            }
            
        except Exception as e:
            logger.error(f"Error analyzing risk correlation: {e}")
            return {}
    
    def _assess_overall_risk(self, risk_counts: Dict[str, int]) -> str:
        """Assess overall risk profile"""
        total_metrics = sum(risk_counts.values())
        if total_metrics == 0:
            return 'unknown'
        
        high_risk_pct = risk_counts.get('high', 0) / total_metrics
        medium_risk_pct = risk_counts.get('medium', 0) / total_metrics
        
        if high_risk_pct > 0.5:
            return 'high_risk'
        elif high_risk_pct + medium_risk_pct > 0.6:
            return 'medium_risk'
        else:
            return 'low_risk'
    
    def _generate_batch_summary(self, batch_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for batch analysis"""
        try:
            summary = {
                'total_metrics_analyzed': len(batch_results),
                'successful_analyses': sum(1 for result in batch_results.values() if 'error' not in result),
                'failed_analyses': sum(1 for result in batch_results.values() if 'error' in result)
            }
            
            # Aggregate insights
            all_insights = []
            all_recommendations = []
            
            for result in batch_results.values():
                if 'financial_insights' in result:
                    insights = result['financial_insights']
                    all_insights.extend(insights.get('insights', []))
                    all_recommendations.extend(insights.get('recommendations', []))
            
            summary['total_insights'] = len(all_insights)
            summary['total_recommendations'] = len(all_recommendations)
            
            # Most common insights
            insight_counts = {}
            for insight in all_insights:
                insight_counts[insight] = insight_counts.get(insight, 0) + 1
            
            if insight_counts:
                summary['most_common_insight'] = max(insight_counts.items(), key=lambda x: x[1])
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating batch summary: {e}")
            return {}


# Utility functions for time series analysis
def create_sample_financial_data(n_periods: int = 20, frequency: str = 'quarterly') -> pd.DataFrame:
    """Create sample financial time series data for testing"""
    try:
        # Generate dates
        if frequency == 'quarterly':
            dates = pd.date_range(start='2020-01-01', periods=n_periods, freq='Q')
        elif frequency == 'monthly':
            dates = pd.date_range(start='2020-01-01', periods=n_periods, freq='M')
        elif frequency == 'annual':
            dates = pd.date_range(start='2015-01-01', periods=n_periods, freq='Y')
        else:
            dates = pd.date_range(start='2020-01-01', periods=n_periods, freq='Q')
        
        # Generate sample data with trend and seasonality
        np.random.seed(42)
        
        # Base trend
        trend = np.linspace(100, 150, n_periods)
        
        # Add seasonality (for quarterly data)
        if frequency == 'quarterly':
            seasonal = 10 * np.sin(2 * np.pi * np.arange(n_periods) / 4)
        elif frequency == 'monthly':
            seasonal = 8 * np.sin(2 * np.pi * np.arange(n_periods) / 12)
        else:
            seasonal = np.zeros(n_periods)
        
        # Add noise
        noise = np.random.normal(0, 5, n_periods)
        
        # Create metrics
        revenue = (trend + seasonal + noise) * 1000000  # Revenue in millions
        net_income = revenue * (0.1 + 0.05 * np.random.normal(0, 1, n_periods))  # 10% margin with variation
        total_assets = revenue * (1.5 + 0.2 * np.random.normal(0, 1, n_periods))
        current_assets = total_assets * (0.4 + 0.1 * np.random.normal(0, 1, n_periods))
        current_liabilities = current_assets * (0.6 + 0.2 * np.random.normal(0, 1, n_periods))
        
        # Calculate ratios
        current_ratio = current_assets / current_liabilities
        roa = net_income / total_assets
        
        # Create DataFrame
        data = pd.DataFrame({
            'date': dates,
            'company_id': 'SAMPLE_001',
            'revenue': revenue,
            'net_income': net_income,
            'total_assets': total_assets,
            'current_assets': current_assets,
            'current_liabilities': current_liabilities,
            'current_ratio': current_ratio,
            'roa': roa
        })
        
        return data
        
    except Exception as e:
        logger.error(f"Error creating sample data: {e}")
        return pd.DataFrame()


# Example usage and testing
if __name__ == "__main__":
    print("=== Testing Time Series Analyzer ===")
    
    # Create sample data
    print("Creating sample financial data...")
    sample_data = create_sample_financial_data(n_periods=16, frequency='quarterly')
    print(f"Sample data created with {len(sample_data)} periods")
    print(f"Columns: {list(sample_data.columns)}")
    
    # Initialize analyzer
    analyzer = FinancialTimeSeriesAnalyzer(frequency='quarterly', confidence_level=0.95)
    
    # Test single metric analysis
    print("\n--- Testing Revenue Analysis ---")
    revenue_analysis = analyzer.analyze_time_series(
        sample_data, 'date', 'revenue', 'SAMPLE_001'
    )
    
    if 'error' not in revenue_analysis:
        print("✓ Revenue analysis completed successfully")
        print(f"  Data period: {revenue_analysis['data_period']['periods']} periods")
        print(f"  Trend type: {revenue_analysis['trend_analysis']['linear_trend']['is_significant']}")
        print(f"  Risk level: {revenue_analysis['financial_insights']['risk_assessment']['risk_level']}")
        
        # Print forecasts
        forecasts = revenue_analysis.get('forecasts', {})
        if forecasts:
            print(f"  Forecast methods: {list(forecasts.keys())}")
    else:
        print(f"✗ Revenue analysis failed: {revenue_analysis['error']}")
    
    # Test batch analysis
    print("\n--- Testing Batch Analysis ---")
    batch_analyzer = TimeSeriesBatchAnalyzer(analyzer)
    
    metrics_to_analyze = ['revenue', 'net_income', 'current_ratio', 'roa']
    batch_results = batch_analyzer.analyze_multiple_series(
        sample_data, 'date', metrics_to_analyze, 'company_id'
    )
    
    if 'error' not in batch_results:
        print("✓ Batch analysis completed successfully")
        summary = batch_results['summary_statistics']
        print(f"  Metrics analyzed: {summary['total_metrics_analyzed']}")
        print(f"  Successful analyses: {summary['successful_analyses']}")
        print(f"  Total insights: {summary['total_insights']}")
        
        # Print cross-series patterns
        patterns = batch_results['cross_series_analysis']['common_patterns']
        if 'dominant_trend' in patterns:
            print(f"  Dominant trend: {patterns['dominant_trend']}")
    else:
        print(f"✗ Batch analysis failed: {batch_results['error']}")
    
    # Test anomaly detection
    print("\n--- Testing Anomaly Detection ---")
    current_ratio_analysis = analyzer.analyze_time_series(
        sample_data, 'date', 'current_ratio', 'SAMPLE_001'
    )
    
    if 'anomalies' in current_ratio_analysis:
        anomalies = current_ratio_analysis['anomalies']
        total_anomalies = anomalies.get('combined', {}).get('total_anomalies', 0)
        print(f"✓ Anomaly detection completed - found {total_anomalies} anomalies")
    
    # Test forecasting
    print("\n--- Testing Forecasting ---")
    if 'forecasts' in revenue_analysis:
        forecasts = revenue_analysis['forecasts']
        for method, forecast_data in forecasts.items():
            if 'forecast' in forecast_data:
                forecast_values = list(forecast_data['forecast'].values())
                print(f"  {method}: {len(forecast_values)} periods forecasted")
                print(f"    Next period forecast: {forecast_values[0]:,.0f}")
    
    print("\n=== Time Series Analyzer testing completed! ===")


# Additional utility class for visualization support
class TimeSeriesVisualizer:
    """
    Helper class for visualizing time series analysis results
    """
    
    def __init__(self):
        """Initialize visualizer"""
        self.figure_size = (12, 8)
        self.style = 'seaborn-v0_8'
    
    def plot_time_series_analysis(self, ts_data: pd.Series, analysis_result: Dict, save_path: Optional[str] = None):
        """Plot comprehensive time series analysis"""
        try:
            plt.style.use(self.style)
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Time Series Analysis: {analysis_result.get("metric", "Unknown")}', fontsize=16)
            
            # Original time series
            axes[0, 0].plot(ts_data.index, ts_data.values, 'b-', linewidth=2)
            axes[0, 0].set_title('Original Time Series')
            axes[0, 0].set_ylabel('Value')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Decomposition (if available)
            decomposition = analysis_result.get('decomposition', {})
            if 'components' in decomposition:
                trend_data = decomposition['components'].get('trend', {})
                if trend_data:
                    trend_series = pd.Series(trend_data)
                    axes[0, 1].plot(trend_series.index, trend_series.values, 'g-', linewidth=2)
                    axes[0, 1].set_title('Trend Component')
                    axes[0, 1].grid(True, alpha=0.3)
            
            # Returns/Growth rates
            returns = ts_data.pct_change().dropna()
            axes[1, 0].plot(returns.index, returns.values, 'r-', alpha=0.7)
            axes[1, 0].set_title('Returns/Growth Rate')
            axes[1, 0].set_ylabel('Percentage Change')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Forecasts (if available)
            forecasts = analysis_result.get('forecasts', {})
            if forecasts:
                # Use the first available forecast method
                forecast_method = list(forecasts.keys())[0]
                forecast_data = forecasts[forecast_method]
                
                if 'forecast' in forecast_data:
                    forecast_series = pd.Series(forecast_data['forecast'])
                    
                    # Plot historical data
                    axes[1, 1].plot(ts_data.index, ts_data.values, 'b-', label='Historical', linewidth=2)
                    
                    # Plot forecast
                    axes[1, 1].plot(forecast_series.index, forecast_series.values, 'r--', 
                                   label=f'Forecast ({forecast_method})', linewidth=2)
                    
                    # Plot confidence intervals if available
                    if 'confidence_interval_lower' in forecast_data and 'confidence_interval_upper' in forecast_data:
                        lower_series = pd.Series(forecast_data['confidence_interval_lower'])
                        upper_series = pd.Series(forecast_data['confidence_interval_upper'])
                        
                        axes[1, 1].fill_between(forecast_series.index, 
                                               lower_series.values, 
                                               upper_series.values, 
                                               alpha=0.3, color='red', label='Confidence Interval')
                    
                    axes[1, 1].set_title('Forecast')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting time series analysis: {e}")
    
    def plot_correlation_matrix(self, batch_results: Dict, save_path: Optional[str] = None):
        """Plot correlation matrix for multiple time series"""
        try:
            # Extract correlations
            correlation_data = {}
            
            for metric, result in batch_results.items():
                if 'correlation_analysis' in result:
                    correlations = result['correlation_analysis'].get('correlations', {})
                    for other_metric, corr_value in correlations.items():
                        if metric not in correlation_data:
                            correlation_data[metric] = {}
                        correlation_data[metric][other_metric] = corr_value
            
            if not correlation_data:
                print("No correlation data available for plotting")
                return
            
            # Create correlation matrix
            correlation_df = pd.DataFrame(correlation_data).fillna(0)
            
            # Plot heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_df, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=0.5)
            plt.title('Time Series Correlation Matrix')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Correlation matrix saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting correlation matrix: {e}")


# Export main classes and functions
__all__ = [
    'FinancialTimeSeriesAnalyzer',
    'TimeSeriesBatchAnalyzer', 
    'TimeSeriesVisualizer',
    'TimeSeriesComponents',
    'ForecastResult',
    'AnomalyDetectionResult',
    'TrendType',
    'SeasonalityType',
    'create_sample_financial_data'
]