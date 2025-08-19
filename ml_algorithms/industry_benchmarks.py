"""
Industry Benchmarks Module
Generates industry-specific benchmarks and comparative analysis
"""

import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceCategory(Enum):
    """Performance categories for companies"""
    EXCELLENT = "excellent"
    MODERATE = "moderate"
    UNDERPERFORMING = "underperforming"

@dataclass
class BenchmarkMetrics:
    """Data class for benchmark metrics"""
    mean: float
    median: float
    std_dev: float
    min_value: float
    max_value: float
    percentile_25: float
    percentile_75: float
    percentile_90: float
    percentile_10: float

@dataclass
class IndustryBenchmark:
    """Data class for industry benchmark results"""
    industry: str
    total_companies: int
    benchmarks: Dict[str, BenchmarkMetrics]
    top_performers: List[Dict]
    category_distribution: Dict[str, int]
    generated_at: datetime

class IndustryBenchmarks:
    """
    Industry benchmarking system for financial analysis
    """
    
    def __init__(self, db_config: Dict[str, str]):
        """
        Initialize industry benchmarks analyzer
        
        Args:
            db_config (Dict): Database configuration
        """
        self.db_config = db_config
        self.conn = None
        
        # Industry mappings
        self.INDUSTRY_SECTORS = {
            'technology': [
                'application_software',
                'it_services', 
                'communication_equipment'
            ],
            'healthcare': [
                'biotechnology',
                'medical_equipment',
                'drug_manufacturing'
            ],
            'industrial': [
                'aerospace_defense',
                'construction_machinery',
                'industrial_electrical'
            ],
            'financial': [
                'asset_management',
                'financial_services',
                'regional_banks'
            ]
        }
        
        # Key financial metrics for benchmarking
        self.BENCHMARK_METRICS = [
            'current_ratio',
            'quick_ratio',
            'debt_to_equity',
            'roa',
            'roe',
            'profit_margin',
            'asset_turnover',
            'inventory_turnover',
            'receivables_turnover',
            'interest_coverage',
            'free_cash_flow_margin',
            'overall_score'
        ]
    
    def connect_to_database(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            logger.info("Connected to database for industry benchmarking")
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    def close_connection(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def generate_industry_benchmark(self, industry: str) -> Dict:
        """
        Generate comprehensive industry benchmark
        
        Args:
            industry (str): Industry name
            
        Returns:
            Dict: Industry benchmark data
        """
        try:
            self.connect_to_database()
            
            # Get industry companies
            companies = self._get_industry_companies(industry)
            if not companies:
                return {'error': f'No companies found for industry: {industry}'}
            
            # Calculate benchmarks
            benchmarks = self._calculate_industry_benchmarks(industry, companies)
            
            # Get top performers
            top_performers = self._get_top_performers(industry, companies)
            
            # Get category distribution
            category_dist = self._get_category_distribution(industry, companies)
            
            # Get industry trends
            trends = self._calculate_industry_trends(industry)
            
            # Get peer comparisons
            peer_analysis = self._generate_peer_analysis(industry, companies)
            
            result = {
                'industry': industry,
                'total_companies': len(companies),
                'benchmarks': benchmarks,
                'top_performers': top_performers,
                'category_distribution': category_dist,
                'industry_trends': trends,
                'peer_analysis': peer_analysis,
                'percentiles': self._calculate_percentiles(industry, companies),
                'averages': self._calculate_averages(industry, companies),
                'generated_at': datetime.now().isoformat(),
                'data_quality': self._assess_data_quality(companies)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating industry benchmark: {e}")
            return {'error': str(e)}
        finally:
            self.close_connection()
    
    def _get_industry_companies(self, industry: str) -> List[Dict]:
        """Get all companies in specified industry"""
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
            SELECT DISTINCT
                c.company_id,
                c.company_name,
                c.industry,
                c.sector,
                c.performance_category,
                bs.current_assets,
                bs.current_liabilities,
                bs.total_assets,
                bs.total_debt,
                bs.shareholders_equity,
                is_.revenue,
                is_.net_income,
                cf.operating_cash_flow,
                cf.free_cash_flow
            FROM companies c
            LEFT JOIN balance_sheets bs ON c.company_id = bs.company_id
            LEFT JOIN income_statements is_ ON c.company_id = is_.company_id  
            LEFT JOIN cash_flows cf ON c.company_id = cf.company_id
            WHERE LOWER(c.industry) = LOWER(%s)
            AND bs.year = (SELECT MAX(year) FROM balance_sheets WHERE company_id = c.company_id)
            AND is_.year = (SELECT MAX(year) FROM income_statements WHERE company_id = c.company_id)
            AND cf.year = (SELECT MAX(year) FROM cash_flows WHERE company_id = c.company_id)
            """
            
            cursor.execute(query, (industry,))
            companies = cursor.fetchall()
            
            logger.info(f"Found {len(companies)} companies in {industry} industry")
            return [dict(company) for company in companies]
            
        except Exception as e:
            logger.error(f"Error getting industry companies: {e}")
            return []
    
    def _calculate_industry_benchmarks(self, industry: str, companies: List[Dict]) -> Dict[str, BenchmarkMetrics]:
        """Calculate statistical benchmarks for industry"""
        benchmarks = {}
        
        try:
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(companies)
            
            for metric in self.BENCHMARK_METRICS:
                if metric == 'overall_score':
                    # Calculate overall score if not available
                    scores = self._calculate_overall_scores(companies)
                    values = np.array(scores)
                elif metric == 'current_ratio':
                    values = self._safe_divide(df['current_assets'], df['current_liabilities'])
                elif metric == 'quick_ratio':
                    # Approximate quick assets as current assets - inventory (if available)
                    quick_assets = df['current_assets'] * 0.8  # Approximation
                    values = self._safe_divide(quick_assets, df['current_liabilities'])
                elif metric == 'debt_to_equity':
                    values = self._safe_divide(df['total_debt'], df['shareholders_equity'])
                elif metric == 'roa':
                    values = self._safe_divide(df['net_income'], df['total_assets'])
                elif metric == 'roe':
                    values = self._safe_divide(df['net_income'], df['shareholders_equity'])
                elif metric == 'profit_margin':
                    values = self._safe_divide(df['net_income'], df['revenue'])
                elif metric == 'asset_turnover':
                    values = self._safe_divide(df['revenue'], df['total_assets'])
                elif metric == 'free_cash_flow_margin':
                    values = self._safe_divide(df['free_cash_flow'], df['revenue'])
                else:
                    continue
                
                # Remove infinite and NaN values
                values = values[np.isfinite(values)]
                
                if len(values) > 0:
                    benchmarks[metric] = BenchmarkMetrics(
                        mean=float(np.mean(values)),
                        median=float(np.median(values)),
                        std_dev=float(np.std(values)),
                        min_value=float(np.min(values)),
                        max_value=float(np.max(values)),
                        percentile_25=float(np.percentile(values, 25)),
                        percentile_75=float(np.percentile(values, 75)),
                        percentile_90=float(np.percentile(values, 90)),
                        percentile_10=float(np.percentile(values, 10))
                    )
            
            return benchmarks
            
        except Exception as e:
            logger.error(f"Error calculating industry benchmarks: {e}")
            return {}
    
    def _safe_divide(self, numerator, denominator):
        """Safely divide arrays, handling zero division"""
        return np.where(denominator != 0, numerator / denominator, 0)
    
    def _calculate_overall_scores(self, companies: List[Dict]) -> List[float]:
        """Calculate overall financial health scores"""
        scores = []
        
        for company in companies:
            try:
                # Simple scoring algorithm
                score = 50  # Base score
                
                # Liquidity score (30% weight)
                current_ratio = self._safe_ratio(company.get('current_assets'), company.get('current_liabilities'))
                if current_ratio >= 2.0:
                    liquidity_score = 30
                elif current_ratio >= 1.5:
                    liquidity_score = 25
                elif current_ratio >= 1.0:
                    liquidity_score = 15
                else:
                    liquidity_score = 0
                
                # Profitability score (40% weight)
                roa = self._safe_ratio(company.get('net_income'), company.get('total_assets'))
                if roa >= 0.15:
                    profitability_score = 40
                elif roa >= 0.10:
                    profitability_score = 30
                elif roa >= 0.05:
                    profitability_score = 20
                elif roa >= 0:
                    profitability_score = 10
                else:
                    profitability_score = 0
                
                # Leverage score (30% weight)
                debt_ratio = self._safe_ratio(company.get('total_debt'), company.get('total_assets'))
                if debt_ratio <= 0.3:
                    leverage_score = 30
                elif debt_ratio <= 0.5:
                    leverage_score = 20
                elif debt_ratio <= 0.7:
                    leverage_score = 10
                else:
                    leverage_score = 0
                
                total_score = liquidity_score + profitability_score + leverage_score
                scores.append(min(100, max(0, total_score)))
                
            except Exception as e:
                logger.warning(f"Error calculating score for company: {e}")
                scores.append(50)  # Default score
        
        return scores
    
    def _safe_ratio(self, numerator, denominator):
        """Safely calculate ratio"""
        try:
            if denominator and float(denominator) != 0:
                return float(numerator or 0) / float(denominator)
            return 0
        except:
            return 0
    
    def _get_top_performers(self, industry: str, companies: List[Dict], top_n: int = 10) -> List[Dict]:
        """Get top performing companies in industry"""
        try:
            # Calculate scores for all companies
            scores = self._calculate_overall_scores(companies)
            
            # Add scores to companies
            for i, company in enumerate(companies):
                company['overall_score'] = scores[i] if i < len(scores) else 50
            
            # Sort by score and get top performers
            sorted_companies = sorted(companies, key=lambda x: x['overall_score'], reverse=True)
            
            top_performers = []
            for company in sorted_companies[:top_n]:
                top_performers.append({
                    'company_id': company['company_id'],
                    'company_name': company['company_name'],
                    'overall_score': company['overall_score'],
                    'performance_category': company.get('performance_category', 'unknown'),
                    'key_metrics': self._extract_key_metrics(company)
                })
            
            return top_performers
            
        except Exception as e:
            logger.error(f"Error getting top performers: {e}")
            return []
    
    def _extract_key_metrics(self, company: Dict) -> Dict:
        """Extract key metrics for a company"""
        try:
            return {
                'current_ratio': self._safe_ratio(company.get('current_assets'), company.get('current_liabilities')),
                'roa': self._safe_ratio(company.get('net_income'), company.get('total_assets')),
                'debt_to_equity': self._safe_ratio(company.get('total_debt'), company.get('shareholders_equity')),
                'revenue': company.get('revenue', 0),
                'net_income': company.get('net_income', 0)
            }
        except Exception as e:
            logger.warning(f"Error extracting key metrics: {e}")
            return {}
    
    def _get_category_distribution(self, industry: str, companies: List[Dict]) -> Dict[str, int]:
        """Get distribution of performance categories"""
        try:
            distribution = {
                'excellent': 0,
                'moderate': 0,
                'underperforming': 0,
                'unknown': 0
            }
            
            for company in companies:
                category = company.get('performance_category', 'unknown').lower()
                if category in distribution:
                    distribution[category] += 1
                else:
                    distribution['unknown'] += 1
            
            return distribution
            
        except Exception as e:
            logger.error(f"Error getting category distribution: {e}")
            return {}
    
    def _calculate_industry_trends(self, industry: str) -> Dict:
        """Calculate industry trends over time"""
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            
            # Get historical data for trend analysis
            query = """
            SELECT 
                bs.year,
                AVG(CASE WHEN bs.current_liabilities > 0 THEN bs.current_assets / bs.current_liabilities END) as avg_current_ratio,
                AVG(CASE WHEN bs.total_assets > 0 THEN is_.net_income / bs.total_assets END) as avg_roa,
                AVG(CASE WHEN bs.shareholders_equity > 0 THEN bs.total_debt / bs.shareholders_equity END) as avg_debt_to_equity,
                COUNT(*) as company_count
            FROM balance_sheets bs
            JOIN income_statements is_ ON bs.company_id = is_.company_id AND bs.year = is_.year
            JOIN companies c ON bs.company_id = c.company_id
            WHERE LOWER(c.industry) = LOWER(%s)
            AND bs.year >= %s
            GROUP BY bs.year
            ORDER BY bs.year
            """
            
            current_year = datetime.now().year
            start_year = current_year - 3  # Last 3 years
            
            cursor.execute(query, (industry, start_year))
            trend_data = cursor.fetchall()
            
            trends = {
                'years': [row['year'] for row in trend_data],
                'avg_current_ratio': [float(row['avg_current_ratio'] or 0) for row in trend_data],
                'avg_roa': [float(row['avg_roa'] or 0) for row in trend_data],
                'avg_debt_to_equity': [float(row['avg_debt_to_equity'] or 0) for row in trend_data],
                'company_count': [int(row['company_count']) for row in trend_data]
            }
            
            # Calculate trend directions
            trends['trend_analysis'] = self._analyze_trends(trends)
            
            return trends
            
        except Exception as e:
            logger.error(f"Error calculating industry trends: {e}")
            return {}
    
    def _analyze_trends(self, trends: Dict) -> Dict:
        """Analyze trend directions"""
        try:
            analysis = {}
            
            for metric in ['avg_current_ratio', 'avg_roa', 'avg_debt_to_equity']:
                values = trends.get(metric, [])
                if len(values) >= 2:
                    if values[-1] > values[0]:
                        direction = 'improving' if metric != 'avg_debt_to_equity' else 'worsening'
                    elif values[-1] < values[0]:
                        direction = 'declining' if metric != 'avg_debt_to_equity' else 'improving'
                    else:
                        direction = 'stable'
                    
                    analysis[metric] = {
                        'direction': direction,
                        'change_percent': ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
                    }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            return {}
    
    def _generate_peer_analysis(self, industry: str, companies: List[Dict]) -> Dict:
        """Generate peer analysis within industry"""
        try:
            # Group companies by performance category
            peer_groups = {
                'excellent': [],
                'moderate': [],
                'underperforming': []
            }
            
            for company in companies:
                category = company.get('performance_category', 'moderate').lower()
                if category in peer_groups:
                    peer_groups[category].append(company)
            
            # Calculate metrics for each peer group
            peer_analysis = {}
            for category, peers in peer_groups.items():
                if peers:
                    df = pd.DataFrame(peers)
                    peer_analysis[category] = {
                        'count': len(peers),
                        'avg_revenue': float(df['revenue'].mean()) if 'revenue' in df.columns else 0,
                        'avg_net_income': float(df['net_income'].mean()) if 'net_income' in df.columns else 0,
                        'avg_total_assets': float(df['total_assets'].mean()) if 'total_assets' in df.columns else 0,
                        'avg_current_ratio': float(self._safe_divide(df['current_assets'], df['current_liabilities']).mean()),
                        'avg_roa': float(self._safe_divide(df['net_income'], df['total_assets']).mean())
                    }
            
            return peer_analysis
            
        except Exception as e:
            logger.error(f"Error generating peer analysis: {e}")
            return {}
    
    def _calculate_percentiles(self, industry: str, companies: List[Dict]) -> Dict:
        """Calculate percentiles for key metrics"""
        try:
            percentiles = {}
            df = pd.DataFrame(companies)
            
            # Calculate overall scores
            scores = self._calculate_overall_scores(companies)
            
            percentiles['overall_score'] = {
                '10th': float(np.percentile(scores, 10)),
                '25th': float(np.percentile(scores, 25)),
                '50th': float(np.percentile(scores, 50)),
                '75th': float(np.percentile(scores, 75)),
                '90th': float(np.percentile(scores, 90))
            }
            
            # Calculate for other key metrics
            metrics = {
                'current_ratio': self._safe_divide(df['current_assets'], df['current_liabilities']),
                'roa': self._safe_divide(df['net_income'], df['total_assets']),
                'debt_to_equity': self._safe_divide(df['total_debt'], df['shareholders_equity'])
            }
            
            for metric_name, values in metrics.items():
                clean_values = values[np.isfinite(values)]
                if len(clean_values) > 0:
                    percentiles[metric_name] = {
                        '10th': float(np.percentile(clean_values, 10)),
                        '25th': float(np.percentile(clean_values, 25)),
                        '50th': float(np.percentile(clean_values, 50)),
                        '75th': float(np.percentile(clean_values, 75)),
                        '90th': float(np.percentile(clean_values, 90))
                    }
            
            return percentiles
            
        except Exception as e:
            logger.error(f"Error calculating percentiles: {e}")
            return {}
    
    def _calculate_averages(self, industry: str, companies: List[Dict]) -> Dict:
        """Calculate industry averages"""
        try:
            averages = {}
            df = pd.DataFrame(companies)
            
            # Overall score average
            scores = self._calculate_overall_scores(companies)
            averages['overall_score'] = {
                'mean': float(np.mean(scores)),
                'median': float(np.median(scores)),
                'std_dev': float(np.std(scores))
            }
            
            # Other metrics
            metrics = {
                'liquidity': self._safe_divide(df['current_assets'], df['current_liabilities']),
                'profitability': self._safe_divide(df['net_income'], df['total_assets']),
                'leverage': self._safe_divide(df['total_debt'], df['shareholders_equity']),
                'efficiency': self._safe_divide(df['revenue'], df['total_assets'])
            }
            
            for metric_name, values in metrics.items():
                clean_values = values[np.isfinite(values)]
                if len(clean_values) > 0:
                    averages[metric_name] = {
                        'mean': float(np.mean(clean_values)),
                        'median': float(np.median(clean_values)),
                        'std_dev': float(np.std(clean_values))
                    }
            
            return averages
            
        except Exception as e:
            logger.error(f"Error calculating averages: {e}")
            return {}
    
    def _assess_data_quality(self, companies: List[Dict]) -> Dict:
        """Assess data quality for benchmark"""
        try:
            total_companies = len(companies)
            
            # Count missing data
            missing_data = {
                'revenue': sum(1 for c in companies if not c.get('revenue')),
                'net_income': sum(1 for c in companies if not c.get('net_income')),
                'total_assets': sum(1 for c in companies if not c.get('total_assets')),
                'current_assets': sum(1 for c in companies if not c.get('current_assets')),
                'current_liabilities': sum(1 for c in companies if not c.get('current_liabilities'))
            }
            
            # Calculate completeness
            completeness = {}
            for field, missing_count in missing_data.items():
                completeness[field] = ((total_companies - missing_count) / total_companies * 100) if total_companies > 0 else 0
            
            # Overall quality score
            avg_completeness = np.mean(list(completeness.values()))
            
            quality_assessment = {
                'total_companies': total_companies,
                'completeness_by_field': completeness,
                'overall_completeness': avg_completeness,
                'quality_score': 'high' if avg_completeness >= 80 else 'medium' if avg_completeness >= 60 else 'low',
                'missing_data_count': missing_data
            }
            
            return quality_assessment
            
        except Exception as e:
            logger.error(f"Error assessing data quality: {e}")
            return {}
    
    def compare_company_to_industry(self, company_id: str, industry: str) -> Dict:
        """Compare a specific company to industry benchmarks"""
        try:
            self.connect_to_database()
            
            # Get company data
            company_data = self._get_company_data(company_id)
            if not company_data:
                return {'error': 'Company not found'}
            
            # Get industry benchmark
            benchmark = self.generate_industry_benchmark(industry)
            if 'error' in benchmark:
                return benchmark
            
            # Perform comparison
            comparison = self._perform_company_benchmark_comparison(company_data, benchmark)
            
            return {
                'company_id': company_id,
                'industry': industry,
                'comparison': comparison,
                'company_percentile': self._calculate_company_percentile(company_data, benchmark),
                'improvement_areas': self._identify_improvement_areas(company_data, benchmark),
                'competitive_position': self._assess_competitive_position(company_data, benchmark)
            }
            
        except Exception as e:
            logger.error(f"Error comparing company to industry: {e}")
            return {'error': str(e)}
        finally:
            self.close_connection()
    
    def _get_company_data(self, company_id: str) -> Optional[Dict]:
        """Get specific company data"""
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
            SELECT 
                c.company_id,
                c.company_name,
                c.industry,
                c.performance_category,
                bs.current_assets,
                bs.current_liabilities,
                bs.total_assets,
                bs.total_debt,
                bs.shareholders_equity,
                is_.revenue,
                is_.net_income,
                cf.operating_cash_flow,
                cf.free_cash_flow
            FROM companies c
            LEFT JOIN balance_sheets bs ON c.company_id = bs.company_id
            LEFT JOIN income_statements is_ ON c.company_id = is_.company_id
            LEFT JOIN cash_flows cf ON c.company_id = cf.company_id
            WHERE c.company_id = %s
            AND bs.year = (SELECT MAX(year) FROM balance_sheets WHERE company_id = c.company_id)
            AND is_.year = (SELECT MAX(year) FROM income_statements WHERE company_id = c.company_id)
            AND cf.year = (SELECT MAX(year) FROM cash_flows WHERE company_id = c.company_id)
            """
            
            cursor.execute(query, (company_id,))
            result = cursor.fetchone()
            
            return dict(result) if result else None
            
        except Exception as e:
            logger.error(f"Error getting company data: {e}")
            return None
    
    def _perform_company_benchmark_comparison(self, company: Dict, benchmark: Dict) -> Dict:
        """Compare company metrics to industry benchmarks"""
        try:
            comparison = {}
            
            # Calculate company metrics
            company_metrics = {
                'current_ratio': self._safe_ratio(company.get('current_assets'), company.get('current_liabilities')),
                'roa': self._safe_ratio(company.get('net_income'), company.get('total_assets')),
                'debt_to_equity': self._safe_ratio(company.get('total_debt'), company.get('shareholders_equity')),
                'overall_score': self._calculate_overall_scores([company])[0]
            }
            
            # Compare to benchmarks
            benchmarks = benchmark.get('benchmarks', {})
            
            for metric, value in company_metrics.items():
                if metric in benchmarks:
                    bench_data = benchmarks[metric]
                    comparison[metric] = {
                        'company_value': value,
                        'industry_mean': bench_data.mean,
                        'industry_median': bench_data.median,
                        'percentile_position': self._calculate_percentile_position(value, bench_data),
                        'vs_mean_percent': ((value - bench_data.mean) / bench_data.mean * 100) if bench_data.mean != 0 else 0,
                        'status': self._get_performance_status(value, bench_data, metric)
                    }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error performing company benchmark comparison: {e}")
            return {}
    
    def _calculate_percentile_position(self, value: float, benchmark: BenchmarkMetrics) -> float:
        """Calculate where company value falls in industry percentiles"""
        try:
            if value <= benchmark.percentile_10:
                return 10
            elif value <= benchmark.percentile_25:
                return 25
            elif value <= benchmark.median:
                return 50
            elif value <= benchmark.percentile_75:
                return 75
            elif value <= benchmark.percentile_90:
                return 90
            else:
                return 95
        except:
            return 50
    
    def _get_performance_status(self, value: float, benchmark: BenchmarkMetrics, metric: str) -> str:
        """Get performance status relative to industry"""
        try:
            # For metrics where higher is better
            higher_better = ['current_ratio', 'roa', 'asset_turnover', 'overall_score']
            
            if metric in higher_better:
                if value >= benchmark.percentile_75:
                    return 'above_average'
                elif value >= benchmark.percentile_25:
                    return 'average'
                else:
                    return 'below_average'
            else:  # For metrics where lower is better (like debt_to_equity)
                if value <= benchmark.percentile_25:
                    return 'above_average'
                elif value <= benchmark.percentile_75:
                    return 'average'
                else:
                    return 'below_average'
        except:
            return 'unknown'
    
    def _calculate_company_percentile(self, company: Dict, benchmark: Dict) -> Dict:
        """Calculate company's percentile ranking in industry"""
        try:
            percentiles = benchmark.get('percentiles', {})
            overall_score = self._calculate_overall_scores([company])[0]
            
            if 'overall_score' in percentiles:
                perc_data = percentiles['overall_score']
                if overall_score >= perc_data['90th']:
                    percentile_rank = 90
                elif overall_score >= perc_data['75th']:
                    percentile_rank = 75
                elif overall_score >= perc_data['50th']:
                    percentile_rank = 50
                elif overall_score >= perc_data['25th']:
                    percentile_rank = 25
                else:
                    percentile_rank = 10
            else:
                percentile_rank = 50
            
            return {
                'overall_percentile': percentile_rank,
                'performance_tier': self._get_performance_tier(percentile_rank),
                'companies_outperforming': self._calculate_outperforming_percentage(percentile_rank),
                'score': overall_score
            }
            
        except Exception as e:
            logger.error(f"Error calculating company percentile: {e}")
            return {}
    
    def _get_performance_tier(self, percentile: float) -> str:
        """Get performance tier based on percentile"""
        if percentile >= 90:
            return 'top_10_percent'
        elif percentile >= 75:
            return 'top_25_percent'
        elif percentile >= 50:
            return 'above_median'
        elif percentile >= 25:
            return 'below_median'
        else:
            return 'bottom_25_percent'
    
    def _calculate_outperforming_percentage(self, percentile: float) -> float:
        """Calculate percentage of companies being outperformed"""
        return 100 - percentile
    
    def _identify_improvement_areas(self, company: Dict, benchmark: Dict) -> List[Dict]:
        """Identify areas where company can improve relative to industry"""
        try:
            improvement_areas = []
            
            # Calculate company metrics
            company_metrics = {
                'current_ratio': self._safe_ratio(company.get('current_assets'), company.get('current_liabilities')),
                'roa': self._safe_ratio(company.get('net_income'), company.get('total_assets')),
                'debt_to_equity': self._safe_ratio(company.get('total_debt'), company.get('shareholders_equity'))
            }
            
            benchmarks = benchmark.get('benchmarks', {})
            
            for metric, value in company_metrics.items():
                if metric in benchmarks:
                    bench_data = benchmarks[metric]
                    
                    # Check if significantly below industry average
                    if metric == 'debt_to_equity':  # Lower is better
                        if value > bench_data.percentile_75:
                            improvement_areas.append({
                                'metric': metric,
                                'current_value': value,
                                'industry_target': bench_data.median,
                                'improvement_potential': ((value - bench_data.median) / value * 100) if value != 0 else 0,
                                'priority': 'high' if value > bench_data.percentile_90 else 'medium',
                                'recommendation': self._get_improvement_recommendation(metric)
                            })
                    else:  # Higher is better
                        if value < bench_data.percentile_25:
                            improvement_areas.append({
                                'metric': metric,
                                'current_value': value,
                                'industry_target': bench_data.median,
                                'improvement_potential': ((bench_data.median - value) / bench_data.median * 100) if bench_data.median != 0 else 0,
                                'priority': 'high' if value < bench_data.percentile_10 else 'medium',
                                'recommendation': self._get_improvement_recommendation(metric)
                            })
            
            # Sort by priority and improvement potential
            improvement_areas.sort(key=lambda x: (x['priority'] == 'high', x['improvement_potential']), reverse=True)
            
            return improvement_areas[:5]  # Top 5 improvement areas
            
        except Exception as e:
            logger.error(f"Error identifying improvement areas: {e}")
            return []
    
    def _get_improvement_recommendation(self, metric: str) -> str:
        """Get specific improvement recommendation for metric"""
        recommendations = {
            'current_ratio': 'Improve working capital management by optimizing inventory levels and accelerating receivables collection',
            'roa': 'Focus on increasing operational efficiency and asset utilization to improve return on assets',
            'debt_to_equity': 'Consider debt reduction strategies or equity financing to improve capital structure',
            'asset_turnover': 'Optimize asset utilization through better inventory management and capacity utilization',
            'profit_margin': 'Review cost structure and pricing strategies to improve profitability'
        }
        return recommendations.get(metric, 'Review and optimize this financial metric based on industry best practices')
    
    def _assess_competitive_position(self, company: Dict, benchmark: Dict) -> Dict:
        """Assess company's competitive position in industry"""
        try:
            overall_score = self._calculate_overall_scores([company])[0]
            
            # Get industry averages
            averages = benchmark.get('averages', {})
            industry_avg = averages.get('overall_score', {}).get('mean', 50)
            
            # Determine competitive position
            if overall_score >= industry_avg * 1.2:
                position = 'market_leader'
                description = 'Significantly outperforms industry average'
            elif overall_score >= industry_avg * 1.1:
                position = 'strong_performer'
                description = 'Above industry average performance'
            elif overall_score >= industry_avg * 0.9:
                position = 'average_performer'
                description = 'Performance in line with industry average'
            elif overall_score >= industry_avg * 0.8:
                position = 'below_average'
                description = 'Performance below industry average'
            else:
                position = 'underperformer'
                description = 'Significantly below industry average'
            
            # Calculate competitive gaps
            top_performers = benchmark.get('top_performers', [])
            gap_to_leader = top_performers[0]['overall_score'] - overall_score if top_performers else 0
            
            return {
                'position': position,
                'description': description,
                'score_vs_industry': overall_score - industry_avg,
                'score_vs_leader': -gap_to_leader,
                'percentile_rank': self._calculate_simple_percentile(overall_score, benchmark),
                'competitive_advantages': self._identify_competitive_advantages(company, benchmark),
                'competitive_threats': self._identify_competitive_threats(company, benchmark)
            }
            
        except Exception as e:
            logger.error(f"Error assessing competitive position: {e}")
            return {}
    
    def _calculate_simple_percentile(self, score: float, benchmark: Dict) -> float:
        """Calculate simple percentile ranking"""
        try:
            percentiles = benchmark.get('percentiles', {}).get('overall_score', {})
            
            if score >= percentiles.get('90th', 90):
                return 90
            elif score >= percentiles.get('75th', 75):
                return 75
            elif score >= percentiles.get('50th', 50):
                return 50
            elif score >= percentiles.get('25th', 25):
                return 25
            else:
                return 10
        except:
            return 50
    
    def _identify_competitive_advantages(self, company: Dict, benchmark: Dict) -> List[str]:
        """Identify company's competitive advantages"""
        try:
            advantages = []
            
            # Calculate company metrics
            company_metrics = {
                'current_ratio': self._safe_ratio(company.get('current_assets'), company.get('current_liabilities')),
                'roa': self._safe_ratio(company.get('net_income'), company.get('total_assets')),
                'revenue': company.get('revenue', 0)
            }
            
            benchmarks = benchmark.get('benchmarks', {})
            
            # Check for above-average performance
            if 'current_ratio' in benchmarks and company_metrics['current_ratio'] > benchmarks['current_ratio'].percentile_75:
                advantages.append('Strong liquidity position')
            
            if 'roa' in benchmarks and company_metrics['roa'] > benchmarks['roa'].percentile_75:
                advantages.append('Excellent asset utilization')
            
            # Check revenue size
            if company_metrics['revenue'] > 1000000000:  # $1B+
                advantages.append('Large scale operations')
            
            return advantages
            
        except Exception as e:
            logger.error(f"Error identifying competitive advantages: {e}")
            return []
    
    def _identify_competitive_threats(self, company: Dict, benchmark: Dict) -> List[str]:
        """Identify competitive threats"""
        try:
            threats = []
            
            # Calculate company metrics
            company_metrics = {
                'debt_to_equity': self._safe_ratio(company.get('total_debt'), company.get('shareholders_equity')),
                'roa': self._safe_ratio(company.get('net_income'), company.get('total_assets'))
            }
            
            benchmarks = benchmark.get('benchmarks', {})
            
            # Check for below-average performance
            if 'debt_to_equity' in benchmarks and company_metrics['debt_to_equity'] > benchmarks['debt_to_equity'].percentile_75:
                threats.append('High financial leverage')
            
            if 'roa' in benchmarks and company_metrics['roa'] < benchmarks['roa'].percentile_25:
                threats.append('Poor asset efficiency')
            
            return threats
            
        except Exception as e:
            logger.error(f"Error identifying competitive threats: {e}")
            return []
    
    def get_available_industries(self) -> List[str]:
        """Get list of available industries for benchmarking"""
        try:
            self.connect_to_database()
            
            cursor = self.conn.cursor()
            query = """
            SELECT DISTINCT industry, COUNT(*) as company_count
            FROM companies 
            WHERE industry IS NOT NULL 
            GROUP BY industry
            HAVING COUNT(*) >= 5
            ORDER BY company_count DESC
            """
            
            cursor.execute(query)
            results = cursor.fetchall()
            
            industries = []
            for row in results:
                industries.append({
                    'industry': row[0],
                    'company_count': row[1]
                })
            
            return industries
            
        except Exception as e:
            logger.error(f"Error getting available industries: {e}")
            return []
        finally:
            self.close_connection()
    
    def generate_sector_comparison(self, sectors: List[str]) -> Dict:
        """Generate comparison between different sectors"""
        try:
            sector_data = {}
            
            for sector in sectors:
                # Get industries in sector
                industries = self.INDUSTRY_SECTORS.get(sector.lower(), [])
                
                if industries:
                    # Aggregate data for all industries in sector
                    sector_benchmark = self._aggregate_sector_data(industries)
                    sector_data[sector] = sector_benchmark
            
            # Generate comparison
            comparison = self._compare_sectors(sector_data)
            
            return {
                'sectors': sectors,
                'sector_data': sector_data,
                'comparison': comparison,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating sector comparison: {e}")
            return {'error': str(e)}
    
    def _aggregate_sector_data(self, industries: List[str]) -> Dict:
        """Aggregate benchmark data for multiple industries in a sector"""
        try:
            self.connect_to_database()
            
            # Get all companies from industries in sector
            all_companies = []
            for industry in industries:
                companies = self._get_industry_companies(industry)
                all_companies.extend(companies)
            
            if not all_companies:
                return {}
            
            # Calculate sector-level benchmarks
            benchmarks = self._calculate_industry_benchmarks('sector', all_companies)
            
            # Calculate sector metrics
            scores = self._calculate_overall_scores(all_companies)
            
            return {
                'total_companies': len(all_companies),
                'industries_included': industries,
                'average_score': float(np.mean(scores)),
                'benchmarks': benchmarks,
                'performance_distribution': self._get_category_distribution('sector', all_companies)
            }
            
        except Exception as e:
            logger.error(f"Error aggregating sector data: {e}")
            return {}
        finally:
            self.close_connection()
    
    def _compare_sectors(self, sector_data: Dict) -> Dict:
        """Compare different sectors"""
        try:
            comparison = {
                'rankings': {},
                'metrics_comparison': {},
                'key_insights': []
            }
            
            # Rank sectors by average score
            sector_scores = {}
            for sector, data in sector_data.items():
                sector_scores[sector] = data.get('average_score', 0)
            
            sorted_sectors = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)
            comparison['rankings']['by_overall_score'] = sorted_sectors
            
            # Compare key metrics across sectors
            metrics = ['current_ratio', 'roa', 'debt_to_equity']
            for metric in metrics:
                metric_comparison = {}
                for sector, data in sector_data.items():
                    benchmarks = data.get('benchmarks', {})
                    if metric in benchmarks:
                        metric_comparison[sector] = benchmarks[metric].mean
                
                if metric_comparison:
                    comparison['metrics_comparison'][metric] = sorted(
                        metric_comparison.items(), 
                        key=lambda x: x[1], 
                        reverse=(metric != 'debt_to_equity')  # debt_to_equity: lower is better
                    )
            
            # Generate insights
            if sorted_sectors:
                best_sector = sorted_sectors[0]
                worst_sector = sorted_sectors[-1]
                
                comparison['key_insights'].append(
                    f"{best_sector[0]} sector leads with average score of {best_sector[1]:.1f}"
                )
                comparison['key_insights'].append(
                    f"{worst_sector[0]} sector has lowest average score of {worst_sector[1]:.1f}"
                )
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing sectors: {e}")
            return {}


# Example usage and testing
if __name__ == "__main__":
    # Database configuration
    db_config = {
        'host': 'localhost',
        'database': 'financial_db',
        'user': 'postgres',
        'password': 'password',
        'port': 5432
    }
    
    # Initialize benchmarks analyzer
    benchmarks = IndustryBenchmarks(db_config)
    
    # Test industry benchmark generation
    print("Testing industry benchmark generation...")
    result = benchmarks.generate_industry_benchmark('technology')
    
    if 'error' not in result:
        print(f"✓ Generated benchmark for {result['total_companies']} companies")
        print(f"✓ Industry: {result['industry']}")
        print(f"✓ Top performer score: {result['top_performers'][0]['overall_score']:.1f}")
    else:
        print(f"✗ Error: {result['error']}")
    
    # Test company comparison
    print("\nTesting company to industry comparison...")
    comparison = benchmarks.compare_company_to_industry('COMP001', 'technology')
    
    if 'error' not in comparison:
        print(f"✓ Company percentile: {comparison['company_percentile']['overall_percentile']}")
        print(f"✓ Competitive position: {comparison['competitive_position']['position']}")
    else:
        print(f"✗ Error: {comparison['error']}")
    
    # Test available industries
    print("\nTesting available industries...")
    industries = benchmarks.get_available_industries()
    print(f"✓ Found {len(industries)} industries available for benchmarking")
    
    print("\nIndustry Benchmarks module test completed!")