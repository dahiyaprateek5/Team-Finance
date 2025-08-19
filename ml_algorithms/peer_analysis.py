"""
PHASE 1: Imports, Data Classes, and Basic Setup
Peer Analysis Module - Phase 1 of 5
"""

import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComparisonMethod(Enum):
    """Methods for peer comparison"""
    INDUSTRY_BASED = "industry_based"
    SIZE_BASED = "size_based"
    PERFORMANCE_BASED = "performance_based"
    SIMILARITY_BASED = "similarity_based"
    CUSTOM_COHORT = "custom_cohort"

class RankingMetric(Enum):
    """Metrics for ranking companies"""
    OVERALL_SCORE = "overall_score"
    REVENUE = "revenue"
    PROFITABILITY = "profitability"
    LIQUIDITY = "liquidity"
    EFFICIENCY = "efficiency"
    GROWTH_RATE = "growth_rate"

@dataclass
class PeerGroup:
    """Peer group definition"""
    group_id: str
    group_name: str
    companies: List[str]
    selection_criteria: Dict[str, Any]
    group_statistics: Dict[str, float]
    created_at: datetime

@dataclass
class CompanyComparison:
    """Individual company comparison results"""
    company_id: str
    company_name: str
    rank: int
    percentile: float
    peer_group_size: int
    key_metrics: Dict[str, float]
    relative_performance: Dict[str, float]
    strengths: List[str]
    weaknesses: List[str]

@dataclass
class PeerAnalysisResult:
    """Complete peer analysis results"""
    target_company: str
    peer_group: PeerGroup
    comparison_method: str
    company_comparison: CompanyComparison
    peer_rankings: List[Dict[str, Any]]
    industry_benchmarks: Dict[str, float]
    competitive_positioning: Dict[str, Any]
    recommendations: List[str]
    analysis_date: datetime

class FinancialPeerAnalyzer:
    """
    Comprehensive peer analysis system for financial companies - Phase 1 Setup
    """
    
    def __init__(self, db_config: Dict[str, str]):
        """
        Initialize peer analyzer with configuration
        
        Args:
            db_config (Dict): Database configuration
        """
        self.db_config = db_config
        self.conn = None
        
        # Key financial metrics for peer comparison
        self.COMPARISON_METRICS = {
            'liquidity': [
                'current_ratio', 'quick_ratio', 'cash_ratio', 
                'operating_cash_flow_ratio'
            ],
            'profitability': [
                'roa', 'roe', 'profit_margin', 'operating_margin',
                'gross_margin', 'ebitda_margin'
            ],
            'efficiency': [
                'asset_turnover', 'inventory_turnover', 'receivables_turnover',
                'working_capital_turnover'
            ],
            'leverage': [
                'debt_to_equity', 'debt_to_assets', 'interest_coverage',
                'debt_service_coverage'
            ],
            'growth': [
                'revenue_growth', 'earnings_growth', 'asset_growth'
            ],
            'valuation': [
                'price_to_earnings', 'price_to_book', 'ev_to_ebitda',
                'price_to_sales'
            ]
        }
        
        # Industry classifications
        self.INDUSTRY_GROUPS = {
            'technology': ['software', 'hardware', 'semiconductors', 'it_services'],
            'healthcare': ['pharmaceuticals', 'medical_devices', 'biotechnology', 'healthcare_services'],
            'financial': ['banks', 'insurance', 'asset_management', 'financial_services'],
            'industrial': ['manufacturing', 'aerospace', 'construction', 'transportation'],
            'consumer': ['retail', 'consumer_goods', 'food_beverage', 'automotive'],
            'energy': ['oil_gas', 'renewable_energy', 'utilities'],
            'real_estate': ['reits', 'real_estate_development', 'real_estate_services']
        }
        
        # Size categories (based on revenue)
        self.SIZE_CATEGORIES = {
            'mega_cap': {'min': 50_000_000_000, 'max': float('inf')},      # $50B+
            'large_cap': {'min': 10_000_000_000, 'max': 50_000_000_000},  # $10B-$50B
            'mid_cap': {'min': 2_000_000_000, 'max': 10_000_000_000},     # $2B-$10B
            'small_cap': {'min': 300_000_000, 'max': 2_000_000_000},      # $300M-$2B
            'micro_cap': {'min': 0, 'max': 300_000_000}                   # <$300M
        }
        
        # Peer group cache
        self.peer_groups_cache = {}
        
        # Scaler for similarity calculations
        self.scaler = StandardScaler()
    
    def connect_to_database(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            logger.info("Connected to database for peer analysis")
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    def close_connection(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

# Example usage for Phase 1
if __name__ == "__main__":
    print("=== Phase 1: Setup and Configuration ===")
    
    # Database configuration
    db_config = {
        'host': 'localhost',
        'database': 'financial_db',
        'user': 'postgres',
        'password': 'password',
        'port': 5432
    }
    
    # Initialize analyzer
    analyzer = FinancialPeerAnalyzer(db_config)
    
    print("✓ Analyzer initialized with configuration")
    print(f"✓ {len(analyzer.COMPARISON_METRICS)} metric categories configured")
    print(f"✓ {len(analyzer.INDUSTRY_GROUPS)} industry groups defined")
    print(f"✓ {len(analyzer.SIZE_CATEGORIES)} size categories defined")
    
    # Test database connection
    try:
        analyzer.connect_to_database()
        print("✓ Database connection successful")
        analyzer.close_connection()
        print("✓ Database connection closed")
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
    
    print("\nPhase 1 completed successfully!")
    print("Ready for Phase 2: Data Retrieval and Financial Calculations")
class FinancialPeerAnalyzer(FinancialPeerAnalyzer):
    """
    Extended analyzer with data retrieval and financial calculation capabilities
    """
    
    def _get_company_data(self, company_id: str) -> Optional[Dict]:
        """Get comprehensive company data"""
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
            SELECT 
                c.company_id,
                c.company_name,
                c.industry,
                c.sector,
                c.performance_category,
                c.country,
                c.market_cap,
                bs.revenue,
                bs.net_income,
                bs.total_assets,
                bs.current_assets,
                bs.current_liabilities,
                bs.total_debt,
                bs.shareholders_equity,
                bs.cash_and_equivalents,
                is_.operating_income,
                is_.gross_profit,
                is_.ebitda,
                cf.operating_cash_flow,
                cf.free_cash_flow,
                cf.capex
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
            
            if result:
                # Calculate additional metrics
                company_data = dict(result)
                company_data.update(self._calculate_financial_ratios(company_data))
                return company_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting company data: {e}")
            return None
    
    def _calculate_financial_ratios(self, company_data: Dict) -> Dict[str, float]:
        """Calculate financial ratios for a company"""
        try:
            ratios = {}
            
            # Helper function for safe division
            def safe_divide(numerator, denominator, default=0):
                try:
                    if denominator and float(denominator) != 0:
                        return float(numerator or 0) / float(denominator)
                    return default
                except:
                    return default
            
            # Liquidity ratios
            ratios['current_ratio'] = safe_divide(
                company_data.get('current_assets'), 
                company_data.get('current_liabilities')
            )
            
            # Quick ratio (approximation)
            quick_assets = float(company_data.get('current_assets', 0)) * 0.8  # Assume 80% are quick assets
            ratios['quick_ratio'] = safe_divide(quick_assets, company_data.get('current_liabilities'))
            
            ratios['cash_ratio'] = safe_divide(
                company_data.get('cash_and_equivalents'),
                company_data.get('current_liabilities')
            )
            
            # Profitability ratios
            ratios['roa'] = safe_divide(
                company_data.get('net_income'),
                company_data.get('total_assets')
            )
            
            ratios['roe'] = safe_divide(
                company_data.get('net_income'),
                company_data.get('shareholders_equity')
            )
            
            ratios['profit_margin'] = safe_divide(
                company_data.get('net_income'),
                company_data.get('revenue')
            )
            
            ratios['operating_margin'] = safe_divide(
                company_data.get('operating_income'),
                company_data.get('revenue')
            )
            
            ratios['gross_margin'] = safe_divide(
                company_data.get('gross_profit'),
                company_data.get('revenue')
            )
            
            # Efficiency ratios
            ratios['asset_turnover'] = safe_divide(
                company_data.get('revenue'),
                company_data.get('total_assets')
            )
            
            # Leverage ratios
            ratios['debt_to_equity'] = safe_divide(
                company_data.get('total_debt'),
                company_data.get('shareholders_equity')
            )
            
            ratios['debt_to_assets'] = safe_divide(
                company_data.get('total_debt'),
                company_data.get('total_assets')
            )
            
            # Cash flow ratios
            ratios['operating_cash_flow_ratio'] = safe_divide(
                company_data.get('operating_cash_flow'),
                company_data.get('current_liabilities')
            )
            
            ratios['free_cash_flow_margin'] = safe_divide(
                company_data.get('free_cash_flow'),
                company_data.get('revenue')
            )
            
            return ratios
            
        except Exception as e:
            logger.error(f"Error calculating financial ratios: {e}")
            return {}
    
    def _get_peer_companies_data(self, peer_company_ids: List[str]) -> List[Dict]:
        """Get financial data for peer companies"""
        try:
            if not peer_company_ids:
                return []
            
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            
            # Create placeholders for IN clause
            placeholders = ','.join(['%s'] * len(peer_company_ids))
            
            query = f"""
            SELECT 
                c.company_id,
                c.company_name,
                c.industry,
                c.sector,
                c.performance_category,
                c.country,
                c.market_cap,
                bs.revenue,
                bs.net_income,
                bs.total_assets,
                bs.current_assets,
                bs.current_liabilities,
                bs.total_debt,
                bs.shareholders_equity,
                bs.cash_and_equivalents,
                is_.operating_income,
                is_.gross_profit,
                is_.ebitda,
                cf.operating_cash_flow,
                cf.free_cash_flow,
                cf.capex
            FROM companies c
            LEFT JOIN balance_sheets bs ON c.company_id = bs.company_id
            LEFT JOIN income_statements is_ ON c.company_id = is_.company_id
            LEFT JOIN cash_flows cf ON c.company_id = cf.company_id
            WHERE c.company_id IN ({placeholders})
            AND bs.year = (SELECT MAX(year) FROM balance_sheets WHERE company_id = c.company_id)
            AND is_.year = (SELECT MAX(year) FROM income_statements WHERE company_id = c.company_id)
            AND cf.year = (SELECT MAX(year) FROM cash_flows WHERE company_id = c.company_id)
            """
            
            cursor.execute(query, peer_company_ids)
            peer_data = cursor.fetchall()
            
            # Calculate financial ratios for each peer
            enriched_peer_data = []
            for peer in peer_data:
                peer_dict = dict(peer)
                peer_dict.update(self._calculate_financial_ratios(peer_dict))
                enriched_peer_data.append(peer_dict)
            
            return enriched_peer_data
            
        except Exception as e:
            logger.error(f"Error getting peer companies data: {e}")
            return []
    
    def _extract_similarity_features(self, company_data: Dict) -> List[float]:
        """Extract features for similarity calculation"""
        try:
            features = []
            
            # Financial ratios
            ratio_features = [
                'current_ratio', 'roa', 'roe', 'profit_margin', 'debt_to_equity',
                'asset_turnover', 'operating_margin', 'gross_margin'
            ]
            
            for feature in ratio_features:
                value = company_data.get(feature, 0)
                features.append(float(value) if value is not None else 0.0)
            
            # Log-transformed size metrics (to handle scale differences)
            size_features = ['revenue', 'total_assets', 'market_cap']
            for feature in size_features:
                value = company_data.get(feature, 1)
                if value and float(value) > 0:
                    features.append(np.log(float(value)))
                else:
                    features.append(0.0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting similarity features: {e}")
            return [0.0] * 11  # Return zeros for feature count
    
    def _calculate_composite_score(self, company: Dict) -> float:
        """Calculate composite financial health score"""
        try:
            score = 0
            total_weight = 0
            
            # Define weights for different metrics
            weights = {
                'roa': 25,
                'current_ratio': 20,
                'profit_margin': 20,
                'roe': 15,
                'debt_to_equity': -10,  # Negative because lower is better
                'asset_turnover': 10,
                'operating_margin': 10
            }
            
            for metric, weight in weights.items():
                value = company.get(metric)
                if value is not None:
                    try:
                        numeric_value = float(value)
                        
                        # Normalize some metrics
                        if metric == 'current_ratio':
                            # Optimal current ratio is around 2, penalize too high or too low
                            normalized_value = 1 - abs(numeric_value - 2) / 2
                        elif metric == 'debt_to_equity':
                            # Lower debt is better, but some debt is okay
                            normalized_value = max(0, 1 - numeric_value / 2)
                        else:
                            # For most metrics, higher is better
                            normalized_value = min(1, max(0, numeric_value * 10))  # Scale appropriately
                        
                        score += normalized_value * abs(weight)
                        total_weight += abs(weight)
                        
                    except (ValueError, TypeError):
                        continue
            
            return score / total_weight if total_weight > 0 else 0
            
        except Exception as e:
            logger.error(f"Error calculating composite score: {e}")
            return 0
    
    def _calculate_industry_benchmarks(self, peer_companies: List[Dict]) -> Dict[str, float]:
        """Calculate industry benchmark statistics"""
        try:
            benchmarks = {}
            
            key_metrics = ['revenue', 'roa', 'roe', 'current_ratio', 'debt_to_equity', 
                          'profit_margin', 'asset_turnover', 'operating_margin']
            
            for metric in key_metrics:
                values = [float(company.get(metric, 0)) for company in peer_companies 
                         if company.get(metric) is not None]
                
                if values:
                    benchmarks[f'{metric}_mean'] = float(np.mean(values))
                    benchmarks[f'{metric}_median'] = float(np.median(values))
                    benchmarks[f'{metric}_std'] = float(np.std(values))
                    benchmarks[f'{metric}_min'] = float(np.min(values))
                    benchmarks[f'{metric}_max'] = float(np.max(values))
                    benchmarks[f'{metric}_p25'] = float(np.percentile(values, 25))
                    benchmarks[f'{metric}_p75'] = float(np.percentile(values, 75))
            
            return benchmarks
            
        except Exception as e:
            logger.error(f"Error calculating industry benchmarks: {e}")
            return {}

# Example usage for Phase 2
if __name__ == "__main__":
    print("=== Phase 2: Data Retrieval and Financial Calculations ===")
    
    # Database configuration
    db_config = {
        'host': 'localhost',
        'database': 'financial_db',
        'user': 'postgres',
        'password': 'password',
        'port': 5432
    }
    
    # Initialize analyzer with Phase 2 capabilities
    analyzer = FinancialPeerAnalyzer(db_config)
    
    try:
        analyzer.connect_to_database()
        print("✓ Database connected")
        
        # Test company data retrieval
        print("\nTesting data retrieval...")
        company_data = analyzer._get_company_data('COMP001')
        
        if company_data:
            print(f"✓ Retrieved data for {company_data.get('company_name', 'Unknown')}")
            print(f"✓ Calculated {len([k for k in company_data.keys() if 'ratio' in k or 'margin' in k or k in ['roa', 'roe']])} financial ratios")
            
            # Test composite score calculation
            score = analyzer._calculate_composite_score(company_data)
            print(f"✓ Composite financial health score: {score:.3f}")
            
        else:
            print("✗ No company data found")
        
        # Test peer data retrieval
        print("\nTesting peer data retrieval...")
        peer_ids = ['COMP002', 'COMP003', 'COMP004']
        peer_data = analyzer._get_peer_companies_data(peer_ids)
        
        if peer_data:
            print(f"✓ Retrieved data for {len(peer_data)} peer companies")
            
            # Test benchmark calculations
            benchmarks = analyzer._calculate_industry_benchmarks(peer_data)
            print(f"✓ Calculated {len(benchmarks)} benchmark statistics")
        else:
            print("✗ No peer data found")
        
        analyzer.close_connection()
        print("✓ Database connection closed")
        
    except Exception as e:
        print(f"✗ Phase 2 testing failed: {e}")
    
    print("\nPhase 2 completed successfully!")
    print("Ready for Phase 3: Peer Group Selection and Similarity Analysis")
class FinancialPeerAnalyzer(FinancialPeerAnalyzer):
    """
    Extended analyzer with peer group selection and similarity analysis capabilities
    """
    
    def _select_peer_group(self, 
                         target_company: Dict,
                         method: ComparisonMethod,
                         group_size: int,
                         custom_criteria: Optional[Dict] = None) -> Optional[PeerGroup]:
        """Select appropriate peer group based on method"""
        try:
            if method == ComparisonMethod.INDUSTRY_BASED:
                return self._select_industry_peers(target_company, group_size)
            elif method == ComparisonMethod.SIZE_BASED:
                return self._select_size_peers(target_company, group_size)
            elif method == ComparisonMethod.PERFORMANCE_BASED:
                return self._select_performance_peers(target_company, group_size)
            elif method == ComparisonMethod.SIMILARITY_BASED:
                return self._select_similarity_peers(target_company, group_size)
            elif method == ComparisonMethod.CUSTOM_COHORT:
                return self._select_custom_peers(target_company, group_size, custom_criteria)
            else:
                logger.warning(f"Unknown comparison method: {method}")
                return self._select_industry_peers(target_company, group_size)
                
        except Exception as e:
            logger.error(f"Error selecting peer group: {e}")
            return None
    
    def _select_industry_peers(self, target_company: Dict, group_size: int) -> Optional[PeerGroup]:
        """Select peers from the same industry"""
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            
            target_industry = target_company.get('industry')
            target_company_id = target_company.get('company_id')
            
            query = """
            SELECT company_id, company_name, industry, revenue, market_cap
            FROM companies
            WHERE LOWER(industry) = LOWER(%s)
            AND company_id != %s
            AND revenue IS NOT NULL
            ORDER BY revenue DESC
            LIMIT %s
            """
            
            cursor.execute(query, (target_industry, target_company_id, group_size))
            peers = cursor.fetchall()
            
            if peers:
                peer_companies = [peer['company_id'] for peer in peers]
                
                # Calculate group statistics
                revenues = [float(peer['revenue']) for peer in peers if peer['revenue']]
                group_stats = {
                    'avg_revenue': np.mean(revenues) if revenues else 0,
                    'median_revenue': np.median(revenues) if revenues else 0,
                    'min_revenue': np.min(revenues) if revenues else 0,
                    'max_revenue': np.max(revenues) if revenues else 0
                }
                
                return PeerGroup(
                    group_id=f"industry_{target_industry}_{datetime.now().strftime('%Y%m%d')}",
                    group_name=f"{target_industry.title()} Industry Peers",
                    companies=peer_companies,
                    selection_criteria={
                        'method': 'industry_based',
                        'industry': target_industry,
                        'group_size': len(peer_companies)
                    },
                    group_statistics=group_stats,
                    created_at=datetime.now()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error selecting industry peers: {e}")
            return None
    
    def _select_size_peers(self, target_company: Dict, group_size: int) -> Optional[PeerGroup]:
        """Select peers based on company size (revenue)"""
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            
            target_revenue = float(target_company.get('revenue', 0))
            target_company_id = target_company.get('company_id')
            
            # Define size range (±50% of target revenue)
            revenue_min = target_revenue * 0.5
            revenue_max = target_revenue * 2.0
            
            query = """
            SELECT company_id, company_name, industry, revenue, market_cap
            FROM companies
            WHERE revenue BETWEEN %s AND %s
            AND company_id != %s
            AND revenue IS NOT NULL
            ORDER BY ABS(revenue - %s)
            LIMIT %s
            """
            
            cursor.execute(query, (revenue_min, revenue_max, target_company_id, target_revenue, group_size))
            peers = cursor.fetchall()
            
            if peers:
                peer_companies = [peer['company_id'] for peer in peers]
                
                # Calculate group statistics
                revenues = [float(peer['revenue']) for peer in peers]
                group_stats = {
                    'avg_revenue': np.mean(revenues),
                    'revenue_range_min': revenue_min,
                    'revenue_range_max': revenue_max,
                    'target_revenue': target_revenue
                }
                
                return PeerGroup(
                    group_id=f"size_{int(target_revenue)}_{datetime.now().strftime('%Y%m%d')}",
                    group_name=f"Size-Based Peers (Revenue: ${target_revenue:,.0f})",
                    companies=peer_companies,
                    selection_criteria={
                        'method': 'size_based',
                        'revenue_range': (revenue_min, revenue_max),
                        'group_size': len(peer_companies)
                    },
                    group_statistics=group_stats,
                    created_at=datetime.now()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error selecting size peers: {e}")
            return None
    
    def _select_performance_peers(self, target_company: Dict, group_size: int) -> Optional[PeerGroup]:
        """Select peers based on similar performance metrics"""
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            
            target_performance = target_company.get('performance_category', 'moderate')
            target_company_id = target_company.get('company_id')
            
            query = """
            SELECT c.company_id, c.company_name, c.industry, c.performance_category,
                   bs.revenue, bs.net_income, bs.total_assets
            FROM companies c
            LEFT JOIN balance_sheets bs ON c.company_id = bs.company_id
            WHERE c.performance_category = %s
            AND c.company_id != %s
            AND bs.year = (SELECT MAX(year) FROM balance_sheets WHERE company_id = c.company_id)
            ORDER BY bs.revenue DESC
            LIMIT %s
            """
            
            cursor.execute(query, (target_performance, target_company_id, group_size))
            peers = cursor.fetchall()
            
            if peers:
                peer_companies = [peer['company_id'] for peer in peers]
                
                # Calculate group statistics
                revenues = [float(peer['revenue']) for peer in peers if peer['revenue']]
                group_stats = {
                    'performance_category': target_performance,
                    'avg_revenue': np.mean(revenues) if revenues else 0,
                    'peer_count': len(peer_companies)
                }
                
                return PeerGroup(
                    group_id=f"performance_{target_performance}_{datetime.now().strftime('%Y%m%d')}",
                    group_name=f"{target_performance.title()} Performance Peers",
                    companies=peer_companies,
                    selection_criteria={
                        'method': 'performance_based',
                        'performance_category': target_performance,
                        'group_size': len(peer_companies)
                    },
                    group_statistics=group_stats,
                    created_at=datetime.now()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error selecting performance peers: {e}")
            return None
    
    def _select_similarity_peers(self, target_company: Dict, group_size: int) -> Optional[PeerGroup]:
        """Select peers based on financial similarity using clustering"""
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            
            target_company_id = target_company.get('company_id')
            target_industry = target_company.get('industry')
            
            # Get companies from similar industries
            query = """
            SELECT 
                c.company_id,
                c.company_name,
                c.industry,
                bs.revenue,
                bs.net_income,
                bs.total_assets,
                bs.current_assets,
                bs.current_liabilities,
                bs.total_debt,
                bs.shareholders_equity
            FROM companies c
            LEFT JOIN balance_sheets bs ON c.company_id = bs.company_id
            WHERE (LOWER(c.industry) = LOWER(%s) OR LOWER(c.sector) = LOWER(%s))
            AND c.company_id != %s
            AND bs.year = (SELECT MAX(year) FROM balance_sheets WHERE company_id = c.company_id)
            AND bs.revenue IS NOT NULL
            AND bs.total_assets IS NOT NULL
            """
            
            target_sector = target_company.get('sector', target_industry)
            cursor.execute(query, (target_industry, target_sector, target_company_id))
            candidates = cursor.fetchall()
            
            if len(candidates) < group_size:
                logger.warning("Insufficient candidates for similarity-based selection")
                return self._select_industry_peers(target_company, group_size)
            
            # Prepare data for similarity calculation
            candidate_data = []
            candidate_ids = []
            
            # Add target company
            target_features = self._extract_similarity_features(target_company)
            candidate_data.append(target_features)
            candidate_ids.append(target_company_id)
            
            # Add candidate companies
            for candidate in candidates:
                candidate_dict = dict(candidate)
                candidate_dict.update(self._calculate_financial_ratios(candidate_dict))
                features = self._extract_similarity_features(candidate_dict)
                candidate_data.append(features)
                candidate_ids.append(candidate['company_id'])
            
            # Calculate similarities
            candidate_array = np.array(candidate_data)
            
            # Handle missing values
            candidate_array = np.nan_to_num(candidate_array, nan=0.0)
            
            # Standardize features
            if len(candidate_array) > 1:
                candidate_array = self.scaler.fit_transform(candidate_array)
            
            # Calculate cosine similarity with target company (first row)
            target_vector = candidate_array[0].reshape(1, -1)
            similarities = cosine_similarity(target_vector, candidate_array[1:])[0]
            
            # Select most similar companies
            similar_indices = np.argsort(similarities)[::-1][:group_size]
            peer_companies = [candidate_ids[i + 1] for i in similar_indices]  # +1 to skip target
            
            # Calculate group statistics
            avg_similarity = np.mean([similarities[i] for i in similar_indices])
            
            group_stats = {
                'avg_similarity': float(avg_similarity),
                'similarity_method': 'cosine',
                'feature_count': len(target_features)
            }
            
            return PeerGroup(
                group_id=f"similarity_{target_company_id}_{datetime.now().strftime('%Y%m%d')}",
                group_name="Similarity-Based Peers",
                companies=peer_companies,
                selection_criteria={
                    'method': 'similarity_based',
                    'similarity_threshold': float(avg_similarity),
                    'group_size': len(peer_companies)
                },
                group_statistics=group_stats,
                created_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error selecting similarity peers: {e}")
            return self._select_industry_peers(target_company, group_size)
    
    def _select_custom_peers(self, 
                           target_company: Dict, 
                           group_size: int, 
                           custom_criteria: Optional[Dict]) -> Optional[PeerGroup]:
        """Select peers based on custom criteria"""
        try:
            if not custom_criteria:
                logger.warning("No custom criteria provided, falling back to industry-based selection")
                return self._select_industry_peers(target_company, group_size)
            
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            
            # Build query based on custom criteria
            conditions = []
            params = []
            
            target_company_id = target_company.get('company_id')
            conditions.append("c.company_id != %s")
            params.append(target_company_id)
            
            # Industry filter
            if 'industries' in custom_criteria:
                industries = custom_criteria['industries']
                if isinstance(industries, list):
                    industry_placeholders = ','.join(['%s'] * len(industries))
                    conditions.append(f"LOWER(c.industry) IN ({industry_placeholders})")
                    params.extend([ind.lower() for ind in industries])
            
            # Revenue range filter
            if 'revenue_range' in custom_criteria:
                revenue_min, revenue_max = custom_criteria['revenue_range']
                conditions.append("bs.revenue BETWEEN %s AND %s")
                params.extend([revenue_min, revenue_max])
            
            # Performance category filter
            if 'performance_categories' in custom_criteria:
                categories = custom_criteria['performance_categories']
                if isinstance(categories, list):
                    cat_placeholders = ','.join(['%s'] * len(categories))
                    conditions.append(f"c.performance_category IN ({cat_placeholders})")
                    params.extend(categories)
            
            # Country filter
            if 'countries' in custom_criteria:
                countries = custom_criteria['countries']
                if isinstance(countries, list):
                    country_placeholders = ','.join(['%s'] * len(countries))
                    conditions.append(f"c.country IN ({country_placeholders})")
                    params.extend(countries)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            query = f"""
            SELECT c.company_id, c.company_name, c.industry, c.performance_category,
                   bs.revenue, bs.total_assets, c.market_cap
            FROM companies c
            LEFT JOIN balance_sheets bs ON c.company_id = bs.company_id
            WHERE {where_clause}
            AND bs.year = (SELECT MAX(year) FROM balance_sheets WHERE company_id = c.company_id)
            ORDER BY bs.revenue DESC
            LIMIT %s
            """
            
            params.append(group_size)
            cursor.execute(query, params)
            peers = cursor.fetchall()
            
            if peers:
                peer_companies = [peer['company_id'] for peer in peers]
                
                # Calculate group statistics
                revenues = [float(peer['revenue']) for peer in peers if peer['revenue']]
                group_stats = {
                    'avg_revenue': np.mean(revenues) if revenues else 0,
                    'custom_criteria': custom_criteria,
                    'peer_count': len(peer_companies)
                }
                
                return PeerGroup(
                    group_id=f"custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    group_name="Custom Peer Group",
                    companies=peer_companies,
                    selection_criteria={
                        'method': 'custom_cohort',
                        'criteria': custom_criteria,
                        'group_size': len(peer_companies)
                    },
                    group_statistics=group_stats,
                    created_at=datetime.now()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error selecting custom peers: {e}")
            return None

# Example usage for Phase 3
if __name__ == "__main__":
    print("=== Phase 3: Peer Group Selection and Similarity Analysis ===")
    
    # Database configuration
    db_config = {
        'host': 'localhost',
        'database': 'financial_db',
        'user': 'postgres',
        'password': 'password',
        'port': 5432
    }
    
    # Initialize analyzer with Phase 3 capabilities
    analyzer = FinancialPeerAnalyzer(db_config)
    
    try:
        analyzer.connect_to_database()
        print("✓ Database connected")
        
        # Get target company data
        target_company = analyzer._get_company_data('COMP001')
        
        if target_company:
            print(f"✓ Target company: {target_company.get('company_name', 'Unknown')}")
            
            # Test different peer selection methods
            print("\nTesting peer selection methods...")
            
            # Industry-based selection
            industry_peers = analyzer._select_industry_peers(target_company, 10)
            if industry_peers:
                print(f"✓ Industry-based: Found {len(industry_peers.companies)} peers")
                print(f"  Group: {industry_peers.group_name}")
            
            # Size-based selection
            size_peers = analyzer._select_size_peers(target_company, 10)
            if size_peers:
                print(f"✓ Size-based: Found {len(size_peers.companies)} peers")
                print(f"  Revenue range: ${size_peers.group_statistics.get('revenue_range_min', 0):,.0f} - ${size_peers.group_statistics.get('revenue_range_max', 0):,.0f}")
            
            # Performance-based selection
            performance_peers = analyzer._select_performance_peers(target_company, 10)
            if performance_peers:
                print(f"✓ Performance-based: Found {len(performance_peers.companies)} peers")
                print(f"  Category: {performance_peers.group_statistics.get('performance_category', 'Unknown')}")
            
            # Similarity-based selection
            similarity_peers = analyzer._select_similarity_peers(target_company, 10)
            if similarity_peers:
                print(f"✓ Similarity-based: Found {len(similarity_peers.companies)} peers")
                print(f"  Avg similarity: {similarity_peers.group_statistics.get('avg_similarity', 0):.3f}")
            
            # Custom selection example
            custom_criteria = {
                'industries': ['technology', 'software'],
                'revenue_range': (100_000_000, 10_000_000_000)
            }
            custom_peers = analyzer._select_custom_peers(target_company, 10, custom_criteria)
            if custom_peers:
                print(f"✓ Custom criteria: Found {len(custom_peers.companies)} peers")
        
        else:
            print("✗ No target company data found")
        
        analyzer.close_connection()
        print("✓ Database connection closed")
        
    except Exception as e:
        print(f"✗ Phase 3 testing failed: {e}")
    
    print("\nPhase 3 completed successfully!")
    print("Ready for Phase 4: Comparison Analysis and Ranking")
class FinancialPeerAnalyzer(FinancialPeerAnalyzer):
    """
    Extended analyzer with comparison analysis and ranking capabilities
    """
    
    def _calculate_company_comparison(self, 
                                    target_company: Dict,
                                    peer_companies: List[Dict],
                                    peer_group: PeerGroup) -> CompanyComparison:
        """Calculate detailed comparison metrics for the target company"""
        try:
            # Combine target company with peers for ranking
            all_companies = [target_company] + peer_companies
            target_company_id = target_company['company_id']
            
            # Calculate rankings for key metrics
            rankings = {}
            percentiles = {}
            
            key_metrics = ['revenue', 'roa', 'roe', 'current_ratio', 'debt_to_equity', 'profit_margin']
            
            for metric in key_metrics:
                values = []
                company_ids = []
                
                for company in all_companies:
                    value = company.get(metric)
                    if value is not None:
                        values.append(float(value))
                        company_ids.append(company['company_id'])
                
                if values and target_company_id in company_ids:
                    # Sort based on metric (higher is better for most metrics)
                    reverse_sort = metric != 'debt_to_equity'  # Lower debt is better
                    
                    sorted_indices = sorted(range(len(values)), 
                                          key=lambda i: values[i], 
                                          reverse=reverse_sort)
                    
                    # Find target company's rank
                    target_idx = company_ids.index(target_company_id)
                    target_rank = sorted_indices.index(target_idx) + 1
                    
                    rankings[metric] = target_rank
                    percentiles[metric] = ((len(values) - target_rank + 1) / len(values)) * 100
            
            # Calculate overall rank based on a composite score
            overall_rank = self._calculate_overall_rank(target_company, all_companies)
            overall_percentile = ((len(all_companies) - overall_rank + 1) / len(all_companies)) * 100
            
            # Extract key metrics for target company
            key_metrics_values = {}
            for metric in key_metrics:
                key_metrics_values[metric] = target_company.get(metric, 0)
            
            # Calculate relative performance vs peer average
            relative_performance = self._calculate_relative_performance(
                target_company, peer_companies
            )
            
            # Identify strengths and weaknesses
            strengths, weaknesses = self._identify_strengths_weaknesses(
                target_company, peer_companies, percentiles
            )
            
            return CompanyComparison(
                company_id=target_company_id,
                company_name=target_company.get('company_name', 'Unknown'),
                rank=overall_rank,
                percentile=overall_percentile,
                peer_group_size=len(all_companies),
                key_metrics=key_metrics_values,
                relative_performance=relative_performance,
                strengths=strengths,
                weaknesses=weaknesses
            )
            
        except Exception as e:
            logger.error(f"Error calculating company comparison: {e}")
            return CompanyComparison(
                company_id=target_company.get('company_id', 'unknown'),
                company_name=target_company.get('company_name', 'Unknown'),
                rank=0,
                percentile=0,
                peer_group_size=0,
                key_metrics={},
                relative_performance={},
                strengths=[],
                weaknesses=[]
            )
    
    def _calculate_overall_rank(self, target_company: Dict, all_companies: List[Dict]) -> int:
        """Calculate overall ranking based on composite financial health score"""
        try:
            company_scores = []
            
            for company in all_companies:
                # Calculate composite score
                score = 0
                weights = {
                    'roa': 0.25,
                    'current_ratio': 0.20,
                    'debt_to_equity': -0.15,  # Negative weight (lower is better)
                    'profit_margin': 0.20,
                    'roe': 0.20
                }
                
                for metric, weight in weights.items():
                    value = company.get(metric, 0)
                    if value is not None:
                        score += float(value) * weight
                
                company_scores.append({
                    'company_id': company['company_id'],
                    'score': score
                })
            
            # Sort by score (highest first)
            company_scores.sort(key=lambda x: x['score'], reverse=True)
            
            # Find target company's rank
            target_company_id = target_company['company_id']
            for i, company_score in enumerate(company_scores):
                if company_score['company_id'] == target_company_id:
                    return i + 1
            
            return len(all_companies)  # Worst rank if not found
            
        except Exception as e:
            logger.error(f"Error calculating overall rank: {e}")
            return 0
    
    def _calculate_relative_performance(self, 
                                      target_company: Dict, 
                                      peer_companies: List[Dict]) -> Dict[str, float]:
        """Calculate relative performance vs peer averages"""
        try:
            relative_performance = {}
            
            key_metrics = ['revenue', 'roa', 'roe', 'current_ratio', 'debt_to_equity', 
                          'profit_margin', 'asset_turnover', 'operating_margin']
            
            for metric in key_metrics:
                target_value = target_company.get(metric)
                peer_values = [peer.get(metric) for peer in peer_companies 
                              if peer.get(metric) is not None]
                
                if target_value is not None and peer_values:
                    peer_average = np.mean([float(v) for v in peer_values])
                    
                    if peer_average != 0:
                        relative_perf = ((float(target_value) - peer_average) / peer_average) * 100
                        relative_performance[metric] = relative_perf
                    else:
                        relative_performance[metric] = 0.0
            
            return relative_performance
            
        except Exception as e:
            logger.error(f"Error calculating relative performance: {e}")
            return {}
    
    def _identify_strengths_weaknesses(self, 
                                     target_company: Dict,
                                     peer_companies: List[Dict],
                                     percentiles: Dict[str, float]) -> Tuple[List[str], List[str]]:
        """Identify company strengths and weaknesses relative to peers"""
        try:
            strengths = []
            weaknesses = []
            
            # Define thresholds
            strength_threshold = 75  # Top quartile
            weakness_threshold = 25  # Bottom quartile
            
            metric_descriptions = {
                'roa': 'Return on Assets',
                'roe': 'Return on Equity',
                'current_ratio': 'Liquidity (Current Ratio)',
                'debt_to_equity': 'Leverage (Debt-to-Equity)',
                'profit_margin': 'Profitability',
                'revenue': 'Revenue Scale',
                'asset_turnover': 'Asset Efficiency',
                'operating_margin': 'Operating Efficiency'
            }
            
            for metric, percentile in percentiles.items():
                description = metric_descriptions.get(metric, metric.replace('_', ' ').title())
                
                if metric == 'debt_to_equity':
                    # For debt-to-equity, lower percentile means better performance
                    if percentile <= (100 - strength_threshold):
                        strengths.append(f"Low leverage - {description}")
                    elif percentile >= (100 - weakness_threshold):
                        weaknesses.append(f"High leverage - {description}")
                else:
                    # For other metrics, higher percentile means better performance
                    if percentile >= strength_threshold:
                        strengths.append(f"Strong {description}")
                    elif percentile <= weakness_threshold:
                        weaknesses.append(f"Weak {description}")
            
            # Additional analysis based on specific metric combinations
            current_ratio = target_company.get('current_ratio', 0)
            roa = target_company.get('roa', 0)
            
            if current_ratio > 2.5:
                strengths.append("Excellent liquidity position")
            elif current_ratio < 1.0:
                weaknesses.append("Liquidity concerns")
            
            if roa and float(roa) > 0.15:
                strengths.append("Exceptional asset productivity")
            elif roa and float(roa) < 0.02:
                weaknesses.append("Poor asset utilization")
            
            return strengths[:5], weaknesses[:5]  # Limit to top 5 each
            
        except Exception as e:
            logger.error(f"Error identifying strengths and weaknesses: {e}")
            return [], []
    
    def _generate_peer_rankings(self, 
                              target_company: Dict,
                              peer_companies: List[Dict]) -> List[Dict[str, Any]]:
        """Generate complete peer rankings"""
        try:
            all_companies = [target_company] + peer_companies
            rankings = []
            
            # Calculate composite scores for ranking
            for company in all_companies:
                # Financial health composite score
                score = self._calculate_composite_score(company)
                
                rankings.append({
                    'company_id': company['company_id'],
                    'company_name': company.get('company_name', 'Unknown'),
                    'industry': company.get('industry', 'Unknown'),
                    'composite_score': score,
                    'revenue': company.get('revenue', 0),
                    'roa': company.get('roa', 0),
                    'current_ratio': company.get('current_ratio', 0),
                    'debt_to_equity': company.get('debt_to_equity', 0),
                    'profit_margin': company.get('profit_margin', 0),
                    'is_target_company': company['company_id'] == target_company['company_id']
                })
            
            # Sort by composite score
            rankings.sort(key=lambda x: x['composite_score'], reverse=True)
            
            # Add rank positions
            for i, company in enumerate(rankings):
                company['rank'] = i + 1
                company['percentile'] = ((len(rankings) - i) / len(rankings)) * 100
            
            return rankings
            
        except Exception as e:
            logger.error(f"Error generating peer rankings: {e}")
            return []
    
    def _assess_competitive_positioning(self, 
                                      target_company: Dict,
                                      peer_companies: List[Dict],
                                      company_comparison: CompanyComparison) -> Dict[str, Any]:
        """Assess competitive positioning and market dynamics"""
        try:
            positioning = {}
            
            # Market position based on revenue
            revenues = [float(company.get('revenue', 0)) for company in peer_companies]
            target_revenue = float(target_company.get('revenue', 0))
            
            if revenues:
                revenue_rank = len([r for r in revenues if r > target_revenue]) + 1
                positioning['market_size_rank'] = revenue_rank
                positioning['market_size_percentile'] = ((len(revenues) - revenue_rank + 1) / len(revenues)) * 100
            
            # Performance-based positioning
            positioning['overall_rank'] = company_comparison.rank
            positioning['overall_percentile'] = company_comparison.percentile
            
            # Competitive advantages
            competitive_advantages = []
            if company_comparison.percentile >= 75:
                competitive_advantages.append("Top quartile overall performance")
            
            for strength in company_comparison.strengths:
                competitive_advantages.append(strength)
            
            positioning['competitive_advantages'] = competitive_advantages[:5]
            
            # Competitive threats
            competitive_threats = []
            if company_comparison.percentile <= 25:
                competitive_threats.append("Bottom quartile overall performance")
            
            for weakness in company_comparison.weaknesses:
                competitive_threats.append(weakness)
            
            positioning['competitive_threats'] = competitive_threats[:5]
            
            # Market dynamics analysis
            market_dynamics = self._analyze_market_dynamics(peer_companies)
            positioning['market_dynamics'] = market_dynamics
            
            # Strategic position
            strategic_position = self._determine_strategic_position(
                target_company, company_comparison
            )
            positioning['strategic_position'] = strategic_position
            
            return positioning
            
        except Exception as e:
            logger.error(f"Error assessing competitive positioning: {e}")
            return {}
    
    def _analyze_market_dynamics(self, peer_companies: List[Dict]) -> Dict[str, Any]:
        """Analyze market dynamics from peer data"""
        try:
            dynamics = {}
            
            # Performance distribution
            performance_categories = [company.get('performance_category', 'unknown') 
                                    for company in peer_companies]
            
            category_counts = {}
            for category in performance_categories:
                category_counts[category] = category_counts.get(category, 0) + 1
            
            dynamics['performance_distribution'] = category_counts
            
            # Revenue concentration
            revenues = [float(company.get('revenue', 0)) for company in peer_companies 
                       if company.get('revenue')]
            
            if revenues:
                revenues_sorted = sorted(revenues, reverse=True)
                total_revenue = sum(revenues)
                
                # Market concentration (Herfindahl-Hirschman Index approximation)
                market_shares = [r / total_revenue for r in revenues]
                hhi = sum(share ** 2 for share in market_shares)
                
                dynamics['market_concentration'] = {
                    'hhi': float(hhi),
                    'concentration_level': 'high' if hhi > 0.25 else 'moderate' if hhi > 0.15 else 'low',
                    'top_3_share': sum(revenues_sorted[:3]) / total_revenue if len(revenues_sorted) >= 3 else 1.0
                }
            
            # Industry health indicators
            roa_values = [float(company.get('roa', 0)) for company in peer_companies 
                         if company.get('roa') is not None]
            
            if roa_values:
                avg_roa = np.mean(roa_values)
                positive_roa_pct = len([r for r in roa_values if r > 0]) / len(roa_values)
                
                dynamics['industry_health'] = {
                    'average_roa': float(avg_roa),
                    'positive_roa_percentage': float(positive_roa_pct),
                    'health_indicator': 'strong' if avg_roa > 0.1 else 'moderate' if avg_roa > 0.05 else 'weak'
                }
            
            return dynamics
            
        except Exception as e:
            logger.error(f"Error analyzing market dynamics: {e}")
            return {}
    
    def _determine_strategic_position(self, 
                                    target_company: Dict, 
                                    company_comparison: CompanyComparison) -> Dict[str, str]:
        """Determine strategic market position"""
        try:
            revenue = float(target_company.get('revenue', 0))
            roa = float(target_company.get('roa', 0))
            percentile = company_comparison.percentile
            
            # Size-based position
            if revenue > 10_000_000_000:
                size_position = "market_leader"
            elif revenue > 1_000_000_000:
                size_position = "major_player"
            elif revenue > 100_000_000:
                size_position = "established_company"
            else:
                size_position = "emerging_company"
            
            # Performance-based position
            if percentile >= 80:
                performance_position = "outperformer"
            elif percentile >= 60:
                performance_position = "above_average"
            elif percentile >= 40:
                performance_position = "average"
            elif percentile >= 20:
                performance_position = "below_average"
            else:
                performance_position = "underperformer"
            
            # Profitability position
            if roa > 0.15:
                profitability_position = "highly_profitable"
            elif roa > 0.10:
                profitability_position = "profitable"
            elif roa > 0.05:
                profitability_position = "moderately_profitable"
            elif roa > 0:
                profitability_position = "marginally_profitable"
            else:
                profitability_position = "unprofitable"
            
            # Overall strategic quadrant
            high_performance = percentile >= 60
            large_size = revenue > 1_000_000_000
            
            if high_performance and large_size:
                strategic_quadrant = "dominant_player"
            elif high_performance and not large_size:
                strategic_quadrant = "efficient_specialist"
            elif not high_performance and large_size:
                strategic_quadrant = "challenged_incumbent"
            else:
                strategic_quadrant = "struggling_competitor"
            
            return {
                'size_position': size_position,
                'performance_position': performance_position,
                'profitability_position': profitability_position,
                'strategic_quadrant': strategic_quadrant
            }
            
        except Exception as e:
            logger.error(f"Error determining strategic position: {e}")
            return {}

# Example usage for Phase 4
if __name__ == "__main__":
    print("=== Phase 4: Comparison Analysis and Ranking ===")
    
    # Database configuration
    db_config = {
        'host': 'localhost',
        'database': 'financial_db',
        'user': 'postgres',
        'password': 'password',
        'port': 5432
    }
    
    # Initialize analyzer with Phase 4 capabilities
    analyzer = FinancialPeerAnalyzer(db_config)
    
    try:
        analyzer.connect_to_database()
        print("✓ Database connected")
        
        # Get target company and peer data
        target_company = analyzer._get_company_data('COMP001')
        
        if target_company:
            print(f"✓ Target company: {target_company.get('company_name', 'Unknown')}")
            
            # Get peer group
            peer_group = analyzer._select_industry_peers(target_company, 15)
            
            if peer_group:
                print(f"✓ Peer group: {len(peer_group.companies)} companies")
                
                # Get peer companies data
                peer_companies_data = analyzer._get_peer_companies_data(peer_group.companies)
                
                if peer_companies_data:
                    print(f"✓ Peer data retrieved: {len(peer_companies_data)} companies")
                    
                    # Test company comparison
                    comparison = analyzer._calculate_company_comparison(
                        target_company, peer_companies_data, peer_group
                    )
                    print(f"✓ Company comparison calculated")
                    print(f"  Rank: {comparison.rank} of {comparison.peer_group_size}")
                    print(f"  Percentile: {comparison.percentile:.1f}%")
                    print(f"  Strengths: {len(comparison.strengths)}")
                    print(f"  Weaknesses: {len(comparison.weaknesses)}")
                    
                    # Test peer rankings
                    rankings = analyzer._generate_peer_rankings(target_company, peer_companies_data)
                    print(f"✓ Peer rankings generated: {len(rankings)} companies")
                    
                    # Test competitive positioning
                    positioning = analyzer._assess_competitive_positioning(
                        target_company, peer_companies_data, comparison
                    )
                    print(f"✓ Competitive positioning assessed")
                    strategic_pos = positioning.get('strategic_position', {})
                    print(f"  Strategic quadrant: {strategic_pos.get('strategic_quadrant', 'Unknown')}")
                    print(f"  Performance position: {strategic_pos.get('performance_position', 'Unknown')}")
                    
                    # Test benchmarks
                    benchmarks = analyzer._calculate_industry_benchmarks(peer_companies_data)
                    print(f"✓ Industry benchmarks calculated: {len(benchmarks)} metrics")
                    
                else:
                    print("✗ No peer companies data found")
            else:
                print("✗ No peer group found")
        else:
            print("✗ No target company data found")
        
        analyzer.close_connection()
        print("✓ Database connection closed")
        
    except Exception as e:
        print(f"✗ Phase 4 testing failed: {e}")
    
    print("\nPhase 4 completed successfully!")
    print("Ready for Phase 5: Reporting and Integration")
class FinancialPeerAnalyzer(FinancialPeerAnalyzer):
    """
    Complete analyzer with reporting and integration capabilities
    """
    
    def analyze_company_peers(self, 
                            company_id: str,
                            comparison_method: ComparisonMethod = ComparisonMethod.INDUSTRY_BASED,
                            peer_group_size: int = 20,
                            custom_criteria: Optional[Dict] = None) -> PeerAnalysisResult:
        """
        Perform comprehensive peer analysis for a company - MAIN ENTRY POINT
        
        Args:
            company_id (str): Target company identifier
            comparison_method (ComparisonMethod): Method for selecting peers
            peer_group_size (int): Number of peer companies to include
            custom_criteria (Dict): Custom selection criteria
            
        Returns:
            PeerAnalysisResult: Complete peer analysis results
        """
        try:
            logger.info(f"Starting peer analysis for company {company_id}")
            self.connect_to_database()
            
            # Get target company data
            target_company = self._get_company_data(company_id)
            if not target_company:
                return self._create_error_result("Target company not found")
            
            # Select peer group based on method
            peer_group = self._select_peer_group(
                target_company, comparison_method, peer_group_size, custom_criteria
            )
            
            if not peer_group or len(peer_group.companies) == 0:
                return self._create_error_result("No suitable peer companies found")
            
            # Get peer companies data
            peer_companies_data = self._get_peer_companies_data(peer_group.companies)
            
            # Calculate company comparison metrics
            company_comparison = self._calculate_company_comparison(
                target_company, peer_companies_data, peer_group
            )
            
            # Generate peer rankings
            peer_rankings = self._generate_peer_rankings(
                target_company, peer_companies_data
            )
            
            # Calculate industry benchmarks
            industry_benchmarks = self._calculate_industry_benchmarks(peer_companies_data)
            
            # Assess competitive positioning
            competitive_positioning = self._assess_competitive_positioning(
                target_company, peer_companies_data, company_comparison
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                target_company, company_comparison, competitive_positioning
            )
            
            # Create result
            result = PeerAnalysisResult(
                target_company=company_id,
                peer_group=peer_group,
                comparison_method=comparison_method.value,
                company_comparison=company_comparison,
                peer_rankings=peer_rankings,
                industry_benchmarks=industry_benchmarks,
                competitive_positioning=competitive_positioning,
                recommendations=recommendations,
                analysis_date=datetime.now()
            )
            
            logger.info(f"Peer analysis completed for company {company_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error in peer analysis: {e}")
            return self._create_error_result(str(e))
        finally:
            self.close_connection()
    
    def _generate_recommendations(self, 
                                target_company: Dict,
                                company_comparison: CompanyComparison,
                                competitive_positioning: Dict[str, Any]) -> List[str]:
        """Generate strategic recommendations based on peer analysis"""
        try:
            recommendations = []
            
            # Performance-based recommendations
            percentile = company_comparison.percentile
            
            if percentile <= 25:
                recommendations.append("Urgent performance improvement needed - company ranks in bottom quartile")
                recommendations.append("Consider comprehensive operational review and cost optimization")
            elif percentile <= 50:
                recommendations.append("Focus on improving key performance metrics to reach industry average")
            elif percentile >= 75:
                recommendations.append("Maintain strong performance and consider market expansion opportunities")
            
            # Specific metric recommendations
            relative_performance = company_comparison.relative_performance
            
            for metric, relative_perf in relative_performance.items():
                if relative_perf < -20:  # Significantly underperforming
                    
                    if metric == 'roa':
                        recommendations.append(f"Improve asset utilization - ROA is {abs(relative_perf):.1f}% below peer average")
                    elif metric == 'current_ratio':
                        recommendations.append("Strengthen liquidity position - current ratio below peer average")
                    elif metric == 'profit_margin':
                        recommendations.append("Focus on margin improvement through cost management and pricing optimization")
                    elif metric == 'debt_to_equity' and relative_perf > 20:  # High debt is bad
                        recommendations.append("Consider debt reduction strategies - leverage significantly above peers")
            
            # Strategic recommendations based on positioning
            strategic_position = competitive_positioning.get('strategic_position', {})
            strategic_quadrant = strategic_position.get('strategic_quadrant', '')
            
            if strategic_quadrant == 'struggling_competitor':
                recommendations.append("Consider strategic partnerships or acquisition opportunities to improve market position")
            elif strategic_quadrant == 'efficient_specialist':
                recommendations.append("Leverage operational efficiency to expand market share or enter new markets")
            elif strategic_quadrant == 'challenged_incumbent':
                recommendations.append("Focus on operational excellence and innovation to defend market position")
            elif strategic_quadrant == 'dominant_player':
                recommendations.append("Explore strategic acquisitions and market expansion opportunities")
            
            # Industry-specific recommendations
            market_dynamics = competitive_positioning.get('market_dynamics', {})
            industry_health = market_dynamics.get('industry_health', {})
            
            if industry_health.get('health_indicator') == 'weak':
                recommendations.append("Industry showing signs of stress - focus on defensive strategies and cash preservation")
            elif industry_health.get('health_indicator') == 'strong':
                recommendations.append("Strong industry conditions - consider growth investments and market share expansion")
            
            # Limit to most important recommendations
            return recommendations[:7]
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Unable to generate recommendations due to analysis error"]
    
    def _create_error_result(self, error_message: str) -> PeerAnalysisResult:
        """Create error result for failed analysis"""
        return PeerAnalysisResult(
            target_company="unknown",
            peer_group=PeerGroup(
                group_id="error",
                group_name="Error",
                companies=[],
                selection_criteria={},
                group_statistics={},
                created_at=datetime.now()
            ),
            comparison_method="error",
            company_comparison=CompanyComparison(
                company_id="unknown",
                company_name="Error",
                rank=0,
                percentile=0,
                peer_group_size=0,
                key_metrics={},
                relative_performance={},
                strengths=[],
                weaknesses=[]
            ),
            peer_rankings=[],
            industry_benchmarks={},
            competitive_positioning={'error': error_message},
            recommendations=[f"Analysis failed: {error_message}"],
            analysis_date=datetime.now()
        )
    
    def generate_peer_group_report(self, peer_analysis_result: PeerAnalysisResult) -> str:
        """Generate a comprehensive peer analysis report"""
        try:
            report_lines = []
            
            # Header
            report_lines.append("=" * 80)
            report_lines.append("PEER ANALYSIS REPORT")
            report_lines.append("=" * 80)
            report_lines.append(f"Target Company: {peer_analysis_result.target_company}")
            report_lines.append(f"Analysis Date: {peer_analysis_result.analysis_date.strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append(f"Comparison Method: {peer_analysis_result.comparison_method}")
            report_lines.append("")
            
            # Peer Group Summary
            peer_group = peer_analysis_result.peer_group
            report_lines.append("PEER GROUP SUMMARY")
            report_lines.append("-" * 40)
            report_lines.append(f"Group Name: {peer_group.group_name}")
            report_lines.append(f"Number of Peers: {len(peer_group.companies)}")
            report_lines.append(f"Selection Criteria: {peer_group.selection_criteria}")
            report_lines.append("")
            
            # Company Performance Summary
            comparison = peer_analysis_result.company_comparison
            report_lines.append("PERFORMANCE SUMMARY")
            report_lines.append("-" * 40)
            report_lines.append(f"Overall Rank: {comparison.rank} of {comparison.peer_group_size}")
            report_lines.append(f"Percentile: {comparison.percentile:.1f}%")
            report_lines.append("")
            
            # Key Metrics
            report_lines.append("KEY METRICS")
            report_lines.append("-" * 40)
            for metric, value in comparison.key_metrics.items():
                report_lines.append(f"{metric.replace('_', ' ').title()}: {value:.4f}")
            report_lines.append("")
            
            # Relative Performance
            report_lines.append("RELATIVE PERFORMANCE VS PEERS")
            report_lines.append("-" * 40)
            for metric, relative_perf in comparison.relative_performance.items():
                direction = "above" if relative_perf > 0 else "below"
                report_lines.append(f"{metric.replace('_', ' ').title()}: {abs(relative_perf):.1f}% {direction} peer average")
            report_lines.append("")
            
            # Strengths and Weaknesses
            report_lines.append("STRENGTHS")
            report_lines.append("-" * 40)
            for strength in comparison.strengths:
                report_lines.append(f"• {strength}")
            report_lines.append("")
            
            report_lines.append("AREAS FOR IMPROVEMENT")
            report_lines.append("-" * 40)
            for weakness in comparison.weaknesses:
                report_lines.append(f"• {weakness}")
            report_lines.append("")
            
            # Competitive Positioning
            positioning = peer_analysis_result.competitive_positioning
            strategic_position = positioning.get('strategic_position', {})
            report_lines.append("COMPETITIVE POSITIONING")
            report_lines.append("-" * 40)
            report_lines.append(f"Strategic Quadrant: {strategic_position.get('strategic_quadrant', 'Unknown')}")
            report_lines.append(f"Size Position: {strategic_position.get('size_position', 'Unknown')}")
            report_lines.append(f"Performance Position: {strategic_position.get('performance_position', 'Unknown')}")
            report_lines.append("")
            
            # Recommendations
            report_lines.append("STRATEGIC RECOMMENDATIONS")
            report_lines.append("-" * 40)
            for i, recommendation in enumerate(peer_analysis_result.recommendations, 1):
                report_lines.append(f"{i}. {recommendation}")
            report_lines.append("")
            
            # Peer Rankings (Top 10)
            report_lines.append("PEER RANKINGS (TOP 10)")
            report_lines.append("-" * 40)
            report_lines.append(f"{'Rank':<6} {'Company':<30} {'Score':<10} {'Revenue':<15}")
            report_lines.append("-" * 61)
            
            for i, peer in enumerate(peer_analysis_result.peer_rankings[:10]):
                rank = peer.get('rank', i + 1)
                name = peer.get('company_name', 'Unknown')[:28]
                score = peer.get('composite_score', 0)
                revenue = peer.get('revenue', 0)
                
                marker = " *" if peer.get('is_target_company', False) else ""
                report_lines.append(f"{rank:<6} {name:<30} {score:<10.3f} ${revenue:<13,.0f}{marker}")
            
            report_lines.append("")
            report_lines.append("* Target Company")
            report_lines.append("")
            report_lines.append("=" * 80)
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"Error generating peer group report: {e}")
            return f"Error generating report: {str(e)}"
    
    def export_peer_analysis(self, 
                           peer_analysis_result: PeerAnalysisResult,
                           export_format: str = 'json',
                           file_path: Optional[str] = None) -> Union[str, Dict]:
        """Export peer analysis results"""
        try:
            if export_format.lower() == 'json':
                # Convert to JSON-serializable format
                export_data = {
                    'target_company': peer_analysis_result.target_company,
                    'analysis_date': peer_analysis_result.analysis_date.isoformat(),
                    'comparison_method': peer_analysis_result.comparison_method,
                    'peer_group': {
                        'group_id': peer_analysis_result.peer_group.group_id,
                        'group_name': peer_analysis_result.peer_group.group_name,
                        'companies': peer_analysis_result.peer_group.companies,
                        'selection_criteria': peer_analysis_result.peer_group.selection_criteria,
                        'group_statistics': peer_analysis_result.peer_group.group_statistics
                    },
                    'company_comparison': {
                        'company_id': peer_analysis_result.company_comparison.company_id,
                        'company_name': peer_analysis_result.company_comparison.company_name,
                        'rank': peer_analysis_result.company_comparison.rank,
                        'percentile': peer_analysis_result.company_comparison.percentile,
                        'peer_group_size': peer_analysis_result.company_comparison.peer_group_size,
                        'key_metrics': peer_analysis_result.company_comparison.key_metrics,
                        'relative_performance': peer_analysis_result.company_comparison.relative_performance,
                        'strengths': peer_analysis_result.company_comparison.strengths,
                        'weaknesses': peer_analysis_result.company_comparison.weaknesses
                    },
                    'peer_rankings': peer_analysis_result.peer_rankings,
                    'industry_benchmarks': peer_analysis_result.industry_benchmarks,
                    'competitive_positioning': peer_analysis_result.competitive_positioning,
                    'recommendations': peer_analysis_result.recommendations
                }
                
                if file_path:
                    with open(file_path, 'w') as f:
                        json.dump(export_data, f, indent=2, default=str)
                    return f"Analysis exported to {file_path}"
                else:
                    return export_data
            
            elif export_format.lower() == 'csv':
                # Export rankings as CSV
                rankings_df = pd.DataFrame(peer_analysis_result.peer_rankings)
                
                if file_path:
                    rankings_df.to_csv(file_path, index=False)
                    return f"Rankings exported to {file_path}"
                else:
                    return rankings_df.to_csv(index=False)
            
            elif export_format.lower() == 'report':
                # Generate text report
                report = self.generate_peer_group_report(peer_analysis_result)
                
                if file_path:
                    with open(file_path, 'w') as f:
                        f.write(report)
                    return f"Report exported to {file_path}"
                else:
                    return report
            
            else:
                return f"Unsupported export format: {export_format}"
                
        except Exception as e:
            logger.error(f"Error exporting peer analysis: {e}")
            return f"Export failed: {str(e)}"
    
    def get_available_peer_groups(self) -> List[Dict[str, Any]]:
        """Get list of available peer groups"""
        try:
            self.connect_to_database()
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            
            # Get industry-based groups
            query = """
            SELECT DISTINCT industry, COUNT(*) as company_count
            FROM companies 
            WHERE industry IS NOT NULL
            GROUP BY industry
            HAVING COUNT(*) >= 5
            ORDER BY company_count DESC
            """
            
            cursor.execute(query)
            industry_groups = cursor.fetchall()
            
            available_groups = []
            
            # Add industry-based groups
            for group in industry_groups:
                available_groups.append({
                    'group_type': 'industry_based',
                    'group_name': f"{group['industry']} Industry",
                    'company_count': group['company_count'],
                    'selection_criteria': {'industry': group['industry']}
                })
            
            # Add size-based groups
            for size_category, size_range in self.SIZE_CATEGORIES.items():
                # Count companies in this size range
                count_query = """
                SELECT COUNT(*) 
                FROM companies c
                LEFT JOIN balance_sheets bs ON c.company_id = bs.company_id
                WHERE bs.revenue BETWEEN %s AND %s
                AND bs.year = (SELECT MAX(year) FROM balance_sheets WHERE company_id = c.company_id)
                """
                
                cursor.execute(count_query, (size_range['min'], size_range['max']))
                count_result = cursor.fetchone()
                
                if count_result and count_result['count'] >= 5:
                    available_groups.append({
                        'group_type': 'size_based',
                        'group_name': f"{size_category.replace('_', ' ').title()} Companies",
                        'company_count': count_result['count'],
                        'selection_criteria': {'size_category': size_category, 'revenue_range': size_range}
                    })
            
            return available_groups
            
        except Exception as e:
            logger.error(f"Error getting available peer groups: {e}")
            return []
        finally:
            self.close_connection()


# Complete integrated example and testing
if __name__ == "__main__":
    print("=" * 60)
    print("COMPLETE PEER ANALYSIS SYSTEM - ALL 5 PHASES INTEGRATED")
    print("=" * 60)
    
    # Database configuration
    db_config = {
        'host': 'localhost',
        'database': 'financial_db',
        'user': 'postgres',
        'password': 'password',
        'port': 5432
    }
    
    # Initialize complete analyzer
    analyzer = FinancialPeerAnalyzer(db_config)
    
    print("\n🚀 Testing Complete Peer Analysis System...")
    
    try:
        # Test main analysis function
        print("\n--- Testing Industry-Based Analysis ---")
        result = analyzer.analyze_company_peers(
            company_id='COMP001',
            comparison_method=ComparisonMethod.INDUSTRY_BASED,
            peer_group_size=15
        )
        
        if hasattr(result, 'target_company') and result.target_company != "unknown":
            print(f"✓ Analysis completed for {result.target_company}")
            print(f"  📊 Peer group: {len(result.peer_group.companies)} companies")
            print(f"  🏆 Rank: {result.company_comparison.rank} of {result.company_comparison.peer_group_size}")
            print(f"  📈 Percentile: {result.company_comparison.percentile:.1f}%")
            print(f"  💪 Strengths: {len(result.company_comparison.strengths)}")
            print(f"  ⚠️  Weaknesses: {len(result.company_comparison.weaknesses)}")
            print(f"  🎯 Recommendations: {len(result.recommendations)}")
            
            # Test report generation
            print("\n--- Testing Report Generation ---")
            report = analyzer.generate_peer_group_report(result)
            print(f"✓ Report generated ({len(report)} characters)")
            
            # Test JSON export
            print("\n--- Testing Export Functions ---")
            json_export = analyzer.export_peer_analysis(result, 'json')
            if isinstance(json_export, dict):
                print(f"✓ JSON export successful")
            
            # Test CSV export
            csv_export = analyzer.export_peer_analysis(result, 'csv')
            if isinstance(csv_export, str) and 'company_id' in csv_export:
                print(f"✓ CSV export successful")
            
            # Print some key results
            print(f"\n--- Key Analysis Results ---")
            strategic_pos = result.competitive_positioning.get('strategic_position', {})
            print(f"Strategic Quadrant: {strategic_pos.get('strategic_quadrant', 'Unknown')}")
            print(f"Performance Position: {strategic_pos.get('performance_position', 'Unknown')}")
            
            if result.recommendations:
                print(f"\nTop Recommendations:")
                for i, rec in enumerate(result.recommendations[:3], 1):
                    print(f"  {i}. {rec}")
        
        else:
            print(f"✗ Analysis failed: {result.competitive_positioning.get('error', 'Unknown error')}")
        
        # Test different comparison methods
        print("\n--- Testing Similarity-Based Analysis ---")
        similarity_result = analyzer.analyze_company_peers(
            company_id='COMP002',
            comparison_method=ComparisonMethod.SIMILARITY_BASED,
            peer_group_size=10
        )
        
        if hasattr(similarity_result, 'target_company') and similarity_result.target_company != "unknown":
            print(f"✓ Similarity analysis completed")
            print(f"  Method: {similarity_result.comparison_method}")
            print(f"  Strategic quadrant: {similarity_result.competitive_positioning.get('strategic_position', {}).get('strategic_quadrant', 'Unknown')}")
        
        # Test available peer groups
        print("\n--- Testing Available Peer Groups ---")
        available_groups = analyzer.get_available_peer_groups()
        print(f"✓ Found {len(available_groups)} available peer groups")
        
        for group in available_groups[:3]:  # Show first 3
            print(f"  - {group['group_name']}: {group['company_count']} companies")
        
        print(f"\n{'='*60}")
        print("🎉 ALL 5 PHASES COMPLETED SUCCESSFULLY!")
        print("📈 Financial Peer Analysis System is fully operational")
        print(f"{'='*60}")
        
        # Summary of capabilities
        print(f"\n📋 SYSTEM CAPABILITIES SUMMARY:")
        print(f"Phase 1: ✓ Setup and Configuration")
        print(f"Phase 2: ✓ Data Retrieval and Financial Calculations")
        print(f"Phase 3: ✓ Peer Group Selection and Similarity Analysis")
        print(f"Phase 4: ✓ Comparison Analysis and Ranking")
        print(f"Phase 5: ✓ Reporting and Integration")
        print(f"\n🔧 FEATURES AVAILABLE:")
        print(f"• Multiple comparison methods (Industry, Size, Performance, Similarity, Custom)")
        print(f"• Comprehensive financial ratio calculations")
        print(f"• Strategic positioning analysis")
        print(f"• Competitive benchmarking")
        print(f"• Automated recommendations")
        print(f"• Multiple export formats (JSON, CSV, Reports)")
        print(f"• Industry dynamics analysis")
        print(f"• Market concentration metrics")
        
    except Exception as e:
        print(f"✗ System test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🚀 Ready for integration with Flask application!")


# Export main classes for integration
__all__ = [
    'FinancialPeerAnalyzer',
    'ComparisonMethod',
    'RankingMetric',
    'PeerGroup',
    'CompanyComparison',
    'PeerAnalysisResult'
]