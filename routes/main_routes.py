from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify
from utils.helpers import format_currency, calculate_enhanced_health_score  
from config.database import db_conn
import traceback

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def home():
    """Main landing page with enhanced system stats"""
    try:
        # Get basic system statistics
        system_stats = {}
        
        # Cash Flow Statement Stats
        cf_stats_query = """
        SELECT 
            COUNT(DISTINCT company_name) as total_companies,
            COUNT(*) as total_records,
            AVG(CASE WHEN net_income > 0 THEN 1 ELSE 0 END) * 100 as profitable_percentage,
            SUM(CASE WHEN liquidation_label = 1 THEN 1 ELSE 0 END) as high_risk_companies,
            MAX(year) as latest_year,
            MIN(year) as earliest_year
        FROM cash_flow_statement
        """
        
        cf_stats = db_conn.execute_query(cf_stats_query)
        system_stats['cash_flow'] = cf_stats[0] if cf_stats else {}
        
        # Balance Sheet Stats
        bs_stats_query = """
        SELECT 
            COUNT(DISTINCT company_name) as total_companies_bs,
            COUNT(*) as total_bs_records,
            AVG(current_ratio) as avg_current_ratio,
            AVG(debt_to_equity_ratio) as avg_debt_ratio
        FROM balance_sheet_1
        """
        
        bs_stats = db_conn.execute_query(bs_stats_query)
        system_stats['balance_sheet'] = bs_stats[0] if bs_stats else {}
        
        # Industry distribution
        industry_query = """
        SELECT industry, COUNT(*) as company_count
        FROM cash_flow_statement 
        GROUP BY industry 
        ORDER BY company_count DESC
        LIMIT 5
        """
        
        industries = db_conn.execute_query(industry_query)
        system_stats['top_industries'] = industries if industries else []
        
        return render_template('index.html', system_stats=system_stats)
        
    except Exception as e:
        flash(f'Error loading home page: {str(e)}', 'error')
        return render_template('index.html', system_stats={})

@main_bp.route('/dashboard')
def dashboard():
    """Enhanced dashboard with both cash flow and balance sheet data"""
    try:
        # Cash Flow Statistics
        cf_stats_query = """
        SELECT 
            COUNT(DISTINCT company_name) as total_companies,
            COUNT(*) as total_records,
            AVG(CASE WHEN net_income > 0 THEN 1 ELSE 0 END) * 100 as profitable_percentage,
            SUM(CASE WHEN liquidation_label = 1 THEN 1 ELSE 0 END) as high_risk_companies,
            AVG(net_cash_from_operating_activities) as avg_operating_cash_flow,
            AVG(free_cash_flow) as avg_free_cash_flow
        FROM cash_flow_statement
        """
        
        cf_stats_result = db_conn.execute_query(cf_stats_query)
        cf_stats = cf_stats_result[0] if cf_stats_result else {
            'total_companies': 0,
            'total_records': 0,
            'profitable_percentage': 0,
            'high_risk_companies': 0,
            'avg_operating_cash_flow': 0,
            'avg_free_cash_flow': 0
        }
        
        # Balance Sheet Statistics
        bs_stats_query = """
        SELECT 
            COUNT(DISTINCT company_name) as total_companies_bs,
            COUNT(*) as total_bs_records,
            AVG(current_ratio) as avg_current_ratio,
            AVG(debt_to_equity_ratio) as avg_debt_ratio,
            AVG(total_assets) as avg_total_assets,
            AVG(net_income) as avg_net_income_bs
        FROM balance_sheet_1
        """
        
        bs_stats_result = db_conn.execute_query(bs_stats_query)
        bs_stats = bs_stats_result[0] if bs_stats_result else {
            'total_companies_bs': 0,
            'total_bs_records': 0,
            'avg_current_ratio': 0,
            'avg_debt_ratio': 0,
            'avg_total_assets': 0,
            'avg_net_income_bs': 0
        }
        
        # Recent Companies with Combined Data
        recent_companies_query = """
        SELECT DISTINCT
            cf.company_name,
            cf.industry,
            cf.net_income,
            cf.net_cash_from_operating_activities,
            cf.free_cash_flow,
            cf.liquidation_label,
            cf.year,
            bs.current_ratio,
            bs.debt_to_equity_ratio,
            bs.total_assets,
            bs.total_liabilities
        FROM cash_flow_statement cf
        LEFT JOIN balance_sheet_1 bs ON cf.company_name = bs.company_name 
        ORDER BY cf.year DESC, cf.company_name
        LIMIT 15
        """
        
        recent_companies = db_conn.execute_query(recent_companies_query)
        
        # Process companies data with enhanced metrics
        processed_companies = []
        if recent_companies:
            for company in recent_companies:
                # Calculate health score using both CF and BS data
                health_score = calculate_enhanced_health_score(company)
                
                processed_company = {
                    'company_name': company.get('company_name', 'Unknown'),
                    'industry': company.get('industry', 'General'),
                    'net_income': format_currency(company.get('net_income')),
                    'operating_cash_flow': format_currency(company.get('net_cash_from_operating_activities')),
                    'free_cash_flow': format_currency(company.get('free_cash_flow')),
                    'total_assets': format_currency(company.get('total_assets')),
                    'current_ratio': round(company.get('current_ratio', 0), 2),
                    'debt_ratio': round(company.get('debt_to_equity_ratio', 0), 2),
                    'health_score': health_score,
                    'risk_level': get_risk_level(health_score),
                    'liquidation_risk': company.get('liquidation_label', 0),
                    'year': company.get('year', 'N/A'),
                    'has_balance_sheet': company.get('current_ratio') is not None
                }
                processed_companies.append(processed_company)
        
        # Risk Distribution Analysis
        risk_distribution_query = """
        SELECT 
            CASE 
                WHEN liquidation_label = 1 THEN 'High Risk'
                WHEN net_income < 0 AND net_cash_from_operating_activities < 0 THEN 'Medium Risk'
                WHEN net_income > 0 AND net_cash_from_operating_activities > 0 THEN 'Low Risk'
                ELSE 'Moderate Risk'
            END as risk_category,
            COUNT(*) as count
        FROM cash_flow_statement
        GROUP BY risk_category
        ORDER BY count DESC
        """
        
        risk_distribution = db_conn.execute_query(risk_distribution_query)
        
        # Industry Performance
        industry_performance_query = """
        SELECT 
            industry,
            COUNT(*) as total_companies,
            AVG(net_income) as avg_net_income,
            AVG(net_cash_from_operating_activities) as avg_operating_cf,
            SUM(CASE WHEN liquidation_label = 1 THEN 1 ELSE 0 END) as high_risk_count
        FROM cash_flow_statement
        GROUP BY industry
        HAVING COUNT(*) >= 3
        ORDER BY avg_net_income DESC
        LIMIT 10
        """
        
        industry_performance = db_conn.execute_query(industry_performance_query)
        
        return render_template('dashboard.html', 
                             cf_stats=cf_stats,
                             bs_stats=bs_stats,
                             recent_companies=processed_companies,
                             risk_distribution=risk_distribution,
                             industry_performance=industry_performance)
                             
    except Exception as e:
        flash(f'Dashboard error: {str(e)}', 'error')
        print(f"Dashboard error: {e}")
        print(traceback.format_exc())
        return redirect(url_for('main.home'))

@main_bp.route('/companies')
def companies_page():
    """Enhanced companies management page with search and filtering"""
    try:
        # Get query parameters
        search_query = request.args.get('search', '')
        industry_filter = request.args.get('industry', '')
        risk_filter = request.args.get('risk', '')
        page = request.args.get('page', 1, type=int)
        per_page = 20
        
        # Build base query
        base_query = """
        SELECT DISTINCT
            cf.company_name,
            cf.industry,
            cf.net_income,
            cf.net_cash_from_operating_activities,
            cf.free_cash_flow,
            cf.liquidation_label,
            cf.year,
            bs.current_ratio,
            bs.debt_to_equity_ratio,
            bs.total_assets
        FROM cash_flow_statement cf
        LEFT JOIN balance_sheet_1 bs ON cf.company_name = bs.company_name
        WHERE 1=1
        """
        
        params = []
        
        # Add filters
        if search_query:
            base_query += " AND LOWER(cf.company_name) LIKE LOWER(%s)"
            params.append(f"%{search_query}%")
        
        if industry_filter:
            base_query += " AND cf.industry = %s"
            params.append(industry_filter)
        
        if risk_filter:
            if risk_filter == 'high':
                base_query += " AND cf.liquidation_label = 1"
            elif risk_filter == 'low':
                base_query += " AND cf.liquidation_label = 0 AND cf.net_income > 0"
        
        # Add pagination
        offset = (page - 1) * per_page
        base_query += f" ORDER BY cf.company_name LIMIT {per_page} OFFSET {offset}"
        
        companies = db_conn.execute_query(base_query, params)
        
        # Get total count for pagination
        count_query = base_query.replace("SELECT DISTINCT cf.company_name, cf.industry, cf.net_income, cf.net_cash_from_operating_activities, cf.free_cash_flow, cf.liquidation_label, cf.year, bs.current_ratio, bs.debt_to_equity_ratio, bs.total_assets", "SELECT COUNT(DISTINCT cf.company_name)")
        count_query = count_query.split("ORDER BY")[0]  # Remove ORDER BY and LIMIT
        
        total_count_result = db_conn.execute_query(count_query, params)
        total_count = total_count_result[0]['count'] if total_count_result else 0
        
        # Get available industries for filter
        industries_query = "SELECT DISTINCT industry FROM cash_flow_statement ORDER BY industry"
        industries = db_conn.execute_query(industries_query)
        
        # Process companies
        processed_companies = []
        if companies:
            for company in companies:
                health_score = calculate_enhanced_health_score(company)
                processed_company = {
                    'company_name': company.get('company_name'),
                    'industry': company.get('industry'),
                    'net_income': format_currency(company.get('net_income')),
                    'operating_cash_flow': format_currency(company.get('net_cash_from_operating_activities')),
                    'free_cash_flow': format_currency(company.get('free_cash_flow')),
                    'total_assets': format_currency(company.get('total_assets')),
                    'current_ratio': round(company.get('current_ratio', 0), 2),
                    'debt_ratio': round(company.get('debt_to_equity_ratio', 0), 2),
                    'health_score': health_score,
                    'risk_level': get_risk_level(health_score),
                    'liquidation_risk': company.get('liquidation_label', 0),
                    'year': company.get('year'),
                    'has_balance_sheet': company.get('current_ratio') is not None
                }
                processed_companies.append(processed_company)
        
        # Pagination info
        pagination = {
            'page': page,
            'per_page': per_page,
            'total': total_count,
            'pages': (total_count + per_page - 1) // per_page,
            'has_prev': page > 1,
            'has_next': page * per_page < total_count,
            'prev_num': page - 1 if page > 1 else None,
            'next_num': page + 1 if page * per_page < total_count else None
        }
        
        return render_template('companies.html',
                             companies=processed_companies,
                             industries=industries,
                             pagination=pagination,
                             search_query=search_query,
                             industry_filter=industry_filter,
                             risk_filter=risk_filter)
    
    except Exception as e:
        flash(f'Error loading companies: {str(e)}', 'error')
        print(f"Companies page error: {e}")
        print(traceback.format_exc())
        return render_template('companies.html', companies=[], industries=[], pagination={})

@main_bp.route('/companies/<company_name>')
def company_detail(company_name):
    """Individual company detail page"""
    try:
        # Get company cash flow data
        cf_query = """
        SELECT * FROM cash_flow_statement 
        WHERE company_name = %s 
        ORDER BY year DESC
        """
        
        cf_data = db_conn.execute_query(cf_query, [company_name])
        
        # Get company balance sheet data
        bs_query = """
        SELECT * FROM balance_sheet_1 
        WHERE company_name = %s 
        ORDER BY year DESC
        """
        
        bs_data = db_conn.execute_query(bs_query, [company_name])
        
        if not cf_data and not bs_data:
            flash(f'Company "{company_name}" not found', 'error')
            return redirect(url_for('main.companies_page'))
        
        # Calculate trends and metrics
        company_analysis = analyze_company_trends(cf_data, bs_data)
        
        return render_template('company_detail.html',
                             company_name=company_name,
                             cash_flow_data=cf_data,
                             balance_sheet_data=bs_data,
                             analysis=company_analysis)
    
    except Exception as e:
        flash(f'Error loading company details: {str(e)}', 'error')
        print(f"Company detail error: {e}")
        return redirect(url_for('main.companies_page'))

@main_bp.route('/analysis')
def analysis_page():
    """Enhanced Financial Analysis Page"""
    try:
        # Get analysis summary data
        analysis_data = get_financial_analysis_data()
        
        return render_template('analysis.html', analysis_data=analysis_data)
    
    except Exception as e:
        flash(f'Error loading analysis: {str(e)}', 'error')
        return render_template('analysis.html', analysis_data={})

@main_bp.route('/api/companies')
def api_companies():
    """API endpoint for companies data"""
    try:
        # Same companies query logic as companies_page()
        companies_query = """
        SELECT DISTINCT
            cf.company_name,
            cf.industry,
            cf.net_income,
            cf.net_cash_from_operating_activities,
            cf.free_cash_flow,
            cf.liquidation_label,
            cf.year
        FROM cash_flow_statement cf
        ORDER BY cf.company_name
        LIMIT 100
        """
        
        companies = fix_database_result(db_conn.execute_query(companies_query))
        
        return jsonify({
            'success': True,
            'companies': companies if companies else []
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@main_bp.route('/api/search-companies')
def api_search_companies():
    """API endpoint for company search autocomplete"""
    try:
        query = request.args.get('q', '')
        limit = request.args.get('limit', 10, type=int)
        
        if not query or len(query) < 2:
            return jsonify({'companies': []})
        
        search_query = """
        SELECT DISTINCT company_name, industry
        FROM cash_flow_statement 
        WHERE LOWER(company_name) LIKE LOWER(%s)
        ORDER BY company_name
        LIMIT %s
        """
        
        companies = db_conn.execute_query(search_query, [f"%{query}%", limit])
        
        return jsonify({
            'success': True,
            'companies': companies if companies else []
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

def fix_database_result(result):
    """Fix database result format"""
    try:
        # Handle nested JSON response
        if isinstance(result, dict):
            if 'cash_flow_data' in result:
                result = result['cash_flow_data']
            elif len(result) == 1 and isinstance(list(result.values())[0], list):
                result = list(result.values())[0]
        
        # Convert string numbers to float
        if isinstance(result, list) and len(result) > 0:
            for row in result:
                if isinstance(row, dict):
                    for key, value in row.items():
                        if isinstance(value, str):
                            try:
                                # Try to convert string numbers to float
                                if value.replace('.', '').replace('-', '').replace('e', '').replace('E', '').replace('+', '').isdigit():
                                    row[key] = float(value)
                            except:
                                pass
        
        return result
    except:
        return result

# Helper Functions
def calculate_enhanced_health_score(company_data):
    """Calculate enhanced health score using both CF and BS data"""
    try:
        score = 0
        max_score = 100
        
        # Cash Flow Health (40 points)
        net_income = company_data.get('net_income', 0) or 0
        operating_cf = company_data.get('net_cash_from_operating_activities', 0) or 0
        free_cf = company_data.get('free_cash_flow', 0) or 0
        
        if net_income > 0:
            score += 15
        if operating_cf > 0:
            score += 15
        if free_cf > 0:
            score += 10
        
        # Balance Sheet Health (40 points)
        current_ratio = company_data.get('current_ratio', 0) or 0
        debt_ratio = company_data.get('debt_to_equity_ratio', 0) or 0
        
        if current_ratio >= 1.5:
            score += 20
        elif current_ratio >= 1.0:
            score += 10
        
        if debt_ratio <= 0.5:
            score += 20
        elif debt_ratio <= 1.0:
            score += 10
        
        # Risk Assessment (20 points)
        liquidation_risk = company_data.get('liquidation_label', 0) or 0
        if liquidation_risk == 0:
            score += 20
        
        return min(score, max_score)
    
    except Exception:
        return 50  # Default moderate score

def get_risk_level(health_score):
    """Convert health score to risk level"""
    if health_score >= 80:
        return 'Low'
    elif health_score >= 60:
        return 'Moderate'
    elif health_score >= 40:
        return 'Medium'
    else:
        return 'High'

def analyze_company_trends(cf_data, bs_data):
    """Analyze company trends over multiple years"""
    try:
        analysis = {
            'cf_trends': {},
            'bs_trends': {},
            'overall_trend': 'stable'
        }
        
        if cf_data and len(cf_data) > 1:
            # Calculate CF trends
            recent_cf = cf_data[0]
            older_cf = cf_data[-1]
            
            analysis['cf_trends'] = {
                'revenue_growth': calculate_growth_rate(
                    older_cf.get('net_income', 0),
                    recent_cf.get('net_income', 0)
                ),
                'operating_cf_growth': calculate_growth_rate(
                    older_cf.get('net_cash_from_operating_activities', 0),
                    recent_cf.get('net_cash_from_operating_activities', 0)
                )
            }
        
        if bs_data and len(bs_data) > 1:
            # Calculate BS trends
            recent_bs = bs_data[0]
            older_bs = bs_data[-1]
            
            analysis['bs_trends'] = {
                'assets_growth': calculate_growth_rate(
                    older_bs.get('total_assets', 0),
                    recent_bs.get('total_assets', 0)
                ),
                'liquidity_trend': 'improving' if recent_bs.get('current_ratio', 0) > older_bs.get('current_ratio', 0) else 'declining'
            }
        
        return analysis
    
    except Exception:
        return {'cf_trends': {}, 'bs_trends': {}, 'overall_trend': 'unknown'}

def calculate_growth_rate(old_value, new_value):
    """Calculate percentage growth rate"""
    try:
        if old_value and old_value != 0:
            return round(((new_value - old_value) / abs(old_value)) * 100, 2)
        return 0
    except Exception:
        return 0

def get_financial_analysis_data():
    """Get comprehensive financial analysis data"""
    try:
        analysis_data = {}
        
        # Industry comparison
        industry_query = """
        SELECT 
            industry,
            COUNT(*) as company_count,
            AVG(net_income) as avg_net_income,
            AVG(net_cash_from_operating_activities) as avg_operating_cf,
            AVG(free_cash_flow) as avg_free_cf,
            STDDEV(net_income) as income_volatility
        FROM cash_flow_statement
        GROUP BY industry
        HAVING COUNT(*) >= 5
        ORDER BY avg_net_income DESC
        """
        
        analysis_data['industry_analysis'] = db_conn.execute_query(industry_query)
        
        # Risk correlation analysis
        risk_query = """
        SELECT 
            liquidation_label,
            AVG(net_income) as avg_income,
            AVG(net_cash_from_operating_activities) as avg_operating_cf,
            AVG(debt_to_equity_ratio) as avg_debt_ratio,
            COUNT(*) as count
        FROM cash_flow_statement cf
        LEFT JOIN balance_sheet_1 bs ON cf.company_name = bs.company_name
        GROUP BY liquidation_label
        """
        
        analysis_data['risk_analysis'] = db_conn.execute_query(risk_query)
        
        # Performance distribution
        performance_query = """
        SELECT 
            CASE 
                WHEN net_income > 1000000 THEN 'High Performer'
                WHEN net_income > 0 THEN 'Profitable'
                WHEN net_income > -500000 THEN 'Marginal'
                ELSE 'Loss Making'
            END as performance_category,
            COUNT(*) as count,
            AVG(net_cash_from_operating_activities) as avg_operating_cf
        FROM cash_flow_statement
        GROUP BY performance_category
        ORDER BY COUNT(*) DESC
        """
        
        analysis_data['performance_distribution'] = db_conn.execute_query(performance_query)
        
        return analysis_data
    
    except Exception as e:
        print(f"Analysis data error: {e}")
        return {}