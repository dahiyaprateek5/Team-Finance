from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DatabaseOperations:
    """Database operations for CRUD functionality using your actual database schema"""
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    # =====================================
    # COMPANY OPERATIONS (Updated for your schema)
    # =====================================
    
    def create_company(self, company_name, company_data=None):
        """Create a new company using your actual companies table schema"""
        try:
            # Your companies table only has: id, company_name, created_at, updated_at
            query = """
            INSERT INTO companies (company_name, created_at, updated_at) 
            VALUES (%s, NOW(), NOW()) 
            RETURNING id
            """
            
            result = self.db.execute_query(query, (company_name,))
            
            if result and len(result) > 0:
                return result[0]['id']
            return None
            
        except Exception as e:
            logger.error(f"Error creating company: {e}")
            return None
    
    def update_company(self, company_id, company_data):
        """Update company using your actual companies table schema"""
        try:
            # Only company_name can be updated in your schema
            if 'company_name' not in company_data:
                logger.warning("No company_name provided for update")
                return False
            
            query = """
            UPDATE companies 
            SET company_name = %s, updated_at = NOW()
            WHERE id = %s
            """
            
            result = self.db.execute_query(query, (company_data['company_name'], company_id))
            return result is not None
            
        except Exception as e:
            logger.error(f"Error updating company: {e}")
            return False
    
    def delete_company(self, company_id):
        """Delete company and all related data using your actual schema"""
        try:
            # Get company name first for cleanup
            company = self.get_company_by_id(company_id)
            if not company:
                logger.warning(f"Company with ID {company_id} not found")
                return False
            
            company_name = company['company_name']
            
            # Delete from all related tables using your actual table names
            delete_queries = [
                # Delete from financial_analysis table
                "DELETE FROM financial_analysis WHERE company_id = %s",
                # Delete from user_sessions table
                "DELETE FROM user_sessions WHERE company_id = %s", 
                # Delete from document_upload table (correct table name)
                "DELETE FROM document_upload WHERE company_id = %s",
                # Delete from uploaded_documents table (uses company_name)
                "DELETE FROM uploaded_documents WHERE company_name = %s",
                # Delete from cash_flow_statement_1 table
                "DELETE FROM cash_flow_statement_1 WHERE company_id = %s",
                # Delete from cash_flow_statement table (if exists)
                "DELETE FROM cash_flow_statement WHERE company_id = %s",
                # Delete from balance_sheet_1 table
                "DELETE FROM balance_sheet_1 WHERE company_id = %s",
                # Finally delete the company
                "DELETE FROM companies WHERE id = %s"
            ]
            
            # Execute deletions in sequence
            for i, query in enumerate(delete_queries):
                try:
                    if i == 3:  # uploaded_documents uses company_name
                        self.db.execute_query(query, (company_name,))
                    else:
                        self.db.execute_query(query, (company_id,))
                except Exception as e:
                    logger.warning(f"Error in delete query {i}: {e}")
                    # Continue with other deletions
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting company: {e}")
            return False
    
    def get_all_companies(self, include_stats=False):
        """Get all companies using your actual schema"""
        try:
            if include_stats:
                query = """
                SELECT 
                    c.*,
                    COUNT(DISTINCT bs.id) as balance_sheet_count,
                    COUNT(DISTINCT cf.id) as cash_flow_count,
                    COUNT(DISTINCT du.id) as document_count,
                    COUNT(DISTINCT ud.id) as uploaded_document_count,
                    GREATEST(
                        COALESCE(MAX(bs.generated_at), c.created_at),
                        COALESCE(MAX(cf.generated_at), c.created_at),
                        COALESCE(MAX(du.created_at), c.created_at),
                        COALESCE(MAX(ud.upload_date), c.created_at),
                        c.updated_at
                    ) as last_activity
                FROM companies c
                LEFT JOIN balance_sheet_1 bs ON c.id = bs.company_id
                LEFT JOIN cash_flow_statement_1 cf ON c.id = cf.company_id
                LEFT JOIN document_upload du ON c.id = du.company_id
                LEFT JOIN uploaded_documents ud ON c.company_name = ud.company_name
                GROUP BY c.id, c.company_name, c.created_at, c.updated_at
                ORDER BY c.company_name
                """
            else:
                query = "SELECT * FROM companies ORDER BY company_name"
            
            return self.db.execute_query(query)
            
        except Exception as e:
            logger.error(f"Error retrieving all companies: {e}")
            return []
    
    def get_company_by_name(self, company_name):
        """Find company by name using your actual schema"""
        try:
            query = "SELECT * FROM companies WHERE company_name = %s"
            results = self.db.execute_query(query, (company_name,))
            return results[0] if results else None
            
        except Exception as e:
            logger.error(f"Error finding company: {e}")
            return None
    
    def get_company_by_id(self, company_id):
        """Find company by ID using your actual schema"""
        try:
            query = "SELECT * FROM companies WHERE id = %s"
            results = self.db.execute_query(query, (company_id,))
            return results[0] if results else None
            
        except Exception as e:
            logger.error(f"Error finding company by ID: {e}")
            return None
    
    def get_company_with_financial_data(self, company_id):
        """Get company with latest financial data using your actual schema"""
        try:
            query = """
            SELECT 
                c.*,
                -- Latest balance sheet data using your actual columns
                bs.total_assets,
                bs.total_liabilities,
                bs.total_equity,
                bs.current_assets,
                bs.current_liabilities,
                bs.cash_and_equivalents,
                bs.property_plant_equipment,
                bs.year as balance_sheet_year,
                -- Latest cash flow data using your actual columns
                cf.net_income,
                cf.net_cash_from_operating_activities,
                cf.free_cash_flow,
                cf.liquidation_label,
                cf.debt_to_equity_ratio,
                cf.interest_coverage_ratio,
                cf.year as cash_flow_year
            FROM companies c
            LEFT JOIN balance_sheet_1 bs ON c.id = bs.company_id 
                AND bs.year = (SELECT MAX(year) FROM balance_sheet_1 WHERE company_id = c.id)
            LEFT JOIN cash_flow_statement_1 cf ON c.id = cf.company_id 
                AND cf.year = (SELECT MAX(year) FROM cash_flow_statement_1 WHERE company_id = c.id)
            WHERE c.id = %s
            """
            
            results = self.db.execute_query(query, (company_id,))
            return results[0] if results else None
            
        except Exception as e:
            logger.error(f"Error getting company with financial data: {e}")
            return None
    
    def search_companies(self, search_term, limit=20):
        """Search companies using your actual schema"""
        try:
            query = """
            SELECT 
                c.*,
                COUNT(DISTINCT bs.id) as balance_sheet_count,
                COUNT(DISTINCT cf.id) as cash_flow_count,
                COUNT(DISTINCT ud.id) as document_count
            FROM companies c
            LEFT JOIN balance_sheet_1 bs ON c.id = bs.company_id
            LEFT JOIN cash_flow_statement_1 cf ON c.id = cf.company_id
            LEFT JOIN uploaded_documents ud ON c.company_name = ud.company_name
            WHERE c.company_name ILIKE %s
            GROUP BY c.id, c.company_name, c.created_at, c.updated_at
            ORDER BY c.company_name
            LIMIT %s
            """
            search_pattern = f"%{search_term}%"
            return self.db.execute_query(query, (search_pattern, limit))
            
        except Exception as e:
            logger.error(f"Error searching companies: {e}")
            return []
    
    # =====================================
    # BALANCE SHEET OPERATIONS (Updated for your schema)
    # =====================================
    
    def save_generated_balance_sheet(self, company_id, year, balance_data):
        """Save balance sheet using your actual balance_sheet_1 schema"""
        try:
            # Your balance_sheet_1 columns
            base_columns = ['company_id', 'year', 'generated_at']
            base_values = [company_id, year, datetime.now()]
            
            # Available columns in your balance_sheet_1 table
            available_columns = [
                'current_assets', 'cash_and_equivalents', 'accounts_receivable', 
                'inventory', 'prepaid_expenses', 'other_current_assets',
                'non_current_assets', 'property_plant_equipment', 'accumulated_depreciation',
                'net_ppe', 'intangible_assets', 'goodwill', 'investments', 'other_non_current_assets',
                'total_assets', 'current_liabilities', 'accounts_payable', 'short_term_debt',
                'accrued_liabilities', 'deferred_revenue', 'other_current_liabilities',
                'non_current_liabilities', 'long_term_debt', 'deferred_tax_liabilities',
                'pension_obligations', 'other_non_current_liabilities', 'total_liabilities',
                'share_capital', 'retained_earnings', 'additional_paid_in_capital',
                'treasury_stock', 'accumulated_other_comprehensive_income', 'total_equity',
                'balance_check', 'accuracy_percentage', 'data_source', 'validation_errors'
            ]
            
            # Build dynamic query
            columns = base_columns.copy()
            values = base_values.copy()
            
            for column in available_columns:
                if column in balance_data and balance_data[column] is not None:
                    columns.append(column)
                    values.append(balance_data[column])
            
            placeholders = ', '.join(['%s'] * len(columns))
            update_cols = ', '.join([f"{col} = EXCLUDED.{col}" for col in columns[3:]])
            
            query = f"""
            INSERT INTO balance_sheet_1 ({', '.join(columns)}) 
            VALUES ({placeholders}) 
            ON CONFLICT (company_id, year) 
            DO UPDATE SET {update_cols}
            RETURNING id
            """
            
            result = self.db.execute_query(query, tuple(values))
            
            if result and len(result) > 0:
                return result[0]['id']
            return None
            
        except Exception as e:
            logger.error(f"Error saving balance sheet: {e}")
            return None
    
    def get_generated_balance_sheets(self, company_id):
        """Get all balance sheets for a company using your actual schema"""
        try:
            query = """
            SELECT * FROM balance_sheet_1 
            WHERE company_id = %s 
            ORDER BY year DESC
            """
            return self.db.execute_query(query, (company_id,))
            
        except Exception as e:
            logger.error(f"Error retrieving balance sheets: {e}")
            return []
    
    def get_latest_balance_sheet(self, company_id):
        """Get latest balance sheet for a company using your actual schema"""
        try:
            query = """
            SELECT * FROM balance_sheet_1 
            WHERE company_id = %s 
            ORDER BY year DESC
            LIMIT 1
            """
            results = self.db.execute_query(query, (company_id,))
            return results[0] if results else None
            
        except Exception as e:
            logger.error(f"Error retrieving latest balance sheet: {e}")
            return None
    
    # =====================================
    # CASH FLOW OPERATIONS (Updated for your schema)
    # =====================================
    
    def save_generated_cash_flow(self, company_id, year, cash_flow_data):
        """Save cash flow using your actual cash_flow_statement_1 schema"""
        try:
            # Your cash_flow_statement_1 columns
            base_columns = ['company_id', 'year', 'generated_at']
            base_values = [company_id, year, datetime.now()]
            
            # Available columns in your cash_flow_statement_1 table
            available_columns = [
                'company_name', 'industry', 'net_income', 'depreciation_and_amortization',
                'stock_based_compensation', 'changes_in_working_capital', 'accounts_receivable',
                'inventory', 'accounts_payable', 'net_cash_from_operating_activities',
                'capital_expenditures', 'acquisitions', 'net_cash_from_investing_activities',
                'dividends_paid', 'share_repurchases', 'net_cash_from_financing_activities',
                'free_cash_flow', 'ocf_to_net_income_ratio', 'liquidation_label',
                'debt_to_equity_ratio', 'interest_coverage_ratio'
            ]
            
            # Build dynamic query
            columns = base_columns.copy()
            values = base_values.copy()
            
            for column in available_columns:
                if column in cash_flow_data and cash_flow_data[column] is not None:
                    columns.append(column)
                    values.append(cash_flow_data[column])
            
            placeholders = ', '.join(['%s'] * len(columns))
            update_cols = ', '.join([f"{col} = EXCLUDED.{col}" for col in columns[3:]])
            
            query = f"""
            INSERT INTO cash_flow_statement_1 ({', '.join(columns)}) 
            VALUES ({placeholders}) 
            ON CONFLICT (company_id, year) 
            DO UPDATE SET {update_cols}
            RETURNING id
            """
            
            result = self.db.execute_query(query, tuple(values))
            
            if result and len(result) > 0:
                return result[0]['id']
            return None
            
        except Exception as e:
            logger.error(f"Error saving cash flow: {e}")
            return None
    
    def get_generated_cash_flows(self, company_id):
        """Get all cash flows for a company using your actual schema"""
        try:
            query = """
            SELECT * FROM cash_flow_statement_1 
            WHERE company_id = %s 
            ORDER BY year DESC
            """
            return self.db.execute_query(query, (company_id,))
            
        except Exception as e:
            logger.error(f"Error retrieving cash flows: {e}")
            return []
    
    def get_latest_cash_flow(self, company_id):
        """Get latest cash flow for a company using your actual schema"""
        try:
            query = """
            SELECT * FROM cash_flow_statement_1 
            WHERE company_id = %s 
            ORDER BY year DESC
            LIMIT 1
            """
            results = self.db.execute_query(query, (company_id,))
            return results[0] if results else None
            
        except Exception as e:
            logger.error(f"Error retrieving latest cash flow: {e}")
            return None
    
    def get_cash_flow_by_year(self, company_id, year):
        """Get specific year cash flow using your actual schema"""
        try:
            query = """
            SELECT * FROM cash_flow_statement_1 
            WHERE company_id = %s AND year = %s
            """
            results = self.db.execute_query(query, (company_id, year))
            return results[0] if results else None
            
        except Exception as e:
            logger.error(f"Error retrieving cash flow for year {year}: {e}")
            return None
    
    # =====================================
    # DOCUMENT OPERATIONS (Updated for your schema)
    # =====================================
    
    def save_document_upload(self, company_id, document_type, filename, file_path, file_size=None):
        """Save document using your actual document_upload schema"""
        try:
            # Your document_upload table columns
            query = """
            INSERT INTO document_upload 
            (company_id, document_type, filename, file_path, upload_status, 
             is_required, file_size, created_at) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, NOW()) 
            RETURNING id
            """
            
            result = self.db.execute_query(query, (
                company_id, document_type, filename, file_path, 
                'uploaded', False, file_size
            ))
            
            if result and len(result) > 0:
                return result[0]['id']
            return None
            
        except Exception as e:
            logger.error(f"Error saving document upload: {e}")
            return None
    
    def save_enhanced_document_upload(self, company_name, document_type, original_filename, stored_filename, file_path, file_size):
        """Save enhanced document using your actual uploaded_documents schema"""
        try:
            # Your uploaded_documents table columns
            query = """
            INSERT INTO uploaded_documents 
            (company_name, document_type, original_filename, stored_filename, 
             file_path, upload_date, file_size, processing_status, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, NOW(), %s, %s, NOW(), NOW())
            RETURNING id
            """
            
            result = self.db.execute_query(query, (
                company_name, document_type, original_filename, stored_filename, 
                file_path, file_size, 'uploaded'
            ))
            
            if result and len(result) > 0:
                return result[0]['id']
            return None
            
        except Exception as e:
            logger.error(f"Error saving enhanced document upload: {e}")
            return None
    
    def get_company_documents(self, company_id):
        """Get all documents for a company using your actual schema"""
        try:
            query = """
            SELECT * FROM document_upload 
            WHERE company_id = %s 
            ORDER BY created_at DESC
            """
            return self.db.execute_query(query, (company_id,))
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def get_company_documents_by_name(self, company_name):
        """Get all documents for a company by name using your actual schema"""
        try:
            query = """
            SELECT * FROM uploaded_documents 
            WHERE company_name = %s 
            ORDER BY upload_date DESC
            """
            return self.db.execute_query(query, (company_name,))
            
        except Exception as e:
            logger.error(f"Error retrieving documents by company name: {e}")
            return []
    
    def update_document_processing_status(self, document_id, status, ai_data=None):
        """Update document processing status using your actual schema"""
        try:
            if ai_data:
                query = """
                UPDATE uploaded_documents 
                SET processing_status = %s, ai_extracted_data = %s, updated_at = NOW()
                WHERE id = %s
                """
                params = (status, ai_data, document_id)
            else:
                query = """
                UPDATE uploaded_documents 
                SET processing_status = %s, updated_at = NOW()
                WHERE id = %s
                """
                params = (status, document_id)
            
            result = self.db.execute_query(query, params)
            return result is not None
            
        except Exception as e:
            logger.error(f"Error updating document status: {e}")
            return False
    
    # =====================================
    # FINANCIAL ANALYSIS OPERATIONS (Updated for your schema)
    # =====================================
    
    def save_financial_analysis(self, company_id, analysis_year, analysis_data):
        """Save financial analysis using your actual financial_analysis schema"""
        try:
            # Your financial_analysis table columns
            available_columns = [
                'current_ratio', 'debt_to_equity_ratio', 'interest_coverage_ratio',
                'risk_score', 'risk_level', 'liquidation_risk', 'recommendations'
            ]
            
            # Build dynamic query
            columns = ['company_id', 'analysis_year', 'created_at']
            values = [company_id, analysis_year, datetime.now()]
            
            for column in available_columns:
                if column in analysis_data and analysis_data[column] is not None:
                    columns.append(column)
                    values.append(analysis_data[column])
            
            placeholders = ', '.join(['%s'] * len(columns))
            update_cols = ', '.join([f"{col} = EXCLUDED.{col}" for col in columns[3:]])
            
            query = f"""
            INSERT INTO financial_analysis ({', '.join(columns)}) 
            VALUES ({placeholders}) 
            ON CONFLICT (company_id, analysis_year) 
            DO UPDATE SET {update_cols}
            RETURNING id
            """
            
            result = self.db.execute_query(query, tuple(values))
            
            if result and len(result) > 0:
                return result[0]['id']
            return None
            
        except Exception as e:
            logger.error(f"Error saving financial analysis: {e}")
            return None
    
    def get_financial_analysis(self, company_id, analysis_year=None):
        """Get financial analysis using your actual schema"""
        try:
            if analysis_year:
                query = """
                SELECT * FROM financial_analysis 
                WHERE company_id = %s AND analysis_year = %s
                """
                results = self.db.execute_query(query, (company_id, analysis_year))
                return results[0] if results else None
            else:
                query = """
                SELECT * FROM financial_analysis 
                WHERE company_id = %s 
                ORDER BY analysis_year DESC
                """
                return self.db.execute_query(query, (company_id,))
                
        except Exception as e:
            logger.error(f"Error retrieving financial analysis: {e}")
            return [] if not analysis_year else None
    
    # =====================================
    # STATISTICS & ANALYTICS (Updated for your schema)
    # =====================================
    
    def get_database_stats(self):
        """Get comprehensive database statistics using your actual schema"""
        try:
            query = """
            SELECT 
                (SELECT COUNT(*) FROM companies) as total_companies,
                (SELECT COUNT(*) FROM companies WHERE created_at >= NOW() - INTERVAL '30 days') as new_companies,
                (SELECT COUNT(*) FROM balance_sheet_1) as total_balance_sheets,
                (SELECT COUNT(*) FROM cash_flow_statement_1) as total_cash_flows,
                (SELECT COUNT(*) FROM cash_flow_statement) as legacy_cash_flows,
                (SELECT COUNT(*) FROM document_upload) as total_document_uploads,
                (SELECT COUNT(*) FROM uploaded_documents) as total_uploaded_documents,
                (SELECT COUNT(*) FROM financial_analysis) as total_analyses,
                (SELECT COUNT(*) FROM user_sessions) as total_sessions,
                (SELECT MAX(upload_date) FROM uploaded_documents) as last_upload_date,
                (SELECT MAX(generated_at) FROM balance_sheet_1) as last_balance_sheet_date,
                (SELECT MAX(generated_at) FROM cash_flow_statement_1) as last_cash_flow_date
            """
            results = self.db.execute_query(query)
            return results[0] if results else {}
            
        except Exception as e:
            logger.error(f"Error getting database statistics: {e}")
            return {}
    
    def get_companies_with_financial_health(self):
        """Get companies with calculated financial health using your actual schema"""
        try:
            query = """
            SELECT 
                c.*,
                bs.total_assets,
                bs.total_liabilities,
                bs.current_assets,
                bs.current_liabilities,
                bs.cash_and_equivalents,
                cf.net_income,
                cf.net_cash_from_operating_activities,
                cf.liquidation_label,
                cf.debt_to_equity_ratio,
                CASE 
                    WHEN bs.total_assets > 0 AND bs.total_liabilities >= 0 THEN
                        LEAST(100, GREATEST(0, 
                            (100 - (bs.total_liabilities::float / NULLIF(bs.total_assets, 0) * 100)) * 0.5 +
                            CASE WHEN cf.net_income > 0 THEN 25 ELSE 0 END +
                            CASE WHEN cf.net_cash_from_operating_activities > 0 THEN 25 ELSE 0 END
                        ))
                    ELSE 50
                END as financial_health_score
            FROM companies c
            LEFT JOIN balance_sheet_1 bs ON c.id = bs.company_id 
                AND bs.year = (SELECT MAX(year) FROM balance_sheet_1 WHERE company_id = c.id)
            LEFT JOIN cash_flow_statement_1 cf ON c.id = cf.company_id 
                AND cf.year = (SELECT MAX(year) FROM cash_flow_statement_1 WHERE company_id = c.id)
            ORDER BY financial_health_score DESC, c.company_name
            """
            return self.db.execute_query(query)
            
        except Exception as e:
            logger.error(f"Error getting companies with financial health: {e}")
            return []
    
    # =====================================
    # LEGACY CASH FLOW OPERATIONS (For backward compatibility)
    # =====================================
    
    def get_companies_from_cash_flow_table(self):
        """Get companies from legacy cash_flow_statement table"""
        try:
            query = """
            SELECT DISTINCT company_name 
            FROM cash_flow_statement 
            WHERE company_name IS NOT NULL 
            ORDER BY company_name
            """
            return self.db.execute_query(query)
            
        except Exception as e:
            logger.error(f"Error retrieving companies from cash flow table: {e}")
            return []
    
    def search_cash_flow_companies(self, search_term):
        """Search companies in legacy cash_flow_statement table"""
        try:
            query = """
            SELECT DISTINCT company_name 
            FROM cash_flow_statement 
            WHERE company_name ILIKE %s 
            ORDER BY company_name
            LIMIT 10
            """
            search_pattern = f"%{search_term}%"
            return self.db.execute_query(query, (search_pattern,))
            
        except Exception as e:
            logger.error(f"Error searching cash flow companies: {e}")
            return []
    
    def get_cash_flow_data_by_company(self, company_name):
        """Get cash flow data from legacy table"""
        try:
            query = """
            SELECT * FROM cash_flow_statement 
            WHERE company_name = %s 
            ORDER BY year DESC
            """
            return self.db.execute_query(query, (company_name,))
            
        except Exception as e:
            logger.error(f"Error retrieving cash flow data for {company_name}: {e}")
            return []
    
    # =====================================
    # USER SESSION OPERATIONS (Updated for your schema)
    # =====================================
    
    def create_user_session(self, session_id, company_id):
        """Create user session using your actual user_sessions schema"""
        try:
            query = """
            INSERT INTO user_sessions (session_id, company_id, last_activity, created_at)
            VALUES (%s, %s, NOW(), NOW())
            ON CONFLICT (session_id) 
            DO UPDATE SET company_id = EXCLUDED.company_id, last_activity = NOW()
            RETURNING id
            """
            
            result = self.db.execute_query(query, (session_id, company_id))
            
            if result and len(result) > 0:
                return result[0]['id']
            return None
            
        except Exception as e:
            logger.error(f"Error creating user session: {e}")
            return None
    
    def update_session_activity(self, session_id):
        """Update session last activity using your actual schema"""
        try:
            query = """
            UPDATE user_sessions 
            SET last_activity = NOW() 
            WHERE session_id = %s
            """
            
            result = self.db.execute_query(query, (session_id,))
            return result is not None
            
        except Exception as e:
            logger.error(f"Error updating session activity: {e}")
            return False
    
    def cleanup_old_sessions(self, days=30):
        """Clean up old sessions using your actual schema"""
        try:
            query = """
            DELETE FROM user_sessions 
            WHERE last_activity < NOW() - INTERVAL '%s days'
            """
            
            result = self.db.execute_query(query, (days,))
            return result is not None
            
        except Exception as e:
            logger.error(f"Error cleaning up old sessions: {e}")
            return False
    
    # =====================================
    # SEARCH OPERATIONS (Updated for your schema)
    # =====================================
    
    def search_companies_comprehensive(self, search_term, limit=20):
        """Comprehensive search across all relevant tables using your actual schema"""
        try:
            query = """
            SELECT DISTINCT
                c.id,
                c.company_name,
                c.created_at,
                c.updated_at,
                -- Latest financial data
                bs.total_assets,
                bs.total_liabilities,
                bs.year as balance_sheet_year,
                cf.net_income,
                cf.liquidation_label,
                cf.year as cash_flow_year,
                -- Industry from cash flow tables
                COALESCE(cf.industry, cf_legacy.industry) as industry,
                -- Document counts
                (SELECT COUNT(*) FROM uploaded_documents ud WHERE ud.company_name = c.company_name) as document_count,
                -- Risk assessment
                fa.risk_score,
                fa.risk_level
            FROM companies c
            LEFT JOIN balance_sheet_1 bs ON c.id = bs.company_id 
                AND bs.year = (SELECT MAX(year) FROM balance_sheet_1 WHERE company_id = c.id)
            LEFT JOIN cash_flow_statement_1 cf ON c.id = cf.company_id 
                AND cf.year = (SELECT MAX(year) FROM cash_flow_statement_1 WHERE company_id = c.id)
            LEFT JOIN cash_flow_statement cf_legacy ON c.company_name = cf_legacy.company_name
            LEFT JOIN financial_analysis fa ON c.id = fa.company_id
                AND fa.created_at = (SELECT MAX(created_at) FROM financial_analysis WHERE company_id = c.id)
            WHERE 
                c.company_name ILIKE %s
                OR COALESCE(cf.industry, cf_legacy.industry) ILIKE %s
                OR COALESCE(cf.company_name, cf_legacy.company_name) ILIKE %s
            ORDER BY c.company_name
            LIMIT %s
            """
            search_pattern = f"%{search_term}%"
            return self.db.execute_query(query, (search_pattern, search_pattern, search_pattern, limit))
            
        except Exception as e:
            logger.error(f"Error in comprehensive search: {e}")
            return []
    
    # =====================================
    # VALIDATION OPERATIONS (Updated for your schema)
    # =====================================
    
    def validate_company_data_completeness(self, company_id):
        """Validate company data completeness using your actual schema"""
        try:
            query = """
            SELECT 
                c.company_name,
                c.created_at,
                c.updated_at,
                COUNT(DISTINCT bs.id) as balance_sheet_count,
                COUNT(DISTINCT cf.id) as cash_flow_count,
                COUNT(DISTINCT du.id) as document_upload_count,
                COUNT(DISTINCT ud.id) as uploaded_document_count,
                COUNT(DISTINCT fa.id) as financial_analysis_count,
                CASE 
                    WHEN COUNT(DISTINCT bs.id) > 0 AND COUNT(DISTINCT cf.id) > 0 THEN 'complete'
                    WHEN COUNT(DISTINCT bs.id) > 0 OR COUNT(DISTINCT cf.id) > 0 THEN 'partial'
                    ELSE 'minimal'
                END as data_completeness_level
            FROM companies c
            LEFT JOIN balance_sheet_1 bs ON c.id = bs.company_id
            LEFT JOIN cash_flow_statement_1 cf ON c.id = cf.company_id
            LEFT JOIN document_upload du ON c.id = du.company_id
            LEFT JOIN uploaded_documents ud ON c.company_name = ud.company_name
            LEFT JOIN financial_analysis fa ON c.id = fa.company_id
            WHERE c.id = %s
            GROUP BY c.id, c.company_name, c.created_at, c.updated_at
            """
            results = self.db.execute_query(query, (company_id,))
            return results[0] if results else None
            
        except Exception as e:
            logger.error(f"Error validating company data completeness: {e}")
            return None
    
    def get_document_upload_stats_by_company(self, company_name):
        """Get upload statistics using your actual schema"""
        try:
            query = """
            SELECT 
                document_type,
                COUNT(*) as file_count,
                SUM(file_size) as total_size,
                MAX(upload_date) as last_upload,
                COUNT(CASE WHEN processing_status = 'processed' THEN 1 END) as processed_count,
                COUNT(CASE WHEN processing_status = 'failed' THEN 1 END) as failed_count,
                COUNT(CASE WHEN processing_status = 'uploaded' THEN 1 END) as pending_count
            FROM uploaded_documents 
            WHERE company_name = %s 
            GROUP BY document_type
            ORDER BY last_upload DESC
            """
            return self.db.execute_query(query, (company_name,))
            
        except Exception as e:
            logger.error(f"Error getting upload stats for {company_name}: {e}")
            return []
    
    # =====================================
    # BULK OPERATIONS (Updated for your schema)
    # =====================================
    
    def bulk_update_processing_status(self, document_ids, new_status):
        """Bulk update document processing status using your actual schema"""
        try:
            if not document_ids:
                return False
            
            placeholders = ','.join(['%s'] * len(document_ids))
            query = f"""
            UPDATE uploaded_documents 
            SET processing_status = %s, updated_at = NOW()
            WHERE id IN ({placeholders})
            """
            params = [new_status] + list(document_ids)
            
            result = self.db.execute_query(query, tuple(params))
            return result is not None
            
        except Exception as e:
            logger.error(f"Error in bulk status update: {e}")
            return False
    
    def bulk_delete_documents(self, document_ids):
        """Bulk delete documents using your actual schema"""
        try:
            if not document_ids:
                return False
            
            placeholders = ','.join(['%s'] * len(document_ids))
            query = f"""
            DELETE FROM uploaded_documents 
            WHERE id IN ({placeholders})
            """
            
            result = self.db.execute_query(query, tuple(document_ids))
            return result is not None
            
        except Exception as e:
            logger.error(f"Error in bulk document deletion: {e}")
            return False
    
    def bulk_analyze_companies(self, company_ids):
        """Bulk financial analysis for companies using your actual schema"""
        try:
            if not company_ids:
                return []
            
            placeholders = ','.join(['%s'] * len(company_ids))
            query = f"""
            SELECT 
                c.id,
                c.company_name,
                bs.total_assets,
                bs.total_liabilities,
                bs.current_assets,
                bs.current_liabilities,
                cf.net_income,
                cf.net_cash_from_operating_activities,
                cf.liquidation_label,
                cf.debt_to_equity_ratio
            FROM companies c
            LEFT JOIN balance_sheet_1 bs ON c.id = bs.company_id 
                AND bs.year = (SELECT MAX(year) FROM balance_sheet_1 WHERE company_id = c.id)
            LEFT JOIN cash_flow_statement_1 cf ON c.id = cf.company_id 
                AND cf.year = (SELECT MAX(year) FROM cash_flow_statement_1 WHERE company_id = c.id)
            WHERE c.id IN ({placeholders})
            """
            
            return self.db.execute_query(query, tuple(company_ids))
            
        except Exception as e:
            logger.error(f"Error in bulk company analysis: {e}")
            return []
    
    # =====================================
    # REPORTING OPERATIONS (Updated for your schema)
    # =====================================
    
    def generate_company_report_data(self, company_id):
        """Generate comprehensive report data using your actual schema"""
        try:
            # Get company basic info
            company = self.get_company_by_id(company_id)
            if not company:
                return None
            
            # Get financial data
            balance_sheets = self.get_generated_balance_sheets(company_id)
            cash_flows = self.get_generated_cash_flows(company_id)
            documents = self.get_company_documents(company_id)
            uploaded_docs = self.get_company_documents_by_name(company['company_name'])
            
            # Get financial analysis
            financial_analysis = self.get_financial_analysis(company_id)
            
            # Calculate summary metrics
            latest_balance_sheet = balance_sheets[0] if balance_sheets else None
            latest_cash_flow = cash_flows[0] if cash_flows else None
            
            # Get years covered
            years_covered = list(set(
                [bs['year'] for bs in balance_sheets if bs.get('year')] + 
                [cf['year'] for cf in cash_flows if cf.get('year')]
            ))
            
            report_data = {
                'company': company,
                'balance_sheets': balance_sheets,
                'cash_flows': cash_flows,
                'documents': documents,
                'uploaded_documents': uploaded_docs,
                'financial_analysis': financial_analysis,
                'latest_balance_sheet': latest_balance_sheet,
                'latest_cash_flow': latest_cash_flow,
                'summary': {
                    'total_balance_sheets': len(balance_sheets),
                    'total_cash_flows': len(cash_flows),
                    'total_documents': len(documents) + len(uploaded_docs),
                    'years_covered': sorted(years_covered, reverse=True),
                    'data_completeness': self.validate_company_data_completeness(company_id)
                }
            }
            
            return report_data
            
        except Exception as e:
            logger.error(f"Error generating company report data: {e}")
            return None
    
    def get_companies_summary_for_dashboard(self):
        """Get companies summary for dashboard using your actual schema"""
        try:
            query = """
            SELECT 
                c.id,
                c.company_name,
                c.created_at,
                c.updated_at,
                COUNT(DISTINCT bs.id) as balance_sheet_count,
                COUNT(DISTINCT cf.id) as cash_flow_count,
                COUNT(DISTINCT du.id) as document_count,
                COUNT(DISTINCT ud.id) as uploaded_document_count,
                GREATEST(
                    COALESCE(MAX(bs.generated_at), c.created_at),
                    COALESCE(MAX(cf.generated_at), c.created_at),
                    COALESCE(MAX(du.created_at), c.created_at),
                    COALESCE(MAX(ud.upload_date), c.created_at),
                    c.updated_at
                ) as last_activity,
                CASE 
                    WHEN bs.total_assets > 0 AND bs.total_liabilities >= 0 THEN
                        LEAST(100, GREATEST(0, 
                            (100 - (bs.total_liabilities::float / NULLIF(bs.total_assets, 0) * 100)) * 0.5 +
                            CASE WHEN cf.net_income > 0 THEN 25 ELSE 0 END +
                            CASE WHEN cf.net_cash_from_operating_activities > 0 THEN 25 ELSE 0 END
                        ))
                    ELSE 50
                END as financial_health_score,
                fa.risk_level,
                fa.liquidation_risk
            FROM companies c
            LEFT JOIN balance_sheet_1 bs ON c.id = bs.company_id 
                AND bs.year = (SELECT MAX(year) FROM balance_sheet_1 WHERE company_id = c.id)
            LEFT JOIN cash_flow_statement_1 cf ON c.id = cf.company_id 
                AND cf.year = (SELECT MAX(year) FROM cash_flow_statement_1 WHERE company_id = c.id)
            LEFT JOIN document_upload du ON c.id = du.company_id
            LEFT JOIN uploaded_documents ud ON c.company_name = ud.company_name
            LEFT JOIN financial_analysis fa ON c.id = fa.company_id
                AND fa.created_at = (SELECT MAX(created_at) FROM financial_analysis WHERE company_id = c.id)
            GROUP BY c.id, c.company_name, c.created_at, c.updated_at, 
                     bs.total_assets, bs.total_liabilities, cf.net_income, 
                     cf.net_cash_from_operating_activities, fa.risk_level, fa.liquidation_risk
            ORDER BY last_activity DESC, c.company_name
            LIMIT 20
            """
            return self.db.execute_query(query)
            
        except Exception as e:
            logger.error(f"Error getting companies summary for dashboard: {e}")
            return []
    
    # =====================================
    # CLEANUP OPERATIONS (Updated for your schema)
    # =====================================
    
    def cleanup_failed_uploads(self, days=7):
        """Clean up failed uploads using your actual schema"""
        try:
            query = """
            DELETE FROM uploaded_documents 
            WHERE processing_status = 'failed' 
            AND upload_date < NOW() - INTERVAL '%s days'
            """
            
            result = self.db.execute_query(query, (days,))
            return result is not None
            
        except Exception as e:
            logger.error(f"Error cleaning up failed uploads: {e}")
            return False
    
    def cleanup_test_data(self):
        """Clean up test data using your actual schema"""
        try:
            # Delete test companies and related data
            query = "DELETE FROM companies WHERE company_name LIKE '%Test%' OR company_name LIKE '%test%'"
            result = self.db.execute_query(query)
            return result is not None
            
        except Exception as e:
            logger.error(f"Error cleaning up test data: {e}")
            return False
    
    def cleanup_orphaned_data(self):
        """Clean up orphaned data using your actual schema"""
        try:
            # Clean up uploaded_documents with no corresponding company
            query1 = """
            DELETE FROM uploaded_documents 
            WHERE company_name NOT IN (SELECT company_name FROM companies)
            """
            
            # Clean up balance_sheet_1 with no corresponding company
            query2 = """
            DELETE FROM balance_sheet_1 
            WHERE company_id NOT IN (SELECT id FROM companies)
            """
            
            # Clean up cash_flow_statement_1 with no corresponding company
            query3 = """
            DELETE FROM cash_flow_statement_1 
            WHERE company_id NOT IN (SELECT id FROM companies)
            """
            
            # Clean up financial_analysis with no corresponding company
            query4 = """
            DELETE FROM financial_analysis 
            WHERE company_id NOT IN (SELECT id FROM companies)
            """
            
            # Clean up document_upload with no corresponding company
            query5 = """
            DELETE FROM document_upload 
            WHERE company_id NOT IN (SELECT id FROM companies)
            """
            
            # Clean up user_sessions with no corresponding company
            query6 = """
            DELETE FROM user_sessions 
            WHERE company_id NOT IN (SELECT id FROM companies)
            """
            
            cleanup_queries = [query1, query2, query3, query4, query5, query6]
            
            for query in cleanup_queries:
                try:
                    self.db.execute_query(query)
                except Exception as e:
                    logger.warning(f"Error in cleanup query: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up orphaned data: {e}")
            return False
    
    # =====================================
    # UTILITY OPERATIONS (Updated for your schema)
    # =====================================
    
    def get_recent_activity(self, limit=20):
        """Get recent activity across all tables using your actual schema"""
        try:
            query = """
            SELECT 
                'company_created' as activity_type,
                company_name as description,
                created_at as activity_date,
                id as reference_id
            FROM companies
            WHERE created_at >= NOW() - INTERVAL '30 days'
            
            UNION ALL
            
            SELECT 
                'document_uploaded' as activity_type,
                company_name || ' - ' || original_filename as description,
                upload_date as activity_date,
                id as reference_id
            FROM uploaded_documents
            WHERE upload_date >= NOW() - INTERVAL '30 days'
            
            UNION ALL
            
            SELECT 
                'balance_sheet_generated' as activity_type,
                (SELECT company_name FROM companies WHERE id = bs.company_id) || ' - Year ' || bs.year as description,
                bs.generated_at as activity_date,
                bs.id as reference_id
            FROM balance_sheet_1 bs
            WHERE bs.generated_at >= NOW() - INTERVAL '30 days'
            
            UNION ALL
            
            SELECT 
                'cash_flow_generated' as activity_type,
                (SELECT company_name FROM companies WHERE id = cf.company_id) || ' - Year ' || cf.year as description,
                cf.generated_at as activity_date,
                cf.id as reference_id
            FROM cash_flow_statement_1 cf
            WHERE cf.generated_at >= NOW() - INTERVAL '30 days'
            
            UNION ALL
            
            SELECT 
                'financial_analysis' as activity_type,
                (SELECT company_name FROM companies WHERE id = fa.company_id) || ' - Analysis ' || fa.analysis_year as description,
                fa.created_at as activity_date,
                fa.id as reference_id
            FROM financial_analysis fa
            WHERE fa.created_at >= NOW() - INTERVAL '30 days'
            
            ORDER BY activity_date DESC
            LIMIT %s
            """
            return self.db.execute_query(query, (limit,))
            
        except Exception as e:
            logger.error(f"Error retrieving recent activity: {e}")
            return []
    
    def check_data_integrity(self):
        """Check data integrity across tables using your actual schema"""
        try:
            integrity_checks = []
            
            # Check for balance sheets without companies
            query1 = """
            SELECT COUNT(*) as orphaned_balance_sheets
            FROM balance_sheet_1 bs
            WHERE bs.company_id NOT IN (SELECT id FROM companies)
            """
            result1 = self.db.execute_query(query1)
            if result1:
                integrity_checks.append({
                    'check': 'orphaned_balance_sheets',
                    'count': result1[0]['orphaned_balance_sheets']
                })
            
            # Check for cash flows without companies
            query2 = """
            SELECT COUNT(*) as orphaned_cash_flows
            FROM cash_flow_statement_1 cf
            WHERE cf.company_id NOT IN (SELECT id FROM companies)
            """
            result2 = self.db.execute_query(query2)
            if result2:
                integrity_checks.append({
                    'check': 'orphaned_cash_flows',
                    'count': result2[0]['orphaned_cash_flows']
                })
            
            # Check for uploaded documents without companies
            query3 = """
            SELECT COUNT(*) as orphaned_documents
            FROM uploaded_documents ud
            WHERE ud.company_name NOT IN (SELECT company_name FROM companies)
            """
            result3 = self.db.execute_query(query3)
            if result3:
                integrity_checks.append({
                    'check': 'orphaned_documents',
                    'count': result3[0]['orphaned_documents']
                })
            
            # Check for financial analysis without companies
            query4 = """
            SELECT COUNT(*) as orphaned_analysis
            FROM financial_analysis fa
            WHERE fa.company_id NOT IN (SELECT id FROM companies)
            """
            result4 = self.db.execute_query(query4)
            if result4:
                integrity_checks.append({
                    'check': 'orphaned_analysis',
                    'count': result4[0]['orphaned_analysis']
                })
            
            return integrity_checks
            
        except Exception as e:
            logger.error(f"Error checking data integrity: {e}")
            return []
    
    def get_table_sizes(self):
        """Get table sizes using your actual schema"""
        try:
            query = """
            SELECT 
                'companies' as table_name,
                COUNT(*) as row_count,
                pg_size_pretty(pg_total_relation_size('companies')) as table_size
            FROM companies
            
            UNION ALL
            
            SELECT 
                'balance_sheet_1' as table_name,
                COUNT(*) as row_count,
                pg_size_pretty(pg_total_relation_size('balance_sheet_1')) as table_size
            FROM balance_sheet_1
            
            UNION ALL
            
            SELECT 
                'cash_flow_statement_1' as table_name,
                COUNT(*) as row_count,
                pg_size_pretty(pg_total_relation_size('cash_flow_statement_1')) as table_size
            FROM cash_flow_statement_1
            
            UNION ALL
            
            SELECT 
                'cash_flow_statement' as table_name,
                COUNT(*) as row_count,
                pg_size_pretty(pg_total_relation_size('cash_flow_statement')) as table_size
            FROM cash_flow_statement
            
            UNION ALL
            
            SELECT 
                'uploaded_documents' as table_name,
                COUNT(*) as row_count,
                pg_size_pretty(pg_total_relation_size('uploaded_documents')) as table_size
            FROM uploaded_documents
            
            UNION ALL
            
            SELECT 
                'document_upload' as table_name,
                COUNT(*) as row_count,
                pg_size_pretty(pg_total_relation_size('document_upload')) as table_size
            FROM document_upload
            
            UNION ALL
            
            SELECT 
                'financial_analysis' as table_name,
                COUNT(*) as row_count,
                pg_size_pretty(pg_total_relation_size('financial_analysis')) as table_size
            FROM financial_analysis
            
            UNION ALL
            
            SELECT 
                'user_sessions' as table_name,
                COUNT(*) as row_count,
                pg_size_pretty(pg_total_relation_size('user_sessions')) as table_size
            FROM user_sessions
            
            ORDER BY row_count DESC
            """
            return self.db.execute_query(query)
            
        except Exception as e:
            logger.error(f"Error getting table sizes: {e}")
            return []
    
    # =====================================
    # ADVANCED ANALYTICS OPERATIONS
    # =====================================
    
    def get_financial_trends_by_company(self, company_id, years=3):
        """Get financial trends for a specific company"""
        try:
            query = """
            SELECT 
                c.company_name,
                bs.year,
                bs.total_assets,
                bs.total_liabilities,
                bs.total_equity,
                bs.current_assets,
                bs.current_liabilities,
                cf.net_income,
                cf.net_cash_from_operating_activities,
                cf.free_cash_flow,
                cf.liquidation_label
            FROM companies c
            LEFT JOIN balance_sheet_1 bs ON c.id = bs.company_id
            LEFT JOIN cash_flow_statement_1 cf ON c.id = cf.company_id AND bs.year = cf.year
            WHERE c.id = %s
            ORDER BY bs.year DESC
            LIMIT %s
            """
            return self.db.execute_query(query, (company_id, years))
            
        except Exception as e:
            logger.error(f"Error getting financial trends: {e}")
            return []
    
    def get_industry_comparison_data(self, company_id):
        """Get industry comparison data for a company"""
        try:
            # First get company's industry
            company_query = """
            SELECT cf.industry 
            FROM companies c
            LEFT JOIN cash_flow_statement_1 cf ON c.id = cf.company_id
            WHERE c.id = %s AND cf.industry IS NOT NULL
            LIMIT 1
            """
            company_result = self.db.execute_query(company_query, (company_id,))
            
            if not company_result or not company_result[0]['industry']:
                return []
            
            industry = company_result[0]['industry']
            
            # Get industry averages
            industry_query = """
            SELECT 
                AVG(bs.total_assets) as avg_total_assets,
                AVG(bs.total_liabilities) as avg_total_liabilities,
                AVG(cf.net_income) as avg_net_income,
                AVG(cf.net_cash_from_operating_activities) as avg_operating_cash_flow,
                COUNT(DISTINCT c.id) as company_count
            FROM companies c
            LEFT JOIN balance_sheet_1 bs ON c.id = bs.company_id
            LEFT JOIN cash_flow_statement_1 cf ON c.id = cf.company_id AND cf.industry = %s
            WHERE cf.industry = %s
            """
            
            return self.db.execute_query(industry_query, (industry, industry))
            
        except Exception as e:
            logger.error(f"Error getting industry comparison: {e}")
            return []
    
    def get_risk_distribution(self):
        """Get risk distribution across all companies"""
        try:
            query = """
            SELECT 
                fa.risk_level,
                COUNT(*) as company_count,
                AVG(fa.risk_score) as avg_risk_score,
                AVG(fa.liquidation_risk) as avg_liquidation_risk
            FROM financial_analysis fa
            WHERE fa.risk_level IS NOT NULL
            GROUP BY fa.risk_level
            ORDER BY AVG(fa.risk_score) DESC
            """
            return self.db.execute_query(query)
            
        except Exception as e:
            logger.error(f"Error getting risk distribution: {e}")
            return []
    
    # =====================================
    # EXPORT OPERATIONS
    # =====================================
    
    def export_company_data(self, company_id, include_financial=True):
        """Export complete company data for backup or analysis"""
        try:
            export_data = {
                'company': self.get_company_by_id(company_id),
                'documents': self.get_company_documents(company_id),
                'uploaded_documents': None,
                'balance_sheets': [],
                'cash_flows': [],
                'financial_analyses': []
            }
            
            if export_data['company']:
                company_name = export_data['company']['company_name']
                export_data['uploaded_documents'] = self.get_company_documents_by_name(company_name)
                
                if include_financial:
                    export_data['balance_sheets'] = self.get_generated_balance_sheets(company_id)
                    export_data['cash_flows'] = self.get_generated_cash_flows(company_id)
                    export_data['financial_analyses'] = self.get_financial_analysis(company_id)
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting company data: {e}")
            return None
    
    def get_system_health_metrics(self):
        """Get system health and performance metrics"""
        try:
            query = """
            SELECT 
                (SELECT COUNT(*) FROM companies) as total_companies,
                (SELECT COUNT(*) FROM uploaded_documents WHERE processing_status = 'failed') as failed_uploads,
                (SELECT COUNT(*) FROM user_sessions WHERE last_activity >= NOW() - INTERVAL '1 hour') as active_sessions,
                (SELECT AVG(file_size) FROM uploaded_documents WHERE file_size > 0) as avg_file_size,
                (SELECT COUNT(*) FROM financial_analysis WHERE created_at >= NOW() - INTERVAL '24 hours') as recent_analyses
            """
            results = self.db.execute_query(query)
            return results[0] if results else {}
            
        except Exception as e:
            logger.error(f"Error getting system health metrics: {e}")
            return {}