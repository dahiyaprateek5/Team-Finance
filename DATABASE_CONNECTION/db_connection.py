import psycopg2
from psycopg2.extras import RealDictCursor
import os
from datetime import datetime
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseConnection:
    """PostgreSQL Database Connection Handler for Financial Risk Assessment Platform"""
    
    def __init__(self):
        self.connection = None
        self.cursor = None
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        
        # PostgreSQL credentials with fallbacks
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '5432')),
            'database': os.getenv('DB_NAME', 'team_finance_db'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'Prateek@2003'),
            'connect_timeout': 10,
            'application_name': 'FinancialRiskAssessment'
        }
        
        logger.info(f"🔧 Database config initialized: {self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}")
    
    def connect(self):
        """Database se connection establish karta hai with retry logic"""
        for attempt in range(self.max_retries):
            try:
                # Check if connection already exists and is valid
                if self.connection and not self.connection.closed:
                    return True
                
                logger.info(f"🔄 Attempting database connection (attempt {attempt + 1}/{self.max_retries})")
                
                # Establish new connection
                self.connection = psycopg2.connect(**self.db_config)
                self.connection.autocommit = False  # Explicit transaction control
                self.cursor = self.connection.cursor(cursor_factory=RealDictCursor)
                
                # Test the connection
                self.cursor.execute("SELECT 1 as test, NOW() as timestamp")
                result = self.cursor.fetchone()
                
                if result and result['test'] == 1:
                    logger.info(f"✅ Database connection established successfully at {result['timestamp']}")
                    return True
                else:
                    raise Exception("Connection test failed")
                    
            except psycopg2.OperationalError as e:
                logger.warning(f"⚠️ Database connection attempt {attempt + 1} failed: {e}")
                self.cleanup_connection()
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    self.retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"❌ Database connection failed after {self.max_retries} attempts")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Unexpected database connection error: {e}")
                self.cleanup_connection()
                return False
        
        return False
    
    def cleanup_connection(self):
        """Clean up connection and cursor objects"""
        try:
            if self.cursor and not self.cursor.closed:
                self.cursor.close()
            if self.connection and not self.connection.closed:
                self.connection.close()
        except:
            pass
        finally:
            self.cursor = None
            self.connection = None
    
    def test_connection(self):
        """Comprehensive database connection test"""
        try:
            logger.info("🧪 Testing database connection...")
            
            if not self.connect():
                return False
            
            # Test basic query
            test_queries = [
                ("SELECT 1 as test", "Basic query test"),
                ("SELECT NOW() as current_time", "Timestamp test"),
                ("SELECT version() as pg_version", "PostgreSQL version check")
            ]
            
            for query, description in test_queries:
                try:
                    self.cursor.execute(query)
                    result = self.cursor.fetchone()
                    logger.info(f"✅ {description}: {result}")
                except Exception as e:
                    logger.error(f"❌ {description} failed: {e}")
                    return False
            
            # Test table access for your schema
            schema_tables = [
                'companies', 'balance_sheet_1', 'cash_flow_statement_1', 
                'financial_analysis', 'uploaded_documents', 'document_upload', 'user_sessions'
            ]
            
            accessible_tables = []
            for table in schema_tables:
                if self.table_exists(table):
                    accessible_tables.append(table)
            
            logger.info(f"✅ Accessible tables: {accessible_tables}")
            
            if len(accessible_tables) > 0:
                logger.info("✅ Database connection test successful - Schema accessible")
                return True
            else:
                logger.warning("⚠️ Connection successful but no expected tables found")
                return False
                
        except Exception as e:
            logger.error(f"❌ Connection test error: {e}")
            return False
    
    def disconnect(self):
        """Database connection close karta hai"""
        try:
            if self.cursor and not self.cursor.closed:
                self.cursor.close()
                logger.debug("🔒 Database cursor closed")
            
            if self.connection and not self.connection.closed:
                self.connection.close()
                logger.info("🔒 Database connection closed")
            
            self.cursor = None
            self.connection = None
            
        except psycopg2.Error as e:
            logger.error(f"❌ Error closing connection: {e}")
        except Exception as e:
            logger.error(f"❌ Unexpected error closing connection: {e}")
    
    def close(self):
        """Alias for disconnect - for compatibility"""
        self.disconnect()
    
    def execute_query(self, query, params=None):
        """Enhanced SQL query execution with better error handling"""
        try:
            # Ensure connection is active
            if not self.connection or self.connection.closed:
                if not self.connect():
                    logger.error("❌ Cannot execute query - connection failed")
                    return False
            
            # Create fresh cursor if needed
            if not self.cursor or self.cursor.closed:
                self.cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            
            # Log query for debugging (be careful with sensitive data)
            query_preview = query[:100] + "..." if len(query) > 100 else query
            logger.debug(f"🔍 Executing query: {query_preview}")
            
            # Execute query
            start_time = time.time()
            self.cursor.execute(query, params)
            execution_time = time.time() - start_time
            
            # Handle different query types
            query_upper = query.strip().upper()
            
            if query_upper.startswith('SELECT'):
                results = self.cursor.fetchall()
                logger.debug(f"✅ SELECT query completed in {execution_time:.3f}s, returned {len(results)} rows")
                return results
                
            elif query_upper.startswith(('INSERT', 'UPDATE', 'DELETE')):
                # Check for RETURNING clause first
                if 'RETURNING' in query_upper:
                    results = self.cursor.fetchall()
                    self.connection.commit()
                    logger.debug(f"✅ {query_upper.split()[0]} query with RETURNING completed in {execution_time:.3f}s")
                    return results
                else:
                    affected_rows = self.cursor.rowcount
                    self.connection.commit()
                    logger.debug(f"✅ {query_upper.split()[0]} query completed in {execution_time:.3f}s, affected {affected_rows} rows")
                    return True
                    
            elif query_upper.startswith(('CREATE', 'ALTER', 'DROP', 'TRUNCATE')):
                self.connection.commit()
                logger.debug(f"✅ DDL query completed in {execution_time:.3f}s")
                return True
                
            else:
                # For other types (GRANT, REVOKE, etc.)
                self.connection.commit()
                logger.debug(f"✅ Query completed in {execution_time:.3f}s")
                return True
                
        except psycopg2.IntegrityError as e:
            if self.connection:
                self.connection.rollback()
            logger.error(f"❌ Integrity constraint violation: {e}")
            return False
            
        except psycopg2.DataError as e:
            if self.connection:
                self.connection.rollback()
            logger.error(f"❌ Data error in query: {e}")
            return False
            
        except psycopg2.OperationalError as e:
            if self.connection:
                self.connection.rollback()
            logger.error(f"❌ Operational error (possibly connection lost): {e}")
            # Try to reconnect for next query
            self.cleanup_connection()
            return False
            
        except psycopg2.Error as e:
            if self.connection:
                self.connection.rollback()
            logger.error(f"❌ PostgreSQL error: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            return False
            
        except Exception as e:
            if self.connection:
                self.connection.rollback()
            logger.error(f"❌ Unexpected error in query execution: {e}")
            return False
    
    def execute_insert(self, query, params=None):
        """Insert query execute karta hai with enhanced return handling"""
        try:
            if not self.connection or self.connection.closed:
                if not self.connect():
                    return None
            
            if not self.cursor or self.cursor.closed:
                self.cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            
            self.cursor.execute(query, params)
            
            # Handle RETURNING clause
            if 'RETURNING' in query.upper():
                result = self.cursor.fetchone()
                self.connection.commit()
                if result:
                    # Return the first column value (usually ID)
                    return result[list(result.keys())[0]]
                else:
                    return None
            else:
                affected_rows = self.cursor.rowcount
                self.connection.commit()
                return affected_rows > 0
                
        except psycopg2.Error as e:
            if self.connection:
                self.connection.rollback()
            logger.error(f"❌ Insert query error: {e}")
            return None
    
    def execute_select(self, query, params=None):
        """Select query execute karta hai with enhanced error handling"""
        try:
            if not self.connection or self.connection.closed:
                if not self.connect():
                    return []
            
            if not self.cursor or self.cursor.closed:
                self.cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            
            self.cursor.execute(query, params)
            results = self.cursor.fetchall()
            
            # Convert to list of dicts for easier handling
            return [dict(row) for row in results]
            
        except psycopg2.Error as e:
            logger.error(f"❌ Select query error: {e}")
            return []
        except Exception as e:
            logger.error(f"❌ Unexpected error in select query: {e}")
            return []
    
    def execute_many(self, query, params_list):
        """Multiple queries execute karta hai (batch processing)"""
        try:
            if not self.connection or self.connection.closed:
                if not self.connect():
                    return False
            
            if not self.cursor or self.cursor.closed:
                self.cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            
            start_time = time.time()
            self.cursor.executemany(query, params_list)
            execution_time = time.time() - start_time
            
            self.connection.commit()
            
            logger.info(f"✅ Batch query completed in {execution_time:.3f}s, processed {len(params_list)} records")
            return True
            
        except psycopg2.Error as e:
            if self.connection:
                self.connection.rollback()
            logger.error(f"❌ Batch query error: {e}")
            return False
    
    def get_table_columns(self, table_name):
        """Table ke columns ki detailed information return karta hai"""
        try:
            query = """
            SELECT 
                column_name, 
                data_type, 
                is_nullable, 
                column_default,
                character_maximum_length,
                numeric_precision,
                numeric_scale
            FROM information_schema.columns 
            WHERE table_name = %s 
            ORDER BY ordinal_position;
            """
            return self.execute_select(query, (table_name,))
            
        except Exception as e:
            logger.error(f"❌ Error getting table columns for {table_name}: {e}")
            return []
    
    def table_exists(self, table_name):
        """Check if table exists in current database"""
        try:
            query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = %s 
                AND table_schema = 'public'
            );
            """
            result = self.execute_select(query, (table_name,))
            return result[0]['exists'] if result else False
            
        except Exception as e:
            logger.error(f"❌ Error checking table existence for {table_name}: {e}")
            return False
    
    def get_database_stats(self):
        """Comprehensive database statistics for your schema"""
        try:
            stats = {}
            
            # Your actual tables
            tables = [
                'companies', 'balance_sheet_1', 'cash_flow_statement_1', 'cash_flow_statement',
                'uploaded_documents', 'document_upload', 'financial_analysis', 'user_sessions'
            ]
            
            for table in tables:
                if self.table_exists(table):
                    try:
                        # Get row count
                        result = self.execute_select(f"SELECT COUNT(*) as count FROM {table}")
                        stats[f'{table}_count'] = result[0]['count'] if result else 0
                        
                        # Get table size
                        size_query = f"SELECT pg_size_pretty(pg_total_relation_size('{table}')) as size"
                        size_result = self.execute_select(size_query)
                        stats[f'{table}_size'] = size_result[0]['size'] if size_result else 'Unknown'
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Could not get stats for table {table}: {e}")
                        stats[f'{table}_count'] = 0
                        stats[f'{table}_size'] = 'Error'
                else:
                    stats[f'{table}_count'] = 0
                    stats[f'{table}_size'] = 'Table not found'
            
            # Get database-level stats
            try:
                db_size_query = "SELECT pg_size_pretty(pg_database_size(current_database())) as db_size"
                db_size_result = self.execute_select(db_size_query)
                stats['database_size'] = db_size_result[0]['db_size'] if db_size_result else 'Unknown'
                
                # Get connection info
                conn_info_query = """
                SELECT 
                    current_database() as database_name,
                    current_user as user_name,
                    version() as postgresql_version,
                    inet_server_addr() as server_ip,
                    inet_server_port() as server_port
                """
                conn_info = self.execute_select(conn_info_query)
                if conn_info:
                    stats.update(conn_info[0])
                    
            except Exception as e:
                logger.warning(f"⚠️ Could not get database-level stats: {e}")
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting database stats: {e}")
            return {}
    
    def check_schema_health(self):
        """Check the health of your database schema"""
        try:
            health_report = {
                'status': 'healthy',
                'issues': [],
                'warnings': [],
                'table_status': {}
            }
            
            # Expected tables for your project
            expected_tables = {
                'companies': ['id', 'company_name', 'created_at', 'updated_at'],
                'balance_sheet_1': ['id', 'company_id', 'year', 'total_assets', 'total_liabilities'],
                'cash_flow_statement_1': ['id', 'company_id', 'year', 'net_income', 'liquidation_label'],
                'financial_analysis': ['id', 'company_id', 'analysis_year', 'risk_score', 'risk_level'],
                'uploaded_documents': ['id', 'company_name', 'document_type', 'upload_date'],
                'document_upload': ['id', 'company_id', 'document_type', 'created_at'],
                'user_sessions': ['id', 'session_id', 'company_id', 'last_activity']
            }
            
            for table, key_columns in expected_tables.items():
                if self.table_exists(table):
                    health_report['table_status'][table] = 'exists'
                    
                    # Check key columns
                    columns = self.get_table_columns(table)
                    existing_columns = [col['column_name'] for col in columns]
                    
                    missing_columns = [col for col in key_columns if col not in existing_columns]
                    if missing_columns:
                        health_report['warnings'].append(f"Table {table} missing columns: {missing_columns}")
                        health_report['table_status'][table] = 'incomplete'
                else:
                    health_report['issues'].append(f"Required table {table} does not exist")
                    health_report['table_status'][table] = 'missing'
                    health_report['status'] = 'unhealthy'
            
            if health_report['warnings'] and health_report['status'] == 'healthy':
                health_report['status'] = 'warning'
            
            return health_report
            
        except Exception as e:
            logger.error(f"❌ Error checking schema health: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_connection_info(self):
        """Get current connection information"""
        try:
            if not self.connection or self.connection.closed:
                return {'status': 'disconnected'}
            
            info = {
                'status': 'connected',
                'host': self.db_config['host'],
                'port': self.db_config['port'],
                'database': self.db_config['database'],
                'user': self.db_config['user'],
                'application_name': self.db_config.get('application_name', 'Unknown'),
                'connection_id': self.connection.get_backend_pid(),
                'autocommit': self.connection.autocommit,
                'isolation_level': self.connection.isolation_level
            }
            
            return info
            
        except Exception as e:
            logger.error(f"❌ Error getting connection info: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def __enter__(self):
        """Context manager support"""
        if self.connect():
            return self
        else:
            raise Exception("Failed to establish database connection")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        try:
            if exc_type:
                if self.connection and not self.connection.closed:
                    self.connection.rollback()
                    logger.info("🔄 Transaction rolled back due to exception")
            else:
                if self.connection and not self.connection.closed:
                    self.connection.commit()
                    logger.debug("✅ Transaction committed successfully")
        except Exception as e:
            logger.error(f"❌ Error in context manager cleanup: {e}")
        finally:
            self.disconnect()
    
    def __del__(self):
        """Destructor to ensure connection is closed"""
        try:
            self.disconnect()
        except:
            pass
            
    def __repr__(self):
        """String representation of the connection"""
        if self.connection and not self.connection.closed:
            return f"<DatabaseConnection: {self.db_config['user']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']} [CONNECTED]>"
        else:
            return f"<DatabaseConnection: {self.db_config['user']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']} [DISCONNECTED]>"