import sys
import os
sys.path.append('DATABASE_CONNECTION')
from db_connection import DatabaseConnection #type: ignore
from db_operations import DatabaseOperations #type: ignore

# Database connection instance
db_conn = DatabaseOperations(DatabaseConnection())