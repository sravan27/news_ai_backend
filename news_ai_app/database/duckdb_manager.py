"""
DuckDB manager for optimized data processing operations.
"""
import os
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import duckdb
import logging
from contextlib import contextmanager

logger = logging.getLogger("duckdb_manager")

class DuckDBManager:
    """Manager for DuckDB operations with optimized data processing."""
    
    def __init__(self, db_path=None, memory_limit="8GB"):
        """Initialize DuckDB manager."""
        self.db_path = db_path
        self.memory_limit = memory_limit
        self.conn = None
    
    def connect(self):
        """Connect to DuckDB."""
        try:
            # Use in-memory database if no path provided
            if self.db_path:
                self.conn = duckdb.connect(database=self.db_path)
            else:
                self.conn = duckdb.connect(database=":memory:")
            
            # Configure settings
            self.conn.execute(f"SET memory_limit='{self.memory_limit}'")
            self.conn.execute("PRAGMA threads=4")
            logger.info(f"Connected to DuckDB with {self.memory_limit} memory limit")
            return True
        except Exception as e:
            logger.error(f"Error connecting to DuckDB: {e}")
            return False
    
    def close(self):
        """Close the DuckDB connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    @contextmanager
    def connection(self):
        """Provide a transactional scope around DuckDB operations."""
        try:
            if not self.conn:
                self.connect()
            yield self.conn
        finally:
            pass  # We don't close the connection here for reuse
    
    def register_parquet(self, name, parquet_path):
        """Register a Parquet file as a DuckDB view."""
        try:
            with self.connection() as conn:
                conn.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM parquet_scan('{parquet_path}')")
                logger.info(f"Registered Parquet file as view: {name}")
                return True
        except Exception as e:
            logger.error(f"Error registering Parquet file: {e}")
            return False
    
    def register_dataframe(self, name, df):
        """Register a pandas DataFrame as a DuckDB view."""
        try:
            with self.connection() as conn:
                conn.register(name, df)
                logger.info(f"Registered DataFrame as view: {name}")
                return True
        except Exception as e:
            logger.error(f"Error registering DataFrame: {e}")
            return False
    
    def execute_query(self, query):
        """Execute a SQL query and return the result as a DataFrame."""
        try:
            with self.connection() as conn:
                result = conn.execute(query).fetch_df()
                return result
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return None
    
    def load_parquet_optimized(self, parquet_path):
        """Load a Parquet file efficiently using DuckDB."""
        try:
            with self.connection() as conn:
                result = conn.execute(f"SELECT * FROM parquet_scan('{parquet_path}')").fetch_df()
                return result
        except Exception as e:
            logger.error(f"Error loading Parquet file: {e}")
            return None
    
    def create_arrow_table(self, sql_query):
        """Execute SQL query and return an Arrow table for efficient processing."""
        try:
            with self.connection() as conn:
                result = conn.execute(sql_query).arrow()
                return result
        except Exception as e:
            logger.error(f"Error creating Arrow table: {e}")
            return None
    
    def execute_batch_processing(self, input_view, process_func, batch_size=10000):
        """Process data in batches using a provided function."""
        try:
            with self.connection() as conn:
                # Get total count
                count_result = conn.execute(f"SELECT COUNT(*) as count FROM {input_view}").fetchone()
                total = count_result[0] if count_result else 0
                
                results = []
                for offset in range(0, total, batch_size):
                    # Fetch batch
                    query = f"SELECT * FROM {input_view} LIMIT {batch_size} OFFSET {offset}"
                    batch = conn.execute(query).fetch_df()
                    
                    # Process batch
                    processed = process_func(batch)
                    results.append(processed)
                
                # Combine results if they're all DataFrames
                if results and all(isinstance(r, pd.DataFrame) for r in results):
                    return pd.concat(results, ignore_index=True)
                return results
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return None
    
    def create_optimized_parquet(self, input_path, output_path, partition_cols=None, filters=None):
        """Create an optimized Parquet file from an existing one."""
        try:
            # First load the data through DuckDB
            with self.connection() as conn:
                # Apply filters if provided
                if filters:
                    filter_sql = " AND ".join(filters)
                    query = f"SELECT * FROM parquet_scan('{input_path}') WHERE {filter_sql}"
                else:
                    query = f"SELECT * FROM parquet_scan('{input_path}')"
                
                # Get Arrow table
                arrow_table = conn.execute(query).arrow()
            
            # Write optimized parquet
            if partition_cols:
                pq.write_to_dataset(
                    arrow_table,
                    root_path=output_path,
                    partition_cols=partition_cols,
                    compression='snappy',
                    version='2.0',
                    row_group_size=100000
                )
            else:
                pq.write_table(
                    arrow_table,
                    output_path,
                    compression='snappy',
                    version='2.0',
                    row_group_size=100000
                )
            
            logger.info(f"Created optimized Parquet at {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error creating optimized Parquet: {e}")
            return False