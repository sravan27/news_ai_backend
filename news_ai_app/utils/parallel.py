"""
Parallel processing utilities using Ray.
"""
import os
import ray
import numpy as np
import pandas as pd
import logging
from functools import wraps
import time

logger = logging.getLogger("parallel")

def init_ray(num_cpus=None, memory_limit=None):
    """Initialize Ray for parallel processing."""
    try:
        if not ray.is_initialized():
            if num_cpus is None:
                num_cpus = os.cpu_count()
            
            # Configure memory limit
            if memory_limit is None:
                memory_limit = "16GB"
                
            # Initialize Ray with specified resources
            ray.init(
                num_cpus=num_cpus,
                _memory=memory_limit,
                ignore_reinit_error=True,
                logging_level=logging.ERROR
            )
            
            logger.info(f"Ray initialized with {num_cpus} CPUs and {memory_limit} memory limit")
            return True
        return True
    except Exception as e:
        logger.error(f"Error initializing Ray: {e}")
        return False

def shutdown_ray():
    """Shutdown Ray."""
    if ray.is_initialized():
        ray.shutdown()
        logger.info("Ray shutdown")

def parallel_map(func, items, num_cpus=None, batch_size=None):
    """Apply function to items in parallel using Ray."""
    try:
        # Initialize Ray if not already done
        if not ray.is_initialized():
            init_ray(num_cpus=num_cpus)
        
        # Define remote function
        @ray.remote
        def _remote_func(batch):
            return [func(item) for item in batch]
        
        # Determine batch size
        if batch_size is None:
            batch_size = max(1, len(items) // (num_cpus or os.cpu_count()))
        
        # Split items into batches
        batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
        
        # Process batches in parallel
        ray_refs = [_remote_func.remote(batch) for batch in batches]
        results_nested = ray.get(ray_refs)
        
        # Flatten results
        results = [item for sublist in results_nested for item in sublist]
        
        return results
    except Exception as e:
        logger.error(f"Error in parallel_map: {e}")
        # Fall back to sequential processing
        logger.info("Falling back to sequential processing")
        return [func(item) for item in items]

def parallel_dataframe(func, df, num_partitions=None, num_cpus=None):
    """Process pandas DataFrame in parallel using Ray."""
    try:
        # Initialize Ray if not already done
        if not ray.is_initialized():
            init_ray(num_cpus=num_cpus)
        
        # Determine number of partitions
        if num_partitions is None:
            num_partitions = num_cpus or os.cpu_count()
        
        # Convert to Ray DataFrame
        df_parts = np.array_split(df, num_partitions)
        
        # Define remote function
        @ray.remote
        def _process_partition(partition):
            return func(partition)
        
        # Process partitions in parallel
        ray_refs = [_process_partition.remote(part) for part in df_parts]
        results = ray.get(ray_refs)
        
        # Combine results if they're DataFrames
        if results and isinstance(results[0], pd.DataFrame):
            return pd.concat(results, ignore_index=True)
        return results
    except Exception as e:
        logger.error(f"Error in parallel_dataframe: {e}")
        # Fall back to sequential processing
        logger.info("Falling back to sequential processing")
        return func(df)

def parallel_decorator(num_cpus=None, batch_size=None):
    """Decorator to make a function execute in parallel using Ray."""
    def decorator(func):
        @wraps(func)
        def wrapper(items, *args, **kwargs):
            # Check if items is iterable
            if not hasattr(items, '__iter__') or isinstance(items, (str, bytes)):
                return func(items, *args, **kwargs)
            
            # Define function to apply to each item
            def apply_func(item):
                return func(item, *args, **kwargs)
            
            # Apply in parallel
            return parallel_map(apply_func, items, num_cpus=num_cpus, batch_size=batch_size)
        return wrapper
    return decorator