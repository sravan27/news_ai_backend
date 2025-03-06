"""
Telemetry and performance monitoring system.
"""
import os
import time
import logging
import json
import threading
import queue
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

logger = logging.getLogger("telemetry")

class Telemetry:
    """Telemetry and performance monitoring system."""
    
    def __init__(self, log_dir=None, auto_flush=True, flush_interval=60):
        """Initialize Telemetry system."""
        self.log_dir = log_dir or Path(__file__).resolve().parent.parent / "logs" / "telemetry"
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Set up metric storage
        self.metrics = {}
        self.events = []
        self.traces = {}
        
        # Set up asynchronous logging
        self.queue = queue.Queue()
        self.should_stop = False
        self.auto_flush = auto_flush
        self.flush_interval = flush_interval
        
        # Start background thread for async logging
        if auto_flush:
            self.thread = threading.Thread(target=self._background_flush)
            self.thread.daemon = True
            self.thread.start()
    
    def _background_flush(self):
        """Background thread for flushing metrics."""
        while not self.should_stop:
            time.sleep(self.flush_interval)
            try:
                self.flush()
            except Exception as e:
                logger.error(f"Error flushing telemetry: {e}")
    
    def increment_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """Increment a counter metric."""
        with threading.Lock():
            metric_key = name
            if metric_key not in self.metrics:
                self.metrics[metric_key] = {
                    "type": "counter",
                    "value": 0,
                    "tags": tags or {}
                }
            
            self.metrics[metric_key]["value"] += value
    
    def gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a gauge metric (current value)."""
        with threading.Lock():
            metric_key = name
            self.metrics[metric_key] = {
                "type": "gauge",
                "value": value,
                "tags": tags or {}
            }
    
    def histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Add a value to a histogram metric."""
        with threading.Lock():
            metric_key = name
            if metric_key not in self.metrics:
                self.metrics[metric_key] = {
                    "type": "histogram",
                    "values": [],
                    "tags": tags or {}
                }
            
            self.metrics[metric_key]["values"].append(value)
    
    def log_event(self, name: str, properties: Dict[str, Any] = None, timestamp=None):
        """Log an event."""
        event = {
            "name": name,
            "timestamp": timestamp or datetime.now().isoformat(),
            "properties": properties or {}
        }
        
        self.events.append(event)
        
        # Log to queue for async processing
        self.queue.put(("event", event))
    
    def start_trace(self, name: str, properties: Dict[str, Any] = None) -> str:
        """Start a trace for timing operations."""
        trace_id = f"{name}_{int(time.time() * 1000)}"
        
        self.traces[trace_id] = {
            "name": name,
            "start_time": time.time(),
            "properties": properties or {}
        }
        
        return trace_id
    
    def end_trace(self, trace_id: str, properties: Dict[str, Any] = None):
        """End a trace and record duration."""
        if trace_id not in self.traces:
            logger.warning(f"Trace {trace_id} not found")
            return
        
        trace = self.traces[trace_id]
        duration = time.time() - trace["start_time"]
        
        # Update properties
        if properties:
            trace["properties"].update(properties)
        
        # Log as histogram
        self.histogram(
            f"{trace['name']}_duration",
            duration,
            tags={"trace_id": trace_id}
        )
        
        # Log event
        self.log_event(
            f"{trace['name']}_completed",
            properties={
                "duration": duration,
                "trace_id": trace_id,
                **trace["properties"]
            }
        )
        
        # Remove trace
        del self.traces[trace_id]
    
    def timed(self, name: str, properties: Dict[str, Any] = None):
        """Decorator for timing functions."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                trace_id = self.start_trace(name, properties)
                try:
                    result = func(*args, **kwargs)
                    self.end_trace(trace_id, {"status": "success"})
                    return result
                except Exception as e:
                    self.end_trace(trace_id, {"status": "error", "error": str(e)})
                    raise e
            return wrapper
        return decorator
    
    def log_model_prediction(self, model_name: str, inputs: Any, outputs: Any, 
                            duration: float = None, metadata: Dict[str, Any] = None):
        """Log model prediction for monitoring."""
        prediction_event = {
            "name": "model_prediction",
            "timestamp": datetime.now().isoformat(),
            "properties": {
                "model_name": model_name,
                "duration": duration,
                **(metadata or {})
            }
        }
        
        # Store only a sample of inputs/outputs to avoid excessive storage
        if isinstance(inputs, pd.DataFrame):
            prediction_event["properties"]["input_sample"] = inputs.head(2).to_dict(orient='records')
        else:
            prediction_event["properties"]["input_sample"] = str(inputs)[:200]
        
        if isinstance(outputs, pd.DataFrame):
            prediction_event["properties"]["output_sample"] = outputs.head(2).to_dict(orient='records')
        else:
            prediction_event["properties"]["output_sample"] = str(outputs)[:200]
        
        self.events.append(prediction_event)
        
        # Log to queue for async processing
        self.queue.put(("model_prediction", prediction_event))
    
    def flush(self):
        """Flush metrics to storage."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save metrics
            if self.metrics:
                metrics_path = self.log_dir / f"metrics_{timestamp}.json"
                with open(metrics_path, "w") as f:
                    json.dump(self.metrics, f, indent=2)
                
                # Clear metrics that can be reset
                with threading.Lock():
                    new_metrics = {}
                    for key, metric in self.metrics.items():
                        if metric["type"] == "histogram":
                            # Keep histograms but clear values
                            new_metrics[key] = {
                                "type": "histogram",
                                "values": [],
                                "tags": metric["tags"]
                            }
                        elif metric["type"] == "counter":
                            # Reset counters
                            new_metrics[key] = {
                                "type": "counter",
                                "value": 0,
                                "tags": metric["tags"]
                            }
                        # Keep current gauge values
                        elif metric["type"] == "gauge":
                            new_metrics[key] = metric
                    
                    self.metrics = new_metrics
            
            # Save events
            if self.events:
                events_path = self.log_dir / f"events_{timestamp}.json"
                with open(events_path, "w") as f:
                    json.dump(self.events, f, indent=2)
                
                # Clear events
                self.events = []
            
            logger.info(f"Flushed telemetry data at {timestamp}")
            return True
        except Exception as e:
            logger.error(f"Error flushing telemetry data: {e}")
            return False
    
    def shutdown(self):
        """Shutdown telemetry system."""
        self.should_stop = True
        if self.auto_flush:
            self.thread.join(timeout=5)
        self.flush()
        logger.info("Telemetry system shutdown")