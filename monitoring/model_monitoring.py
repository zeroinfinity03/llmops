#!/usr/bin/env python3
"""
Model Monitoring for Algospeak Content Moderation System

Provides basic monitoring capabilities for production deployment:
- Endpoint health checks
- Prediction metrics logging
- System status reporting
- Performance tracking
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

class ModelMonitor:
    """
    Basic model monitoring for demonstration of MLOps best practices
    
    In production, this would integrate with CloudWatch, Prometheus, or similar
    monitoring systems. For demonstration, it provides local logging and metrics.
    """
    
    def __init__(self, log_dir: str = "logs"):
        """Initialize monitoring with local logging directory"""
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize metrics storage
        self.prediction_log = self.log_dir / "predictions.jsonl"
        self.health_log = self.log_dir / "health_checks.jsonl"
        
        print(f"✅ Model monitoring initialized with log directory: {log_dir}")
    
    def check_endpoint_health(self, endpoint_name: str) -> Dict:
        """
        Check if SageMaker endpoint is responding
        
        Args:
            endpoint_name: Name of SageMaker endpoint to check
            
        Returns:
            Dictionary with health status information
        """
        
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "endpoint_name": endpoint_name,
            "status": "unknown",
            "response_time_ms": 0,
            "error": None
        }
        
        try:
            import boto3
            
            start_time = time.time()
            
            # Test endpoint with simple request
            runtime = boto3.client('sagemaker-runtime')
            response = runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=json.dumps({"inputs": "health check"})
            )
            
            response_time = (time.time() - start_time) * 1000
            
            # Validate response format
            result = json.loads(response['Body'].read().decode())
            
            health_status.update({
                "status": "healthy",
                "response_time_ms": response_time,
                "response_valid": isinstance(result, dict)
            })
            
        except Exception as e:
            health_status.update({
                "status": "unhealthy",
                "error": str(e)
            })
        
        # Log health check result
        self._log_health_check(health_status)
        
        return health_status
    
    def log_prediction_metrics(self, input_text: str, result: Dict):
        """
        Log prediction for monitoring and analysis
        
        Args:
            input_text: Original input text
            result: Moderation result dictionary
        """
        
        prediction_entry = {
            "timestamp": datetime.now().isoformat(),
            "input_length": len(input_text),
            "algospeak_detected": result.get('algospeak_detected', False),
            "classification": result.get('classification', 'unknown'),
            "detected_patterns": result.get('detected_patterns', []),
            "processing_time_ms": result.get('processing_time_ms', 0),
            "model_used": result.get('model_used', 'unknown'),
            "mock_mode": result.get('mock_mode', False)
        }
        
        # Log prediction entry
        self._log_prediction(prediction_entry)
    
    def get_system_status(self) -> Dict:
        """
        Get overall system health status
        
        Returns:
            Dictionary with comprehensive system status
        """
        
        # Read recent health checks
        recent_health = self._get_recent_health_checks(limit=10)
        
        # Read recent predictions
        recent_predictions = self._get_recent_predictions(limit=100)
        
        # Calculate metrics
        total_predictions = len(recent_predictions)
        algospeak_detected = sum(1 for p in recent_predictions if p.get('algospeak_detected', False))
        harmful_classifications = sum(1 for p in recent_predictions if 'harmful' in p.get('classification', ''))
        
        avg_response_time = 0
        if recent_predictions:
            response_times = [p.get('processing_time_ms', 0) for p in recent_predictions]
            avg_response_time = sum(response_times) / len(response_times)
        
        system_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy" if recent_health and recent_health[0].get('status') == 'healthy' else "degraded",
            "recent_predictions": total_predictions,
            "algospeak_detection_rate": (algospeak_detected / total_predictions * 100) if total_predictions > 0 else 0,
            "harmful_content_rate": (harmful_classifications / total_predictions * 100) if total_predictions > 0 else 0,
            "avg_response_time_ms": avg_response_time,
            "last_health_check": recent_health[0] if recent_health else None
        }
        
        return system_status
    
    def _log_health_check(self, health_status: Dict):
        """Log health check result to file"""
        
        with open(self.health_log, 'a') as f:
            f.write(json.dumps(health_status) + '\n')
    
    def _log_prediction(self, prediction_entry: Dict):
        """Log prediction entry to file"""
        
        with open(self.prediction_log, 'a') as f:
            f.write(json.dumps(prediction_entry) + '\n')
    
    def _get_recent_health_checks(self, limit: int = 10) -> List[Dict]:
        """Get recent health check entries"""
        
        if not self.health_log.exists():
            return []
        
        entries = []
        with open(self.health_log, 'r') as f:
            for line in f:
                try:
                    entries.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        
        return entries[-limit:] if entries else []
    
    def _get_recent_predictions(self, limit: int = 100) -> List[Dict]:
        """Get recent prediction entries"""
        
        if not self.prediction_log.exists():
            return []
        
        entries = []
        with open(self.prediction_log, 'r') as f:
            for line in f:
                try:
                    entries.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        
        return entries[-limit:] if entries else []

# Simple usage example for testing
if __name__ == "__main__":
    monitor = ModelMonitor()
    
    # Example health check
    health = monitor.check_endpoint_health("test-endpoint")
    print(f"Health check result: {health}")
    
    # Example prediction logging
    sample_result = {
        "algospeak_detected": True,
        "classification": "harmful",
        "detected_patterns": ["unalive"],
        "processing_time_ms": 150,
        "model_used": "mock_classifier",
        "mock_mode": True
    }
    
    monitor.log_prediction_metrics("I want to unalive myself", sample_result)
    
    # Get system status
    status = monitor.get_system_status()
    print(f"System status: {json.dumps(status, indent=2)}")