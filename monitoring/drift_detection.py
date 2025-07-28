#!/usr/bin/env python3
"""
Drift Detection for Algospeak Content Moderation System

Provides basic drift detection capabilities:
- Input distribution analysis
- Classification pattern monitoring
- Algospeak pattern evolution tracking
- Alert generation for significant changes
"""

import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import Counter
from pathlib import Path

class DriftDetector:
    """
    Basic drift detection for demonstration of MLOps monitoring
    
    In production, this would use statistical tests and ML-based drift detection.
    For demonstration, it provides simple pattern-based analysis.
    """
    
    def __init__(self, baseline_window_days: int = 7, alert_threshold: float = 0.2):
        """
        Initialize drift detector
        
        Args:
            baseline_window_days: Days of data to use as baseline
            alert_threshold: Threshold for triggering drift alerts (0.0-1.0)
        """
        
        self.baseline_window_days = baseline_window_days
        self.alert_threshold = alert_threshold
        
        print(f"✅ Drift detector initialized with {baseline_window_days}-day baseline window")
    
    def analyze_input_distribution(self, texts: List[str]) -> Dict:
        """
        Analyze if input text patterns are changing
        
        Args:
            texts: List of input texts to analyze
            
        Returns:
            Dictionary with input distribution analysis
        """
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(texts),
            "metrics": {},
            "alerts": []
        }
        
        if not texts:
            analysis["alerts"].append("No input data available for analysis")
            return analysis
        
        # Analyze text length distribution
        lengths = [len(text) for text in texts]
        analysis["metrics"]["avg_length"] = sum(lengths) / len(lengths)
        analysis["metrics"]["min_length"] = min(lengths)
        analysis["metrics"]["max_length"] = max(lengths)
        
        # Analyze character patterns
        char_patterns = self._analyze_character_patterns(texts)
        analysis["metrics"]["character_patterns"] = char_patterns
        
        # Analyze word patterns
        word_patterns = self._analyze_word_patterns(texts)
        analysis["metrics"]["word_patterns"] = word_patterns
        
        # Check for potential algospeak evolution
        algospeak_indicators = self._detect_algospeak_indicators(texts)
        analysis["metrics"]["algospeak_indicators"] = algospeak_indicators
        
        # Generate alerts based on thresholds
        if char_patterns["special_char_ratio"] > 0.3:
            analysis["alerts"].append("High special character usage detected - possible new evasion patterns")
        
        if algospeak_indicators["potential_new_patterns"]:
            analysis["alerts"].append(f"Potential new algospeak patterns detected: {algospeak_indicators['potential_new_patterns'][:5]}")
        
        return analysis
    
    def detect_classification_drift(self, results: List[Dict]) -> Dict:
        """
        Check if classification patterns are shifting
        
        Args:
            results: List of moderation result dictionaries
            
        Returns:
            Dictionary with classification drift analysis
        """
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "total_predictions": len(results),
            "metrics": {},
            "alerts": []
        }
        
        if not results:
            analysis["alerts"].append("No prediction data available for analysis")
            return analysis
        
        # Analyze classification distribution
        classifications = [r.get('classification', 'unknown') for r in results]
        class_distribution = dict(Counter(classifications))
        
        analysis["metrics"]["classification_distribution"] = class_distribution
        analysis["metrics"]["harmful_rate"] = sum(1 for c in classifications if 'harmful' in c) / len(classifications)
        analysis["metrics"]["algospeak_detection_rate"] = sum(1 for r in results if r.get('algospeak_detected', False)) / len(results)
        
        # Analyze response times
        response_times = [r.get('processing_time_ms', 0) for r in results if r.get('processing_time_ms', 0) > 0]
        if response_times:
            analysis["metrics"]["avg_response_time_ms"] = sum(response_times) / len(response_times)
            analysis["metrics"]["max_response_time_ms"] = max(response_times)
        
        # Analyze detected patterns
        all_patterns = []
        for r in results:
            patterns = r.get('detected_patterns', [])
            all_patterns.extend(patterns)
        
        pattern_frequency = dict(Counter(all_patterns))
        analysis["metrics"]["top_detected_patterns"] = dict(sorted(pattern_frequency.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # Generate drift alerts
        harmful_rate = analysis["metrics"]["harmful_rate"]
        if harmful_rate > 0.5:
            analysis["alerts"].append(f"High harmful content rate detected: {harmful_rate:.1%}")
        elif harmful_rate < 0.05:
            analysis["alerts"].append(f"Unusually low harmful content rate: {harmful_rate:.1%} - possible model degradation")
        
        algospeak_rate = analysis["metrics"]["algospeak_detection_rate"]
        if algospeak_rate > 0.3:
            analysis["alerts"].append(f"High algospeak detection rate: {algospeak_rate:.1%} - possible new evasion tactics")
        
        return analysis
    
    def compare_with_baseline(self, current_data: List[Dict], baseline_data: List[Dict]) -> Dict:
        """
        Compare current data with baseline to detect drift
        
        Args:
            current_data: Recent prediction results
            baseline_data: Baseline prediction results
            
        Returns:
            Dictionary with drift comparison analysis
        """
        
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "current_samples": len(current_data),
            "baseline_samples": len(baseline_data),
            "drift_detected": False,
            "metrics": {},
            "alerts": []
        }
        
        if not baseline_data or not current_data:
            comparison["alerts"].append("Insufficient data for baseline comparison")
            return comparison
        
        # Compare classification distributions
        current_classes = [r.get('classification', 'unknown') for r in current_data]
        baseline_classes = [r.get('classification', 'unknown') for r in baseline_data]
        
        current_dist = Counter(current_classes)
        baseline_dist = Counter(baseline_classes)
        
        # Calculate distribution shift (Jensen-Shannon divergence approximation)
        all_classes = set(current_dist.keys()) | set(baseline_dist.keys())
        distribution_shift = 0
        
        for cls in all_classes:
            current_rate = current_dist.get(cls, 0) / len(current_data)
            baseline_rate = baseline_dist.get(cls, 0) / len(baseline_data)
            distribution_shift += abs(current_rate - baseline_rate)
        
        # Normalize by 2 to get proper distance metric (0-1 range)
        distribution_shift = distribution_shift / 2
        
        comparison["metrics"]["distribution_shift"] = distribution_shift
        
        # Compare algospeak detection rates
        current_algospeak_rate = sum(1 for r in current_data if r.get('algospeak_detected', False)) / len(current_data)
        baseline_algospeak_rate = sum(1 for r in baseline_data if r.get('algospeak_detected', False)) / len(baseline_data)
        
        algospeak_drift = abs(current_algospeak_rate - baseline_algospeak_rate)
        comparison["metrics"]["algospeak_rate_drift"] = algospeak_drift
        
        # Determine if drift is significant
        if distribution_shift > self.alert_threshold:
            comparison["drift_detected"] = True
            comparison["alerts"].append(f"Significant classification drift detected: {distribution_shift:.3f}")
        
        if algospeak_drift > self.alert_threshold:
            comparison["drift_detected"] = True
            comparison["alerts"].append(f"Significant algospeak detection drift: {algospeak_drift:.3f}")
        
        return comparison
    
    def _analyze_character_patterns(self, texts: List[str]) -> Dict:
        """Analyze character-level patterns in texts"""
        
        total_chars = sum(len(text) for text in texts)
        special_chars = sum(len(re.findall(r'[^a-zA-Z0-9\s]', text)) for text in texts)
        numbers = sum(len(re.findall(r'\d', text)) for text in texts)
        
        return {
            "special_char_ratio": special_chars / total_chars if total_chars > 0 else 0,
            "number_ratio": numbers / total_chars if total_chars > 0 else 0,
            "avg_special_chars_per_text": special_chars / len(texts) if texts else 0
        }
    
    def _analyze_word_patterns(self, texts: List[str]) -> Dict:
        """Analyze word-level patterns in texts"""
        
        all_words = []
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend(words)
        
        word_freq = Counter(all_words)
        
        return {
            "unique_words": len(word_freq),
            "total_words": len(all_words),
            "avg_words_per_text": len(all_words) / len(texts) if texts else 0,
            "top_words": dict(word_freq.most_common(10))
        }
    
    def _detect_algospeak_indicators(self, texts: List[str]) -> Dict:
        """Detect potential new algospeak patterns"""
        
        # Common algospeak indicators
        substitution_patterns = [
            r'\w*\d+\w*',  # Numbers mixed with letters
            r'\w*[!@#$%^&*]+\w*',  # Special characters in words
            r'\b\w*x+\w*\b',  # Multiple x's (common substitution)
            r'\b\w*z+\w*\b',  # Multiple z's (leetspeak)
        ]
        
        potential_patterns = []
        
        for text in texts:
            for pattern in substitution_patterns:
                matches = re.findall(pattern, text.lower())
                potential_patterns.extend(matches)
        
        # Filter out common words and focus on unusual patterns
        unusual_patterns = [p for p in potential_patterns if len(p) > 2 and not p.isdigit()]
        
        return {
            "potential_new_patterns": list(set(unusual_patterns)),
            "substitution_indicators": len(potential_patterns),
            "unusual_pattern_rate": len(unusual_patterns) / len(texts) if texts else 0
        }

# Simple usage example for testing
if __name__ == "__main__":
    detector = DriftDetector()
    
    # Example input analysis
    sample_texts = [
        "I want to unalive myself",
        "This is normal content",
        "Let's have some seggs tonight",
        "Going to commit sudoku",
        "Regular message here"
    ]
    
    input_analysis = detector.analyze_input_distribution(sample_texts)
    print(f"Input analysis: {json.dumps(input_analysis, indent=2)}")
    
    # Example classification analysis
    sample_results = [
        {"classification": "extremely_harmful", "algospeak_detected": True, "detected_patterns": ["unalive"], "processing_time_ms": 120},
        {"classification": "safe", "algospeak_detected": False, "detected_patterns": [], "processing_time_ms": 80},
        {"classification": "harmful", "algospeak_detected": True, "detected_patterns": ["seggs"], "processing_time_ms": 110},
        {"classification": "extremely_harmful", "algospeak_detected": True, "detected_patterns": ["sudoku"], "processing_time_ms": 130},
        {"classification": "safe", "algospeak_detected": False, "detected_patterns": [], "processing_time_ms": 75}
    ]
    
    classification_analysis = detector.detect_classification_drift(sample_results)
    print(f"Classification analysis: {json.dumps(classification_analysis, indent=2)}")