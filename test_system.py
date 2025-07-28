#!/usr/bin/env python3
"""
Basic Testing Infrastructure for LLMOps Algospeak System

Provides testing capabilities without requiring AWS credentials:
- Mock SageMaker classifier testing
- API endpoint testing
- Monitoring system testing
- Integration testing
"""

import json
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def test_algospeak_normalization():
    """Test the algospeak normalization functionality"""
    
    print("🧪 Testing Algospeak Normalization...")
    
    try:
        from moderator import AlgospeakNormalizer
        
        normalizer = AlgospeakNormalizer()
        
        test_cases = [
            ("I want to unalive myself", "I want to kill myself", True),
            ("This is seggs content", "This is sex content", True),
            ("I killed it at work today", "I succeeded at work today", True),  # Safe context
            ("Hello, how are you?", "Hello, how are you?", False),
        ]
        
        passed = 0
        for original, expected_normalized, expected_algospeak in test_cases:
            normalized, patterns, has_algospeak = normalizer.normalize(original)
            
            if normalized == expected_normalized and has_algospeak == expected_algospeak:
                print(f"   ✅ '{original}' → '{normalized}' (algospeak: {has_algospeak})")
                passed += 1
            else:
                print(f"   ❌ '{original}' → '{normalized}' (expected: '{expected_normalized}')")
        
        print(f"   Normalization tests: {passed}/{len(test_cases)} passed")
        return passed == len(test_cases)
        
    except Exception as e:
        print(f"   ❌ Normalization test failed: {e}")
        return False

def test_mock_classification():
    """Test the mock classification functionality"""
    
    print("\n🧪 Testing Mock Classification...")
    
    try:
        from moderator import ContentModerator
        
        # Initialize with mock endpoint (will use mock mode)
        moderator = ContentModerator("mock-endpoint")
        
        test_cases = [
            ("I want to unalive myself", "extremely_harmful"),
            ("This is seggs content", "harmful"),
            ("I killed it at work today", "safe"),  # Safe context should be handled
            ("Hello, how are you?", "safe"),
        ]
        
        passed = 0
        for text, expected_classification in test_cases:
            result = moderator.moderate_content(text)
            
            classification = result.get('classification', 'unknown')
            
            if expected_classification in classification:
                print(f"   ✅ '{text}' → {classification}")
                passed += 1
            else:
                print(f"   ❌ '{text}' → {classification} (expected: {expected_classification})")
        
        print(f"   Classification tests: {passed}/{len(test_cases)} passed")
        return passed == len(test_cases)
        
    except Exception as e:
        print(f"   ❌ Classification test failed: {e}")
        return False

def test_api_response_format():
    """Test that API responses match AlgoSpeak format"""
    
    print("\n🧪 Testing API Response Format...")
    
    try:
        from main import ModerationResponse
        from moderator import ContentModerator
        
        moderator = ContentModerator("mock-endpoint")
        result = moderator.moderate_content("I want to unalive myself")
        
        # Convert to API format
        stage1_status = "algospeak_normalized" if result['algospeak_detected'] else "no_algospeak_found"
        stage2_status = "sagemaker_classified" if 'error' not in result else "sagemaker_unavailable"
        
        api_response = ModerationResponse(
            original_text=result['original_text'],
            normalized_text=result['normalized_text'],
            algospeak_detected=result['algospeak_detected'],
            classification=result['classification'],
            stage1_status=stage1_status,
            stage2_status=stage2_status
        )
        
        # Check required fields
        required_fields = ['original_text', 'normalized_text', 'algospeak_detected', 'classification', 'stage1_status', 'stage2_status']
        
        response_dict = api_response.dict()
        missing_fields = [field for field in required_fields if field not in response_dict]
        
        if not missing_fields:
            print(f"   ✅ API response format correct: {list(response_dict.keys())}")
            return True
        else:
            print(f"   ❌ Missing fields: {missing_fields}")
            return False
        
    except Exception as e:
        print(f"   ❌ API format test failed: {e}")
        return False

def test_monitoring_system():
    """Test the monitoring system functionality"""
    
    print("\n🧪 Testing Monitoring System...")
    
    try:
        from monitoring.model_monitoring import ModelMonitor
        from monitoring.drift_detection import DriftDetector
        
        # Test model monitor
        monitor = ModelMonitor(log_dir="test_logs")
        
        # Test prediction logging
        sample_result = {
            "algospeak_detected": True,
            "classification": "harmful",
            "detected_patterns": ["unalive"],
            "processing_time_ms": 150,
            "model_used": "mock_classifier",
            "mock_mode": True
        }
        
        monitor.log_prediction_metrics("I want to unalive myself", sample_result)
        
        # Test system status
        status = monitor.get_system_status()
        
        if 'timestamp' in status and 'overall_status' in status:
            print(f"   ✅ Model monitoring working: {status['overall_status']}")
        else:
            print(f"   ❌ Model monitoring failed: missing status fields")
            return False
        
        # Test drift detector
        detector = DriftDetector()
        
        sample_texts = ["I want to unalive myself", "This is normal content"]
        input_analysis = detector.analyze_input_distribution(sample_texts)
        
        if 'timestamp' in input_analysis and 'metrics' in input_analysis:
            print(f"   ✅ Drift detection working: {len(input_analysis['alerts'])} alerts")
            return True
        else:
            print(f"   ❌ Drift detection failed: missing analysis fields")
            return False
        
    except Exception as e:
        print(f"   ❌ Monitoring test failed: {e}")
        return False

def test_error_handling():
    """Test error handling and graceful degradation"""
    
    print("\n🧪 Testing Error Handling...")
    
    try:
        from moderator import ContentModerator
        
        # Test with invalid endpoint (should use mock mode)
        moderator = ContentModerator("invalid-endpoint-name")
        
        result = moderator.moderate_content("Test content")
        
        # Should still return a valid result in mock mode
        required_fields = ['original_text', 'normalized_text', 'classification', 'algospeak_detected']
        missing_fields = [field for field in required_fields if field not in result]
        
        if not missing_fields and result.get('mock_mode', False):
            print(f"   ✅ Graceful degradation working: mock_mode = {result['mock_mode']}")
            return True
        else:
            print(f"   ❌ Error handling failed: {missing_fields}")
            return False
        
    except Exception as e:
        print(f"   ❌ Error handling test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and provide summary"""
    
    print("🚀 Running LLMOps System Tests")
    print("=" * 50)
    
    tests = [
        ("Algospeak Normalization", test_algospeak_normalization),
        ("Mock Classification", test_mock_classification),
        ("API Response Format", test_api_response_format),
        ("Monitoring System", test_monitoring_system),
        ("Error Handling", test_error_handling),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f"   ❌ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("✅ All tests passed! System is ready for demonstration.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)