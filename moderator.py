#!/usr/bin/env python3
"""
🛡️ COMPLETE ALGOSPEAK CONTENT MODERATION SYSTEM
Two stages in one file: Normalization + SageMaker Classification

Stage 1: Convert "unalive" → "kill" (your exact algorithm)
Stage 2: AWS SageMaker classification (replaces Ollama)
"""

import json
import re
import time
import boto3
from pathlib import Path
from typing import Dict, List, Tuple, Any

# =============================================================================
# STAGE 1: ALGOSPEAK NORMALIZATION
# =============================================================================
# Your exact normalization algorithm from local version

class AlgospeakNormalizer:
    """Stage 1: Algospeak pattern replacer (your exact local algorithm)"""
    
    def __init__(self, patterns_path: str = "data/dataset/algospeak_patterns.json"):
        """Load algospeak patterns from JSON (your exact loading logic)"""
        
        print("Loading algospeak patterns...")
        
        # Load patterns from centralized dataset location
        with open(patterns_path, 'r') as f:
            data = json.load(f)
        
        # Extract all pattern mappings (your exact pattern loading)
        self.patterns = {}
        self.patterns.update(data.get("direct_mappings", {}))
        self.patterns.update(data.get("homophones", {}))
        self.patterns.update(data.get("misspellings", {}))
        self.patterns.update(data.get("leetspeak", {}))
        
        # Load safe context patterns
        self.safe_patterns = data.get("safe_context_patterns", {})
        
        # Your exact debugging output
        print(f"✅ Loaded {len(self.patterns)} algospeak patterns")
        print(f"✅ Loaded {len(self.safe_patterns)} safe context patterns")
    
    def normalize(self, text: str) -> Tuple[str, List[str], bool]:
        """
        Your exact normalization algorithm
        
        Args:
            text: Input text like "I want to unalive myself"
            
        Returns:
            Tuple of (normalized_text, detected_patterns, has_algospeak)
        """
        normalized = text
        replacements_made = []
        
        # First, check for safe context patterns (longer phrases first)
        for safe_phrase, safe_replacement in sorted(self.safe_patterns.items(), key=len, reverse=True):
            pattern = re.escape(safe_phrase)
            
            if re.search(pattern, normalized, re.IGNORECASE):
                normalized = re.sub(pattern, safe_replacement, normalized, flags=re.IGNORECASE)
                replacements_made.append(f'"{safe_phrase}" → "{safe_replacement}" (safe context)')
        
        # Then replace algospeak patterns (case-insensitive, whole words)
        for algospeak, normal in self.patterns.items():
            pattern = r'\b' + re.escape(algospeak) + r'\b'
            
            if re.search(pattern, normalized, re.IGNORECASE):
                normalized = re.sub(pattern, normal, normalized, flags=re.IGNORECASE)
                replacements_made.append(f'"{algospeak}" → "{normal}"')
        
        # Your exact debugging output
        if replacements_made:
            print(f"🔄 Normalized: {', '.join(replacements_made)}")
        
        has_algospeak = len(replacements_made) > 0
        detected_patterns = [item.split('"')[1] for item in replacements_made if '"' in item]
        
        return normalized, detected_patterns, has_algospeak

# =============================================================================
# STAGE 2: SAGEMAKER CLASSIFICATION
# =============================================================================
# Replaces your Ollama classifier with SageMaker

class SageMakerClassifier:
    """Stage 2: AI classifier using SageMaker (replaces Ollama)"""
    
    def __init__(self, endpoint_name: str):
        """Initialize SageMaker classifier with graceful degradation"""
        
        self.endpoint_name = endpoint_name
        self.normalizer = AlgospeakNormalizer()
        self.mock_mode = False
        
        # Test SageMaker connection and enable mock mode if unavailable
        try:
            self.runtime = boto3.client('sagemaker-runtime')
            # Simple test call to check if endpoint is available
            test_response = self.runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=json.dumps({"inputs": "test"})
            )
            print(f"✅ SageMaker endpoint '{endpoint_name}' is responding")
        except Exception as e:
            print(f"⚠️ SageMaker unavailable, enabling mock mode: {e}")
            print("💡 For production, ensure SageMaker endpoint is deployed and AWS credentials are configured")
            self.mock_mode = True
            self.runtime = None
    
    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classify text using SageMaker (same interface as your Ollama version)
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with classification results (same format as your Ollama version)
        """
        
        # Step 1: Normalize algospeak (your exact approach)
        normalized_text, detected_patterns, algospeak_detected = self.normalizer.normalize(text)
        
        # Step 2: Build classification prompt (same structure as your Ollama version)
        prompt = f"""Classify this text as harmful or safe. Give a short answer.

Text: "{normalized_text}"
Classification:"""
        
        # Step 3: Call SageMaker API or use mock mode
        if self.mock_mode:
            return self._mock_classification(text, normalized_text, detected_patterns, algospeak_detected)
        
        try:
            response = self.runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Body=json.dumps({"inputs": prompt})
            )
            
            result = json.loads(response['Body'].read().decode())
            classification = result.get('generated_text', '').strip()
            
            # Clean up response (remove prompt echo)
            if 'Classification:' in classification:
                classification = classification.split('Classification:')[-1].strip()
            
            # Return same format as your Ollama version
            return {
                'original_text': text,
                'normalized_text': normalized_text,
                'detected_patterns': detected_patterns,
                'algospeak_detected': algospeak_detected,
                'classification': classification,
                'model_used': f'sagemaker:{self.endpoint_name}',
                'normalization_applied': text != normalized_text
            }
            
        except Exception as e:
            # Fallback to mock mode on error
            print(f"⚠️ SageMaker error, falling back to mock mode: {e}")
            return self._mock_classification(text, normalized_text, detected_patterns, algospeak_detected)
    
    def _mock_classification(self, text: str, normalized_text: str, detected_patterns: list, algospeak_detected: bool) -> dict:
        """
        Mock classification for demonstration when SageMaker unavailable
        
        Provides realistic responses based on text patterns for demo purposes
        """
        
        # Simulate realistic classification based on normalized text patterns
        harmful_keywords = [
            "kill", "suicide", "death", "harm", "hate", "violence", 
            "sex", "porn", "drug", "cocaine", "marijuana"
        ]
        
        extremely_harmful_keywords = [
            "kill myself", "suicide", "unalive myself", "end my life", 
            "want to die", "self harm"
        ]
        
        normalized_lower = normalized_text.lower()
        
        # Check for extremely harmful content
        if any(keyword in normalized_lower for keyword in extremely_harmful_keywords):
            classification = "extremely_harmful"
        # Check for generally harmful content
        elif any(keyword in normalized_lower for keyword in harmful_keywords):
            classification = "harmful"
        # Check for potentially harmful patterns
        elif algospeak_detected:
            classification = "potentially_harmful"
        else:
            classification = "safe"
        
        return {
            'original_text': text,
            'normalized_text': normalized_text,
            'detected_patterns': detected_patterns,
            'algospeak_detected': algospeak_detected,
            'classification': classification,
            'model_used': 'mock_classifier',
            'normalization_applied': text != normalized_text,
            'mock_mode': True
        }

# =============================================================================
# COMPLETE CONTENT MODERATOR (Simple wrapper like your original)
# =============================================================================

class ContentModerator:
    """Complete two-stage content moderation system"""
    
    def __init__(self, endpoint_name: str):
        """Initialize complete system"""
        print("🔧 Initializing Algospeak Content Moderation...")
        self.classifier = SageMakerClassifier(endpoint_name)
        print("✅ Content moderation system ready!")
    
    def moderate_content(self, text: str) -> Dict[str, Any]:
        """
        Complete moderation pipeline (main function for FastAPI)
        
        Args:
            text: Input content to moderate
            
        Returns:
            Complete moderation results
        """
        return self.classifier.classify(text)

# =============================================================================
# SIMPLE WRAPPER FUNCTIONS (Like your original approach)
# =============================================================================

def moderate_text(text: str, endpoint_name: str) -> Dict[str, Any]:
    """Simple function to moderate text (like your classify_text wrapper)"""
    moderator = ContentModerator(endpoint_name)
    return moderator.moderate_content(text)

# Simple testing when run directly
if __name__ == "__main__":
    print("🛡️ ALGOSPEAK CONTENT MODERATION SYSTEM")
    print("Two-stage pipeline: Normalization + SageMaker Classification")
    print()
    
    # Test cases (your debugging approach)
    test_cases = [
        "I want to unalive myself",
        "Going to commit sudoku tonight", 
        "I killed it at work today",
        "This content looks totally normal"
    ]
    
    # Replace with your actual endpoint name
    ENDPOINT_NAME = "algospeak-endpoint-20250127-140523"  # Update this!
    
    try:
        print(f"Testing with endpoint: {ENDPOINT_NAME}")
        moderator = ContentModerator(ENDPOINT_NAME)
        
        for text in test_cases:
            print(f"\n📝 Testing: '{text}'")
            result = moderator.moderate_content(text)
            
            if 'error' not in result:
                print(f"   Normalized: '{result['normalized_text']}'")
                print(f"   Algospeak: {result['algospeak_detected']}")
                print(f"   Classification: {result['classification']}")
            else:
                print(f"   Error: {result['error']}")
    
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        print("Make sure to update ENDPOINT_NAME with your actual endpoint!")
