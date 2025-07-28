#!/usr/bin/env python3
"""
ALGOSPEAK INFERENCE API - ALTERNATIVE DEPLOYMENT SERVER

This is an alternative API server focused on direct SageMaker inference.
It provides different response formats and is optimized for batch processing.

Use this server when:
- You need batch processing capabilities
- You want detailed confidence scores
- You're integrating with systems that expect different response formats

Primary server: main.py (AlgoSpeak-compatible)
Alternative server: deployment/inference.py (this file)

Usage: python deployment/inference.py
"""

import os
import json
import boto3
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Algospeak Content Moderation API",
    description="Production API for algospeak content classification using fine-tuned Qwen2.5-3B",
    version="1.0.0"
)

# Enable CORS for web applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SageMaker runtime client
sagemaker_runtime = boto3.client('sagemaker-runtime')

# Configuration
ENDPOINT_NAME = os.getenv('SAGEMAKER_ENDPOINT_NAME')  # Set this after deployment
if not ENDPOINT_NAME:
    print("WARNING: SAGEMAKER_ENDPOINT_NAME not set. Update .env file after deployment.")

# =============================================================================
# PYDANTIC MODELS FOR API
# =============================================================================

class ContentRequest(BaseModel):
    """Request model for content classification"""
    content: str = Field(..., min_length=1, max_length=10000, description="Content to classify")
    include_confidence: bool = Field(default=True, description="Include confidence score in response")

class BatchContentRequest(BaseModel):
    """Request model for batch content classification"""
    contents: List[str] = Field(..., min_items=1, max_items=100, description="List of content to classify")
    include_confidence: bool = Field(default=True, description="Include confidence scores in response")

class ClassificationResponse(BaseModel):
    """Response model for content classification"""
    content: str
    classification: str
    confidence: Optional[float] = None
    is_algospeak: bool
    processing_time_ms: float

class BatchClassificationResponse(BaseModel):
    """Response model for batch content classification"""
    results: List[ClassificationResponse]
    total_processed: int
    processing_time_ms: float

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    endpoint_name: str
    model_ready: bool

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def classify_content(content: str, include_confidence: bool = True) -> Dict:
    """Classify single piece of content using SageMaker endpoint"""
    
    if not ENDPOINT_NAME:
        raise HTTPException(status_code=500, detail="SageMaker endpoint not configured")
    
    try:
        import time
        start_time = time.time()
        
        # Call SageMaker endpoint
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/json',
            Body=json.dumps({"inputs": content})
        )
        
        # Parse response
        result = json.loads(response['Body'].read().decode())
        processing_time = (time.time() - start_time) * 1000
        
        # Extract classification results
        classification = result.get('classification', 'unknown')
        confidence = result.get('confidence', 0.0) if include_confidence else None
        is_algospeak = result.get('is_algospeak', False)
        
        return {
            'content': content,
            'classification': classification,
            'confidence': confidence,
            'is_algospeak': is_algospeak,
            'processing_time_ms': processing_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Algospeak Content Moderation API",
        "version": "1.0.0",
        "endpoints": {
            "classify": "/classify",
            "batch_classify": "/batch_classify",
            "health": "/health"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    
    model_ready = False
    if ENDPOINT_NAME:
        try:
            # Test endpoint with simple content
            test_response = sagemaker_runtime.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType='application/json',
                Body=json.dumps({"inputs": "test"})
            )
            model_ready = True
        except:
            model_ready = False
    
    return HealthResponse(
        status="healthy" if model_ready else "degraded",
        endpoint_name=ENDPOINT_NAME or "not_configured",
        model_ready=model_ready
    )

@app.post("/classify", response_model=ClassificationResponse)
async def classify_single_content(request: ContentRequest):
    """
    Classify a single piece of content for algospeak detection
    
    Returns classification with confidence score and algospeak detection.
    """
    
    result = classify_content(request.content, request.include_confidence)
    return ClassificationResponse(**result)

@app.post("/batch_classify", response_model=BatchClassificationResponse)
async def classify_batch_content(request: BatchContentRequest):
    """
    Classify multiple pieces of content in batch
    
    Processes up to 100 pieces of content and returns results for each.
    """
    
    import time
    start_time = time.time()
    
    results = []
    for content in request.contents:
        try:
            result = classify_content(content, request.include_confidence)
            results.append(ClassificationResponse(**result))
        except Exception as e:
            # Handle individual failures gracefully
            results.append(ClassificationResponse(
                content=content,
                classification="error",
                confidence=None,
                is_algospeak=False,
                processing_time_ms=0.0
            ))
    
    total_time = (time.time() - start_time) * 1000
    
    return BatchClassificationResponse(
        results=results,
        total_processed=len(results),
        processing_time_ms=total_time
    )

# =============================================================================
# MAIN APPLICATION
# =============================================================================

if __name__ == "__main__":
    """Run FastAPI server"""
    
    print("Starting Algospeak Content Moderation API...")
    print(f"Endpoint: {ENDPOINT_NAME}")
    print("Access API docs at: http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload during development
        access_log=True
    )

