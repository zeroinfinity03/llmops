#!/usr/bin/env python3
"""
🛡️ ALGOSPEAK CONTENT MODERATION API - PRIMARY PRODUCTION SERVER

This is the main API server for the LLMOps algospeak content moderation system.
It provides AlgoSpeak-compatible endpoints for seamless frontend integration.

Primary endpoint:
POST /moderate - Runs complete two-stage algospeak content moderation

Note: deployment/inference.py provides an alternative API server with different
response formats for specialized use cases.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Import content moderator (fixed import path)
from moderator import ContentModerator

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="Algospeak Content Moderation API",
    description="Production API for algospeak content moderation using fine-tuned Qwen2.5-3B",
    version="1.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ModerationRequest(BaseModel):
    text: str

class ModerationResponse(BaseModel):
    original_text: str
    normalized_text: str
    algospeak_detected: bool
    classification: str
    stage1_status: str
    stage2_status: str

# Initialize content moderator
ENDPOINT_NAME = os.getenv('SAGEMAKER_ENDPOINT_NAME')

if not ENDPOINT_NAME:
    print("⚠️ WARNING: SAGEMAKER_ENDPOINT_NAME not found in environment")
    print("   System will run in mock mode for demonstration")
    print("   For production: Add SAGEMAKER_ENDPOINT_NAME=your-endpoint-name to .env")
    # Initialize with placeholder endpoint name - will use mock mode
    moderator = ContentModerator("mock-endpoint")
else:
    try:
        moderator = ContentModerator(ENDPOINT_NAME)
        print("✅ Content Moderation System Ready")
        print(f"   Endpoint: {ENDPOINT_NAME}")
    except Exception as e:
        print(f"⚠️ SageMaker initialization failed, using mock mode: {e}")
        print("   For production: Ensure AWS credentials and SageMaker endpoint are configured")
        moderator = ContentModerator("mock-endpoint")

@app.post("/moderate", response_model=ModerationResponse)
async def moderate_content(request: ModerationRequest):
    """
    Complete algospeak content moderation.
    
    Takes any text input and returns:
    - Normalized text (algospeak converted to normal words)
    - Classification (safe, harmful, extremely_harmful, etc.)
    - Detected algospeak patterns
    - Processing metrics
    """
    
    # Moderator is always available (either real or mock mode)
    if not moderator:
        raise HTTPException(
            status_code=503, 
            detail="Content moderator initialization failed. Check system logs."
        )
    
    try:
        # Run complete two-stage moderation
        result = moderator.moderate_content(request.text)
        
        # Convert to AlgoSpeak-compatible format
        stage1_status = "algospeak_normalized" if result['algospeak_detected'] else "no_algospeak_found"
        stage2_status = "sagemaker_classified" if 'error' not in result else "sagemaker_unavailable"
        
        # Return structured response (AlgoSpeak-compatible format)
        return ModerationResponse(
            original_text=result['original_text'],
            normalized_text=result['normalized_text'],
            algospeak_detected=result['algospeak_detected'],
            classification=result['classification'],
            stage1_status=stage1_status,
            stage2_status=stage2_status
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Content moderation failed: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting Algospeak Content Moderation API")
    print("📋 Endpoint: POST /moderate")
    print("📖 API docs: http://localhost:8000/docs")
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
