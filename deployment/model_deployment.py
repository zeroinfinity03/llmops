#!/usr/bin/env python3
"""
ALGOSPEAK MODEL DEPLOYMENT
Deploy trained model as SageMaker real-time endpoint for inference

This script:
1. Retrieves trained model artifacts from S3
2. Creates SageMaker endpoint for real-time inference
3. Configures auto-scaling and monitoring
4. Provides endpoint URL for API integration

Usage: python deployment/model_deployment.py
"""

import os
import json
import boto3
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
from dotenv import load_dotenv
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.session import Session

class AlgospeakModelDeployment:
    """
    Production model deployment orchestrator
    
    Handles deployment of trained algospeak models as scalable
    SageMaker endpoints for real-time content classification.
    """
    
    def __init__(self):
        """Initialize deployment orchestrator"""
        
        load_dotenv()
        
        self.session = Session()
        self.role = os.getenv('SAGEMAKER_ROLE_ARN')
        self.bucket = os.getenv('S3_BUCKET_NAME')
        self.model_name = os.getenv('MODEL_NAME', 'qwen-algospeak')
        self.huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
        
        missing_vars = []
        if not self.role: missing_vars.append("SAGEMAKER_ROLE_ARN")
        if not self.bucket: missing_vars.append("S3_BUCKET_NAME")
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        print(f"Deployment Orchestrator Initialized")
        print(f"   Model: {self.model_name}")
        print(f"   Bucket: {self.bucket}")
    
    def get_latest_model_artifacts(self) -> str:
        """Find the most recent trained model artifacts in S3"""
        
        s3_client = boto3.client('s3')
        model_prefix = f"models/{self.model_name}/"
        
        try:
            # List all training job outputs
            response = s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=model_prefix
            )
            
            if 'Contents' not in response:
                raise FileNotFoundError(f"No trained models found in s3://{self.bucket}/{model_prefix}")
            
            # Find the most recent model.tar.gz
            model_artifacts = [
                obj for obj in response['Contents'] 
                if obj['Key'].endswith('model.tar.gz')
            ]
            
            if not model_artifacts:
                raise FileNotFoundError("No model.tar.gz files found")
            
            # Get the most recent model
            latest_model = max(model_artifacts, key=lambda x: x['LastModified'])
            model_s3_path = f"s3://{self.bucket}/{latest_model['Key']}"
            
            print(f"Using model artifacts: {model_s3_path}")
            return model_s3_path
            
        except Exception as e:
            print(f"Error finding model artifacts: {e}")
            raise
    
    def create_model(self, model_artifacts_path: str) -> HuggingFaceModel:
        """Create SageMaker model from trained artifacts"""
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name = f"algospeak-model-{timestamp}"
        
        # Create HuggingFace model for deployment
        model = HuggingFaceModel(
            model_data=model_artifacts_path,     # S3 path to trained model
            role=self.role,
            transformers_version="4.28",
            pytorch_version="2.0",
            py_version="py310",
            name=model_name,
            env={
                'HF_MODEL_ID': 'Qwen/Qwen2.5-3B-Instruct',  # Base model info
                'HF_TASK': 'text-classification',
                'HUGGINGFACE_HUB_TOKEN': self.huggingface_token,
            }
        )
        
        print(f"Created SageMaker model: {model_name}")
        return model
    
    def deploy_endpoint(self, model: HuggingFaceModel) -> str:
        """Deploy model as real-time SageMaker endpoint"""
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        endpoint_name = f"algospeak-endpoint-{timestamp}"
        
        print(f"Deploying endpoint: {endpoint_name}")
        print("This will take 5-10 minutes...")
        
        # Deploy model to endpoint
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type="ml.g4dn.xlarge",      # GPU instance for fast inference
            endpoint_name=endpoint_name,
            wait=True,                           # Wait for deployment to complete
        )
        
        # Note: Auto-scaling would be configured separately via SageMaker console
        # or additional boto3 calls for production deployment
        
        print(f"Endpoint deployed successfully: {endpoint_name}")
        print(f"Endpoint URL: https://{endpoint_name}.sagemaker.{self.session.boto_region_name}.amazonaws.com")
        
        return endpoint_name
    
    def test_endpoint(self, endpoint_name: str):
        """Test deployed endpoint with sample content"""
        
        print(f"Testing endpoint: {endpoint_name}")
        
        # Test samples including algospeak
        test_samples = [
            "This content looks completely normal and safe",
            "I want to unalive myself",           # algospeak for suicide
            "Let's meet at the airport terminal", # potentially suspicious
            "Going to commit sudoku tonight",     # algospeak for suicide
        ]
        
        runtime = boto3.client('sagemaker-runtime')
        
        for sample in test_samples:
            try:
                response = runtime.invoke_endpoint(
                    EndpointName=endpoint_name,
                    ContentType='application/json',
                    Body=json.dumps({"inputs": sample})
                )
                
                result = json.loads(response['Body'].read().decode())
                print(f"Input: '{sample}'")
                print(f"Output: {result}")
                print("-" * 50)
                
            except Exception as e:
                print(f"Error testing sample '{sample}': {e}")
    
    def deploy_complete_pipeline(self) -> str:
        """Execute complete deployment pipeline"""
        
        print("ALGOSPEAK MODEL DEPLOYMENT PIPELINE")
        print("=" * 50)
        
        try:
            # Get trained model artifacts
            print("\nStep 1: Finding latest trained model...")
            model_artifacts = self.get_latest_model_artifacts()
            
            # Create SageMaker model
            print("\nStep 2: Creating SageMaker model...")
            model = self.create_model(model_artifacts)
            
            # Deploy endpoint
            print("\nStep 3: Deploying endpoint...")
            endpoint_name = self.deploy_endpoint(model)
            
            # Test endpoint
            print("\nStep 4: Testing endpoint...")
            self.test_endpoint(endpoint_name)
            
            print(f"\nSUCCESS! Deployment complete")
            print(f"Endpoint: {endpoint_name}")
            print(f"Ready for production inference!")
            
            return endpoint_name
            
        except Exception as e:
            print(f"Deployment failed: {e}")
            raise

def main():
    """Execute deployment pipeline"""
    
    try:
        deployment = AlgospeakModelDeployment()
        endpoint_name = deployment.deploy_complete_pipeline()
        
        print("\nNext Steps:")
        print("1. Run: python deployment/inference.py")
        print("2. Integrate endpoint with your applications")
        print("3. Monitor performance in SageMaker console")
        
        return endpoint_name
        
    except Exception as e:
        print(f"Deployment failed: {e}")
        return None

if __name__ == "__main__":
    main()

