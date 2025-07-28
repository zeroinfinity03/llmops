#!/usr/bin/env python3
"""
ALGOSPEAK MODEL TRAINING ORCHESTRATOR
Runs on your local machine to create and manage SageMaker training jobs

This script:
1. Reads dataset paths from your data ingestion manifest
2. Configures SageMaker HuggingFace estimator
3. Launches training jobs with proper resource allocation
4. Monitors training progress and handles failures

Usage: python training/model_training.py
"""

import os
import json
import boto3
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
from dotenv import load_dotenv
from sagemaker.huggingface import HuggingFace
from sagemaker.session import Session

# =============================================================================
# ENVIRONMENT SETUP AND VALIDATION
# =============================================================================
# Initialize the training orchestrator and validate environment configuration

class AlgospeakTrainingOrchestrator:
    """
    Production training orchestrator for algospeak content moderation models
    
    Manages the complete training lifecycle:
    - Dataset path resolution from data ingestion manifests
    - SageMaker training job configuration and launch
    - Training monitoring and error handling
    - Model artifact management and versioning
    """
    
    def __init__(self):
        """Initialize and validate environment configuration"""
        
        # Load environment variables from .env file
        load_dotenv()
        
        # Initialize SageMaker session for job management
        self.session = Session()
        
        # Extract required configuration from environment
        self.role = os.getenv('SAGEMAKER_ROLE_ARN')
        self.bucket = os.getenv('S3_BUCKET_NAME')
        self.model_name = os.getenv('MODEL_NAME', 'qwen-algospeak')
        self.base_model = os.getenv('BASE_MODEL', 'Qwen/Qwen2.5-3B-Instruct')
        self.huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
        
        # Validate required environment variables
        missing_vars = []
        if not self.role: missing_vars.append("SAGEMAKER_ROLE_ARN")
        if not self.bucket: missing_vars.append("S3_BUCKET_NAME") 
        if not self.huggingface_token: missing_vars.append("HUGGINGFACE_TOKEN")
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        print(f"Training Orchestrator Initialized")
        print(f"   Model: {self.base_model}")
        print(f"   Bucket: {self.bucket}")
        print(f"   Role: {self.role}")

# =============================================================================
# DATASET PATH RESOLUTION
# =============================================================================
# Connect to data ingestion output and extract S3 paths for training

    def get_latest_dataset_paths(self) -> Dict[str, str]:
        """
        Get dataset paths from the latest data ingestion manifest
        
        Finds the most recent data ingestion manifest file and extracts
        S3 URLs for training dataset and patterns. This creates the link
        between data ingestion output and training input.
        """
        
        # Look for manifest files created by data ingestion pipeline
        manifest_files = list(Path("data").glob("upload_manifest_v*.json"))
        
        # Ensure data ingestion ran successfully
        if not manifest_files:
            raise FileNotFoundError("No data ingestion manifests found. Run data/data_ingestion.py first.")
        
        # Get the most recent manifest (latest dataset version)
        latest_manifest = max(manifest_files, key=lambda x: x.stat().st_mtime)
        
        print(f"Using dataset manifest: {latest_manifest}")
        
        # Parse manifest to extract S3 dataset locations
        with open(latest_manifest, 'r') as f:
            manifest = json.load(f)
        
        # Extract S3 URLs for required training files
        files = manifest.get('files', {})
        
        training_data_url = files.get('training_dataset.json', {}).get('s3_url')
        patterns_url = files.get('algospeak_patterns.json', {}).get('s3_url')
        
        # Validate that all required datasets are available
        missing_files = []
        if not training_data_url: missing_files.append('training_dataset.json')
        if not patterns_url: missing_files.append('algospeak_patterns.json')
        
        if missing_files:
            raise ValueError(f"Required dataset files not found in manifest: {', '.join(missing_files)}. Check data ingestion results.")
        
        print(f"Training data: {training_data_url}")
        print(f"Patterns: {patterns_url}")
        
        return {
            'training_data': training_data_url,
            'patterns': patterns_url,
            'version': manifest.get('version', 'unknown')
        }

# =============================================================================
# SAGEMAKER TRAINING JOB CONFIGURATION
# =============================================================================
# Create SageMaker training job with optimized hyperparameters

    def create_training_job(self, dataset_paths: Dict[str, str]) -> HuggingFace:
        """
        Create and configure SageMaker HuggingFace estimator
        
        Sets up hyperparameters optimized for Qwen2.5-3B + QLoRA fine-tuning,
        configures GPU instance type and resource allocation, and specifies
        input data paths from data ingestion output.
        """
        
        # Generate unique training job name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_name = f"algospeak-{self.model_name}-{timestamp}"
        
        # Hyperparameters optimized for Qwen2.5-3B + QLoRA
        hyperparameters = {
            # Model configuration - HuggingFace model identifier
            'model_name': self.base_model,  # This tells SageMaker which model to download from HF
            'dataset_path': dataset_paths['training_data'],
            'patterns_path': dataset_paths['patterns'],
            'output_dir': '/opt/ml/model',
            
            # Training parameters
            'num_train_epochs': 3,
            'per_device_train_batch_size': 2,  # Conservative for 3B model
            'gradient_accumulation_steps': 4,   # Effective batch size = 8
            'learning_rate': 2e-4,
            'warmup_ratio': 0.1,
            'logging_steps': 10,
            'save_steps': 500,
            'evaluation_strategy': 'steps',
            'eval_steps': 500,
            'save_total_limit': 2,
            'load_best_model_at_end': True,
            'metric_for_best_model': 'eval_loss',
            'greater_is_better': False,
            
            # QLoRA specific parameters for memory efficiency
            'use_lora': True,
            'lora_r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
            'target_modules': 'q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj',
            
            # Memory optimization
            'gradient_checkpointing': True,
            'dataloader_pin_memory': False,
            'bf16': True,  # Better than fp16 for training stability
        }
        
        # Environment variables passed to SageMaker training container
        # IMPORTANT: HuggingFace token enables automatic model download from HF Hub to AWS
        environment = {
            'HUGGINGFACE_HUB_TOKEN': self.huggingface_token,  # Authenticates HF download in SageMaker
            'TOKENIZERS_PARALLELISM': 'false',
        }
        
        # Create HuggingFace estimator with production configuration
        huggingface_estimator = HuggingFace(
            entry_point='train_script.py',     # Script that runs inside SageMaker
            source_dir='training',
            instance_type='ml.g4dn.xlarge',    # Cost-effective GPU instance
            instance_count=1,
            role=self.role,
            volume_size=50,  # GB storage
            max_run=24*3600,  # 24 hours max runtime
            transformers_version='4.28',
            pytorch_version='2.0',
            py_version='py310',
            hyperparameters=hyperparameters,
            environment=environment,           # Passes HF token to training container
            base_job_name=job_name,
            output_path=f's3://{self.bucket}/models/{self.model_name}/',
            
            # Monitoring configuration
            enable_sagemaker_metrics=True,
            metric_definitions=[
                {'Name': 'train_loss', 'Regex': 'train_loss: ([0-9\\.]+)'},
                {'Name': 'eval_loss', 'Regex': 'eval_loss: ([0-9\\.]+)'},
                {'Name': 'learning_rate', 'Regex': 'learning_rate: ([0-9\\.E-]+)'},
            ],
            disable_profiler=True,
            debugger_hook_config=False,
        )
        
        print(f"Created training job configuration: {job_name}")
        print(f"   Instance: ml.g4dn.xlarge")
        print(f"   Max runtime: 24 hours")
        print(f"   Output: s3://{self.bucket}/models/{self.model_name}/")
        
        return huggingface_estimator

# =============================================================================
# TRAINING JOB EXECUTION AND MONITORING
# =============================================================================
# Launch training job and provide monitoring capabilities

    def launch_training(self) -> str:
        """
        Execute the complete training pipeline
        
        Resolves dataset paths, creates training job configuration,
        and launches the training job on AWS SageMaker infrastructure.
        """
        
        print("ALGOSPEAK MODEL TRAINING PIPELINE")
        print("=" * 50)
        
        try:
            # Connect to data ingestion output
            print("\nStep 1: Resolving dataset paths...")
            dataset_paths = self.get_latest_dataset_paths()
            
            # Configure training job
            print("\nStep 2: Configuring SageMaker training job...")
            estimator = self.create_training_job(dataset_paths)
            
            # Launch training job - this transitions to cloud execution
            print("\nStep 3: Launching training job...")
            print("This will take several hours. Monitor progress in SageMaker console.")
            
            # Start training - SageMaker will now handle model download and training
            estimator.fit({
                'training': dataset_paths['training_data'],
                'patterns': dataset_paths['patterns']
            })
            
            job_name = estimator.latest_training_job.name
            print(f"\nTraining job launched successfully: {job_name}")
            print(f"Monitor at: https://console.aws.amazon.com/sagemaker/home#/jobs/{job_name}")
            
            return job_name
            
        except Exception as e:
            print(f"\nERROR: Training job failed to launch: {e}")
            print("CHECK: Verify your AWS credentials and SageMaker permissions")
            raise
    
    def get_training_status(self, job_name: str) -> str:
        """Check status of running training job"""
        
        sagemaker_client = boto3.client('sagemaker')
        
        try:
            response = sagemaker_client.describe_training_job(TrainingJobName=job_name)
            status = response['TrainingJobStatus']
            
            if status == 'Completed':
                model_artifacts = response['ModelArtifacts']['S3ModelArtifacts']
                print(f"Training completed! Model artifacts: {model_artifacts}")
            elif status == 'Failed':
                failure_reason = response.get('FailureReason', 'Unknown failure')
                print(f"Training failed: {failure_reason}")
            else:
                print(f"Training status: {status}")
            
            return status
            
        except Exception as e:
            print(f"Error checking training status: {e}")
            return "Unknown"

# =============================================================================
# MAIN EXECUTION ENTRY POINT
# =============================================================================

def main():
    """Execute training orchestration workflow"""
    
    try:
        orchestrator = AlgospeakTrainingOrchestrator()
        job_name = orchestrator.launch_training()
        
        print("\nNext Steps:")
        print(f"1. Monitor training: aws sagemaker describe-training-job --training-job-name {job_name}")
        print("2. Once complete, run: python deployment/model_deployment.py")
        print("3. Test inference with your algospeak content")
        
    except Exception as e:
        print(f"Training orchestration failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
