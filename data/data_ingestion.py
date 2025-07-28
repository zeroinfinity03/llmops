#!/usr/bin/env python3
"""
COMPLETE ALGOSPEAK DATA INGESTION PIPELINE
Single script containing all components for uploading processed datasets to S3

This script handles:
1. S3 upload management with retry logic
2. Dataset validation and processing
3. Version management and manifest creation
4. Complete pipeline orchestration

Usage: python data/data_ingestion.py
"""

import os
import json
import boto3
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from dotenv import load_dotenv

# =============================================================================
# ROBUST UPLOAD MANAGER COMPONENT
# =============================================================================
# This section handles S3 uploads with retry logic and error handling
# Ensures reliable file uploads even with network issues or temporary failures

class RobustUploadManager:
    """
    Handles S3 uploads with automatic retry on failure
    
    Features:
    - Automatic retry with exponential backoff (2s, 4s, 8s)
    - Upload verification to ensure files actually reach S3
    - Detailed metadata tracking for each upload
    - Comprehensive error handling and logging
    """
    
    def __init__(self, max_retries: int = 3):
        """
        Initialize upload manager with S3 client and retry configuration
        
        Args:
            max_retries: Maximum number of upload attempts per file
        """
        self.s3_client = boto3.client('s3')
        self.max_retries = max_retries
        self.bucket = os.getenv('S3_BUCKET_NAME')
        
        if not self.bucket:
            raise ValueError("S3_BUCKET_NAME not found in environment variables")
        
        print(f"Upload Manager initialized for bucket: {self.bucket}")
        
    def upload_with_retry(self, local_path: str, s3_key: str) -> Optional[str]:
        """
        Upload single file with automatic retry on failure
        
        Args:
            local_path: Path to local file to upload
            s3_key: S3 object key (path) where file will be stored
            
        Returns:
            S3 URL if successful, None if all retries failed
        """
        local_file = Path(local_path)
        
        # Pre-upload validation
        if not local_file.exists():
            print(f"ERROR: File not found: {local_path}")
            return None
        
        file_size_mb = local_file.stat().st_size / (1024 * 1024)
        print(f"Uploading {local_file.name} ({file_size_mb:.1f} MB)")
        
        # Retry loop with exponential backoff
        for attempt in range(1, self.max_retries + 1):
            try:
                print(f"   Attempt {attempt}/{self.max_retries}...")
                
                # Upload file with comprehensive metadata
                self.s3_client.upload_file(
                    str(local_path),
                    self.bucket,
                    s3_key,
                    ExtraArgs={
                        'Metadata': {
                            'upload-attempt': str(attempt),
                            'uploaded-at': str(int(time.time())),
                            'original-filename': local_file.name,
                            'file-size-mb': f"{file_size_mb:.1f}",
                            'upload-source': 'algospeak-data-pipeline'
                        }
                    }
                )
                
                # Verify upload actually succeeded by checking object existence
                self.s3_client.head_object(Bucket=self.bucket, Key=s3_key)
                
                s3_url = f"s3://{self.bucket}/{s3_key}"
                print(f"   SUCCESS: Upload successful: {s3_url}")
                return s3_url
                
            except Exception as e:
                print(f"   FAILED: Attempt {attempt} failed: {e}")
                
                # If this was the last attempt, give up
                if attempt == self.max_retries:
                    print(f"   FINAL FAILURE: All {self.max_retries} attempts failed!")
                    return None
                
                # Exponential backoff: 2s, 4s, 8s
                wait_time = 2 ** attempt
                print(f"   RETRY: Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        
        return None
    
    def upload_multiple_files(self, file_mappings: Dict[str, str]) -> Dict[str, Optional[str]]:
        """
        Upload multiple files with mapping local_path -> s3_key
        
        Args:
            file_mappings: Dictionary mapping local file paths to S3 keys
            
        Returns:
            Dictionary mapping local paths to S3 URLs (or None for failures)
        """
        results = {}
        
        print(f"\nStarting batch upload of {len(file_mappings)} files...")
        
        for local_path, s3_key in file_mappings.items():
            print(f"\nProcessing: {Path(local_path).name}")
            results[local_path] = self.upload_with_retry(local_path, s3_key)
        
        return results
    
    def verify_s3_access(self) -> bool:
        """
        Test S3 bucket access before attempting uploads
        
        Returns:
            True if bucket is accessible, False otherwise
        """
        try:
            print("Verifying S3 bucket access...")
            self.s3_client.head_bucket(Bucket=self.bucket)
            print("SUCCESS: S3 bucket accessible")
            return True
            
        except Exception as e:
            print(f"ERROR: S3 access failed: {e}")
            print("CHECK: Verify your AWS credentials and bucket name in .env")
            return False

# =============================================================================
# MAIN DATA INGESTION ORCHESTRATOR
# =============================================================================
# This section coordinates the entire data pipeline:
# - Validates processed datasets from Jupyter preprocessing
# - Manages dataset versioning with timestamps
# - Orchestrates uploads using the upload manager
# - Creates comprehensive tracking manifests

class AlgospeakDataIngestion:
    """
    Production data ingestion orchestrator for algospeak content moderation
    
    Coordinates the complete pipeline:
    1. Validates that preprocessed datasets exist and are valid
    2. Creates versioned upload mappings for S3 storage
    3. Executes reliable uploads using RobustUploadManager
    4. Generates comprehensive manifests for tracking and reproducibility
    5. Prepares datasets for SageMaker training workflows
    """
    
    def __init__(self):
        """Initialize data ingestion pipeline with environment configuration"""
        
        # Load environment variables from .env file
        load_dotenv()
        
        self.bucket = os.getenv('S3_BUCKET_NAME')
        self.uploader = RobustUploadManager()
        
        # Create version timestamp for dataset tracking
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.version_id = f"v{self.timestamp}"
        
        print(f"Data Ingestion Pipeline Initialized")
        print(f"   Version: {self.version_id}")
        print(f"   Target Bucket: {self.bucket}")
        
    def validate_dataset_files(self) -> tuple:
        """
        Validate that all required dataset files exist and have valid content
        
        Returns:
            Tuple of (validation_results_dict, all_required_files_present)
        """
        print("\nValidating dataset files...")
        
        # Define files to check - required vs optional
        files_to_check = {
            "training_dataset.json": {
                "path": "data/dataset/training_dataset.json",
                "required": True,
                "description": "Processed instruction dataset for fine-tuning"
            },
            "algospeak_patterns.json": {
                "path": "data/dataset/algospeak_patterns.json", 
                "required": True,
                "description": "Algospeak pattern database"
            },
            "train.csv": {
                "path": "data/dataset/train.csv",
                "required": False,
                "description": "Raw Jigsaw training data"
            },
            "test.csv": {
                "path": "data/dataset/test.csv",
                "required": False,
                "description": "Raw Jigsaw test data"
            }
        }
        
        validation_results = {}
        all_required_present = True
        
        # Check each file for existence, size, and basic content validation
        for filename, info in files_to_check.items():
            file_path = Path(info["path"])
            
            if file_path.exists():
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                
                # Perform basic content validation for JSON files
                content_valid = True
                sample_stats = {}
                
                if filename.endswith('.json'):
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        
                        # Extract statistics for training dataset
                        if filename == "training_dataset.json":
                            if isinstance(data, list):
                                sample_stats = {
                                    "total_samples": len(data),
                                    "sample_fields": list(data[0].keys()) if data else [],
                                    "algospeak_samples": sum(1 for item in data if item.get('is_algospeak', False))
                                }
                            else:
                                sample_stats = {"structure": "non-list format"}
                        
                        # Extract statistics for pattern database
                        elif filename == "algospeak_patterns.json":
                            total_patterns = sum(len(v) for v in data.values() if isinstance(v, dict))
                            sample_stats = {
                                "total_patterns": total_patterns,
                                "categories": list(data.keys())
                            }
                                
                    except json.JSONDecodeError:
                        content_valid = False
                        sample_stats = {"error": "Invalid JSON format"}
                
                # Store validation results
                validation_results[filename] = {
                    "exists": True,
                    "size_mb": file_size_mb,
                    "content_valid": content_valid,
                    "required": info["required"],
                    "stats": sample_stats,
                    "path": str(file_path)
                }
                
                # Display validation results
                status = "VALID" if content_valid else "WARNING"
                req_status = "[REQUIRED]" if info["required"] else "[OPTIONAL]"
                print(f"   {status} {filename} {req_status}: {file_size_mb:.1f} MB")
                
                # Show detailed statistics if available
                if sample_stats:
                    for key, value in sample_stats.items():
                        print(f"      {key}: {value}")
                
            else:
                # File not found
                validation_results[filename] = {
                    "exists": False,
                    "required": info["required"],
                    "path": str(file_path)
                }
                
                if info["required"]:
                    print(f"   ERROR {filename} [REQUIRED]: NOT FOUND - {file_path}")
                    all_required_present = False
                else:
                    print(f"   SKIP {filename} [OPTIONAL]: Not found (skipping)")
        
        return validation_results, all_required_present
    
    def create_upload_mapping(self, validation_results: Dict) -> Dict[str, str]:
        """
        Create mapping of local files to versioned S3 keys
        
        Args:
            validation_results: Results from file validation
            
        Returns:
            Dictionary mapping local file paths to S3 keys
        """
        upload_mapping = {}
        
        for filename, info in validation_results.items():
            # Only include files that exist and have valid content
            if info["exists"] and info.get("content_valid", True):
                local_path = info["path"]
                
                # Define S3 key structure based on file type
                if filename == "training_dataset.json":
                    s3_key = f"datasets/algospeak-training/{self.version_id}/training_data.json"
                elif filename == "algospeak_patterns.json":
                    s3_key = f"patterns/algospeak_patterns_{self.version_id}.json"
                elif filename.endswith('.csv'):
                    s3_key = f"datasets/raw-jigsaw/{self.version_id}/{filename}"
                else:
                    s3_key = f"datasets/misc/{self.version_id}/{filename}"
                
                upload_mapping[local_path] = s3_key
        
        return upload_mapping
    
    def execute_uploads(self, upload_mapping: Dict[str, str]) -> Dict[str, Optional[str]]:
        """
        Execute all file uploads using the robust upload manager
        
        Args:
            upload_mapping: Dictionary mapping local paths to S3 keys
            
        Returns:
            Dictionary mapping local paths to S3 URLs (or None for failures)
        """
        if not upload_mapping:
            print("WARNING: No files to upload")
            return {}
        
        print(f"\nStarting uploads for version {self.version_id}...")
        print(f"   Files to upload: {len(upload_mapping)}")
        
        # Use our upload manager for reliable uploads with retry logic
        upload_results = self.uploader.upload_multiple_files(upload_mapping)
        
        return upload_results
    
    def create_upload_manifest(self, upload_results: Dict[str, Optional[str]], 
                             validation_results: Dict) -> str:
        """
        Create comprehensive upload manifest for tracking and reproducibility
        
        Args:
            upload_results: Results from upload operations
            validation_results: Results from file validation
            
        Returns:
            Path to created manifest file
        """
        # Build comprehensive manifest with all relevant information
        manifest = {
            "version": self.version_id,
            "created_at": datetime.now().isoformat(),
            "bucket": self.bucket,
            "upload_summary": {
                "total_files": len(upload_results),
                "successful_uploads": sum(1 for result in upload_results.values() if result),
                "failed_uploads": sum(1 for result in upload_results.values() if not result)
            },
            "files": {}
        }
        
        # Add detailed information for each file
        for local_path, s3_url in upload_results.items():
            filename = Path(local_path).name
            file_info = validation_results.get(filename, {})
            
            manifest["files"][filename] = {
                "local_path": local_path,
                "s3_url": s3_url,
                "upload_successful": s3_url is not None,
                "file_size_mb": file_info.get("size_mb", 0),
                "file_stats": file_info.get("stats", {}),
                "validation_passed": file_info.get("content_valid", False)
            }
        
        # Determine if ready for SageMaker training
        required_files = ["training_dataset.json", "algospeak_patterns.json"]
        uploaded_required = [
            filename for filename in required_files 
            if manifest["files"].get(filename, {}).get("upload_successful", False)
        ]
        
        manifest["ready_for_training"] = len(uploaded_required) == len(required_files)
        manifest["training_requirements"] = {
            "required_files": required_files,
            "uploaded_required": uploaded_required,
            "missing_required": [f for f in required_files if f not in uploaded_required]
        }
        
        # Save manifest to local file for reference
        manifest_filename = f"upload_manifest_{self.version_id}.json"
        manifest_path = Path("data") / manifest_filename
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"\nUpload manifest created: {manifest_path}")
        return str(manifest_path)
    
    def generate_summary_report(self, upload_results: Dict[str, Optional[str]], 
                              manifest_path: str) -> None:
        """
        Generate comprehensive final summary report
        
        Args:
            upload_results: Results from upload operations
            manifest_path: Path to created manifest file
        """
        print("\n" + "=" * 60)
        print("DATA INGESTION SUMMARY")
        print("=" * 60)
        
        print(f"Version: {self.version_id}")
        print(f"Bucket: {self.bucket}")
        print(f"Manifest: {manifest_path}")
        
        # Categorize upload results
        successful = [path for path, url in upload_results.items() if url]
        failed = [path for path, url in upload_results.items() if not url]
        
        print(f"\nUpload Results:")
        print(f"   Total files processed: {len(upload_results)}")
        print(f"   Successful uploads: {len(successful)}")
        print(f"   Failed uploads: {len(failed)}")
        
        # Display successful uploads
        if successful:
            print(f"\nSuccessfully uploaded:")
            for local_path in successful:
                filename = Path(local_path).name
                s3_url = upload_results[local_path]
                print(f"   - {filename}: {s3_url}")
        
        # Display failed uploads
        if failed:
            print(f"\nFailed uploads:")
            for local_path in failed:
                filename = Path(local_path).name
                print(f"   - {filename}: {local_path}")
        
        # Check training readiness
        required_successful = [
            path for path in successful 
            if Path(path).name in ["training_dataset.json", "algospeak_patterns.json"]
        ]
        
        if len(required_successful) == 2:
            print(f"\nSUCCESS! All required files uploaded.")
            print(f"Ready for SageMaker training!")
            print(f"\nNext Steps:")
            print(f"   1. Run: python training/model_training.py")
            print(f"   2. Monitor training in SageMaker console")
            print(f"   3. Deploy model using deployment/model_deployment.py")
        else:
            print(f"\nWARNING: Training requirements not met.")
            print(f"   Required files: training_dataset.json, algospeak_patterns.json")
            print(f"   Successfully uploaded: {len(required_successful)}/2")

# =============================================================================
# MAIN EXECUTION PIPELINE
# =============================================================================
# This section ties everything together and executes the complete workflow

def main():
    """
    Execute the complete algospeak data ingestion pipeline
    
    Workflow:
    1. Initialize pipeline components
    2. Verify AWS S3 access
    3. Validate local dataset files
    4. Create versioned upload mappings
    5. Execute reliable uploads with retry logic
    6. Generate tracking manifests
    7. Provide comprehensive summary and next steps
    """
    print("ALGOSPEAK DATA INGESTION PIPELINE")
    print("Preprocessing completed in Jupyter - uploading processed datasets")
    print("=" * 70)
    
    try:
        # Initialize the complete ingestion pipeline
        ingestion = AlgospeakDataIngestion()
        
        # Verify S3 bucket access before proceeding
        if not ingestion.uploader.verify_s3_access():
            print("ERROR: Cannot proceed without S3 access")
            print("CHECK: Verify your AWS credentials and S3_BUCKET_NAME in .env")
            return
        
        # Step 1: Validate that all required local files exist and are valid
        validation_results, all_required_present = ingestion.validate_dataset_files()
        
        if not all_required_present:
            print("\nERROR: Required files missing - cannot proceed")
            print("CHECK: Ensure training_dataset.json and algospeak_patterns.json exist in data/dataset/")
            return
        
        # Step 2: Create versioned upload mapping for S3 storage
        upload_mapping = ingestion.create_upload_mapping(validation_results)
        
        if not upload_mapping:
            print("\nWARNING: No valid files found for upload")
            return
        
        # Step 3: Execute uploads using robust upload manager
        upload_results = ingestion.execute_uploads(upload_mapping)
        
        # Step 4: Create comprehensive tracking manifest
        manifest_path = ingestion.create_upload_manifest(upload_results, validation_results)
        
        # Step 5: Generate final summary report and next steps
        ingestion.generate_summary_report(upload_results, manifest_path)
        
    except KeyboardInterrupt:
        print("\nWARNING: Pipeline interrupted by user")
        
    except Exception as e:
        print(f"\nERROR: Pipeline failed with error: {e}")
        print("CHECK: Verify your .env configuration and file paths")
        print("CHECK: Ensure all required files exist in data/dataset/")

# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
