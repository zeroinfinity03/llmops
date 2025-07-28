#!/usr/bin/env python3
"""
ALGOSPEAK MODEL EVALUATION
Evaluate trained model performance on test datasets

This script:
1. Loads trained model from SageMaker artifacts
2. Evaluates on test datasets with algospeak samples
3. Generates comprehensive performance metrics
4. Creates evaluation reports for model validation

Usage: python training/model_evaluation.py
"""

import os
import json
import boto3
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# EVALUATION ORCHESTRATOR
# =============================================================================

class AlgospeakModelEvaluator:
    """
    Production model evaluation system
    
    Evaluates trained algospeak models on test datasets to validate
    performance before deployment and monitor model quality over time.
    """
    
    def __init__(self):
        """Initialize evaluation system"""
        
        load_dotenv()
        
        self.bucket = os.getenv('S3_BUCKET_NAME')
        self.model_name = os.getenv('MODEL_NAME', 'qwen-algospeak')
        self.endpoint_name = os.getenv('SAGEMAKER_ENDPOINT_NAME')
        
        # Initialize SageMaker runtime for inference
        self.sagemaker_runtime = boto3.client('sagemaker-runtime')
        
        print(f"Model Evaluator Initialized")
        print(f"   Model: {self.model_name}")
        print(f"   Endpoint: {self.endpoint_name}")
    
    def load_test_dataset(self) -> List[Dict]:
        """Load test dataset for evaluation"""
        
        # Look for test dataset from data preparation
        test_files = [
            "data/dataset/test_dataset.json",
            "data/dataset/training_dataset.json"  # Use subset if no separate test set
        ]
        
        for test_file in test_files:
            if Path(test_file).exists():
                print(f"Loading test data from: {test_file}")
                with open(test_file, 'r') as f:
                    data = json.load(f)
                
                # If using training data, take a subset for evaluation
                if "training_dataset" in test_file:
                    # Take last 1000 samples as test set
                    data = data[-1000:]
                    print(f"Using last 1000 training samples as test set")
                
                print(f"Loaded {len(data)} test samples")
                return data
        
        raise FileNotFoundError("No test dataset found. Run data preparation first.")
    
    def create_evaluation_samples(self, test_data: List[Dict]) -> List[Dict]:
        """Create balanced evaluation samples including algospeak variants"""
        
        # Separate by label for balanced evaluation
        samples_by_label = {}
        algospeak_samples = []
        
        for item in test_data:
            label = item.get('label', 'unknown')
            if label not in samples_by_label:
                samples_by_label[label] = []
            samples_by_label[label].append(item)
            
            # Track algospeak samples separately
            if item.get('is_algospeak', False):
                algospeak_samples.append(item)
        
        # Create balanced test set (max 200 per label)
        evaluation_samples = []
        for label, items in samples_by_label.items():
            sample_count = min(200, len(items))
            evaluation_samples.extend(items[:sample_count])
        
        print(f"Created evaluation set:")
        print(f"   Total samples: {len(evaluation_samples)}")
        print(f"   Algospeak samples: {len(algospeak_samples)}")
        for label, items in samples_by_label.items():
            count = min(200, len(items))
            print(f"   {label}: {count} samples")
        
        return evaluation_samples
    
    def predict_sample(self, content: str) -> Dict:
        """Get model prediction for single sample"""
        
        if not self.endpoint_name:
            raise ValueError("SAGEMAKER_ENDPOINT_NAME not configured")
        
        try:
            response = self.sagemaker_runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Body=json.dumps({"inputs": content})
            )
            
            result = json.loads(response['Body'].read().decode())
            
            return {
                'predicted_label': result.get('classification', 'unknown'),
                'confidence': result.get('confidence', 0.0),
                'is_algospeak_detected': result.get('is_algospeak', False)
            }
            
        except Exception as e:
            print(f"Prediction error for content '{content[:50]}...': {e}")
            return {
                'predicted_label': 'error',
                'confidence': 0.0,
                'is_algospeak_detected': False
            }
    
    def evaluate_model(self, evaluation_samples: List[Dict]) -> Dict:
        """Run comprehensive model evaluation"""
        
        print(f"Evaluating model on {len(evaluation_samples)} samples...")
        
        # Collect predictions and ground truth
        predictions = []
        ground_truth = []
        algospeak_predictions = []
        algospeak_ground_truth = []
        confidence_scores = []
        
        for i, sample in enumerate(evaluation_samples):
            if i % 50 == 0:
                print(f"   Progress: {i}/{len(evaluation_samples)}")
            
            content = sample.get('input', '')
            true_label = sample.get('label', 'unknown')
            is_algospeak = sample.get('is_algospeak', False)
            
            # Get model prediction
            prediction = self.predict_sample(content)
            
            predictions.append(prediction['predicted_label'])
            ground_truth.append(true_label)
            confidence_scores.append(prediction['confidence'])
            
            # Track algospeak detection separately
            algospeak_predictions.append(prediction['is_algospeak_detected'])
            algospeak_ground_truth.append(is_algospeak)
        
        # Calculate comprehensive metrics
        metrics = self.calculate_metrics(
            ground_truth, predictions, confidence_scores,
            algospeak_ground_truth, algospeak_predictions
        )
        
        return metrics
    
    def calculate_metrics(self, ground_truth: List[str], predictions: List[str],
                         confidence_scores: List[float], algospeak_gt: List[bool],
                         algospeak_pred: List[bool]) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        
        # Overall classification metrics
        accuracy = accuracy_score(ground_truth, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            ground_truth, predictions, average='weighted'
        )
        
        # Per-class metrics
        class_report = classification_report(ground_truth, predictions, output_dict=True)
        
        # Algospeak detection metrics
        algospeak_accuracy = accuracy_score(algospeak_gt, algospeak_pred)
        algospeak_precision, algospeak_recall, algospeak_f1, _ = precision_recall_fscore_support(
            algospeak_gt, algospeak_pred, average='binary'
        )
        
        # Confidence analysis
        avg_confidence = np.mean(confidence_scores)
        confidence_by_correctness = {
            'correct': np.mean([conf for conf, gt, pred in zip(confidence_scores, ground_truth, predictions) if gt == pred]),
            'incorrect': np.mean([conf for conf, gt, pred in zip(confidence_scores, ground_truth, predictions) if gt != pred])
        }
        
        return {
            'overall_metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'average_confidence': avg_confidence
            },
            'per_class_metrics': class_report,
            'algospeak_detection': {
                'accuracy': algospeak_accuracy,
                'precision': algospeak_precision,
                'recall': algospeak_recall,
                'f1_score': algospeak_f1
            },
            'confidence_analysis': confidence_by_correctness,
            'confusion_matrix': confusion_matrix(ground_truth, predictions).tolist()
        }
    
    def generate_evaluation_report(self, metrics: Dict, output_dir: str = "evaluation_results"):
        """Generate comprehensive evaluation report"""
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics as JSON
        metrics_file = Path(output_dir) / f"evaluation_metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Generate text report
        report_file = Path(output_dir) / f"evaluation_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write("ALGOSPEAK MODEL EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Evaluation Date: {datetime.now().isoformat()}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Endpoint: {self.endpoint_name}\n\n")
            
            # Overall performance
            overall = metrics['overall_metrics']
            f.write("OVERALL PERFORMANCE:\n")
            f.write(f"   Accuracy: {overall['accuracy']:.3f}\n")
            f.write(f"   Precision: {overall['precision']:.3f}\n")
            f.write(f"   Recall: {overall['recall']:.3f}\n")
            f.write(f"   F1-Score: {overall['f1_score']:.3f}\n")
            f.write(f"   Avg Confidence: {overall['average_confidence']:.3f}\n\n")
            
            # Algospeak detection
            algospeak = metrics['algospeak_detection']
            f.write("ALGOSPEAK DETECTION:\n")
            f.write(f"   Accuracy: {algospeak['accuracy']:.3f}\n")
            f.write(f"   Precision: {algospeak['precision']:.3f}\n")
            f.write(f"   Recall: {algospeak['recall']:.3f}\n")
            f.write(f"   F1-Score: {algospeak['f1_score']:.3f}\n\n")
            
            # Per-class performance
            f.write("PER-CLASS PERFORMANCE:\n")
            for class_name, class_metrics in metrics['per_class_metrics'].items():
                if isinstance(class_metrics, dict) and 'precision' in class_metrics:
                    f.write(f"   {class_name}:\n")
                    f.write(f"      Precision: {class_metrics['precision']:.3f}\n")
                    f.write(f"      Recall: {class_metrics['recall']:.3f}\n")
                    f.write(f"      F1-Score: {class_metrics['f1-score']:.3f}\n")
                    f.write(f"      Support: {class_metrics['support']}\n\n")
        
        print(f"Evaluation report saved:")
        print(f"   Metrics: {metrics_file}")
        print(f"   Report: {report_file}")
        
        return str(report_file)
    
    def run_complete_evaluation(self) -> str:
        """Execute complete evaluation pipeline"""
        
        print("ALGOSPEAK MODEL EVALUATION PIPELINE")
        print("=" * 50)
        
        try:
            # Load test dataset
            print("\nStep 1: Loading test dataset...")
            test_data = self.load_test_dataset()
            
            # Create evaluation samples
            print("\nStep 2: Creating evaluation samples...")
            evaluation_samples = self.create_evaluation_samples(test_data)
            
            # Run evaluation
            print("\nStep 3: Running model evaluation...")
            metrics = self.evaluate_model(evaluation_samples)
            
            # Generate report
            print("\nStep 4: Generating evaluation report...")
            report_path = self.generate_evaluation_report(metrics)
            
            # Summary
            overall = metrics['overall_metrics']
            algospeak = metrics['algospeak_detection']
            
            print(f"\nEVALUATION COMPLETE!")
            print(f"   Overall Accuracy: {overall['accuracy']:.3f}")
            print(f"   Overall F1-Score: {overall['f1_score']:.3f}")
            print(f"   Algospeak Detection F1: {algospeak['f1_score']:.3f}")
            print(f"   Report: {report_path}")
            
            return report_path
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
            raise

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Execute model evaluation pipeline"""
    
    try:
        evaluator = AlgospeakModelEvaluator()
        report_path = evaluator.run_complete_evaluation()
        
        print("\nNext Steps:")
        print("1. Review evaluation report for model performance")
        print("2. If performance is acceptable, proceed with deployment")
        print("3. If performance needs improvement, retrain with more data")
        
        return report_path
        
    except Exception as e:
        print(f"Model evaluation failed: {e}")
        return None

if __name__ == "__main__":
    main()