#!/usr/bin/env python3
"""
ALGOSPEAK MODEL TRAINING WORKER
Executes actual model fine-tuning inside SageMaker training instances

This script runs inside SageMaker training containers and:
1. Downloads Qwen2.5-3B model directly from HuggingFace Hub to AWS
2. Implements QLoRA fine-tuning for memory efficiency
3. Processes algospeak instruction dataset
4. Saves fine-tuned model artifacts to S3

IMPORTANT: This is automatically executed by SageMaker - do not run directly.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset as HFDataset

# =============================================================================
# SAGEMAKER ENVIRONMENT INITIALIZATION
# =============================================================================

# Configure logging for SageMaker monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# DATASET PROCESSING AND FORMATTING
# =============================================================================

class AlgospeakInstructionDataset:
    """
    Dataset handler for algospeak instruction fine-tuning data
    
    Processes instruction-input-output triplets from your data ingestion pipeline
    and formats them for QLoRA fine-tuning of the content moderation model.
    """
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        """Initialize dataset processing"""
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load your instruction dataset uploaded by data ingestion
        logger.info(f"Loading training data from: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} training samples")
        
        # Log dataset statistics for monitoring
        self.log_dataset_stats()
    
    def log_dataset_stats(self):
        """Analyze dataset for training monitoring"""
        
        if not self.data:
            return
        
        # Analyze dataset composition
        label_counts = {}
        algospeak_count = 0
        
        for item in self.data:
            label = item.get('label', 'unknown')
            label_counts[label] = label_counts.get(label, 0) + 1
            
            if item.get('is_algospeak', False):
                algospeak_count += 1
        
        # Log comprehensive dataset statistics
        logger.info("Dataset Statistics:")
        logger.info(f"  Total samples: {len(self.data)}")
        logger.info(f"  Algospeak samples: {algospeak_count} ({algospeak_count/len(self.data)*100:.1f}%)")
        logger.info("  Label distribution:")
        for label, count in label_counts.items():
            logger.info(f"    {label}: {count} ({count/len(self.data)*100:.1f}%)")
    
    def format_instruction(self, item: Dict) -> str:
        """Format individual training examples for instruction fine-tuning"""
        
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output_text = item.get('output', '')
        
        # Format for instruction fine-tuning (standard format for LLMs)
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        # Complete training example with EOS token
        full_text = prompt + output_text + self.tokenizer.eos_token
        
        return full_text
    
    def create_hf_dataset(self) -> HFDataset:
        """Create HuggingFace Dataset for training"""
        
        # Format all training examples
        formatted_texts = []
        for item in self.data:
            formatted_text = self.format_instruction(item)
            formatted_texts.append(formatted_text)
        
        # Create HuggingFace dataset from formatted text
        dataset = HFDataset.from_dict({'text': formatted_texts})
        
        # Tokenize dataset for model consumption
        def tokenize_function(examples):
            """Convert text to tokens that the model can understand"""
            tokenized = self.tokenizer(
                examples['text'],
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_overflowing_tokens=False,
            )
            # Create labels for causal language modeling
            tokenized['labels'] = tokenized['input_ids'].copy()
            return tokenized
        
        # Apply tokenization to entire dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text'],
            desc="Tokenizing dataset"
        )
        
        logger.info(f"Created tokenized dataset with {len(tokenized_dataset)} examples")
        
        return tokenized_dataset

# =============================================================================
# MODEL LOADING AND QLORA SETUP
# =============================================================================

class AlgospeakModelTrainer:
    """
    Main training coordinator within SageMaker
    
    Handles model loading, QLoRA configuration, training execution,
    and model saving with proper error handling and monitoring.
    """
    
    def __init__(self, args):
        """Initialize model trainer with hyperparameters"""
        
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Log training environment
        logger.info(f"Initializing trainer with device: {self.device}")
        logger.info(f"Model: {args.model_name}")
        logger.info(f"Training epochs: {args.num_train_epochs}")
        logger.info(f"Batch size: {args.per_device_train_batch_size}")
        logger.info(f"Learning rate: {args.learning_rate}")
        
        # Initialize model components
        self.tokenizer = self.load_tokenizer()
        self.model = self.load_model()
        
        # Configure QLoRA for memory efficiency
        if args.use_lora:
            self.model = self.setup_lora()
        
        logger.info("Trainer initialization complete")
    
    def load_tokenizer(self):
        """Load and configure tokenizer from HuggingFace Hub"""
        
        logger.info(f"Loading tokenizer: {self.args.model_name}")
        
        # DIRECT DOWNLOAD: Downloads tokenizer from HuggingFace Hub to SageMaker
        # Uses HUGGINGFACE_HUB_TOKEN environment variable for authentication
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name,        # 'Qwen/Qwen2.5-3B-Instruct'
            trust_remote_code=True,      # Allow custom tokenizer code
            cache_dir="/tmp/model_cache"  # Download location within SageMaker instance
        )
        
        # Configure padding token for batch processing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        logger.info("Tokenizer loaded successfully")
        return tokenizer
    
    def load_model(self):
        """
        CRITICAL: Download and load base model directly from HuggingFace Hub to AWS
        
        This function performs the AUTOMATIC MODEL DOWNLOAD that you asked about.
        The model (Qwen2.5-3B-Instruct) is downloaded directly from HuggingFace Hub
        into the SageMaker training instance - no manual download or S3 storage needed.
        """
        
        logger.info(f"Loading model: {self.args.model_name}")
        
        # AUTOMATIC HUGGINGFACE MODEL DOWNLOAD TO AWS SAGEMAKER
        # This line downloads Qwen2.5-3B-Instruct directly from HuggingFace Hub
        # to the SageMaker training instance using your HUGGINGFACE_TOKEN
        model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,        # 'Qwen/Qwen2.5-3B-Instruct' - HF model identifier
            trust_remote_code=True,      # Allow custom model code from HuggingFace
            cache_dir="/tmp/model_cache", # Downloads directly to SageMaker instance storage
            torch_dtype=torch.bfloat16 if self.args.bf16 else torch.float16,  # Memory optimization
            device_map="auto" if torch.cuda.is_available() else None,  # Automatic GPU placement
        )
        
        # Enable gradient checkpointing for memory efficiency
        if self.args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        logger.info("Model downloaded and loaded successfully from HuggingFace Hub")
        return model
    
    def setup_lora(self):
        """Configure QLoRA for efficient fine-tuning"""
        
        logger.info("Setting up QLoRA configuration")
        
        # Prepare model for k-bit training (quantization)
        model = prepare_model_for_kbit_training(self.model)
        
        # LoRA configuration optimized for Qwen2.5-3B
        lora_config = LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            target_modules=self.args.target_modules.split(','),
            lora_dropout=self.args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Apply LoRA configuration to model
        model = get_peft_model(model, lora_config)
        
        # Calculate and log parameter efficiency
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        logger.info(f"QLoRA configuration complete")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
        
        return model
    
    def create_dataset(self):
        """Prepare dataset for training"""
        
        dataset_handler = AlgospeakInstructionDataset(
            self.args.dataset_path,
            self.tokenizer,
            max_length=512
        )
        
        return dataset_handler.create_hf_dataset()

# =============================================================================
# TRAINING EXECUTION
# =============================================================================

    def train(self):
        """Execute model fine-tuning"""
        
        logger.info("Starting model training")
        
        # Create training dataset from your uploaded data
        train_dataset = self.create_dataset()
        
        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            num_train_epochs=self.args.num_train_epochs,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            learning_rate=self.args.learning_rate,
            warmup_ratio=self.args.warmup_ratio,
            logging_steps=self.args.logging_steps,
            save_steps=self.args.save_steps,
            save_total_limit=self.args.save_total_limit,
            remove_unused_columns=False,
            dataloader_pin_memory=self.args.dataloader_pin_memory,
            gradient_checkpointing=self.args.gradient_checkpointing,
            bf16=self.args.bf16,
            report_to=None,  # Disable external logging
        )
        
        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )
        
        # Create HuggingFace Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Execute training - this is where the actual learning happens
        logger.info("Beginning training...")
        trainer.train()
        
        # Save final trained model to S3
        logger.info("Saving trained model")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.args.output_dir)
        
        logger.info("Training completed successfully")

# =============================================================================
# ARGUMENT PARSING AND MAIN EXECUTION
# =============================================================================

def parse_arguments():
    """Parse training configuration from orchestrator"""
    
    parser = argparse.ArgumentParser(description="Algospeak Model Training Worker")
    
    # Model and data arguments
    parser.add_argument('--model_name', type=str, required=True, help='Base model name')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to training dataset')
    parser.add_argument('--patterns_path', type=str, help='Path to algospeak patterns')
    parser.add_argument('--output_dir', type=str, default='/opt/ml/model', help='Output directory')
    
    # Training arguments
    parser.add_argument('--num_train_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--per_device_train_batch_size', type=int, default=2, help='Batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--logging_steps', type=int, default=10, help='Logging frequency')
    parser.add_argument('--save_steps', type=int, default=500, help='Save frequency')
    parser.add_argument('--save_total_limit', type=int, default=2, help='Max saved checkpoints')
    
    # QLoRA arguments
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA fine-tuning')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
    parser.add_argument('--target_modules', type=str, default='q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj', help='LoRA target modules')
    
    # Optimization arguments
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Enable gradient checkpointing')
    parser.add_argument('--dataloader_pin_memory', action='store_true', help='Pin memory in dataloader')
    parser.add_argument('--bf16', action='store_true', help='Use bfloat16 precision')
    
    return parser.parse_args()

def main():
    """Main training execution coordinator"""
    
    # Parse configuration from orchestrator
    args = parse_arguments()
    
    # Log environment information
    logger.info("=== Algospeak Model Training Worker ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Transformers version: {transformers.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
    
    try:
        # Initialize trainer with configuration
        trainer = AlgospeakModelTrainer(args)
        
        # Execute complete training pipeline
        trainer.train()
        
        logger.info("Training worker completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()



