#!/usr/bin/env python3
"""
Main Runner Script for Llama3 Medical Assistant Finetuning
This script orchestrates the entire pipeline from data preparation to training
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import time
from datetime import datetime

def check_requirements():
    """Check if all required packages are installed"""
    print("Checking requirements...")
    
    required_packages = [
        "torch", "transformers", "datasets", "accelerate", 
        "peft", "trl", "bitsandbytes", "wandb"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} - MISSING")
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("All requirements satisfied!")
    return True

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU available: {gpu_name} (Count: {gpu_count})")
            return True
        else:
            print("✗ No GPU available - Training will be very slow!")
            return False
    except ImportError:
        print("✗ PyTorch not available")
        return False

def check_data():
    """Check if data files exist"""
    print("Checking data files...")
    
    data_path = Path("../datacreation/dialogues.csv")
    if data_path.exists():
        print(f"✓ Data file found: {data_path}")
        return True
    else:
        print(f"✗ Data file not found: {data_path}")
        return False

def run_data_preparation():
    """Run data preparation step"""
    print("\n" + "="*50)
    print("STEP 1: DATA PREPARATION")
    print("="*50)
    
    try:
        result = subprocess.run([
            sys.executable, "data_preparation.py"
        ], capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print("✓ Data preparation completed successfully!")
            print(result.stdout)
            return True
        else:
            print("✗ Data preparation failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"✗ Error running data preparation: {e}")
        return False

def run_finetuning():
    """Run the finetuning step"""
    print("\n" + "="*50)
    print("STEP 2: MODEL FINETUNING")
    print("="*50)
    
    try:
        result = subprocess.run([
            sys.executable, "finetune_llama3.py"
        ], capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print("✓ Finetuning completed successfully!")
            print(result.stdout)
            return True
        else:
            print("✗ Finetuning failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"✗ Error running finetuning: {e}")
        return False

def run_inference():
    """Run inference to test the model"""
    print("\n" + "="*50)
    print("STEP 3: MODEL INFERENCE")
    print("="*50)
    
    try:
        result = subprocess.run([
            sys.executable, "inference.py", "--mode", "test"
        ], capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print("✓ Inference completed successfully!")
            print(result.stdout)
            return True
        else:
            print("✗ Inference failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"✗ Error running inference: {e}")
        return False

def create_training_summary():
    """Create a summary of the training process"""
    summary = f"""
# Medical Assistant Finetuning Summary

## Training Configuration
- Model: Llama3-8B with Unsloth optimizations
- Dataset: Medical dialogues from {Path("../datacreation/dialogues.csv")}
- Training Method: LoRA (Low-Rank Adaptation)
- Quantization: 4-bit quantization for memory efficiency

## Files Created
- `medical_dialogue_dataset/`: Prepared dataset
- `medical_llama3_finetuned/`: Finetuned model
- `formatted_medical_data.json`: Formatted training data
- `inference_results.json`: Test results

## Usage
1. Interactive chat: `python inference.py --mode interactive`
2. Test examples: `python inference.py --mode test`

## Training Parameters
- Learning rate: 2e-4
- Batch size: 2 (per device)
- Gradient accumulation: 4
- Max sequence length: 2048
- LoRA rank: 16

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open("training_summary.md", "w") as f:
        f.write(summary)
    
    print("✓ Training summary created: training_summary.md")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Medical Assistant Finetuning Pipeline")
    parser.add_argument("--skip-checks", action="store_true", 
                       help="Skip environment checks")
    parser.add_argument("--skip-data-prep", action="store_true",
                       help="Skip data preparation (use existing dataset)")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip training (use existing model)")
    parser.add_argument("--skip-inference", action="store_true",
                       help="Skip inference testing")
    
    args = parser.parse_args()
    
    print("="*60)
    print("MEDICAL ASSISTANT LLAMA3 FINETUNING PIPELINE")
    print("="*60)
    
    # Environment checks
    if not args.skip_checks:
        print("\nPerforming environment checks...")
        
        if not check_requirements():
            print("Please install missing requirements and try again.")
            return
        
        if not check_gpu():
            print("Warning: No GPU detected. Training will be very slow!")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                return
        
        if not check_data():
            print("Please ensure the data file exists and try again.")
            return
    
    # Create output directories
    os.makedirs("medical_llama3_finetuned", exist_ok=True)
    
    # Run pipeline steps
    success = True
    
    # Step 1: Data Preparation
    if not args.skip_data_prep:
        if not run_data_preparation():
            success = False
            print("Pipeline failed at data preparation step.")
            return
    else:
        print("Skipping data preparation...")
    
    # Step 2: Finetuning
    if not args.skip_training:
        if not run_finetuning():
            success = False
            print("Pipeline failed at finetuning step.")
            return
    else:
        print("Skipping training...")
    
    # Step 3: Inference
    if not args.skip_inference:
        if not run_inference():
            success = False
            print("Pipeline failed at inference step.")
            return
    else:
        print("Skipping inference...")
    
    # Create summary
    create_training_summary()
    
    if success:
        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY! 🎉")
        print("="*60)
        print("\nYour medical assistant is ready!")
        print("\nNext steps:")
        print("1. Test the model: python inference.py --mode interactive")
        print("2. Check the results in inference_results.json")
        print("3. Review the training summary in training_summary.md")
    else:
        print("\nPipeline failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 