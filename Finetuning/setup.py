#!/usr/bin/env python3
"""
Setup script for Medical Assistant Llama3 Finetuning
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA {cuda_version} available")
            print(f"✅ GPU: {gpu_name} (Count: {gpu_count})")
            return True
        else:
            print("⚠️  CUDA not available - Training will be slow on CPU")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing requirements...")
    
    try:
        # Install Unsloth first
        print("Installing Unsloth...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
        ])
        
        # Install other requirements
        print("Installing other requirements...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        
        print("✅ Requirements installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        "medical_dialogue_dataset",
        "medical_llama3_finetuned",
        "logs",
        "checkpoints"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def check_data_file():
    """Check if the data file exists"""
    data_path = Path("../datacreation/dialogues.csv")
    if data_path.exists():
        print(f"✅ Data file found: {data_path}")
        return True
    else:
        print(f"❌ Data file not found: {data_path}")
        print("Please ensure the medical dialogue dataset exists.")
        return False

def setup_wandb():
    """Setup Weights & Biases"""
    print("\n🔧 Setting up Weights & Biases...")
    print("To use WandB for experiment tracking:")
    print("1. Create an account at https://wandb.ai")
    print("2. Run: wandb login")
    print("3. Or set WANDB_MODE=disabled to skip WandB")

def create_config_file():
    """Create a configuration file"""
    config = {
        "model": {
            "name": "unsloth/llama-3-8b-bnb-4bit",
            "max_seq_length": 2048,
            "load_in_4bit": True
        },
        "training": {
            "num_epochs": 3,
            "batch_size": 2,
            "learning_rate": 2e-4,
            "gradient_accumulation_steps": 4
        },
        "data": {
            "source_path": "../datacreation/dialogues.csv",
            "max_samples": 10000
        }
    }
    
    import json
    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Created config.json")

def main():
    """Main setup function"""
    print("="*60)
    print("🏥 Medical Assistant Llama3 Finetuning Setup")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check CUDA
    check_cuda()
    
    # Check data file
    if not check_data_file():
        print("\nPlease ensure the data file exists before proceeding.")
        return
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Create config file
    print("\n⚙️  Creating configuration...")
    create_config_file()
    
    # Setup WandB
    setup_wandb()
    
    # Install requirements
    install_choice = input("\n🤔 Install requirements now? (y/N): ")
    if install_choice.lower() == 'y':
        if not install_requirements():
            print("❌ Setup failed during requirements installation")
            return
    
    print("\n" + "="*60)
    print("🎉 Setup completed successfully!")
    print("="*60)
    
    print("\n📋 Next steps:")
    print("1. Run the complete pipeline:")
    print("   python run_finetuning.py")
    print("\n2. Or run individual steps:")
    print("   python data_preparation.py")
    print("   python finetune_llama3.py")
    print("   python inference.py --mode test")
    print("\n3. For interactive chat:")
    print("   python inference.py --mode interactive")
    
    print("\n📚 For more information, see README.md")

if __name__ == "__main__":
    main() 