#!/usr/bin/env python3
"""
Setup script for the contrastive learning project.

This script helps users set up the project environment,
download datasets, and run initial tests.
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse


def run_command(command: str, description: str) -> bool:
    """
    Run a command and return success status.
    
    Args:
        command: Command to run
        description: Description of what the command does
        
    Returns:
        True if command succeeded, False otherwise
    """
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"❌ Python 3.10+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor} is compatible")
    return True


def install_dependencies() -> bool:
    """Install project dependencies."""
    commands = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install -r requirements.txt", "Installing dependencies"),
        ("pip install -e .", "Installing package in development mode"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    
    return True


def setup_pre_commit() -> bool:
    """Setup pre-commit hooks."""
    commands = [
        ("pip install pre-commit", "Installing pre-commit"),
        ("pre-commit install", "Installing pre-commit hooks"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    
    return True


def create_directories() -> bool:
    """Create necessary directories."""
    directories = [
        "data",
        "checkpoints",
        "logs",
        "outputs",
        "assets",
        "assets/visualizations",
        "assets/embeddings",
        "assets/models",
        "configs/local",
        "experiments",
        "results",
    ]
    
    print("🔄 Creating directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created successfully!")
    return True


def run_tests() -> bool:
    """Run unit tests."""
    commands = [
        ("python -m pytest tests/ -v", "Running unit tests"),
        ("python -m pytest tests/ --cov=src --cov-report=html", "Running tests with coverage"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    
    return True


def download_sample_data() -> bool:
    """Download sample data for testing."""
    print("🔄 Downloading sample data...")
    
    # Create a simple script to download CIFAR-10
    download_script = """
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

print("Downloading CIFAR-10 dataset...")
dataset = datasets.CIFAR10(root='./data', train=True, download=True)
print(f"Downloaded {len(dataset)} training samples")

dataset = datasets.CIFAR10(root='./data', train=False, download=True)
print(f"Downloaded {len(dataset)} test samples")
print("Sample data download completed!")
"""
    
    with open("download_data.py", "w") as f:
        f.write(download_script)
    
    success = run_command("python download_data.py", "Downloading sample data")
    
    # Clean up
    if os.path.exists("download_data.py"):
        os.remove("download_data.py")
    
    return success


def run_demo() -> bool:
    """Run the demo application."""
    print("🔄 Starting demo application...")
    print("📱 The demo will be available at http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the demo")
    
    try:
        subprocess.run(["streamlit", "run", "demo.py"], check=True)
        return True
    except subprocess.CalledProcessError:
        print("❌ Demo failed to start!")
        return False
    except KeyboardInterrupt:
        print("🛑 Demo stopped by user")
        return True


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup contrastive learning project")
    
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="Skip dependency installation"
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running tests"
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip downloading sample data"
    )
    parser.add_argument(
        "--demo-only",
        action="store_true",
        help="Only run the demo application"
    )
    parser.add_argument(
        "--pre-commit",
        action="store_true",
        help="Setup pre-commit hooks"
    )
    
    args = parser.parse_args()
    
    print("🚀 Setting up Contrastive Learning Project")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    if args.demo_only:
        return run_demo()
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    # Install dependencies
    if not args.skip_deps:
        if not install_dependencies():
            print("❌ Dependency installation failed!")
            sys.exit(1)
    
    # Setup pre-commit
    if args.pre_commit:
        if not setup_pre_commit():
            print("❌ Pre-commit setup failed!")
            sys.exit(1)
    
    # Download sample data
    if not args.skip_data:
        if not download_sample_data():
            print("❌ Sample data download failed!")
            sys.exit(1)
    
    # Run tests
    if not args.skip_tests:
        if not run_tests():
            print("❌ Tests failed!")
            sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📚 Next steps:")
    print("1. Run training: python train.py model=simclr data=cifar10")
    print("2. Run evaluation: python scripts/evaluate.py --model_path checkpoints/best_model.pth --model_type simclr")
    print("3. Run visualization: python scripts/visualize.py --model_path checkpoints/best_model.pth --model_type simclr")
    print("4. Run demo: streamlit run demo.py")
    print("\n📖 For more information, see README.md")


if __name__ == "__main__":
    main()
