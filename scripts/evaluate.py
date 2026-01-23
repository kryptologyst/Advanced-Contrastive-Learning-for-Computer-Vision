#!/usr/bin/env python3
"""
Script to run evaluation on trained models.

This script provides a command-line interface for evaluating
contrastive learning models with various metrics.
"""

import argparse
import torch
import logging
from pathlib import Path
import json

from src.models.contrastive import SimCLR, MoCo, BYOL
from src.data.datasets import create_data_loaders
from src.eval.evaluator import ContrastiveEvaluator
from src.utils import get_device, setup_logging


def load_model(model_path: str, model_type: str) -> torch.nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        model_type: Type of model ('simclr', 'moco', 'byol')
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create model based on type
    if model_type == "simclr":
        model = SimCLR(backbone="resnet50", pretrained=False)
    elif model_type == "moco":
        model = MoCo(backbone="resnet50", pretrained=False)
    elif model_type == "byol":
        model = BYOL(backbone="resnet50", pretrained=False)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate contrastive learning models")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["simclr", "moco", "byol"],
        required=True,
        help="Type of model to evaluate"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100"],
        help="Dataset to evaluate on"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory containing dataset"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for evaluation"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Get device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, args.model_type)
    model.to(device)
    model.eval()
    
    # Create data loaders
    logger.info(f"Creating data loaders for {args.dataset}")
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_name=args.dataset,
        root=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        augmentation_type=args.model_type,
        input_size=224,
    )
    
    # Get number of classes
    num_classes = {"cifar10": 10, "cifar100": 100}[args.dataset]
    
    # Create evaluator
    evaluator = ContrastiveEvaluator(
        model=model,
        device=device,
        save_dir=args.output_dir,
    )
    
    # Run evaluation
    logger.info("Starting evaluation...")
    results = evaluator.analyze_representations(
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        save_results=True,
    )
    
    # Print results
    logger.info("Evaluation completed!")
    logger.info(f"Linear probe accuracy: {results['linear_probe_accuracy']:.4f}")
    logger.info(f"Linear probe precision: {results['linear_probe_precision']:.4f}")
    logger.info(f"Linear probe recall: {results['linear_probe_recall']:.4f}")
    logger.info(f"Linear probe F1: {results['linear_probe_f1']:.4f}")
    logger.info(f"k-NN accuracy: {results['knn_accuracy']:.4f}")
    logger.info(f"k-NN precision: {results['knn_precision']:.4f}")
    logger.info(f"k-NN recall: {results['knn_recall']:.4f}")
    logger.info(f"k-NN F1: {results['knn_f1']:.4f}")
    
    # Save summary
    summary = {
        "model_type": args.model_type,
        "dataset": args.dataset,
        "linear_probe_accuracy": results['linear_probe_accuracy'],
        "knn_accuracy": results['knn_accuracy'],
    }
    
    summary_path = Path(args.output_dir) / "evaluation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Evaluation summary saved to {summary_path}")


if __name__ == "__main__":
    main()
