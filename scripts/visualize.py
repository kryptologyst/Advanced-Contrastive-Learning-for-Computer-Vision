#!/usr/bin/env python3
"""
Script to visualize learned representations.

This script provides visualization tools for analyzing
contrastive learning model representations.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pathlib import Path
import json

from src.models.contrastive import SimCLR, MoCo, BYOL
from src.data.datasets import create_data_loaders
from src.eval.evaluator import ContrastiveEvaluator
from src.utils import get_device, setup_logging


def load_model(model_path: str, model_type: str) -> torch.nn.Module:
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if model_type == "simclr":
        model = SimCLR(backbone="resnet50", pretrained=False)
    elif model_type == "moco":
        model = MoCo(backbone="resnet50", pretrained=False)
    elif model_type == "byol":
        model = BYOL(backbone="resnet50", pretrained=False)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model


def create_visualization(features: np.ndarray, labels: np.ndarray, method: str, save_path: str):
    """Create visualization of features."""
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    elif method == "pca":
        reducer = PCA(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    embeddings = reducer.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=labels,
        cmap='tab10',
        alpha=0.7,
        s=50,
    )
    plt.colorbar(scatter)
    plt.xlabel(f"{method.upper()} Component 1")
    plt.ylabel(f"{method.upper()} Component 2")
    plt.title(f"Learned Representations Visualization ({method.upper()})")
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description="Visualize contrastive learning representations")
    
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
        help="Type of model to visualize"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100"],
        help="Dataset to visualize"
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
        help="Batch size for feature extraction"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./visualizations",
        help="Directory to save visualizations"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["tsne", "pca"],
        choices=["tsne", "pca"],
        help="Visualization methods to use"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Maximum number of samples to visualize"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    # Get device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
        augmentation_type="none",  # No augmentation for visualization
        input_size=224,
    )
    
    # Create evaluator
    evaluator = ContrastiveEvaluator(
        model=model,
        device=device,
        save_dir=str(output_dir),
    )
    
    # Extract features
    logger.info("Extracting features...")
    features, labels = evaluator.extract_features(val_loader)
    
    # Limit samples if requested
    if len(features) > args.max_samples:
        indices = np.random.choice(len(features), args.max_samples, replace=False)
        features = features[indices]
        labels = labels[indices]
    
    logger.info(f"Extracted features for {len(features)} samples")
    
    # Create visualizations
    for method in args.methods:
        logger.info(f"Creating {method.upper()} visualization...")
        
        save_path = output_dir / f"embeddings_{method}.png"
        create_visualization(features, labels, method, str(save_path))
        
        logger.info(f"Visualization saved to {save_path}")
    
    # Create similarity matrix
    logger.info("Creating similarity matrix...")
    
    # Normalize features
    normalized_features = features / np.linalg.norm(features, axis=1, keepdims=True)
    
    # Compute similarity matrix
    similarity_matrix = np.dot(normalized_features, normalized_features.T)
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        similarity_matrix,
        cmap='viridis',
        square=True,
        cbar_kws={'label': 'Cosine Similarity'},
    )
    plt.title("Feature Similarity Matrix")
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    plt.tight_layout()
    
    similarity_path = output_dir / "similarity_matrix.png"
    plt.savefig(similarity_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Similarity matrix saved to {similarity_path}")
    
    # Create feature distribution plot
    logger.info("Creating feature distribution plot...")
    
    plt.figure(figsize=(12, 8))
    
    # Plot feature statistics
    plt.subplot(2, 2, 1)
    plt.hist(features.flatten(), bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel("Feature Value")
    plt.ylabel("Frequency")
    plt.title("Feature Value Distribution")
    
    # Plot feature norms
    plt.subplot(2, 2, 2)
    feature_norms = np.linalg.norm(features, axis=1)
    plt.hist(feature_norms, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel("Feature Norm")
    plt.ylabel("Frequency")
    plt.title("Feature Norm Distribution")
    
    # Plot per-class feature means
    plt.subplot(2, 2, 3)
    unique_labels = np.unique(labels)
    class_means = []
    for label in unique_labels:
        class_features = features[labels == label]
        class_mean = np.mean(class_features, axis=0)
        class_means.append(np.linalg.norm(class_mean))
    
    plt.bar(range(len(unique_labels)), class_means)
    plt.xlabel("Class")
    plt.ylabel("Mean Feature Norm")
    plt.title("Per-Class Feature Norms")
    plt.xticks(range(len(unique_labels)), unique_labels)
    
    # Plot feature variance
    plt.subplot(2, 2, 4)
    feature_vars = np.var(features, axis=0)
    plt.plot(feature_vars)
    plt.xlabel("Feature Dimension")
    plt.ylabel("Variance")
    plt.title("Feature Variance Across Dimensions")
    
    plt.tight_layout()
    
    distribution_path = output_dir / "feature_distributions.png"
    plt.savefig(distribution_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Feature distributions saved to {distribution_path}")
    
    # Save feature statistics
    stats = {
        "num_samples": len(features),
        "feature_dim": features.shape[1],
        "num_classes": len(unique_labels),
        "mean_feature_norm": float(np.mean(feature_norms)),
        "std_feature_norm": float(np.std(feature_norms)),
        "mean_similarity": float(np.mean(similarity_matrix)),
        "std_similarity": float(np.std(similarity_matrix)),
    }
    
    stats_path = output_dir / "feature_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Feature statistics saved to {stats_path}")
    logger.info("Visualization completed!")


if __name__ == "__main__":
    main()
