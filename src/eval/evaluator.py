"""
Evaluation utilities for contrastive learning.

This module provides evaluation metrics, visualization tools, and
analysis utilities for contrastive learning models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple, List
import logging
from pathlib import Path
import json


class ContrastiveEvaluator:
    """
    Evaluator class for contrastive learning models.
    
    Provides comprehensive evaluation including linear probing, k-NN evaluation,
    and visualization of learned representations.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        save_dir: str = "./evaluation_results",
    ) -> None:
        """
        Initialize contrastive evaluator.
        
        Args:
            model: Trained contrastive learning model
            device: Device to use for evaluation
            save_dir: Directory to save evaluation results
        """
        self.model = model
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
    
    def extract_features(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from the model.
        
        Args:
            data_loader: Data loader for feature extraction
            
        Returns:
            Tuple of (features, labels)
        """
        features = []
        labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                view1, view2, batch_labels = batch
                view1 = view1.to(self.device)
                
                # Extract features (use first view)
                if hasattr(self.model, 'encode'):
                    batch_features = self.model.encode(view1)
                else:
                    batch_features, _ = self.model(view1)
                
                features.append(batch_features.cpu().numpy())
                labels.append(batch_labels.numpy())
        
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        
        return features, labels
    
    def linear_probe(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_classes: int,
        learning_rate: float = 1e-3,
        max_epochs: int = 100,
    ) -> Dict[str, float]:
        """
        Perform linear probing evaluation.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_classes: Number of classes
            learning_rate: Learning rate for linear probe
            max_epochs: Maximum number of epochs
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.logger.info("Starting linear probe evaluation...")
        
        # Extract features
        train_features, train_labels = self.extract_features(train_loader)
        val_features, val_labels = self.extract_features(val_loader)
        
        # Create linear probe model
        feature_dim = train_features.shape[1]
        linear_probe = nn.Linear(feature_dim, num_classes).to(self.device)
        optimizer = torch.optim.Adam(linear_probe.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Convert to tensors
        train_features = torch.FloatTensor(train_features).to(self.device)
        train_labels = torch.LongTensor(train_labels).to(self.device)
        val_features = torch.FloatTensor(val_features).to(self.device)
        val_labels = torch.LongTensor(val_labels).to(self.device)
        
        # Training loop
        best_acc = 0.0
        for epoch in range(max_epochs):
            linear_probe.train()
            
            # Forward pass
            logits = linear_probe(train_features)
            loss = criterion(logits, train_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Validation
            if epoch % 10 == 0:
                linear_probe.eval()
                with torch.no_grad():
                    val_logits = linear_probe(val_features)
                    val_preds = torch.argmax(val_logits, dim=1)
                    val_acc = accuracy_score(val_labels.cpu(), val_preds.cpu())
                    
                    if val_acc > best_acc:
                        best_acc = val_acc
                    
                    self.logger.info(f"Epoch {epoch}: Val Acc: {val_acc:.4f}")
        
        # Final evaluation
        linear_probe.eval()
        with torch.no_grad():
            val_logits = linear_probe(val_features)
            val_preds = torch.argmax(val_logits, dim=1)
            
            accuracy = accuracy_score(val_labels.cpu(), val_preds.cpu())
            precision, recall, f1, _ = precision_recall_fscore_support(
                val_labels.cpu(), val_preds.cpu(), average='weighted'
            )
        
        results = {
            "linear_probe_accuracy": accuracy,
            "linear_probe_precision": precision,
            "linear_probe_recall": recall,
            "linear_probe_f1": f1,
        }
        
        self.logger.info(f"Linear probe results: {results}")
        
        return results
    
    def knn_evaluation(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        k: int = 5,
    ) -> Dict[str, float]:
        """
        Perform k-NN evaluation.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            k: Number of nearest neighbors
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.logger.info(f"Starting k-NN evaluation with k={k}...")
        
        # Extract features
        train_features, train_labels = self.extract_features(train_loader)
        val_features, val_labels = self.extract_features(val_loader)
        
        # Normalize features
        train_features = F.normalize(torch.FloatTensor(train_features), dim=1)
        val_features = F.normalize(torch.FloatTensor(val_features), dim=1)
        
        # Compute similarities
        similarities = torch.matmul(val_features, train_features.T)
        
        # Get k nearest neighbors
        _, top_k_indices = torch.topk(similarities, k, dim=1)
        
        # Get labels of k nearest neighbors
        top_k_labels = train_labels[top_k_indices.cpu().numpy()]
        
        # Predict labels using majority voting
        predictions = []
        for neighbors in top_k_labels:
            unique, counts = np.unique(neighbors, return_counts=True)
            prediction = unique[np.argmax(counts)]
            predictions.append(prediction)
        
        predictions = np.array(predictions)
        
        # Compute metrics
        accuracy = accuracy_score(val_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_labels, predictions, average='weighted'
        )
        
        results = {
            "knn_accuracy": accuracy,
            "knn_precision": precision,
            "knn_recall": recall,
            "knn_f1": f1,
        }
        
        self.logger.info(f"k-NN results: {results}")
        
        return results
    
    def visualize_embeddings(
        self,
        data_loader: DataLoader,
        method: str = "tsne",
        n_components: int = 2,
        perplexity: float = 30.0,
        n_iter: int = 1000,
        save_plot: bool = True,
    ) -> np.ndarray:
        """
        Visualize learned embeddings.
        
        Args:
            data_loader: Data loader for visualization
            method: Dimensionality reduction method ('tsne', 'pca')
            n_components: Number of components for reduction
            perplexity: Perplexity for t-SNE
            n_iter: Number of iterations for t-SNE
            save_plot: Whether to save the plot
            
        Returns:
            Reduced embeddings
        """
        self.logger.info(f"Visualizing embeddings using {method}...")
        
        # Extract features
        features, labels = self.extract_features(data_loader)
        
        # Reduce dimensionality
        if method.lower() == "tsne":
            reducer = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                n_iter=n_iter,
                random_state=42,
            )
        elif method.lower() == "pca":
            reducer = PCA(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        embeddings = reducer.fit_transform(features)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        
        if n_components == 2:
            scatter = plt.scatter(
                embeddings[:, 0],
                embeddings[:, 1],
                c=labels,
                cmap='tab10',
                alpha=0.7,
            )
            plt.colorbar(scatter)
            plt.xlabel(f"{method.upper()} Component 1")
            plt.ylabel(f"{method.upper()} Component 2")
        elif n_components == 3:
            ax = plt.axes(projection='3d')
            scatter = ax.scatter3D(
                embeddings[:, 0],
                embeddings[:, 1],
                embeddings[:, 2],
                c=labels,
                cmap='tab10',
                alpha=0.7,
            )
            plt.colorbar(scatter)
            ax.set_xlabel(f"{method.upper()} Component 1")
            ax.set_ylabel(f"{method.upper()} Component 2")
            ax.set_zlabel(f"{method.upper()} Component 3")
        
        plt.title(f"Learned Embeddings Visualization ({method.upper()})")
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.save_dir / f"embeddings_{method}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Embedding visualization saved to {plot_path}")
        
        plt.show()
        
        return embeddings
    
    def compute_similarity_matrix(
        self,
        data_loader: DataLoader,
        save_matrix: bool = True,
    ) -> np.ndarray:
        """
        Compute similarity matrix between all pairs of samples.
        
        Args:
            data_loader: Data loader for similarity computation
            save_matrix: Whether to save the similarity matrix
            
        Returns:
            Similarity matrix
        """
        self.logger.info("Computing similarity matrix...")
        
        # Extract features
        features, labels = self.extract_features(data_loader)
        
        # Normalize features
        features = F.normalize(torch.FloatTensor(features), dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T).numpy()
        
        # Create visualization
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
        
        if save_matrix:
            matrix_path = self.save_dir / "similarity_matrix.png"
            plt.savefig(matrix_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Similarity matrix saved to {matrix_path}")
        
        plt.show()
        
        return similarity_matrix
    
    def analyze_representations(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_classes: int,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of learned representations.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_classes: Number of classes
            save_results: Whether to save results
            
        Returns:
            Dictionary with all evaluation results
        """
        self.logger.info("Starting comprehensive representation analysis...")
        
        results = {}
        
        # Linear probe evaluation
        linear_results = self.linear_probe(train_loader, val_loader, num_classes)
        results.update(linear_results)
        
        # k-NN evaluation
        knn_results = self.knn_evaluation(train_loader, val_loader)
        results.update(knn_results)
        
        # Visualize embeddings
        tsne_embeddings = self.visualize_embeddings(val_loader, method="tsne")
        pca_embeddings = self.visualize_embeddings(val_loader, method="pca")
        
        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(val_loader)
        
        # Add visualization results
        results["tsne_embeddings"] = tsne_embeddings.tolist()
        results["pca_embeddings"] = pca_embeddings.tolist()
        results["similarity_matrix"] = similarity_matrix.tolist()
        
        # Save results
        if save_results:
            results_path = self.save_dir / "evaluation_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Evaluation results saved to {results_path}")
        
        return results


def evaluate_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    device: str = "cuda",
    save_dir: str = "./evaluation_results",
) -> Dict[str, Any]:
    """
    Evaluate a contrastive learning model.
    
    Args:
        model: Trained contrastive learning model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_classes: Number of classes
        device: Device to use for evaluation
        save_dir: Directory to save evaluation results
        
    Returns:
        Dictionary with evaluation results
    """
    evaluator = ContrastiveEvaluator(model, device, save_dir)
    results = evaluator.analyze_representations(train_loader, val_loader, num_classes)
    
    return results
