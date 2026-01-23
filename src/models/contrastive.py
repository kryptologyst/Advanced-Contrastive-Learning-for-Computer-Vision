"""
Modern SimCLR implementation for contrastive learning.

This module implements the SimCLR (Simple Contrastive Learning of Representations)
framework with proper augmentations, temperature scaling, and modern PyTorch practices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Optional, Dict, Any
import math


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 128,
        num_layers: int = 3,
    ) -> None:
        """
        Initialize projection head.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output projection dimension
            num_layers: Number of layers in projection head
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ])
            prev_dim = hidden_dim
        
        # Final layer without activation
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.projection = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through projection head."""
        return self.projection(x)


class SimCLR(nn.Module):
    """
    SimCLR model implementation.
    
    SimCLR learns representations by maximizing agreement between differently
    augmented views of the same data example via a contrastive loss in the
    latent space.
    """
    
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        projection_dim: int = 128,
        hidden_dim: int = 2048,
        temperature: float = 0.07,
        freeze_backbone: bool = False,
    ) -> None:
        """
        Initialize SimCLR model.
        
        Args:
            backbone: Backbone architecture ('resnet18', 'resnet50', 'resnet101')
            pretrained: Whether to use pretrained backbone
            projection_dim: Dimension of projection head output
            hidden_dim: Hidden dimension in projection head
            temperature: Temperature parameter for contrastive loss
            freeze_backbone: Whether to freeze backbone parameters
        """
        super().__init__()
        
        self.temperature = temperature
        
        # Load backbone
        if backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif backbone == "resnet101":
            self.backbone = models.resnet101(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove final classification layer
        self.backbone.fc = nn.Identity()
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Projection head
        self.projection_head = ProjectionHead(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim,
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tuple of (features, projections) where:
            - features: Backbone features of shape (batch_size, feature_dim)
            - projections: Projected features of shape (batch_size, projection_dim)
        """
        # Extract features
        features = self.backbone(x)
        
        # Project to contrastive space
        projections = self.projection_head(features)
        
        return features, projections
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to feature space (for inference).
        
        Args:
            x: Input tensor
            
        Returns:
            Feature representations
        """
        return self.backbone(x)
    
    def contrastive_loss(
        self,
        projections1: torch.Tensor,
        projections2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute NT-Xent contrastive loss.
        
        Args:
            projections1: Projections from first augmented view
            projections2: Projections from second augmented view
            
        Returns:
            Contrastive loss
        """
        batch_size = projections1.size(0)
        
        # Normalize projections
        projections1 = F.normalize(projections1, dim=1)
        projections2 = F.normalize(projections2, dim=1)
        
        # Concatenate projections
        projections = torch.cat([projections1, projections2], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(projections, projections.T) / self.temperature
        
        # Create labels for positive pairs
        labels = torch.arange(batch_size).to(projections.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        # Mask to remove self-similarity
        mask = torch.eye(2 * batch_size, device=projections.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        # Compute loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss
    
    def get_similarity_matrix(
        self,
        projections1: torch.Tensor,
        projections2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute similarity matrix between two sets of projections.
        
        Args:
            projections1: First set of projections
            projections2: Second set of projections
            
        Returns:
            Similarity matrix
        """
        projections1 = F.normalize(projections1, dim=1)
        projections2 = F.normalize(projections2, dim=1)
        
        return torch.matmul(projections1, projections2.T) / self.temperature


class MoCo(nn.Module):
    """
    Momentum Contrast (MoCo) implementation.
    
    MoCo maintains a queue of negative samples and uses momentum updates
    for the key encoder to maintain consistency.
    """
    
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        projection_dim: int = 128,
        hidden_dim: int = 2048,
        temperature: float = 0.07,
        momentum: float = 0.999,
        queue_size: int = 65536,
    ) -> None:
        """
        Initialize MoCo model.
        
        Args:
            backbone: Backbone architecture
            pretrained: Whether to use pretrained backbone
            projection_dim: Dimension of projection head output
            hidden_dim: Hidden dimension in projection head
            temperature: Temperature parameter for contrastive loss
            momentum: Momentum coefficient for key encoder updates
            queue_size: Size of the negative sample queue
        """
        super().__init__()
        
        self.temperature = temperature
        self.momentum = momentum
        self.queue_size = queue_size
        
        # Load backbone
        if backbone == "resnet18":
            self.backbone_q = models.resnet18(pretrained=pretrained)
            self.backbone_k = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone == "resnet50":
            self.backbone_q = models.resnet50(pretrained=pretrained)
            self.backbone_k = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove final classification layer
        self.backbone_q.fc = nn.Identity()
        self.backbone_k.fc = nn.Identity()
        
        # Projection heads
        self.projection_head_q = ProjectionHead(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim,
        )
        self.projection_head_k = ProjectionHead(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim,
        )
        
        # Initialize key encoder with query encoder weights
        for param_q, param_k in zip(
            self.backbone_q.parameters(), self.backbone_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        for param_q, param_k in zip(
            self.projection_head_q.parameters(), self.projection_head_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        # Initialize queue
        self.register_buffer("queue", torch.randn(projection_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(
            self.backbone_q.parameters(), self.backbone_k.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
        
        for param_q, param_k in zip(
            self.projection_head_q.parameters(), self.projection_head_k.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        """Update the queue with new keys."""
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity
        
        # Replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        
        self.queue_ptr[0] = ptr
    
    def forward(
        self,
        im_q: torch.Tensor,
        im_k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            im_q: Query images
            im_k: Key images
            
        Returns:
            Tuple of (logits, labels) for contrastive loss
        """
        batch_size = im_q.size(0)
        
        # Compute query features
        q = self.backbone_q(im_q)
        q = self.projection_head_q(q)
        q = F.normalize(q, dim=1)
        
        # Compute key features
        with torch.no_grad():
            self._momentum_update_key_encoder()
            
            k = self.backbone_k(im_k)
            k = self.projection_head_k(k)
            k = F.normalize(k, dim=1)
        
        # Compute positive logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        # Compute negative logits
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        # Combine positive and negative logits
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        
        # Labels: positive samples are at index 0
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)
        
        # Dequeue and enqueue
        self._dequeue_and_enqueue(k)
        
        return logits, labels
    
    def contrastive_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contrastive loss."""
        return F.cross_entropy(logits, labels)


class BYOL(nn.Module):
    """
    Bootstrap Your Own Latent (BYOL) implementation.
    
    BYOL learns representations by predicting one augmented view of an image
    from another augmented view using an online and target network.
    """
    
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        projection_dim: int = 256,
        hidden_dim: int = 4096,
        prediction_dim: int = 256,
        momentum: float = 0.996,
    ) -> None:
        """
        Initialize BYOL model.
        
        Args:
            backbone: Backbone architecture
            pretrained: Whether to use pretrained backbone
            projection_dim: Dimension of projection head output
            hidden_dim: Hidden dimension in projection head
            prediction_dim: Dimension of prediction head output
            momentum: Momentum coefficient for target network updates
        """
        super().__init__()
        
        self.momentum = momentum
        
        # Load backbone
        if backbone == "resnet18":
            self.backbone_online = models.resnet18(pretrained=pretrained)
            self.backbone_target = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone == "resnet50":
            self.backbone_online = models.resnet50(pretrained=pretrained)
            self.backbone_target = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove final classification layer
        self.backbone_online.fc = nn.Identity()
        self.backbone_target.fc = nn.Identity()
        
        # Projection heads
        self.projection_online = ProjectionHead(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim,
        )
        self.projection_target = ProjectionHead(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim,
        )
        
        # Prediction head (only for online network)
        self.prediction = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, prediction_dim),
        )
        
        # Initialize target network with online network weights
        for param_online, param_target in zip(
            self.backbone_online.parameters(), self.backbone_target.parameters()
        ):
            param_target.data.copy_(param_online.data)
            param_target.requires_grad = False
        
        for param_online, param_target in zip(
            self.projection_online.parameters(), self.projection_target.parameters()
        ):
            param_target.data.copy_(param_online.data)
            param_target.requires_grad = False
    
    @torch.no_grad()
    def _momentum_update_target_network(self) -> None:
        """Momentum update of the target network."""
        for param_online, param_target in zip(
            self.backbone_online.parameters(), self.backbone_target.parameters()
        ):
            param_target.data = param_target.data * self.momentum + param_online.data * (1.0 - self.momentum)
        
        for param_online, param_target in zip(
            self.projection_online.parameters(), self.projection_target.parameters()
        ):
            param_target.data = param_target.data * self.momentum + param_online.data * (1.0 - self.momentum)
    
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x1: First augmented view
            x2: Second augmented view
            
        Returns:
            Tuple of (prediction1, target2, prediction2, target1)
        """
        # Online network forward pass
        h1_online = self.backbone_online(x1)
        z1_online = self.projection_online(h1_online)
        p1 = self.prediction(z1_online)
        
        h2_online = self.backbone_online(x2)
        z2_online = self.projection_online(h2_online)
        p2 = self.prediction(z2_online)
        
        # Target network forward pass
        with torch.no_grad():
            self._momentum_update_target_network()
            
            h1_target = self.backbone_target(x1)
            z1_target = self.projection_target(h1_target)
            
            h2_target = self.backbone_target(x2)
            z2_target = self.projection_target(h2_target)
        
        return p1, z2_target, p2, z1_target
    
    def loss_fn(
        self,
        p1: torch.Tensor,
        z2_target: torch.Tensor,
        p2: torch.Tensor,
        z1_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute BYOL loss.
        
        Args:
            p1: Prediction from first view
            z2_target: Target from second view
            p2: Prediction from second view
            z1_target: Target from first view
            
        Returns:
            BYOL loss
        """
        # Normalize targets
        z1_target = F.normalize(z1_target, dim=1)
        z2_target = F.normalize(z2_target, dim=1)
        
        # Compute losses
        loss1 = F.mse_loss(p1, z2_target)
        loss2 = F.mse_loss(p2, z1_target)
        
        return loss1 + loss2
