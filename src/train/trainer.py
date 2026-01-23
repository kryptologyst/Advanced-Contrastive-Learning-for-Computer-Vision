"""
Training utilities for contrastive learning.

This module provides training loops, loss functions, and optimization
utilities for contrastive learning experiments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple, List
import time
import logging
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm


class ContrastiveTrainer:
    """
    Trainer class for contrastive learning models.
    
    Handles training loops, validation, checkpointing, and logging
    for contrastive learning experiments.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cuda",
        save_dir: str = "./checkpoints",
        log_interval: int = 100,
        save_interval: int = 1000,
        max_epochs: int = 100,
        gradient_clip_val: Optional[float] = None,
        mixed_precision: bool = False,
        accumulation_steps: int = 1,
    ) -> None:
        """
        Initialize contrastive trainer.
        
        Args:
            model: Contrastive learning model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler
            device: Device to use for training
            save_dir: Directory to save checkpoints
            log_interval: Interval for logging
            save_interval: Interval for saving checkpoints
            max_epochs: Maximum number of epochs
            gradient_clip_val: Gradient clipping value
            mixed_precision: Whether to use mixed precision training
            accumulation_steps: Number of accumulation steps
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = Path(save_dir)
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.max_epochs = max_epochs
        self.gradient_clip_val = gradient_clip_val
        self.mixed_precision = mixed_precision
        self.accumulation_steps = accumulation_steps
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup mixed precision scaler
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Setup loss function based on model type
        self._setup_loss_function()
    
    def _setup_loss_function(self) -> None:
        """Setup loss function based on model type."""
        model_name = self.model.__class__.__name__.lower()
        
        if "simclr" in model_name:
            self.loss_fn = self._simclr_loss
        elif "moco" in model_name:
            self.loss_fn = self._moco_loss
        elif "byol" in model_name:
            self.loss_fn = self._byol_loss
        else:
            # Default to SimCLR loss
            self.loss_fn = self._simclr_loss
    
    def _simclr_loss(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Compute SimCLR loss."""
        view1, view2, _ = batch
        
        # Forward pass
        _, projections1 = self.model(view1)
        _, projections2 = self.model(view2)
        
        # Compute contrastive loss
        if hasattr(self.model, 'contrastive_loss'):
            loss = self.model.contrastive_loss(projections1, projections2)
        else:
            # Fallback to manual computation
            loss = self._compute_nt_xent_loss(projections1, projections2)
        
        return loss
    
    def _moco_loss(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Compute MoCo loss."""
        view1, view2, _ = batch
        
        # Forward pass
        logits, labels = self.model(view1, view2)
        
        # Compute contrastive loss
        if hasattr(self.model, 'contrastive_loss'):
            loss = self.model.contrastive_loss(logits, labels)
        else:
            loss = nn.CrossEntropyLoss()(logits, labels)
        
        return loss
    
    def _byol_loss(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Compute BYOL loss."""
        view1, view2, _ = batch
        
        # Forward pass
        p1, z2_target, p2, z1_target = self.model(view1, view2)
        
        # Compute BYOL loss
        if hasattr(self.model, 'loss_fn'):
            loss = self.model.loss_fn(p1, z2_target, p2, z1_target)
        else:
            # Fallback to manual computation
            loss = self._compute_byol_loss(p1, z2_target, p2, z1_target)
        
        return loss
    
    def _compute_nt_xent_loss(
        self,
        projections1: torch.Tensor,
        projections2: torch.Tensor,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """Compute NT-Xent loss manually."""
        batch_size = projections1.size(0)
        
        # Normalize projections
        projections1 = nn.functional.normalize(projections1, dim=1)
        projections2 = nn.functional.normalize(projections2, dim=1)
        
        # Concatenate projections
        projections = torch.cat([projections1, projections2], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(projections, projections.T) / temperature
        
        # Create labels for positive pairs
        labels = torch.arange(batch_size).to(projections.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        # Mask to remove self-similarity
        mask = torch.eye(2 * batch_size, device=projections.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        # Compute loss
        loss = nn.CrossEntropyLoss()(similarity_matrix, labels)
        
        return loss
    
    def _compute_byol_loss(
        self,
        p1: torch.Tensor,
        z2_target: torch.Tensor,
        p2: torch.Tensor,
        z1_target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute BYOL loss manually."""
        # Normalize targets
        z1_target = nn.functional.normalize(z1_target, dim=1)
        z2_target = nn.functional.normalize(z2_target, dim=1)
        
        # Compute losses
        loss1 = nn.functional.mse_loss(p1, z2_target)
        loss2 = nn.functional.mse_loss(p2, z1_target)
        
        return loss1 + loss2
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = tuple(tensor.to(self.device) for tensor in batch)
            
            # Forward pass with mixed precision
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    loss = self.loss_fn(batch)
                    loss = loss / self.accumulation_steps
            else:
                loss = self.loss_fn(batch)
                loss = loss / self.accumulation_steps
            
            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.gradient_clip_val is not None:
                    if self.mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip_val
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip_val
                        )
                
                if self.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update learning rate
                if self.scheduler is not None:
                    self.scheduler.step()
                
                self.global_step += 1
            
            # Update statistics
            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item() * self.accumulation_steps:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}",
            })
            
            # Logging
            if self.global_step % self.log_interval == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch}, Step {self.global_step}, "
                    f"Loss: {loss.item() * self.accumulation_steps:.4f}"
                )
            
            # Save checkpoint
            if self.global_step % self.save_interval == 0:
                self.save_checkpoint(f"checkpoint_step_{self.global_step}.pth")
        
        return total_loss / num_batches
    
    def validate(self) -> float:
        """Validate the model."""
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                batch = tuple(tensor.to(self.device) for tensor in batch)
                
                # Forward pass
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        loss = self.loss_fn(batch)
                else:
                    loss = self.loss_fn(batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self) -> Dict[str, List[float]]:
        """Train the model."""
        self.logger.info("Starting training...")
        
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Log epoch results
            self.logger.info(
                f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
            
            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint("best_model.pth")
            
            # Save epoch checkpoint
            self.save_checkpoint(f"epoch_{epoch}.pth")
        
        self.logger.info("Training completed!")
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_loss": self.best_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }
        
        checkpoint_path = self.save_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        checkpoint_path = self.save_dir / filename
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["best_loss"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = "adam",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    **kwargs: Any,
) -> optim.Optimizer:
    """
    Create optimizer for the model.
    
    Args:
        model: Model to optimize
        optimizer_name: Name of optimizer ('adam', 'sgd', 'adamw')
        learning_rate: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer arguments
        
    Returns:
        Optimizer instance
    """
    if optimizer_name.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs,
        )
    elif optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs,
        )
    elif optimizer_name.lower() == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    return optimizer


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str = "cosine",
    max_epochs: int = 100,
    **kwargs: Any,
) -> Any:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_name: Name of scheduler ('cosine', 'step', 'exponential')
        max_epochs: Maximum number of epochs
        **kwargs: Additional scheduler arguments
        
    Returns:
        Scheduler instance
    """
    if scheduler_name.lower() == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
            **kwargs,
        )
    elif scheduler_name.lower() == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max_epochs // 3,
            gamma=0.1,
            **kwargs,
        )
    elif scheduler_name.lower() == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.95,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
    return scheduler
