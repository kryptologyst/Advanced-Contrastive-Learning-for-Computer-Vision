"""
Main training script for contrastive learning experiments.

This script provides a complete training pipeline with Hydra configuration
management, logging, and checkpointing.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import logging
import os
import random
import numpy as np
from pathlib import Path
import wandb
from typing import Dict, Any

from src.models.contrastive import SimCLR, MoCo, BYOL
from src.data.datasets import create_data_loaders
from src.train.trainer import ContrastiveTrainer, create_optimizer, create_scheduler
from src.eval.evaluator import ContrastiveEvaluator


def setup_logging(log_level: str = "INFO", log_dir: str = "./logs") -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level
        log_dir: Directory for log files
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "training.log"),
            logging.StreamHandler(),
        ],
    )


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_config: str) -> torch.device:
    """
    Get the appropriate device for training.
    
    Args:
        device_config: Device configuration ('auto', 'cuda', 'mps', 'cpu')
        
    Returns:
        PyTorch device
    """
    if device_config == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_config)
    
    return device


def create_model(model_config: DictConfig) -> nn.Module:
    """
    Create model from configuration.
    
    Args:
        model_config: Model configuration
        
    Returns:
        Model instance
    """
    model_class = hydra.utils.get_class(model_config._target_)
    model = hydra.utils.instantiate(model_config)
    
    return model


def setup_wandb(cfg: DictConfig) -> None:
    """
    Setup Weights & Biases logging.
    
    Args:
        cfg: Configuration object
    """
    if cfg.logging.wandb.enabled:
        wandb.init(
            project=cfg.logging.wandb.project,
            entity=cfg.logging.wandb.entity,
            name=cfg.experiment.name,
            tags=cfg.experiment.tags,
            notes=cfg.experiment.notes,
            config=OmegaConf.to_container(cfg, resolve=True),
        )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training function.
    
    Args:
        cfg: Hydra configuration object
    """
    # Setup logging
    setup_logging(cfg.logging.level, cfg.logging.log_dir)
    logger = logging.getLogger(__name__)
    
    # Set seed for reproducibility
    set_seed(cfg.seed)
    
    # Get device
    device = get_device(cfg.device)
    logger.info(f"Using device: {device}")
    
    # Setup Weights & Biases
    setup_wandb(cfg)
    
    # Create directories
    for path_key in ["data_dir", "checkpoint_dir", "output_dir", "assets_dir"]:
        Path(cfg.paths[path_key]).mkdir(parents=True, exist_ok=True)
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_name=cfg.data.dataset_name,
        root=cfg.paths.data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        augmentation_type=cfg.data.augmentation_type,
        input_size=cfg.data.input_size,
        train_split=cfg.data.train_split,
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
    )
    
    logger.info(f"Train loader: {len(train_loader)} batches")
    if val_loader:
        logger.info(f"Val loader: {len(val_loader)} batches")
    if test_loader:
        logger.info(f"Test loader: {len(test_loader)} batches")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(cfg.model)
    logger.info(f"Model created: {model.__class__.__name__}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer
    logger.info("Creating optimizer...")
    optimizer = create_optimizer(
        model=model,
        optimizer_name=cfg.training.optimizer_name,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        betas=cfg.training.betas,
        eps=cfg.training.eps,
    )
    
    # Create scheduler
    logger.info("Creating scheduler...")
    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_name=cfg.training.scheduler.scheduler_name,
        max_epochs=cfg.training.max_epochs,
    )
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = ContrastiveTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=cfg.paths.checkpoint_dir,
        log_interval=cfg.training.log_interval,
        save_interval=cfg.training.save_interval,
        max_epochs=cfg.training.max_epochs,
        gradient_clip_val=cfg.training.gradient_clip_val,
        mixed_precision=cfg.training.mixed_precision,
        accumulation_steps=cfg.training.accumulation_steps,
    )
    
    # Start training
    logger.info("Starting training...")
    training_results = trainer.train()
    
    # Save training results
    results_path = Path(cfg.paths.output_dir) / "training_results.json"
    import json
    with open(results_path, 'w') as f:
        json.dump(training_results, f, indent=2)
    logger.info(f"Training results saved to {results_path}")
    
    # Evaluation
    if val_loader:
        logger.info("Starting evaluation...")
        
        # Get dataset info for evaluation
        dataset_info = {
            "cifar10": 10,
            "cifar100": 100,
            "imagenet": 1000,
        }
        num_classes = dataset_info.get(cfg.data.dataset_name, 10)
        
        # Create evaluator
        evaluator = ContrastiveEvaluator(
            model=model,
            device=device,
            save_dir=cfg.paths.output_dir,
        )
        
        # Run evaluation
        eval_results = evaluator.analyze_representations(
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            save_results=True,
        )
        
        logger.info("Evaluation completed!")
        logger.info(f"Linear probe accuracy: {eval_results['linear_probe_accuracy']:.4f}")
        logger.info(f"k-NN accuracy: {eval_results['knn_accuracy']:.4f}")
        
        # Log to wandb
        if cfg.logging.wandb.enabled:
            wandb.log(eval_results)
    
    # Finish wandb run
    if cfg.logging.wandb.enabled:
        wandb.finish()
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
