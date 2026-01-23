"""
Unit tests for contrastive learning models and utilities.

This module contains comprehensive tests for all components of the
contrastive learning framework.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock

from src.models.contrastive import SimCLR, MoCo, BYOL, ProjectionHead
from src.data.augmentation import SimCLRAugmentation, MoCoAugmentation, ContrastiveTransform
from src.data.datasets import CIFAR10Contrastive, CustomImageDataset
from src.train.trainer import ContrastiveTrainer, create_optimizer, create_scheduler
from src.eval.evaluator import ContrastiveEvaluator
from src.utils import get_device, set_seed, count_parameters, format_time, format_bytes


class TestProjectionHead:
    """Test cases for ProjectionHead."""
    
    def test_projection_head_creation(self):
        """Test projection head creation."""
        projection_head = ProjectionHead(
            input_dim=512,
            hidden_dim=1024,
            output_dim=128,
            num_layers=3,
        )
        
        assert isinstance(projection_head, nn.Module)
        assert len(projection_head.projection) == 7  # 3 layers * 2 + 1 final layer
    
    def test_projection_head_forward(self):
        """Test projection head forward pass."""
        projection_head = ProjectionHead(input_dim=512, output_dim=128)
        
        x = torch.randn(32, 512)
        output = projection_head(x)
        
        assert output.shape == (32, 128)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestSimCLR:
    """Test cases for SimCLR model."""
    
    def test_simclr_creation(self):
        """Test SimCLR model creation."""
        model = SimCLR(
            backbone="resnet18",
            pretrained=False,
            projection_dim=128,
            hidden_dim=1024,
            temperature=0.07,
        )
        
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'backbone')
        assert hasattr(model, 'projection_head')
        assert model.temperature == 0.07
    
    def test_simclr_forward(self):
        """Test SimCLR forward pass."""
        model = SimCLR(backbone="resnet18", pretrained=False)
        
        x = torch.randn(4, 3, 224, 224)
        features, projections = model(x)
        
        assert features.shape[0] == 4
        assert projections.shape == (4, 128)
        assert not torch.isnan(features).any()
        assert not torch.isnan(projections).any()
    
    def test_simclr_contrastive_loss(self):
        """Test SimCLR contrastive loss computation."""
        model = SimCLR(backbone="resnet18", pretrained=False)
        
        projections1 = torch.randn(4, 128)
        projections2 = torch.randn(4, 128)
        
        loss = model.contrastive_loss(projections1, projections2)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_simclr_encode(self):
        """Test SimCLR encoding method."""
        model = SimCLR(backbone="resnet18", pretrained=False)
        
        x = torch.randn(4, 3, 224, 224)
        features = model.encode(x)
        
        assert features.shape[0] == 4
        assert features.shape[1] == 512  # ResNet-18 feature dimension


class TestMoCo:
    """Test cases for MoCo model."""
    
    def test_moco_creation(self):
        """Test MoCo model creation."""
        model = MoCo(
            backbone="resnet18",
            pretrained=False,
            projection_dim=128,
            temperature=0.07,
            momentum=0.999,
            queue_size=1024,
        )
        
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'backbone_q')
        assert hasattr(model, 'backbone_k')
        assert hasattr(model, 'queue')
        assert model.queue_size == 1024
    
    def test_moco_forward(self):
        """Test MoCo forward pass."""
        model = MoCo(backbone="resnet18", pretrained=False, queue_size=1024)
        
        im_q = torch.randn(4, 3, 224, 224)
        im_k = torch.randn(4, 3, 224, 224)
        
        logits, labels = model(im_q, im_k)
        
        assert logits.shape[0] == 4
        assert logits.shape[1] == 1024 + 1  # queue_size + 1 positive
        assert labels.shape[0] == 4
        assert torch.all(labels == 0)  # All labels should be 0 (positive samples)
    
    def test_moco_contrastive_loss(self):
        """Test MoCo contrastive loss computation."""
        model = MoCo(backbone="resnet18", pretrained=False, queue_size=1024)
        
        logits = torch.randn(4, 1025)
        labels = torch.zeros(4, dtype=torch.long)
        
        loss = model.contrastive_loss(logits, labels)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert not torch.isnan(loss)


class TestBYOL:
    """Test cases for BYOL model."""
    
    def test_byol_creation(self):
        """Test BYOL model creation."""
        model = BYOL(
            backbone="resnet18",
            pretrained=False,
            projection_dim=256,
            hidden_dim=2048,
            prediction_dim=256,
            momentum=0.996,
        )
        
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'backbone_online')
        assert hasattr(model, 'backbone_target')
        assert hasattr(model, 'prediction')
        assert model.momentum == 0.996
    
    def test_byol_forward(self):
        """Test BYOL forward pass."""
        model = BYOL(backbone="resnet18", pretrained=False)
        
        x1 = torch.randn(4, 3, 224, 224)
        x2 = torch.randn(4, 3, 224, 224)
        
        p1, z2_target, p2, z1_target = model(x1, x2)
        
        assert p1.shape == (4, 256)
        assert z2_target.shape == (4, 256)
        assert p2.shape == (4, 256)
        assert z1_target.shape == (4, 256)
    
    def test_byol_loss_fn(self):
        """Test BYOL loss function."""
        model = BYOL(backbone="resnet18", pretrained=False)
        
        p1 = torch.randn(4, 256)
        z2_target = torch.randn(4, 256)
        p2 = torch.randn(4, 256)
        z1_target = torch.randn(4, 256)
        
        loss = model.loss_fn(p1, z2_target, p2, z1_target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert not torch.isnan(loss)


class TestAugmentation:
    """Test cases for data augmentation."""
    
    def test_simclr_augmentation(self):
        """Test SimCLR augmentation."""
        aug = SimCLRAugmentation(input_size=224)
        
        image = torch.randn(3, 224, 224)
        augmented = aug(image)
        
        assert augmented.shape == (3, 224, 224)
        assert not torch.isnan(augmented).any()
    
    def test_moco_augmentation(self):
        """Test MoCo augmentation."""
        aug = MoCoAugmentation(input_size=224)
        
        image = torch.randn(3, 224, 224)
        augmented = aug(image)
        
        assert augmented.shape == (3, 224, 224)
        assert not torch.isnan(augmented).any()
    
    def test_contrastive_transform(self):
        """Test contrastive transform."""
        transform = ContrastiveTransform(augmentation_type="simclr", input_size=224)
        
        image = torch.randn(3, 224, 224)
        view1, view2 = transform(image)
        
        assert view1.shape == (3, 224, 224)
        assert view2.shape == (3, 224, 224)
        assert not torch.isnan(view1).any()
        assert not torch.isnan(view2).any()


class TestDatasets:
    """Test cases for datasets."""
    
    @patch('src.data.datasets.datasets.CIFAR10')
    def test_cifar10_contrastive(self, mock_cifar10):
        """Test CIFAR-10 contrastive dataset."""
        # Mock CIFAR-10 dataset
        mock_dataset = MagicMock()
        mock_dataset.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer']
        mock_dataset.class_to_idx = {cls: i for i, cls in enumerate(mock_dataset.classes)}
        mock_dataset.__getitem__ = MagicMock(return_value=(torch.randn(3, 32, 32), 0))
        mock_cifar10.return_value = mock_dataset
        
        dataset = CIFAR10Contrastive(root="./data", train=True, download=True)
        
        assert len(dataset) == len(mock_dataset)
        assert dataset.classes == mock_dataset.classes
        
        # Test getting an item
        view1, view2, label = dataset[0]
        assert view1.shape == (3, 32, 32)
        assert view2.shape == (3, 32, 32)
        assert label == 0


class TestTrainer:
    """Test cases for trainer."""
    
    def test_create_optimizer(self):
        """Test optimizer creation."""
        model = nn.Linear(10, 1)
        
        optimizer = create_optimizer(
            model=model,
            optimizer_name="adam",
            learning_rate=0.001,
            weight_decay=1e-4,
        )
        
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]['lr'] == 0.001
        assert optimizer.param_groups[0]['weight_decay'] == 1e-4
    
    def test_create_scheduler(self):
        """Test scheduler creation."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        scheduler = create_scheduler(
            optimizer=optimizer,
            scheduler_name="cosine",
            max_epochs=100,
        )
        
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        assert scheduler.T_max == 100


class TestEvaluator:
    """Test cases for evaluator."""
    
    def test_contrastive_evaluator_creation(self):
        """Test ContrastiveEvaluator creation."""
        model = SimCLR(backbone="resnet18", pretrained=False)
        
        evaluator = ContrastiveEvaluator(model, device="cpu")
        
        assert isinstance(evaluator, ContrastiveEvaluator)
        assert evaluator.device == torch.device("cpu")


class TestUtils:
    """Test cases for utility functions."""
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device("cpu")
        assert device == torch.device("cpu")
        
        device = get_device("auto")
        assert isinstance(device, torch.device)
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Test that random numbers are reproducible
        torch.manual_seed(42)
        rand1 = torch.randn(10)
        
        torch.manual_seed(42)
        rand2 = torch.randn(10)
        
        assert torch.allclose(rand1, rand2)
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = nn.Linear(10, 5)
        
        counts = count_parameters(model)
        
        assert counts['total_parameters'] == 55  # 10*5 + 5
        assert counts['trainable_parameters'] == 55
        assert counts['non_trainable_parameters'] == 0
    
    def test_format_time(self):
        """Test time formatting."""
        assert format_time(30) == "30.00s"
        assert format_time(90) == "1.50m"
        assert format_time(7200) == "2.00h"
    
    def test_format_bytes(self):
        """Test bytes formatting."""
        assert format_bytes(1024) == "1.00KB"
        assert format_bytes(1024**2) == "1.00MB"
        assert format_bytes(1024**3) == "1.00GB"


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_training_step(self):
        """Test end-to-end training step."""
        # Create model
        model = SimCLR(backbone="resnet18", pretrained=False)
        
        # Create dummy data
        batch_size = 4
        view1 = torch.randn(batch_size, 3, 224, 224)
        view2 = torch.randn(batch_size, 3, 224, 224)
        labels = torch.randint(0, 10, (batch_size,))
        
        # Forward pass
        _, projections1 = model(view1)
        _, projections2 = model(view2)
        
        # Compute loss
        loss = model.contrastive_loss(projections1, projections2)
        
        # Backward pass
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_model_serialization(self):
        """Test model serialization and loading."""
        # Create and save model
        model1 = SimCLR(backbone="resnet18", pretrained=False)
        torch.save(model1.state_dict(), "test_model.pth")
        
        # Load model
        model2 = SimCLR(backbone="resnet18", pretrained=False)
        model2.load_state_dict(torch.load("test_model.pth"))
        
        # Test that models are equivalent
        x = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            features1, projections1 = model1(x)
            features2, projections2 = model2(x)
        
        assert torch.allclose(features1, features2)
        assert torch.allclose(projections1, projections2)
        
        # Cleanup
        import os
        os.remove("test_model.pth")


if __name__ == "__main__":
    pytest.main([__file__])
