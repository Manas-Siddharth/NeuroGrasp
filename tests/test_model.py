import torch
import pytest
from src.model import NeuroGraspNet
from src.losses import NeuroGraspLoss


def test_model_forward_pass():
    model = NeuroGraspNet(backbone_name="resnet18", pretrained=False)
    model.eval()

    # Dummy batch of 4 RGB images (4, 3, 224, 224)
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)

    # Output shape should be (4, 7)
    assert output.shape == (4, 7)
    assert not torch.isnan(output).any()


def test_loss_computation():
    criterion = NeuroGraspLoss()
    
    # Dummy predictions and targets
    pred = torch.randn(4, 7)
    target = torch.randn(4, 7)

    total_loss, loss_dict = criterion(pred, target)

    assert isinstance(total_loss, torch.Tensor)
    assert total_loss.item() >= 0
    assert "loss_xy" in loss_dict
    assert "loss_wh" in loss_dict
    assert "loss_angle" in loss_dict
    assert "loss_conf" in loss_dict
