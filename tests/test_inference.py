import os
import tempfile
from pathlib import Path
import pytest
import numpy as np
from PIL import Image
import torch

from src.model import NeuroGraspNet
from src.inference import GraspPredictor


def test_inference_pipeline():
    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_dir = Path(tmp_dir) / "checkpoints"
        checkpoint_dir.mkdir()
        checkpoint_path = checkpoint_dir / "test_model.pth"

        model = NeuroGraspNet(backbone_name="resnet18", pretrained=False)
        torch.save({
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "val_loss": 0.5,
            "config": {}
        }, checkpoint_path)

        # Create dummy RGB image
        dummy_img = Image.fromarray(np.uint8(np.random.randint(0, 255, (224, 224, 3))))

        predictor = GraspPredictor(checkpoint_path=str(checkpoint_path))
        res = predictor.predict(dummy_img)

        assert "x" in res
        assert "y" in res
        assert "width" in res
        assert "height" in res
        assert "angle_deg" in res
        assert "confidence" in res
        assert "annotated_image" in res

        assert isinstance(res["annotated_image"], np.ndarray)
        assert res["annotated_image"].shape == (224, 224, 3)
