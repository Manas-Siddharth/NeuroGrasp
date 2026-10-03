import os
import tempfile
from pathlib import Path
import torch
import pytest
from src.dataset import JacquardGraspDataset, parse_jacquard_annotation


def test_parse_jacquard_annotation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        anno_file = Path(tmp_dir) / "test_grasps.txt"
        anno_file.write_text("512.5;480.0;45.0;85.2;20.0\n300.0;200.0;-30.0;90.0;25.0\n")

        grasps = parse_jacquard_annotation(str(anno_file))
        assert len(grasps) == 2
        x, y, w, h, theta_rad = grasps[0]
        assert x == 512.5
        assert y == 480.0
        assert w == 85.2
        assert h == 20.0
        assert pytest.approx(theta_rad, 1e-3) == 0.785398  # 45 deg in rad


def test_dataset_loading():
    data_dir = "data/sample_dataset"
    if not os.path.exists(data_dir):
        pytest.skip("Sample dataset not present")

    dataset = JacquardGraspDataset(data_dir=data_dir, target_size=224)
    assert len(dataset) > 0

    img_tensor, target_tensor = dataset[0]
    assert isinstance(img_tensor, torch.Tensor)
    assert isinstance(target_tensor, torch.Tensor)

    # Expected image tensor shape: [3, 224, 224]
    assert img_tensor.shape == (3, 224, 224)
    # Expected target tensor shape: [7] -> [x, y, w, h, sin_t, cos_t, conf]
    assert target_tensor.shape == (7,)
