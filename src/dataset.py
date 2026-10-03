import os
import glob
import math
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def parse_jacquard_annotation(grasp_file_path: str):
    """
    Parse Jacquard annotation file (*_grasps.txt).
    
    Format per line:
    x;y;theta;opening;jaws_size
    
    - x, y: center coordinates (pixels)
    - theta: rotation angle in degrees
    - opening: gripper width w (pixels)
    - jaws_size: gripper jaw height h (pixels)
    
    Returns:
        List of tuples: [(x, y, w, h, theta_rad), ...]
    """
    grasps = []
    if not os.path.exists(grasp_file_path):
        return grasps

    with open(grasp_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Handle both ';' and space/comma delimiters for robust parsing
            if ";" in line:
                parts = line.split(";")
            elif "," in line:
                parts = line.split(",")
            else:
                parts = line.split()
                
            if len(parts) >= 5:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    theta_deg = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    
                    theta_rad = math.radians(theta_deg)
                    grasps.append((x, y, w, h, theta_rad))
                except ValueError:
                    continue
    return grasps


class JacquardGraspDataset(Dataset):
    """
    PyTorch Dataset for Jacquard Grasp Detection.
    
    Loads RGB images and parses corresponding grasp annotations.
    Resizes image to target resolution and rescales grasp coordinates accordingly.
    """
    def __init__(self, data_dir: str, target_size: int = 224, is_train: bool = True, transform=None):
        self.data_dir = data_dir
        self.target_size = target_size
        self.is_train = is_train
        self.transform = transform

        # Find all RGB images in data_dir (search recursively or flat)
        pattern = os.path.join(data_dir, "**", "*_RGB.png")
        self.image_paths = sorted(glob.glob(pattern, recursive=True))
        
        # Fallback to search any .png or .jpg if no _RGB.png files found
        if not self.image_paths:
            self.image_paths = sorted(
                glob.glob(os.path.join(data_dir, "*.png")) +
                glob.glob(os.path.join(data_dir, "*.jpg"))
            )
            # Filter out mask/depth files
            self.image_paths = [
                p for p in self.image_paths 
                if not ("mask" in p or "depth" in p)
            ]

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No image files found in dataset directory: {data_dir}")

        # ImageNet normalization transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Determine corresponding grasp file path
        if img_path.endswith("_RGB.png"):
            grasp_path = img_path.replace("_RGB.png", "_grasps.txt")
        else:
            base_name, _ = os.path.splitext(img_path)
            grasp_path = f"{base_name}_grasps.txt"

        # Load RGB Image
        image_pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image_pil.size
        
        # Resize image to target resolution
        resized_pil = image_pil.resize((self.target_size, self.target_size), Image.BILINEAR)
        img_tensor = self.transform(resized_pil)

        # Scale factors for spatial coordinates
        scale_x = self.target_size / orig_w
        scale_y = self.target_size / orig_h

        # Parse grasps
        grasps = parse_jacquard_annotation(grasp_path)

        if len(grasps) > 0:
            # Deterministic selection: pick middle or first grasp to ensure reproducibility
            target_grasp = grasps[len(grasps) // 2]
            raw_x, raw_y, raw_w, raw_h, raw_theta = target_grasp
            
            # Scale spatial parameters to resized image frame
            x_scaled = raw_x * scale_x
            y_scaled = raw_y * scale_y
            w_scaled = raw_w * scale_x
            h_scaled = raw_h * scale_y
            theta_rad = raw_theta
            
            # Normalize target values to [0, 1] range for neural network regression
            x_norm = x_scaled / self.target_size
            y_norm = y_scaled / self.target_size
            w_norm = w_scaled / self.target_size
            h_norm = h_scaled / self.target_size
            sin_theta = math.sin(theta_rad)
            cos_theta = math.cos(theta_rad)
            confidence = 1.0  # Valid annotated grasp
        else:
            # Fallback default target if no annotation found
            x_norm, y_norm, w_norm, h_norm = 0.5, 0.5, 0.2, 0.1
            sin_theta, cos_theta = 0.0, 1.0
            confidence = 0.0

        # Target tensor format: [x_norm, y_norm, w_norm, h_norm, sin_theta, cos_theta, confidence]
        target_tensor = torch.tensor(
            [x_norm, y_norm, w_norm, h_norm, sin_theta, cos_theta, confidence],
            dtype=torch.float32
        )

        return img_tensor, target_tensor


def create_dataloaders(data_dir: str, target_size: int = 224, batch_size: int = 16, val_split: float = 0.2, num_workers: int = 0):
    """
    Create train and validation DataLoaders with reproducible split.
    """
    full_dataset = JacquardGraspDataset(data_dir=data_dir, target_size=target_size)
    total_len = len(full_dataset)
    
    val_len = int(total_len * val_split)
    train_len = total_len - val_len

    # If dataset is very small (e.g. sample dataset < 5 items), keep at least 1 in val if possible
    if total_len > 1 and val_len == 0:
        val_len = 1
        train_len = total_len - 1

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_len, val_len], generator=generator
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader
