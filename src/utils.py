import os
import random
import math
import yaml
import numpy as np
import torch
import cv2


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration dictionary from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def grasp_to_corners(x: float, y: float, w: float, h: float, theta_rad: float) -> np.ndarray:
    """
    Convert grasp parameters (x, y, width, height, theta_rad) to 4 corner coordinates (4x2).
    
    Args:
        x, y: Center position (pixels)
        w, h: Gripper opening width and jaw height (pixels)
        theta_rad: Rotation angle in radians
    
    Returns:
        np.ndarray of shape (4, 2) representing (x, y) of corners.
    """
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)
    
    dx = w / 2.0
    dy = h / 2.0
    
    # 4 local corner offsets before rotation
    local_corners = np.array([
        [-dx, -dy],
        [ dx, -dy],
        [ dx,  dy],
        [-dx,  dy]
    ])
    
    # Rotation matrix
    R = np.array([
        [cos_t, -sin_t],
        [sin_t,  cos_t]
    ])
    
    # Rotate and translate
    corners = np.dot(local_corners, R.T) + np.array([x, y])
    return corners


def calculate_grasp_iou(grasp1: tuple, grasp2: tuple) -> float:
    """
    Calculate Intersection over Union (IoU) between two rotated grasp rectangles.
    grasp format: (x, y, w, h, theta_rad)
    
    Returns:
        float IoU value in [0.0, 1.0].
    """
    corners1 = grasp_to_corners(*grasp1).astype(np.float32)
    corners2 = grasp_to_corners(*grasp2).astype(np.float32)
    
    # Use OpenCV rotated rectangle intersection
    rect1 = cv2.minAreaRect(corners1)
    rect2 = cv2.minAreaRect(corners2)
    
    int_type, intersection_pts = cv2.rotatedRectangleIntersection(rect1, rect2)
    if int_type == cv2.INTERSECT_NONE or intersection_pts is None:
        return 0.0
    
    # Order points to compute contour area
    intersection_pts = cv2.convexHull(intersection_pts, returnPoints=True)
    inter_area = cv2.contourArea(intersection_pts)
    
    area1 = grasp1[2] * grasp1[3]
    area2 = grasp2[2] * grasp2[3]
    
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    
    return float(inter_area / union_area)


def draw_grasp(
    image: np.ndarray,
    grasp: tuple,
    confidence: float = None,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
    draw_center: bool = True,
    draw_orientation: bool = True
) -> np.ndarray:
    """
    Draw rotated grasp rectangle, center point, orientation direction, and label on an image.
    
    Args:
        image: RGB image (H, W, 3) numpy array
        grasp: (x, y, w, h, theta_rad)
        confidence: Confidence score to overlay
        color: BGR/RGB tuple for grasp lines
        thickness: Line thickness
    
    Returns:
        Annotated image numpy array
    """
    img_out = image.copy()
    x, y, w, h, theta_rad = grasp
    
    corners = grasp_to_corners(x, y, w, h, theta_rad).astype(np.int32)
    
    # Draw rotated bounding box
    cv2.polylines(img_out, [corners], isClosed=True, color=color, thickness=thickness)
    
    # Draw center point
    cx, cy = int(round(x)), int(round(y))
    if draw_center:
        cv2.circle(img_out, (cx, cy), radius=4, color=(255, 0, 0), thickness=-1)
    
    # Draw orientation vector line from center along grasp angle direction
    if draw_orientation:
        vec_len = max(w / 2.0, 15.0)
        end_x = int(round(cx + vec_len * math.cos(theta_rad)))
        end_y = int(round(cy + vec_len * math.sin(theta_rad)))
        cv2.arrowedLine(img_out, (cx, cy), (end_x, end_y), color=(0, 0, 255), thickness=thickness, tipLength=0.3)
    
    # Overlay text label
    if confidence is not None:
        theta_deg = math.degrees(theta_rad)
        label = f"Conf: {confidence:.2f} | Ang: {theta_deg:.1f}deg"
        cv2.putText(
            img_out, label, (max(10, cx - 50), max(20, cy - 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
        )
        
    return img_out
