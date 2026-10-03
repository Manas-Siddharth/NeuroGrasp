import os
import sys
import math
import numpy as np
import torch
import cv2

# Ensure root workspace directory is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import grasp_to_corners, calculate_grasp_iou
from src.dataset import create_dataloaders
from src.model import NeuroGraspNet
from src.inference import GraspPredictor


def independent_polygon_area(corners: np.ndarray) -> float:
    """Compute area of a 2D convex polygon using Shoelace formula."""
    x = corners[:, 0]
    y = corners[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def clip_polygon_with_halfplane(polygon: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman polygon clipping against a single directed line (half-plane)."""
    def is_inside(p, p1, p2):
        return (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0]) >= 0

    def intersection(seg_p1, seg_p2, line_p1, line_p2):
        dc = np.array([line_p1[0] - line_p2[0], line_p1[1] - line_p2[1]])
        dp = np.array([seg_p1[0] - seg_p2[0], seg_p1[1] - seg_p2[1]])
        n1 = line_p1[0] * line_p2[1] - line_p1[1] * line_p2[0]
        n2 = seg_p1[0] * seg_p2[1] - seg_p1[1] * seg_p2[0]
        denom = dc[0] * dp[1] - dc[1] * dp[0]
        if abs(denom) < 1e-9:
            return seg_p1
        x = (n1 * dp[0] - n2 * dc[0]) / denom
        y = (n1 * dp[1] - n2 * dc[1]) / denom
        return np.array([x, y])

    output_list = list(polygon)
    if len(output_list) == 0:
        return np.empty((0, 2))

    input_list = output_list
    output_list = []
    s = input_list[-1]

    for e in input_list:
        if is_inside(e, p1, p2):
            if not is_inside(s, p1, p2):
                output_list.append(intersection(s, e, p1, p2))
            output_list.append(e)
        elif is_inside(s, p1, p2):
            output_list.append(intersection(s, e, p1, p2))
        s = e

    return np.array(output_list) if len(output_list) > 0 else np.empty((0, 2))


def sutherland_hodgman_polygon_iou(grasp1: tuple, grasp2: tuple) -> float:
    """
    Independent rotated rectangle IoU calculation using Sutherland-Hodgman polygon clipping algorithm.
    """
    c1 = grasp_to_corners(*grasp1)
    c2 = grasp_to_corners(*grasp2)

    clipped = c1.copy()
    # Clip c1 against all 4 edges of c2
    for i in range(4):
        p1 = c2[i]
        p2 = c2[(i + 1) % 4]
        clipped = clip_polygon_with_halfplane(clipped, p1, p2)
        if len(clipped) == 0:
            break

    if len(clipped) < 3:
        inter_area = 0.0
    else:
        inter_area = independent_polygon_area(clipped)

    area1 = grasp1[2] * grasp1[3]
    area2 = grasp2[2] * grasp2[3]
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0

    return float(inter_area / union_area)


def main():
    print("=" * 60)
    print(" NEUROGRASP INDEPENDENT GEOMETRY & IOU VERIFICATION AUDIT ")
    print("=" * 60)

    # 1. Independent IoU Comparison across 5 validation samples
    predictor = GraspPredictor("checkpoints/best_model.pth")
    _, val_loader = create_dataloaders("data/sample_dataset")
    val_dataset = val_loader.dataset

    print("\n--- INDEPENDENT IOU COMPARISON (Sutherland-Hodgman vs OpenCV) ---")
    iou_diffs = []
    
    for idx in range(min(5, len(val_dataset))):
        img_tensor, target_tensor = val_dataset[idx]
        
        # Extract ground truth grasp in 224x224 frame
        gt_x = target_tensor[0].item() * 224.0
        gt_y = target_tensor[1].item() * 224.0
        gt_w = max(target_tensor[2].item() * 224.0, 1.0)
        gt_h = max(target_tensor[3].item() * 224.0, 1.0)
        gt_theta = math.atan2(target_tensor[4].item(), target_tensor[5].item())
        gt_grasp = (gt_x, gt_y, gt_w, gt_h, gt_theta)

        # Model prediction
        img_input = img_tensor.numpy()
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        img_np = np.clip((img_input * std + mean).transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)

        pred_res = predictor.predict(img_np)
        pred_x = pred_res["x"]
        pred_y = pred_res["y"]
        pred_w = pred_res["width"]
        pred_h = pred_res["height"]
        pred_theta = pred_res["angle_rad"]
        pred_grasp = (pred_x, pred_y, pred_w, pred_h, pred_theta)

        iou_opencv = calculate_grasp_iou(pred_grasp, gt_grasp)
        iou_sutherland = sutherland_hodgman_polygon_iou(pred_grasp, gt_grasp)
        diff = abs(iou_opencv - iou_sutherland)
        iou_diffs.append(diff)

        print(
            f"Sample #{idx+1:02d} | OpenCV IoU: {iou_opencv:.5f} | "
            f"Independent Sutherland-Hodgman IoU: {iou_sutherland:.5f} | Diff: {diff:.6f}"
        )

    max_diff = max(iou_diffs)
    print(f"\n[+] Maximum difference between implementations: {max_diff:.6f}")
    if max_diff < 1e-4:
        print("[+] Rotated Rectangle IoU Implementation mathematically VERIFIED EXACT!")
    else:
        print("[!] Discrepancy observed between IoU implementations.")

    # 2. Programmatic Data Leakage Check
    print("\n--- PROGRAMMATIC DATA LEAKAGE CHECK ---")
    train_loader, val_loader = create_dataloaders("data/sample_dataset")
    train_set = set(train_loader.dataset.indices)
    val_set = set(val_loader.dataset.indices)
    overlap = train_set.intersection(val_set)
    print(f"Train Dataset Indices Count: {len(train_set)}")
    print(f"Val Dataset Indices Count:   {len(val_set)}")
    print(f"Overlap Index Count:         {len(overlap)}")
    assert len(overlap) == 0, "DATA LEAKAGE DETECTED!"
    print("[+] Data Leakage Check PASSED: Train and Validation sets are 100% disjoint.")

    # 3. Success Rate Formula Verification
    print("\n--- GRASP SUCCESS RATE FORMULA AUDIT ---")
    iou_thresh = 0.25
    angle_thresh_deg = 30.0
    
    success_count = 0
    total_val = len(val_dataset)

    for idx in range(total_val):
        img_tensor, target_tensor = val_dataset[idx]
        gt_x = target_tensor[0].item() * 224.0
        gt_y = target_tensor[1].item() * 224.0
        gt_w = max(target_tensor[2].item() * 224.0, 1.0)
        gt_h = max(target_tensor[3].item() * 224.0, 1.0)
        gt_theta = math.atan2(target_tensor[4].item(), target_tensor[5].item())
        gt_grasp = (gt_x, gt_y, gt_w, gt_h, gt_theta)

        img_input = img_tensor.numpy()
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        img_np = np.clip((img_input * std + mean).transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)

        pred_res = predictor.predict(img_np)
        pred_grasp = (pred_res["x"], pred_res["y"], pred_res["width"], pred_res["height"], pred_res["angle_rad"])

        iou = calculate_grasp_iou(pred_grasp, gt_grasp)
        angle_err_deg = math.degrees(abs(math.atan2(math.sin(pred_res["angle_rad"] - gt_theta), math.cos(pred_res["angle_rad"] - gt_theta))))

        is_success = (iou >= iou_thresh) and (angle_err_deg <= angle_thresh_deg)
        if is_success:
            success_count += 1

    success_rate = (success_count / total_val) * 100.0
    print(f"Successful Samples: {success_count} / {total_val}")
    print(f"Calculated Success Rate: {success_rate:.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()
