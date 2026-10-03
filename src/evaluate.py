import os
import json
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

from src.utils import set_seed, get_device, load_config, calculate_grasp_iou, grasp_to_corners, draw_grasp
from src.dataset import create_dataloaders
from src.model import NeuroGraspNet
from src.losses import NeuroGraspLoss


def evaluate_model(model, val_loader, criterion, device, iou_thresh=0.25, angle_thresh_deg=30.0, img_size=224):
    """
    Evaluates the model on validation set and computes metrics.
    """
    model.eval()
    total_loss = 0.0
    
    pos_errors = []
    width_errors = []
    height_errors = []
    angle_errors_deg = []
    ious = []
    successes = []
    inference_times = []

    visual_samples = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            start_time = time.time()
            outputs = model(images)
            infer_time = (time.time() - start_time) / images.size(0)
            inference_times.append(infer_time)

            loss, _ = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)

            # Unpack predictions and targets for each sample in batch
            outputs_cpu = outputs.cpu().numpy()
            targets_cpu = targets.cpu().numpy()

            for i in range(images.size(0)):
                pred_vec = outputs_cpu[i]
                gt_vec = targets_cpu[i]

                # Denormalize predictions and ground truth to pixel values
                px_pred = pred_vec[0] * img_size
                py_pred = pred_vec[1] * img_size
                pw_pred = max(pred_vec[2] * img_size, 1.0)
                ph_pred = max(pred_vec[3] * img_size, 1.0)
                p_theta = math.atan2(pred_vec[4], pred_vec[5])
                p_conf = 1.0 / (1.0 + math.exp(-pred_vec[6]))  # Sigmoid

                px_gt = gt_vec[0] * img_size
                py_gt = gt_vec[1] * img_size
                pw_gt = max(gt_vec[2] * img_size, 1.0)
                ph_gt = max(gt_vec[3] * img_size, 1.0)
                gt_theta = math.atan2(gt_vec[4], gt_vec[5])

                # Position Error (pixels)
                pos_err = math.sqrt((px_pred - px_gt) ** 2 + (py_pred - py_gt) ** 2)
                pos_errors.append(pos_err)

                # Width & Height Error (pixels)
                width_errors.append(abs(pw_pred - pw_gt))
                height_errors.append(abs(ph_pred - ph_gt))

                # Angle Error (degrees)
                angle_diff_rad = abs(p_theta - gt_theta)
                angle_diff_rad = math.atan2(math.sin(angle_diff_rad), math.cos(angle_diff_rad))
                angle_diff_deg = math.degrees(abs(angle_diff_rad))
                angle_errors_deg.append(angle_diff_deg)

                # IoU calculation
                grasp_pred = (px_pred, py_pred, pw_pred, ph_pred, p_theta)
                grasp_gt = (px_gt, py_gt, pw_gt, ph_gt, gt_theta)
                iou = calculate_grasp_iou(grasp_pred, grasp_gt)
                ious.append(iou)

                # Grasp Success Criterion (IoU >= threshold AND angle error <= threshold)
                is_success = (iou >= iou_thresh) and (angle_diff_deg <= angle_thresh_deg)
                successes.append(float(is_success))

                # Collect visual samples for prediction visualization (up to 10)
                if len(visual_samples) < 10:
                    img_tensor = images[i].cpu()
                    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                    img_np = img_tensor.numpy() * std + mean
                    img_np = np.clip(img_np.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)

                    visual_samples.append({
                        "sample_idx": len(visual_samples) + 1,
                        "image": img_np,
                        "pred": (px_pred, py_pred, pw_pred, ph_pred, p_theta),
                        "gt": (px_gt, py_gt, pw_gt, ph_gt, gt_theta),
                        "conf": p_conf,
                        "iou": iou,
                        "angle_err_deg": angle_diff_deg,
                        "pos_err_px": pos_err
                    })

    num_samples = len(val_loader.dataset)
    avg_val_loss = total_loss / num_samples

    metrics = {
        "val_loss": float(avg_val_loss),
        "mean_position_error_px": float(np.mean(pos_errors)),
        "mean_width_error_px": float(np.mean(width_errors)),
        "mean_height_error_px": float(np.mean(height_errors)),
        "mean_angle_error_deg": float(np.mean(angle_errors_deg)),
        "mean_iou": float(np.mean(ious)),
        "grasp_success_rate": float(np.mean(successes)),
        "avg_inference_latency_ms": float(np.mean(inference_times) * 1000.0),
        "total_validation_samples": num_samples,
        "success_criteria": {
            "iou_threshold": iou_thresh,
            "angle_threshold_deg": angle_thresh_deg
        }
    }

    return metrics, visual_samples


def save_individual_prediction_samples(visual_samples, output_dir="results/predictions"):
    """
    Save individual side-by-side ground truth vs predicted visual comparison images under results/predictions/.
    """
    os.makedirs(output_dir, exist_ok=True)
    for sample in visual_samples:
        idx = sample["sample_idx"]
        img_np = sample["image"]
        pred_grasp = sample["pred"]
        gt_grasp = sample["gt"]
        conf = sample["conf"]
        iou = sample["iou"]

        # Overlay Ground Truth (Blue) & Prediction (Green)
        gt_vis = draw_grasp(img_np.copy(), gt_grasp, confidence=1.0, color=(0, 120, 255))
        pred_vis = draw_grasp(img_np.copy(), pred_grasp, confidence=conf, color=(0, 255, 0))

        # Side-by-side combined image
        combined = np.hstack((gt_vis, pred_vis))
        
        # Add Header label
        header_h = 30
        header = np.zeros((header_h, combined.shape[1], 3), dtype=np.uint8)
        text = f"Sample #{idx} | GT (Left, Blue) vs Pred (Right, Green) | IoU: {iou:.2f} | Conf: {conf:.2f}"
        cv2.putText(header, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        final_img = np.vstack((header, combined))
        
        save_path = os.path.join(output_dir, f"sample_{idx:02d}.png")
        cv2.imwrite(save_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
    
    print(f"[+] Saved {len(visual_samples)} prediction visualization samples to: {output_dir}/")


def plot_validation_results(visual_samples, save_path="results/validation_results.png"):
    """
    Generate composite comparison figure showing Ground Truth vs Predicted Grasp annotations.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    num_display = min(len(visual_samples), 4)
    if num_display == 0:
        return

    fig, axes = plt.subplots(num_display, 2, figsize=(10, 3.5 * num_display))
    if num_display == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_display):
        sample = visual_samples[i]
        img_np = sample["image"]
        pred_grasp = sample["pred"]
        gt_grasp = sample["gt"]
        conf = sample["conf"]
        iou = sample["iou"]

        gt_vis = draw_grasp(img_np.copy(), gt_grasp, confidence=1.0, color=(0, 120, 255))
        pred_vis = draw_grasp(img_np.copy(), pred_grasp, confidence=conf, color=(0, 255, 0))

        axes[i, 0].imshow(cv2.cvtColor(gt_vis, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"Sample #{i+1} Ground Truth Grasp", fontsize=10, fontweight="bold")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(cv2.cvtColor(pred_vis, cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title(f"Sample #{i+1} Predicted Grasp (IoU: {iou:.2f})", fontsize=10, fontweight="bold")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Validation visualization saved to: {save_path}")


def main():
    config = load_config("config.yaml")
    set_seed(config["training"]["seed"])
    device = get_device()

    checkpoint_path = os.path.join(config["training"]["checkpoint_dir"], config["training"]["best_model_name"])
    if not os.path.exists(checkpoint_path):
        print(f"[-] Checkpoint not found at {checkpoint_path}. Train the model first!")
        return

    print(f"[*] Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = NeuroGraspNet(
        backbone_name=config["model"]["backbone"],
        pretrained=False,
        dropout_rate=config["model"]["dropout"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = NeuroGraspLoss(
        lambda_xy=config["loss"]["lambda_xy"],
        lambda_wh=config["loss"]["lambda_wh"],
        lambda_angle=config["loss"]["lambda_angle"],
        lambda_conf=config["loss"]["lambda_conf"]
    ).to(device)

    _, val_loader = create_dataloaders(
        data_dir=config["data"]["data_dir"],
        target_size=config["data"]["image_size"],
        batch_size=config["training"]["batch_size"],
        val_split=config["data"]["val_split"]
    )

    print(f"[*] Evaluating NeuroGrasp model performance across {len(val_loader.dataset)} validation samples...")
    metrics, visual_samples = evaluate_model(
        model, val_loader, criterion, device,
        iou_thresh=config["evaluation"]["iou_threshold"],
        angle_thresh_deg=config["evaluation"]["angle_threshold_deg"],
        img_size=config["data"]["image_size"]
    )

    # Save metrics JSON
    metrics_path = os.path.join(config["evaluation"]["results_dir"], "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"[+] Metrics saved to {metrics_path}")

    # Display Metrics Summary
    print("\n" + "=" * 50)
    print("           NEUROGRASP EVALUATION METRICS          ")
    print("=" * 50)
    print(f" Validation Samples:       {metrics['total_validation_samples']}")
    print(f" Validation Loss:          {metrics['val_loss']:.4f}")
    print(f" Mean Position Error:      {metrics['mean_position_error_px']:.2f} px")
    print(f" Mean Width Error:         {metrics['mean_width_error_px']:.2f} px")
    print(f" Mean Height Error:        {metrics['mean_height_error_px']:.2f} px")
    print(f" Mean Angle Error:         {metrics['mean_angle_error_deg']:.2f} deg")
    print(f" Mean Grasp IoU:           {metrics['mean_iou']:.4f}")
    print(f" Grasp Success Rate:       {metrics['grasp_success_rate'] * 100:.1f} %")
    print(f" Avg Latency per Image:    {metrics['avg_inference_latency_ms']:.2f} ms")
    print("=" * 50 + "\n")

    # Save visual comparison figure & individual samples under results/predictions/
    val_vis_path = os.path.join(config["evaluation"]["results_dir"], "validation_results.png")
    plot_validation_results(visual_samples, save_path=val_vis_path)
    save_individual_prediction_samples(visual_samples, output_dir="results/predictions")


if __name__ == "__main__":
    main()
