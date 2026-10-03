import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils import set_seed, get_device, load_config
from src.dataset import create_dataloaders
from src.model import NeuroGraspNet
from src.losses import NeuroGraspLoss


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Run one epoch of model training."""
    model.train()
    total_loss = 0.0
    loss_components = {"loss_xy": 0.0, "loss_wh": 0.0, "loss_angle": 0.0, "loss_conf": 0.0}

    for images, targets in tqdm(dataloader, desc="Training Batch", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss, batch_components = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        for k in loss_components:
            loss_components[k] += batch_components[k] * images.size(0)

    num_samples = len(dataloader.dataset)
    avg_loss = total_loss / num_samples
    avg_components = {k: v / num_samples for k, v in loss_components.items()}
    return avg_loss, avg_components


def validate_epoch(model, dataloader, criterion, device):
    """Run one epoch of model validation."""
    model.eval()
    total_loss = 0.0
    loss_components = {"loss_xy": 0.0, "loss_wh": 0.0, "loss_angle": 0.0, "loss_conf": 0.0}

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validation Batch", leave=False):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss, batch_components = criterion(outputs, targets)

            total_loss += loss.item() * images.size(0)
            for k in loss_components:
                loss_components[k] += batch_components[k] * images.size(0)

    num_samples = len(dataloader.dataset)
    avg_loss = total_loss / num_samples
    avg_components = {k: v / num_samples for k, v in loss_components.items()}
    return avg_loss, avg_components


def plot_training_history(train_losses, val_losses, save_path="results/training_loss.png"):
    """Generate and save training vs validation loss curve."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, "b-o", label="Training Loss", linewidth=2)
    plt.plot(epochs, val_losses, "r-s", label="Validation Loss", linewidth=2)
    
    plt.title("NeuroGrasp Model Training & Validation Loss", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Multi-Task Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Training loss plot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train NeuroGrasp model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-epoch quick verification smoke test")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    device = get_device()
    print(f"[*] Executing on device: {device}")

    # Prepare directories
    checkpoint_dir = config["training"]["checkpoint_dir"]
    results_dir = config["evaluation"]["results_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Data loaders
    data_dir = config["data"]["data_dir"]
    batch_size = config["training"]["batch_size"]
    target_size = config["data"]["image_size"]

    print(f"[*] Loading dataset from: {data_dir}")
    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        target_size=target_size,
        batch_size=batch_size,
        val_split=config["data"]["val_split"]
    )
    print(f"[+] Dataset loaded: {len(train_loader.dataset)} training samples, {len(val_loader.dataset)} validation samples.")

    # Model & Loss Function
    model = NeuroGraspNet(
        backbone_name=config["model"]["backbone"],
        pretrained=config["model"]["pretrained"],
        dropout_rate=config["model"]["dropout"]
    ).to(device)

    criterion = NeuroGraspLoss(
        lambda_xy=config["loss"]["lambda_xy"],
        lambda_wh=config["loss"]["lambda_wh"],
        lambda_angle=config["loss"]["lambda_angle"],
        lambda_conf=config["loss"]["lambda_conf"]
    ).to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    epochs = 1 if args.smoke_test else (args.epochs or config["training"]["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Smoke Test Mode
    if args.smoke_test:
        print("[*] Running Smoke Test (Forward pass, backward pass, optimizer step)...")
        train_loss, _ = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _ = validate_epoch(model, val_loader, criterion, device)
        print(f"[+] Smoke test PASSED successfully! Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        return

    # Full Training Loop
    best_val_loss = float("inf")
    train_history, val_history = [], []
    best_model_path = os.path.join(checkpoint_dir, config["training"]["best_model_name"])

    print(f"[*] Starting NeuroGrasp training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        train_loss, train_comp = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_comp = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        train_history.append(train_loss)
        val_history.append(val_loss)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} (xy: {train_comp['loss_xy']:.4f}, wh: {train_comp['loss_wh']:.4f}, angle: {train_comp['loss_angle']:.4f}) | "
            f"Val Loss: {val_loss:.4f}"
        )

        # Save Best Model Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": config
            }, best_model_path)
            print(f"    --> Saved new best checkpoint to {best_model_path} (Val Loss: {val_loss:.4f})")

    # Plot & Save Loss Curve
    plot_training_path = os.path.join(results_dir, "training_loss.png")
    plot_training_history(train_history, val_history, save_path=plot_training_path)
    print(f"[+] Training complete. Best Validation Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
