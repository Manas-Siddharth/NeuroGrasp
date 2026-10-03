import os
import argparse
import math
import numpy as np
from PIL import Image
import cv2
import torch
from torchvision import transforms

from src.utils import get_device, load_config, draw_grasp
from src.model import NeuroGraspNet


class GraspPredictor:
    """
    Inference Engine for NeuroGrasp model.
    Loads trained checkpoint and performs grasp prediction on single RGB image.
    """
    def __init__(self, checkpoint_path: str = "checkpoints/best_model.pth", config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.device = get_device()
        self.image_size = self.config["data"]["image_size"]

        # Load model architecture
        self.model = NeuroGraspNet(
            backbone_name=self.config["model"]["backbone"],
            pretrained=False,
            dropout_rate=0.0
        ).to(self.device)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image_input):
        """
        Predict grasp parameters for an RGB image input.
        
        Args:
            image_input: File path (str), PIL.Image, or numpy.ndarray (RGB)
            
        Returns:
            dict containing predicted parameters and annotated visualization image:
            {
                "x": float,
                "y": float,
                "width": float,
                "height": float,
                "angle_rad": float,
                "angle_deg": float,
                "confidence": float,
                "annotated_image": np.ndarray (RGB)
            }
        """
        if isinstance(image_input, str):
            image_pil = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            image_pil = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image_pil = image_input.convert("RGB")
        else:
            raise TypeError("Unsupported image_input type.")

        orig_w, orig_h = image_pil.size

        # Preprocess image for neural net
        resized_pil = image_pil.resize((self.image_size, self.image_size), Image.BILINEAR)
        img_tensor = self.transform(resized_pil).unsqueeze(0).to(self.device)

        # Model Forward Pass
        with torch.no_grad():
            output = self.model(img_tensor).squeeze(0).cpu().numpy()

        # Unpack normalized predictions
        x_norm, y_norm, w_norm, h_norm, sin_t, cos_t, conf_logit = output

        # Map back to original image coordinate frame
        x_orig = x_norm * orig_w
        y_orig = y_norm * orig_h
        w_orig = max(w_norm * orig_w, 5.0)
        h_orig = max(h_norm * orig_h, 5.0)
        
        angle_rad = math.atan2(sin_t, cos_t)
        angle_deg = math.degrees(angle_rad)
        confidence = float(1.0 / (1.0 + math.exp(-conf_logit)))  # Sigmoid

        # Draw grasp visualization on original image
        orig_np = np.array(image_pil)
        grasp_tuple = (x_orig, y_orig, w_orig, h_orig, angle_rad)
        annotated_img = draw_grasp(orig_np, grasp_tuple, confidence=confidence, color=(0, 255, 0))

        return {
            "x": float(x_orig),
            "y": float(y_orig),
            "width": float(w_orig),
            "height": float(h_orig),
            "angle_rad": float(angle_rad),
            "angle_deg": float(angle_deg),
            "confidence": float(confidence),
            "annotated_image": annotated_img
        }


def main():
    parser = argparse.ArgumentParser(description="Predict robotic grasp on an RGB image")
    parser.add_argument("--image", type=str, required=True, help="Path to input RGB image")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to trained model checkpoint")
    parser.add_argument("--output", type=str, default="results/grasp_predictions.png", help="Path to save prediction visualization")
    args = parser.parse_args()

    predictor = GraspPredictor(checkpoint_path=args.checkpoint)
    result = predictor.predict(args.image)

    print("\n" + "=" * 45)
    print("        NEUROGRASP INFERENCE RESULT         ")
    print("=" * 45)
    print(f" Image Path:       {args.image}")
    print(f" Grasp Center X:   {result['x']:.2f} px")
    print(f" Grasp Center Y:   {result['y']:.2f} px")
    print(f" Gripper Width:    {result['width']:.2f} px")
    print(f" Gripper Height:   {result['height']:.2f} px")
    print(f" Grasp Angle:      {result['angle_deg']:.2f} degrees")
    print(f" Confidence Score: {result['confidence']:.4f}")
    print("=" * 45 + "\n")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    # Save annotated RGB image
    out_bgr = cv2.cvtColor(result["annotated_image"], cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, out_bgr)
    print(f"[+] Prediction visualization saved to: {args.output}")


if __name__ == "__main__":
    main()
