import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuroGraspLoss(nn.Module):
    """
    Multi-task loss function for planar grasp regression and confidence scoring.
    
    Combines:
    - Position Loss (SmoothL1 on x, y)
    - Size Loss (SmoothL1 on width, height)
    - Angle Loss (SmoothL1 on sin(theta), cos(theta))
    - Confidence Loss (BCEWithLogits on grasp confidence)
    """
    def __init__(
        self,
        lambda_xy: float = 1.0,
        lambda_wh: float = 1.0,
        lambda_angle: float = 1.0,
        lambda_conf: float = 0.5
    ):
        super(NeuroGraspLoss, self).__init__()
        self.lambda_xy = lambda_xy
        self.lambda_wh = lambda_wh
        self.lambda_angle = lambda_angle
        self.lambda_conf = lambda_conf

        self.smooth_l1 = nn.SmoothL1Loss(beta=0.1)
        self.bce_logits = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple:
        """
        Args:
            pred: Tensor (B, 7) -> [x, y, w, h, sin_t, cos_t, conf_logit]
            target: Tensor (B, 7) -> [x, y, w, h, sin_t, cos_t, target_conf]
            
        Returns:
            tuple: (total_loss, dict of individual loss component values)
        """
        # Split predictions and targets
        pred_xy = pred[:, 0:2]
        target_xy = target[:, 0:2]

        pred_wh = pred[:, 2:4]
        target_wh = target[:, 2:4]

        pred_angle = pred[:, 4:6]
        target_angle = target[:, 4:6]

        pred_conf_logit = pred[:, 6]
        target_conf = target[:, 6]

        # Compute individual losses
        loss_xy = self.smooth_l1(pred_xy, target_xy)
        loss_wh = self.smooth_l1(pred_wh, target_wh)
        loss_angle = self.smooth_l1(pred_angle, target_angle)
        loss_conf = self.bce_logits(pred_conf_logit, target_conf)

        # Weighted Total Loss
        total_loss = (
            self.lambda_xy * loss_xy +
            self.lambda_wh * loss_wh +
            self.lambda_angle * loss_angle +
            self.lambda_conf * loss_conf
        )

        loss_dict = {
            "total_loss": total_loss.item(),
            "loss_xy": loss_xy.item(),
            "loss_wh": loss_wh.item(),
            "loss_angle": loss_angle.item(),
            "loss_conf": loss_conf.item()
        }

        return total_loss, loss_dict
