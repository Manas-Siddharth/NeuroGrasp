import torch
import torch.nn as nn
import torchvision.models as models


class NeuroGraspNet(nn.Module):
    """
    NeuroGrasp Deep Neural Network for Robotic Grasp Detection.
    
    Uses ResNet18 CNN backbone to extract 512-dim visual features,
    followed by dual prediction heads:
    1. Grasp Head: Predicts planar grasp parameters [x, y, width, height, sin(theta), cos(theta)]
    2. Confidence Head: Predicts grasp confidence logit
    """
    def __init__(self, backbone_name: str = "resnet18", pretrained: bool = True, dropout_rate: float = 0.2):
        super(NeuroGraspNet, self).__init__()
        
        # Load backbone
        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            in_features = self.backbone.fc.in_features
            # Replace final classification layer with identity
            self.backbone.fc = nn.Identity()
        else:
            # Fallback to MobileNetV3 if specified
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v3_small(weights=weights)
            in_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()

        # Shared Feature Processing
        self.feature_dropout = nn.Dropout(p=dropout_rate)

        # Grasp Head: Regression for (x, y, w, h, sin(theta), cos(theta))
        self.grasp_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 6)
        )

        # Confidence Head: Logit prediction for grasp validity/confidence
        self.confidence_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Input:
            x: Tensor of shape (B, 3, H, W)
        Output:
            Tensor of shape (B, 7) containing:
            [x_norm, y_norm, w_norm, h_norm, sin_theta, cos_theta, confidence_logit]
        """
        features = self.backbone(x)
        features = self.feature_dropout(features)

        grasp_preds = self.grasp_head(features)
        conf_logits = self.confidence_head(features)

        # Concatenate outputs: (B, 6) + (B, 1) -> (B, 7)
        output = torch.cat([grasp_preds, conf_logits], dim=1)
        return output
