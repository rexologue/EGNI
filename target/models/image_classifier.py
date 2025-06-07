from typing import Literal

import torch
from torch import nn

from models.resnet import ResNet


class ImageClassifier(nn.Module):
    def __init__(
        self,
        classes_num: int,
        backbone: Literal[
            "resnet50",
            "resnet101",
            "resnet152",
        ]
    ):
        super(ImageClassifier, self).__init__()

        if backbone.startswith("resnet"):
            blocks_amount = int(backbone.split("resnet")[1])
            self.backbone = ResNet(blocks_amount)

        else:
            raise ValueError(f"Unsupported bacbone type: {backbone}")
        
        self.classes_num = classes_num
        self.fc = nn.Linear(self.backbone.output_dim, classes_num)

    
    def load_backbone_state_dict(self, state_dict: dict):
        self.backbone.load_state_dict(state_dict)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation"""
        return self.fc(self.backbone(x))
