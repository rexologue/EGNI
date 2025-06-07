import torch
from torch import nn


class ResNetStemBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stem_channels: int = 32
    ):

        """
        Makes ResNet-D stem module.

        Scheme: Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> MaxPool

        Args:
            in_channels: The number of input channels.
            out_channels: The number of otput channels.
            stem_channels: The number of channels inside block.
        """

        super(ResNetStemBlock, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(stem_channels, stem_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(stem_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation."""
        return self.stem(x)
