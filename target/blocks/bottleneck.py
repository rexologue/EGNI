from torch.nn import (Conv2d, AvgPool2d,
                      SiLU, 
                      Sequential, Module) 

from .utility_layers import ConvLayerNorm

class Bottleneck(Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 downsample=True):
        """
        Pre-activation ResNet Bottleneck Block (ResNet v2 style).

        Components:
        
        1. Shortcut Path:
        - If downsample=True: applies spatial downsampling using AvgPool2d + 1x1 convolution (ResNet-D style).
        - If in_channels != out_channels: uses a 1x1 convolution to align the channel dimensions.
        - This path allows identity mappings, enabling better gradient flow.

        2. Residual Path:
        - Bottleneck architecture: 1x1 → 3x3 → 1x1 convolutions.
        - Pre-activation order: BatchNorm → SiLU → Conv (before each convolution).

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            downsample (bool): Whether to halve the spatial resolution (default: True).
        """
        super(Bottleneck, self).__init__()
        
        #################
        # Shortcut Path #
        #################
        self.shortcut_layer = Sequential()

        # Downsample with AvgPool if needed
        if downsample:
            self.shortcut_layer.add_module("pool", AvgPool2d(kernel_size=2, stride=2))

        # Match channel dimensions if needed
        if in_channels != out_channels:
            self.shortcut_layer.add_module("conv", Conv2d(
                in_channels, out_channels, 
                kernel_size=1, stride=1, 
                bias=False
            ))

        #################
        # Residual Path #
        #################
        bottleneck_channels = out_channels // 4
        stride = 2 if downsample else 1

        self.res_layer = Sequential(
            # Phase 1: Reduce channels
            ConvLayerNorm(in_channels),
            SiLU(inplace=True),
            Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1, bias=False),
            
            # Phase 2: Spatial processing
            ConvLayerNorm(bottleneck_channels),
            SiLU(inplace=True),
            Conv2d(bottleneck_channels, bottleneck_channels, 
                   kernel_size=3, stride=stride, padding=1, bias=False),
            
            # Phase 3: Restore channels
            ConvLayerNorm(bottleneck_channels),
            SiLU(inplace=True),
            Conv2d(bottleneck_channels, out_channels, kernel_size=1, stride=1, bias=False),
            ConvLayerNorm(out_channels)
        )

    def forward(self, x):
        """Forward pass: F(x) + Shortcut(x)"""
        return self.res_layer(x) + self.shortcut_layer(x)
