import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Literal 

import torch
from torch.nn import (Linear, Conv2d, 
                      SiLU, 
                      Flatten, AdaptiveAvgPool2d, 
                      Sequential, Module) 

from blocks.bottleneck import Bottleneck
from blocks.utility_layers import ConvLayerNorm


class ResidualStemBlock(Module):
    def __init__(self):
        """
        Residual Stem Block for ResNet. 


        Returns:
            torch.Tensor: A tensor of the same shape as the input tensor.
        """
        super(ResidualStemBlock, self).__init__()

        self.stem = Sequential(
            Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            ConvLayerNorm(64),
            SiLU(inplace=True),

            Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            ConvLayerNorm(64),
            SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class ResNet(Module):
    def __init__(self, blocks_amount: Literal[50, 101, 152]):
        """
        ResNet model. 

        In constrast with original implementation that one uses LayerNorm,
        Swish activation and a full-pre-activation residual block.

        Args:
            blocks_amount (Literal[50, 101, 152]): Number of blocks in the model

        Returns:
            torch.Tensor: Features vector of the input image.
        """
        super(ResNet, self).__init__()
       
        self.blocks_amount = blocks_amount

        if blocks_amount == 50:
            blocks_per_stage = [3, 4, 6, 3]
        elif blocks_amount == 101:
            blocks_per_stage = [3, 4, 23, 3]
        elif blocks_amount == 152:
            blocks_per_stage = [3, 8, 36, 3]

        # During creation loop we will substract 1 from current index block
        # so, at the start we will have -1 - the last element of the list.
        # That is why we have 64 at the end of the list.
        channels_per_stage = [256, 512, 1024, 2048, 64]

        # Stem block
        self.stem = ResidualStemBlock()

        # Main body of the model
        self.stages = Sequential()

        for stage_idx in range(4):
            current_blocks_amount    = blocks_per_stage[stage_idx]
            previous_channels_amount = channels_per_stage[stage_idx - 1]
            current_channels_amount  = channels_per_stage[stage_idx]

            stage = self._build_stage(
                num_blocks=current_blocks_amount,
                in_channels=previous_channels_amount,
                out_channels=current_channels_amount
            )

            self.stages.add_module(f"stage_{stage_idx}", stage)

        # Final pooling layer
        self.final_pool = Sequential(
            AdaptiveAvgPool2d(1),
            Flatten()
        )

        # Initialize weights for all layers
        for m in self.modules():
            if isinstance(m, Conv2d) or isinstance(m, Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

            elif isinstance(m, ConvLayerNorm):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def _build_stage(self,
                    num_blocks, 
                    in_channels, 
                    out_channels
        ) -> Sequential:
        """
        Creates a stage of the network from several Bottleneck blocks.
        
        Args:
            num_blocks (int): Number of blocks in the stage
            in_channels (int): Number of input channels for the first block
            out_channels (int): Number of output channels for all blocks

        Returns:
            Sequential: A sequential container with the stage blocks
        """
        stage = Sequential()
        
        # First block downsamples the input
        stage.add_module(f'block_0', Bottleneck(
            in_channels=in_channels,
            out_channels=out_channels,
            downsample=True
        ))
        
        # Subsequent blocks work without changing the size
        for block_idx in range(1, num_blocks):
            stage.add_module(f'block_{block_idx}', Bottleneck(
                in_channels=out_channels,
                out_channels=out_channels,
                downsample=False
            ))
            
        return stage


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.stages(out)
        out = self.final_pool(out)

        return out


class ResNetClassifier(Module):
    def __init__(self, blocks_amount: Literal[50, 101, 152], num_classes: int):
        super(ResNetClassifier, self).__init__()

        self.resnet = ResNet(blocks_amount)
        self.classifier = Linear(2048, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.resnet(x))

