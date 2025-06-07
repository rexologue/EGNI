import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Literal

import torch
from torch import nn

from blocks.stem import ResNetStemBlock
from blocks.bottleneck import Bottleneck


class ResNet(nn.Module):
    """
    ResNet model.

    A class for implementing ResNet-50/101/152 in ResNet v2 style-like.
    """

    def __init__(
        self,
        blocks_amount: Literal[50, 101, 152]
    ):

        super(ResNet, self).__init__()

        if blocks_amount not in [50, 101, 152]:
            raise ValueError(f"Incorrect blocks amount parameter! It must be only one of 50, 101 or 152!")

        self.blocks_amount = blocks_amount

        self._init_layers()
        self.apply(self._init_weights)

        self.output_dim = 2048


    def _make_stage(
        self,
        stage_idx: int,
        num_blocks: int,
        in_channels: int,
        out_channels: int,
    ):

        """
        Creates a stage of the network from several blocks.

        Args:
            stage_idx (int); Current stage index
            num_blocks (int): Number of blocks in the stage
            in_channels (int): Number of input channels for the first block
            out_channels (int): Number of output channels for all blocks
        """
        self.layers.add_module(
            f"stage_{stage_idx}_block_0",
            Bottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                downsample=(stage_idx != 0),
                after_stem=(stage_idx == 0)
            )
        )

        # Subsequent blocks work without changing the size
        for block_idx in range(1, num_blocks):
            self.layers.add_module(
                f"stage_{stage_idx}_block_{block_idx}",
                Bottleneck(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    downsample=False
                )
            )


    def _init_layers(self) -> None:
        """
        Layers initialization.

        Args:
            in_channels: The number of model input channels.
            classes_num: The number of classes.
        """
        self.layers = nn.Sequential()

        # Get arch specs
        blocks_per_stage, channels_per_stage = self._get_arch_specs()

        # Build stem
        self.layers.add_module(
            "stem",
            ResNetStemBlock(3, channels_per_stage[0])
        )

        # Build stages
        for stage_idx in range(4):
            current_blocks_amount    = blocks_per_stage[stage_idx]

            previous_channels_amount = channels_per_stage[stage_idx]
            current_channels_amount  = channels_per_stage[stage_idx+1]

            self._make_stage(
                stage_idx=stage_idx,
                num_blocks=current_blocks_amount,
                in_channels=previous_channels_amount,
                out_channels=current_channels_amount
            )


    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        """Layer parameters initialization.

        Args:
            module: A model layer.
        """
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(
                module.weight,
                mode='fan_in',
                nonlinearity='relu'
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.BatchNorm2d):
            nn.init.zeros_(module.bias)

            if getattr(module, "is_last", None) is not None:
                nn.init.zeros_(module.weight)
            else:
                nn.init.ones_(module.weight)

        else:
            pass


    def _get_arch_specs(self) -> tuple[list[int], list[int]]:
        if self.blocks_amount == 50:
            blocks_per_stage = [3, 4, 6, 3]

        elif self.blocks_amount == 101:
            blocks_per_stage = [3, 4, 23, 3]

        elif self.blocks_amount == 152:
            blocks_per_stage = [3, 8, 36, 3]

        channels_per_stage = [64, 256, 512, 1024, 2048]

        return blocks_per_stage, channels_per_stage


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation"""
        return self.layers(x)
