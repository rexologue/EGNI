import torch
from torch import nn 


class Bottleneck(nn.Module):
    """
    A 'bottleneck' block used as a building block for ResNet-50/101/152

    This block is usually repeated several times within each of the model stages.

    At each repeat the block has two paths for the input:
        - Path A: BN -> ReLU -> Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> Conv (skipping first BN and ReLU when
            there is a stem before)
                  Three convolutions defined as follows:
                      1. The same depth for the first two convolutions and depth x 4 for the third convolution
                      2. (1, 1) kernel size for the first and the third convolutions and (3, 3) - for the second
                      3. Padding = 1 only for the second convolution and 0 for all others
                      4. For the first and the third convolutions: stride = 1 always
                      5. For the second convolution: stride = 2 only at first repeat at all stages except for the first
                            one and stride = 1 otherwise

        - Path B: Identity or Conv or (AvgPool -> Conv)
                  A shortcut connection is defined as follows:
                      1. 'Downsampling' layer that consists of:
                            - Avg Pool layer with kernel = (2, 2) and stride = 2 and convolution with the same depth
                                    as for the last convolution from path A, kernel = (1, 1) and stride = 1
                            - both only at first repeat at all stages except for the first one.
                            Note: shortcut convolutions in this block are not followed by BN!
                      2. Identity layer (nn.Identity) at all other repeats

    Inputs are passed through the both paths, then results are summed up.
    """

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        downsample: bool = True,
        after_stem: bool = False
    ):
        
        """
        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            downsample: If to use downsampling layer.
            after_stem: If this block follows the stem block.
        """
        super(Bottleneck, self).__init__()

        #################
        # Shortcut Path #
        #################

        stride = 2 if downsample else 1

        if downsample:
            self.shortcut_path = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            )
            
        elif in_channels != out_channels:
            self.shortcut_path = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            
        else:
            self.shortcut_path = nn.Identity()

        #############
        # Main Path #
        #############

        if after_stem:
            pre_layers = []
        
        else:
            pre_layers = [
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            ]

        bottleneck_channels = out_channels // 4

        self.main_path = nn.Sequential(
            # Phase 1: "Bottleneck" channels
            *pre_layers,
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1, padding=0, bias=False),

            # Phase 2: Downsample if needed
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1, bias=False),

            # Phase 3: Unsqueeze channels
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )

        self._set_last_bn()


    def _set_last_bn(self) -> None:
        last_bn = None

        for m in self.main_path.modules():
            if isinstance(m, nn.BatchNorm2d):
                last_bn = m

        last_bn.is_last = True


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation."""
        return self.shortcut_path(x) + self.main_path(x)