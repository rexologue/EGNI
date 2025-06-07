import torch
from typing import Literal


#########################
# NORMALIZE INPUT LAYER #
#########################


class Normalize(torch.nn.Module):
    def __init__(
            self,
            norm_type: Literal['l1', 'l2'],
            axis=1
        ):
        """
        Module that implements L1 or L2 normalization across given axis

        Args:
            norm_type (str): Must be 'l1' or 'l2'. Responsible for which type of normalization will be applied.
            axis (int): Axis across which one normalization will be applied.

        Raises:
            ValueError: if norm_type is not equal 'l1' or 'l2'.
        """
        super(Normalize, self).__init__()

        if norm_type not in ['l1', 'l2']:
            raise ValueError("Provided norm_type argument is incorrect. It must take one of the 'l1', 'l2' values")

        self.norm = 1 if norm_type == 'l1' else 2
        self.axis = axis


    def forward(self, x: torch.Tensor):
        # Compute norm
        norm = torch.norm(x, p=self.norm, dim=self.axis, keepdim=True)

        # Clamp result for stability
        norm = norm.clamp(min=1e-8)

        # Normalize
        return x / norm


#####################################
# CONVOLUTIONAL LAYER NORMALIZATION #
#####################################


class ConvLayerNorm(torch.nn.GroupNorm):
    def __init__(self, channels):
        super(ConvLayerNorm, self).__init__(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Module that implements Group Normalization for convolutional layers.

        Args:
            channels (int): Number of channels in the input tensor.

        Returns:
            torch.Tensor: A tensor of the same shape as the input tensor.
        """
        return super().forward(x)
