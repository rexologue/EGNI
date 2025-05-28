import torch
from torch.nn import Conv2d, SiLU, Sigmoid, AdaptiveAvgPool2d, Module


class SEModule(Module):
    def __init__(self, channels: int, reduction: int):
        """
        The Squeeze-and-Excitation (SE) mechanism is a module for adaptive recalibration of channel-wise feature responses.
        Its core idea is to dynamically emphasize more informative channels and suppress less useful ones.
        Working principle of the SE module:

        1. Squeeze
        At this stage, spatial information is reduced to a vector that captures global characteristics for each channel.
        Global Average Pooling is applied.
        - Goal: Aggregate spatial information across the entire feature map for each channel.
        - Result: For an input tensor of shape H x W x C, the result is a 1 x 1 x C vector, where each value reflects the global importance of the corresponding channel.

        2. Excitation
        After squeezing, the excitation step recalibrates the information for each channel.
        - Process: The squeezed vector is passed through a small neural network, typically with two fully connected layers and nonlinearities.
        - First layer: Reduces dimensionality (using the reduction ratio), allowing the model to learn cross-channel interactions.
        - Second layer: Restores the original channel dimension.
        - Activation functions: ReLU is commonly used after the first layer and sigmoid after the second to obtain weights in the [0, 1] range.
        - Result: A vector of channel-wise weights representing their relative importance.

        3. Recalibration
        Finally, the original features are scaled by the learned channel weights.
        - Operation: Each channel of the input tensor is multiplied by the corresponding scalar from the excitation vector.
        - Effect: This allows the network to recalibrate channel activations—enhancing important ones and suppressing less informative ones,
                 improving feature representation for subsequent layers.

        Summary: The SE module enables the network to dynamically adjust channel importance based on global context.
        """
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)  # Reduces all feature maps to a single value per channel

        # Here, convolution acts as a linear layer that reduces dimensionality.
        # Since we use Global Average Pooling, the feature maps are of size 1x1,
        # so this avoids the need to reshape tensors.
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0)

        # Second layer restores original number of channels
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0)

        # Initialize weights
        torch.nn.init.xavier_uniform_(self.fc1.weight.data)
        torch.nn.init.xavier_uniform_(self.fc2.weight.data)

        # Activations
        self.relu = SiLU(inplace=True)
        self.sigmoid = Sigmoid()
        

    def forward(self, x: torch.Tensor):
        # Squeeze
        out = self.avg_pool(x)

        # Excitation
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        # Recalibration
        return out * x
