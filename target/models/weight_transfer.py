from typing import Dict, Tuple, Optional
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.parameter import Parameter

from .resnet import ResNet, ResNetClassifier


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_pretrained_resnet(variant: int) -> nn.Module:
    """
    Get the pretrained ResNet model from torchvision.

    Args:
        variant (int): ResNet variant (50, 101, or 152)

    Returns:
        nn.Module: Pretrained ResNet model
    """
    if variant == 50:
        return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    elif variant == 101:
        return models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
    elif variant == 152:
        return models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
    else:
        raise ValueError(f"Unsupported ResNet variant: {variant}")


def convert_bn_to_ln_weights(bn_weight: Parameter,
                           bn_bias: Parameter,
                           bn_mean: Parameter,
                           bn_var: Parameter) -> Tuple[Parameter, Parameter]:
    """
    Convert BatchNorm weights to LayerNorm weights.
    This is an approximation as the normalization methods are different.

    Args:
        bn_weight: BatchNorm weight parameter
        bn_bias: BatchNorm bias parameter
        bn_mean: BatchNorm running mean
        bn_var: BatchNorm running variance

    Returns:
        Tuple[Parameter, Parameter]: LayerNorm weight and bias parameters
    """
    # Initialize LayerNorm parameters
    ln_weight = Parameter(bn_weight.clone())
    ln_bias = Parameter(bn_bias.clone())

    # Adjust for running statistics
    ln_weight = ln_weight / torch.sqrt(bn_var + 1e-5)
    ln_bias = ln_bias - bn_mean * ln_weight

    return ln_weight, ln_bias


def transfer_conv_weights(source_conv: nn.Conv2d,
                         target_conv: nn.Conv2d) -> None:
    """
    Transfer weights between convolutional layers.

    Args:
        source_conv: Source convolution layer
        target_conv: Target convolution layer
    """
    if source_conv.weight.shape != target_conv.weight.shape:
        raise ValueError(f"Incompatible conv shapes: {source_conv.weight.shape} vs {target_conv.weight.shape}")

    target_conv.weight.data.copy_(source_conv.weight.data)
    if source_conv.bias is not None and target_conv.bias is not None:
        target_conv.bias.data.copy_(source_conv.bias.data)


def transfer_stage_weights(source_stage: nn.Sequential,
                         target_stage: nn.Sequential,
                         stage_idx: int) -> None:
    """
    Transfer weights for a complete ResNet stage.

    Args:
        source_stage: Source ResNet stage
        target_stage: Target ResNet stage
        stage_idx: Stage index for logging
    """
    for block_idx, (source_block, target_block) in enumerate(zip(source_stage, target_stage)):
        # Transfer shortcut weights
        if hasattr(source_block, 'downsample') and source_block.downsample is not None:
            if block_idx == 0:  # First block in stage
                # Transfer conv weights in shortcut
                source_conv = source_block.downsample[0]
                target_conv = target_block.shortcut_layer._modules.get('conv', None)

                if target_conv is not None:
                    transfer_conv_weights(source_conv, target_conv)
                    logger.info(f"Transferred shortcut conv weights for stage {stage_idx}, block {block_idx}")

        # Transfer main path weights
        for i, (source_conv, target_conv) in enumerate(zip(
            [m for m in source_block.modules() if isinstance(m, nn.Conv2d)],
            [m for m in target_block.modules() if isinstance(m, nn.Conv2d)]
        )):
            transfer_conv_weights(source_conv, target_conv)
            logger.info(f"Transferred conv weights for stage {stage_idx}, block {block_idx}, conv {i}")

        # Convert and transfer normalization weights
        source_bns = [m for m in source_block.modules() if isinstance(m, nn.BatchNorm2d)]
        target_lns = [m for m in target_block.modules() if isinstance(m, nn.LayerNorm)]

        for i, (bn, ln) in enumerate(zip(source_bns, target_lns)):
            ln_weight, ln_bias = convert_bn_to_ln_weights(
                bn.weight, bn.bias, bn.running_mean, bn.running_var
            )
            ln.weight.data.copy_(ln_weight.data)
            ln.bias.data.copy_(ln_bias.data)
            logger.info(f"Converted and transferred norm weights for stage {stage_idx}, block {block_idx}, norm {i}")


def transfer_stem_weights(source_model: nn.Module,
                        target_model: ResNet) -> None:
    """
    Transfer and adapt weights for the stem block.
    Note: This is an approximation as architectures differ.

    Args:
        source_model: Source ResNet model
        target_model: Target custom ResNet model
    """
    # Source has 7x7 conv, target has two 3x3 convs
    source_conv = source_model.conv1
    target_convs = [m for m in target_model.stem.modules() if isinstance(m, nn.Conv2d)]

    # Approximate 7x7 conv as two 3x3 convs
    # First 3x3 conv gets central weights
    center = source_conv.weight.data[:, :, 2:5, 2:5]
    target_convs[0].weight.data.copy_(center)
    logger.info("Transferred approximated stem weights to first 3x3 conv")

    # Second 3x3 conv gets identity-like weights
    nn.init.kaiming_normal_(target_convs[1].weight)
    logger.info("Initialized second stem conv with Kaiming initialization")


def transfer_weights(source_variant: int, target_model: ResNet) -> None:
    """
    Main function to transfer weights from a pre-trained ResNet to custom ResNet.

    Args:
        source_variant (int): Source ResNet variant (50, 101, or 152)
        target_model (ResNet): Target custom ResNet model
    """
    logger.info(f"Starting weight transfer from ResNet-{source_variant} to custom model")

    # Get pre-trained model
    source_model = get_pretrained_resnet(source_variant)
    source_model.eval()

    # Transfer stem weights
    transfer_stem_weights(source_model, target_model)

    # Transfer stage weights
    source_stages = [
        nn.Sequential(*list(source_model.layer1)),
        nn.Sequential(*list(source_model.layer2)),
        nn.Sequential(*list(source_model.layer3)),
        nn.Sequential(*list(source_model.layer4))
    ]

    for stage_idx, (source_stage, target_stage) in enumerate(
        zip(source_stages, target_model.stages)
    ):
        transfer_stage_weights(source_stage, target_stage, stage_idx)
        logger.info(f"Completed transfer for stage {stage_idx}")

    logger.info("Weight transfer completed successfully")


def transfer_weights_to_classifier(source_variant: int,
                                target_model: ResNetClassifier,
                                num_classes: Optional[int] = None) -> None:
    """
    Transfer weights to the classifier model, including the classification head if possible.

    Args:
        source_variant (int): Source ResNet variant (50, 101, or 152)
        target_model (ResNetClassifier): Target classifier model
        num_classes (Optional[int]): Number of classes in target model
    """
    # Transfer backbone weights
    transfer_weights(source_variant, target_model.resnet)

    # Transfer classifier weights if number of classes matches
    source_model = get_pretrained_resnet(source_variant)
    if num_classes == 1000:  # ImageNet classes
        target_model.classifier.weight.data.copy_(source_model.fc.weight.data)
        target_model.classifier.bias.data.copy_(source_model.fc.bias.data)
        logger.info("Transferred classification head weights")
    else:
        logger.info("Skipped classification head transfer due to different number of classes")
