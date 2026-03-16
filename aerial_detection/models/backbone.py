"""Backbone networks for feature extraction."""

from typing import Dict, List
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models._utils import IntermediateLayerGetter


class BackboneWithFPN(nn.Module):
    """
    Backbone wrapper that returns multi-scale feature maps.
    
    Returns features from multiple stages for FPN.
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        return_layers: Dict[str, str],
        in_channels_list: List[int]
    ):
        super().__init__()
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.in_channels_list = in_channels_list
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract multi-scale features.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Dict mapping layer names to feature tensors
        """
        return self.body(x)


def build_resnet_backbone(
    name: str = 'resnet50',
    pretrained: bool = True
) -> BackboneWithFPN:
    """
    Build ResNet backbone for feature extraction.
    
    Args:
        name: ResNet variant ('resnet50', 'resnet101')
        pretrained: Whether to use ImageNet pretrained weights
        
    Returns:
        BackboneWithFPN instance
    """
    if name == 'resnet50':
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)
        in_channels_list = [256, 512, 1024, 2048]
    elif name == 'resnet101':
        weights = models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet101(weights=weights)
        in_channels_list = [256, 512, 1024, 2048]
    else:
        raise ValueError(f"Unknown ResNet variant: {name}")
    
    # Return features from layer1, layer2, layer3, layer4
    return_layers = {
        'layer1': 'c2',  # stride 4
        'layer2': 'c3',  # stride 8
        'layer3': 'c4',  # stride 16
        'layer4': 'c5',  # stride 32
    }
    
    return BackboneWithFPN(backbone, return_layers, in_channels_list)


def build_backbone(
    name: str = 'resnet50',
    pretrained: bool = True
) -> BackboneWithFPN:
    """
    Build backbone network.
    
    Args:
        name: Backbone name ('resnet50', 'resnet101', 'swin_t', 'swin_s')
        pretrained: Whether to use pretrained weights
        
    Returns:
        BackboneWithFPN instance
    """
    if name.startswith('resnet'):
        return build_resnet_backbone(name, pretrained)
    elif name.startswith('swin'):
        return build_swin_backbone(name, pretrained)
    else:
        raise ValueError(f"Unknown backbone: {name}")


def build_swin_backbone(
    name: str = 'swin_t',
    pretrained: bool = True
) -> BackboneWithFPN:
    """
    Build Swin Transformer backbone.
    
    Args:
        name: Swin variant ('swin_t', 'swin_s', 'swin_b')
        pretrained: Whether to use pretrained weights
        
    Returns:
        BackboneWithFPN instance
    """
    if name == 'swin_t':
        weights = models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.swin_t(weights=weights)
        in_channels_list = [96, 192, 384, 768]
    elif name == 'swin_s':
        weights = models.Swin_S_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.swin_s(weights=weights)
        in_channels_list = [96, 192, 384, 768]
    elif name == 'swin_b':
        weights = models.Swin_B_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.swin_b(weights=weights)
        in_channels_list = [128, 256, 512, 1024]
    else:
        raise ValueError(f"Unknown Swin variant: {name}")
    
    # Swin uses features module
    return_layers = {
        'features.1': 'c2',
        'features.3': 'c3',
        'features.5': 'c4',
        'features.7': 'c5',
    }
    
    return BackboneWithFPN(backbone, return_layers, in_channels_list)
