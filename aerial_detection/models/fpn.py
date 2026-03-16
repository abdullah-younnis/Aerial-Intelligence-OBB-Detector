"""Feature Pyramid Network for multi-scale feature extraction."""

from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network (FPN) for multi-scale feature fusion.
    
    Takes features from backbone (C2-C5) and produces P3-P7 feature maps.
    """
    
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256,
        extra_blocks: bool = True
    ):
        """
        Initialize FPN.
        
        Args:
            in_channels_list: List of input channels from backbone [C2, C3, C4, C5]
            out_channels: Number of output channels for all FPN levels
            extra_blocks: Whether to add P6 and P7 levels
        """
        super().__init__()
        
        self.out_channels = out_channels
        self.extra_blocks = extra_blocks
        
        # Lateral connections (1x1 conv to reduce channels)
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
        
        # Output convolutions (3x3 conv to smooth features)
        self.output_convs = nn.ModuleList()
        for _ in in_channels_list:
            self.output_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )
        
        # Extra blocks for P6 and P7
        if extra_blocks:
            # P6 from C5 with stride 2
            self.p6_conv = nn.Conv2d(
                in_channels_list[-1], out_channels, kernel_size=3, stride=2, padding=1
            )
            # P7 from P6 with stride 2
            self.p7_conv = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=2, padding=1
            )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize convolution weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self, 
        features: Dict[str, torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Forward pass through FPN.
        
        Args:
            features: Dict with keys 'c2', 'c3', 'c4', 'c5' from backbone
            
        Returns:
            List of feature tensors [P3, P4, P5, P6, P7] (or [P3, P4, P5] if no extra blocks)
        """
        # Get features in order
        feature_list = [features['c2'], features['c3'], features['c4'], features['c5']]
        
        # Build top-down pathway
        laterals = []
        for i, (feature, lateral_conv) in enumerate(zip(feature_list, self.lateral_convs)):
            laterals.append(lateral_conv(feature))
        
        # Top-down fusion
        for i in range(len(laterals) - 1, 0, -1):
            # Upsample and add
            upsampled = F.interpolate(
                laterals[i], 
                size=laterals[i-1].shape[-2:],
                mode='nearest'
            )
            laterals[i-1] = laterals[i-1] + upsampled
        
        # Apply output convolutions
        outputs = []
        for lateral, output_conv in zip(laterals, self.output_convs):
            outputs.append(output_conv(lateral))
        
        # We want P3, P4, P5 (skip P2 as it's too large)
        # outputs[0] = P2, outputs[1] = P3, outputs[2] = P4, outputs[3] = P5
        pyramid = outputs[1:]  # [P3, P4, P5]
        
        # Add extra blocks
        if self.extra_blocks:
            # P6 from C5
            p6 = self.p6_conv(feature_list[-1])
            pyramid.append(p6)
            
            # P7 from P6
            p7 = self.p7_conv(F.relu(p6))
            pyramid.append(p7)
        
        return pyramid


class FPNWithBackbone(nn.Module):
    """Combined backbone and FPN module."""
    
    def __init__(
        self,
        backbone: nn.Module,
        fpn_out_channels: int = 256,
        extra_blocks: bool = True
    ):
        """
        Initialize backbone + FPN.
        
        Args:
            backbone: Backbone module with in_channels_list attribute
            fpn_out_channels: Output channels for FPN
            extra_blocks: Whether to add P6, P7
        """
        super().__init__()
        
        self.backbone = backbone
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=backbone.in_channels_list,
            out_channels=fpn_out_channels,
            extra_blocks=extra_blocks
        )
        self.out_channels = fpn_out_channels
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            List of feature tensors [P3, P4, P5, P6, P7]
        """
        features = self.backbone(x)
        return self.fpn(features)
