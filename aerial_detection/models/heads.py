"""Detection heads for classification and regression."""

from typing import List
import torch
import torch.nn as nn
import math


class ClassificationHead(nn.Module):
    """
    Classification head for object detection.
    
    Predicts class probabilities for each anchor.
    """
    
    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        num_classes: int,
        num_convs: int = 4,
        prior_prob: float = 0.01
    ):
        """
        Initialize classification head.
        
        Args:
            in_channels: Number of input channels from FPN
            num_anchors: Number of anchors per spatial location
            num_classes: Number of object classes
            num_convs: Number of conv layers before output
            prior_prob: Prior probability for bias initialization
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Shared conv layers
        conv_layers = []
        for _ in range(num_convs):
            conv_layers.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
            )
            conv_layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv_layers)
        
        # Output layer
        self.cls_logits = nn.Conv2d(
            in_channels, 
            num_anchors * num_classes, 
            kernel_size=3, 
            padding=1
        )
        
        # Initialize weights
        self._init_weights(prior_prob)
    
    def _init_weights(self, prior_prob: float):
        """Initialize weights with proper bias for focal loss."""
        for layer in self.conv:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)
        
        nn.init.normal_(self.cls_logits.weight, std=0.01)
        # Initialize bias for prior probability
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_value)
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through classification head.
        
        Args:
            features: List of FPN feature tensors
            
        Returns:
            Classification logits of shape (B, total_anchors, num_classes)
        """
        all_cls_logits = []
        
        for feature in features:
            cls_logits = self.conv(feature)
            cls_logits = self.cls_logits(cls_logits)
            
            # Reshape: (B, A*C, H, W) -> (B, H*W*A, C)
            B, _, H, W = cls_logits.shape
            cls_logits = cls_logits.view(B, self.num_anchors, self.num_classes, H, W)
            cls_logits = cls_logits.permute(0, 3, 4, 1, 2)  # (B, H, W, A, C)
            cls_logits = cls_logits.reshape(B, -1, self.num_classes)  # (B, H*W*A, C)
            
            all_cls_logits.append(cls_logits)
        
        return torch.cat(all_cls_logits, dim=1)


class RegressionHead(nn.Module):
    """
    Regression head for rotated bounding box prediction.
    
    Predicts (dx, dy, dw, dh, dtheta) offsets for each anchor.
    """
    
    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        num_convs: int = 4
    ):
        """
        Initialize regression head.
        
        Args:
            in_channels: Number of input channels from FPN
            num_anchors: Number of anchors per spatial location
            num_convs: Number of conv layers before output
        """
        super().__init__()
        
        self.num_anchors = num_anchors
        
        # Shared conv layers
        conv_layers = []
        for _ in range(num_convs):
            conv_layers.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
            )
            conv_layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv_layers)
        
        # Output layer: 5 values per anchor (dx, dy, dw, dh, dtheta)
        self.bbox_reg = nn.Conv2d(
            in_channels, 
            num_anchors * 5, 
            kernel_size=3, 
            padding=1
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for layer in self.conv:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)
        
        nn.init.normal_(self.bbox_reg.weight, std=0.01)
        nn.init.constant_(self.bbox_reg.bias, 0)
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through regression head.
        
        Args:
            features: List of FPN feature tensors
            
        Returns:
            Box regression of shape (B, total_anchors, 5)
        """
        all_bbox_reg = []
        
        for feature in features:
            bbox_reg = self.conv(feature)
            bbox_reg = self.bbox_reg(bbox_reg)
            
            # Reshape: (B, A*5, H, W) -> (B, H*W*A, 5)
            B, _, H, W = bbox_reg.shape
            bbox_reg = bbox_reg.view(B, self.num_anchors, 5, H, W)
            bbox_reg = bbox_reg.permute(0, 3, 4, 1, 2)  # (B, H, W, A, 5)
            bbox_reg = bbox_reg.reshape(B, -1, 5)  # (B, H*W*A, 5)
            
            all_bbox_reg.append(bbox_reg)
        
        return torch.cat(all_bbox_reg, dim=1)


class RetinaNetHead(nn.Module):
    """
    Combined classification and regression head for RetinaNet.
    """
    
    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        num_classes: int,
        num_convs: int = 4,
        prior_prob: float = 0.01
    ):
        """
        Initialize RetinaNet head.
        
        Args:
            in_channels: Number of input channels from FPN
            num_anchors: Number of anchors per spatial location
            num_classes: Number of object classes
            num_convs: Number of conv layers
            prior_prob: Prior probability for classification bias
        """
        super().__init__()
        
        self.cls_head = ClassificationHead(
            in_channels, num_anchors, num_classes, num_convs, prior_prob
        )
        self.reg_head = RegressionHead(
            in_channels, num_anchors, num_convs
        )
    
    def forward(
        self, 
        features: List[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            features: List of FPN feature tensors
            
        Returns:
            cls_logits: (B, total_anchors, num_classes)
            bbox_reg: (B, total_anchors, 5)
        """
        cls_logits = self.cls_head(features)
        bbox_reg = self.reg_head(features)
        return cls_logits, bbox_reg
