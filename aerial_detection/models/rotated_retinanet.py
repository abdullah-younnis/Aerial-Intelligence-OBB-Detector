"""Rotated RetinaNet model for oriented object detection."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from .backbone import build_backbone
from .fpn import FPNWithBackbone
from .anchor_generator import RotatedAnchorGenerator
from .heads import RetinaNetHead
from .losses import RotatedRetinaNetLoss, decode_boxes
from ..geometry import rotated_nms


class RotatedRetinaNet(nn.Module):
    """
    Rotated RetinaNet for oriented object detection.
    
    Combines backbone, FPN, anchor generator, and detection heads
    for end-to-end rotated object detection.
    """
    
    def __init__(
        self,
        num_classes: int,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        fpn_channels: int = 256,
        anchor_sizes: List[int] = [32, 64, 128, 256, 512],
        anchor_ratios: List[float] = [0.5, 1.0, 2.0],
        anchor_angles: List[float] = [-90, -60, -30, 0, 30, 60],
        strides: List[int] = [8, 16, 32, 64, 128],
        # Loss parameters
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        positive_iou_threshold: float = 0.5,
        negative_iou_threshold: float = 0.4,
        # Inference parameters
        score_threshold: float = 0.05,
        nms_threshold: float = 0.5,
        max_detections: int = 1000
    ):
        """
        Initialize Rotated RetinaNet.
        
        Args:
            num_classes: Number of object classes (excluding background)
            backbone: Backbone network name
            pretrained: Use pretrained backbone weights
            fpn_channels: Number of FPN output channels
            anchor_sizes: Base anchor sizes for each FPN level
            anchor_ratios: Anchor aspect ratios
            anchor_angles: Anchor rotation angles in degrees
            strides: Feature map strides
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter
            positive_iou_threshold: IoU threshold for positive anchors
            negative_iou_threshold: IoU threshold for negative anchors
            score_threshold: Minimum score for detections
            nms_threshold: NMS IoU threshold
            max_detections: Maximum detections per image
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        
        # Build backbone + FPN
        backbone_module = build_backbone(backbone, pretrained)
        self.backbone_fpn = FPNWithBackbone(
            backbone_module,
            fpn_out_channels=fpn_channels,
            extra_blocks=True
        )
        
        # Anchor generator
        self.anchor_generator = RotatedAnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=anchor_ratios,
            angles=anchor_angles,
            strides=strides
        )
        
        # Detection head
        num_anchors = len(anchor_ratios) * len(anchor_angles)
        self.head = RetinaNetHead(
            in_channels=fpn_channels,
            num_anchors=num_anchors,
            num_classes=num_classes
        )
        
        # Loss function
        self.loss_fn = RotatedRetinaNetLoss(
            num_classes=num_classes,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            positive_iou_threshold=positive_iou_threshold,
            negative_iou_threshold=negative_iou_threshold
        )
    
    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: Input images (B, C, H, W)
            targets: Optional list of target dicts for training
            
        Returns:
            During training: Dict with losses
            During inference: Dict with detections
        """
        # Extract features
        features = self.backbone_fpn(images)
        
        # Generate anchors
        image_size = (images.shape[2], images.shape[3])
        anchors = self.anchor_generator(features, image_size)
        
        # Get predictions
        cls_logits, box_regression = self.head(features)
        
        if self.training and targets is not None:
            # Compute losses
            return self.loss_fn(cls_logits, box_regression, anchors, targets)
        else:
            # Decode predictions
            return self.postprocess(cls_logits, box_regression, anchors, image_size)
    
    def postprocess(
        self,
        cls_logits: torch.Tensor,
        box_regression: torch.Tensor,
        anchors: torch.Tensor,
        image_size: Tuple[int, int]
    ) -> Dict[str, torch.Tensor]:
        """
        Post-process predictions to get final detections.
        
        Args:
            cls_logits: (B, N, num_classes) classification logits
            box_regression: (B, N, 5) box deltas
            anchors: (N, 5) anchors
            image_size: (H, W) image dimensions
            
        Returns:
            Dict with 'boxes', 'scores', 'labels' for each image
        """
        device = cls_logits.device
        batch_size = cls_logits.shape[0]
        
        results = {
            'boxes': [],
            'scores': [],
            'labels': []
        }
        
        for b in range(batch_size):
            # Get scores
            scores = torch.sigmoid(cls_logits[b])  # (N, num_classes)
            
            # Decode boxes
            boxes = decode_boxes(anchors, box_regression[b])  # (N, 5)
            
            # Filter by score threshold
            max_scores, max_labels = scores.max(dim=1)
            keep_mask = max_scores > self.score_threshold
            
            if keep_mask.sum() == 0:
                results['boxes'].append(torch.zeros(0, 5, device=device))
                results['scores'].append(torch.zeros(0, device=device))
                results['labels'].append(torch.zeros(0, dtype=torch.long, device=device))
                continue
            
            filtered_boxes = boxes[keep_mask]
            filtered_scores = max_scores[keep_mask]
            filtered_labels = max_labels[keep_mask]
            
            # Apply NMS
            keep_indices = rotated_nms(
                filtered_boxes.cpu().numpy(),
                filtered_scores.cpu().numpy(),
                self.nms_threshold
            )
            
            # Limit detections
            if len(keep_indices) > self.max_detections:
                keep_indices = keep_indices[:self.max_detections]
            
            keep_indices = torch.from_numpy(keep_indices).to(device)
            
            results['boxes'].append(filtered_boxes[keep_indices])
            results['scores'].append(filtered_scores[keep_indices])
            results['labels'].append(filtered_labels[keep_indices])
        
        return results
    
    def get_detections(
        self,
        images: torch.Tensor
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Get detections for a batch of images.
        
        Args:
            images: Input images (B, C, H, W)
            
        Returns:
            List of dicts with 'boxes', 'scores', 'labels' per image
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(images)
        
        # Convert to list of dicts
        batch_size = len(outputs['boxes'])
        detections = []
        for b in range(batch_size):
            detections.append({
                'boxes': outputs['boxes'][b],
                'scores': outputs['scores'][b],
                'labels': outputs['labels'][b]
            })
        
        return detections


def build_rotated_retinanet(
    num_classes: int,
    backbone: str = 'resnet50',
    pretrained: bool = True,
    **kwargs
) -> RotatedRetinaNet:
    """
    Build Rotated RetinaNet model.
    
    Args:
        num_classes: Number of object classes
        backbone: Backbone name
        pretrained: Use pretrained weights
        **kwargs: Additional model parameters
        
    Returns:
        RotatedRetinaNet model
    """
    return RotatedRetinaNet(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        **kwargs
    )
