"""Rotated anchor generator for object detection."""

from typing import List, Tuple
import torch
import torch.nn as nn
import numpy as np


class RotatedAnchorGenerator(nn.Module):
    """
    Generates rotated anchors at multiple scales, ratios, and angles.
    
    Anchors are generated for each spatial location in feature maps
    at multiple FPN levels.
    """
    
    def __init__(
        self,
        sizes: List[int] = [32, 64, 128, 256, 512],
        aspect_ratios: List[float] = [0.5, 1.0, 2.0],
        angles: List[float] = [-90, -60, -30, 0, 30, 60],
        strides: List[int] = [8, 16, 32, 64, 128]
    ):
        """
        Initialize rotated anchor generator.
        
        Args:
            sizes: Base anchor sizes for each FPN level
            aspect_ratios: Width/height ratios
            angles: Rotation angles in degrees
            strides: Feature map strides for each FPN level
        """
        super().__init__()
        
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.angles = angles
        self.strides = strides
        
        # Number of anchors per location
        self.num_anchors_per_location = len(aspect_ratios) * len(angles)
        
        # Pre-compute base anchors for each level
        self.register_buffer(
            'base_anchors',
            self._generate_base_anchors()
        )
    
    def _generate_base_anchors(self) -> torch.Tensor:
        """
        Generate base anchors (centered at origin).
        
        Returns:
            Tensor of shape (num_levels, num_anchors_per_loc, 5)
            where 5 = (x_center, y_center, width, height, theta)
        """
        num_levels = len(self.sizes)
        num_anchors = self.num_anchors_per_location
        
        base_anchors = torch.zeros(num_levels, num_anchors, 5)
        
        for level_idx, size in enumerate(self.sizes):
            anchor_idx = 0
            for ratio in self.aspect_ratios:
                # Compute width and height for this ratio
                # area = size^2, w/h = ratio
                # w * h = size^2, w = ratio * h
                # ratio * h * h = size^2
                h = size / np.sqrt(ratio)
                w = size * np.sqrt(ratio)
                
                for angle in self.angles:
                    base_anchors[level_idx, anchor_idx] = torch.tensor([
                        0.0,  # x_center (will be shifted)
                        0.0,  # y_center (will be shifted)
                        w,
                        h,
                        angle
                    ])
                    anchor_idx += 1
        
        return base_anchors
    
    def forward(
        self,
        feature_maps: List[torch.Tensor],
        image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Generate anchors for all feature map levels.
        
        Args:
            feature_maps: List of feature tensors from FPN
            image_size: (height, width) of input image
            
        Returns:
            Tensor of shape (total_anchors, 5) with (x, y, w, h, theta)
        """
        device = feature_maps[0].device
        dtype = feature_maps[0].dtype
        
        all_anchors = []
        
        for level_idx, feature_map in enumerate(feature_maps):
            _, _, feat_h, feat_w = feature_map.shape
            stride = self.strides[level_idx]
            
            # Generate grid of anchor centers
            shifts_x = (torch.arange(feat_w, device=device, dtype=dtype) + 0.5) * stride
            shifts_y = (torch.arange(feat_h, device=device, dtype=dtype) + 0.5) * stride
            
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            
            # Get base anchors for this level
            base = self.base_anchors[level_idx].to(device=device, dtype=dtype)
            
            # Broadcast: (num_locations, 1, 5) + (1, num_anchors, 5)
            num_locations = len(shift_x)
            
            # Create shifts tensor
            shifts = torch.zeros(num_locations, 1, 5, device=device, dtype=dtype)
            shifts[:, 0, 0] = shift_x
            shifts[:, 0, 1] = shift_y
            
            # Add shifts to base anchors
            anchors = base.unsqueeze(0) + shifts  # (num_locations, num_anchors, 5)
            anchors = anchors.reshape(-1, 5)  # (num_locations * num_anchors, 5)
            
            all_anchors.append(anchors)
        
        return torch.cat(all_anchors, dim=0)
    
    def num_anchors_per_level(self, feature_maps: List[torch.Tensor]) -> List[int]:
        """Get number of anchors at each FPN level."""
        counts = []
        for feature_map in feature_maps:
            _, _, h, w = feature_map.shape
            counts.append(h * w * self.num_anchors_per_location)
        return counts


def generate_anchors_for_image(
    image_size: Tuple[int, int],
    sizes: List[int] = [32, 64, 128, 256, 512],
    aspect_ratios: List[float] = [0.5, 1.0, 2.0],
    angles: List[float] = [-90, -60, -30, 0, 30, 60],
    strides: List[int] = [8, 16, 32, 64, 128]
) -> torch.Tensor:
    """
    Generate all anchors for an image without feature maps.
    
    Useful for visualization and debugging.
    
    Args:
        image_size: (height, width)
        sizes, aspect_ratios, angles, strides: Anchor parameters
        
    Returns:
        Tensor of shape (total_anchors, 5)
    """
    generator = RotatedAnchorGenerator(sizes, aspect_ratios, angles, strides)
    
    h, w = image_size
    
    # Create dummy feature maps
    feature_maps = []
    for stride in strides:
        feat_h = (h + stride - 1) // stride
        feat_w = (w + stride - 1) // stride
        feature_maps.append(torch.zeros(1, 1, feat_h, feat_w))
    
    return generator(feature_maps, image_size)
