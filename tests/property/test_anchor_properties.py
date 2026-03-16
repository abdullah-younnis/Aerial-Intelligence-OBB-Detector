"""Property-based tests for anchor generator.

**Validates: Requirements 6.5**
"""

import torch
import numpy as np
from hypothesis import given, strategies as st, settings

from aerial_detection.models.anchor_generator import RotatedAnchorGenerator


# Feature: aerial-object-detection, Property 12: Anchor Format Validity
@settings(max_examples=100)
@given(
    image_h=st.integers(min_value=256, max_value=1024),
    image_w=st.integers(min_value=256, max_value=1024),
    num_ratios=st.integers(min_value=1, max_value=4),
    num_angles=st.integers(min_value=1, max_value=6)
)
def test_anchor_format_validity(image_h, image_w, num_ratios, num_angles):
    """
    Property 12: For any anchor configuration (scales, ratios, angles) 
    and feature map, all generated anchors SHALL have the format 
    (x_center, y_center, width, height, theta) with positive width/height 
    and theta in [-90°, 90°).
    
    **Validates: Requirements 6.5**
    """
    # Generate random but valid configuration
    ratios = [0.5 + i * 0.5 for i in range(num_ratios)]
    angles = [-90 + i * (180 / (num_angles + 1)) for i in range(1, num_angles + 1)]
    
    generator = RotatedAnchorGenerator(
        sizes=[32, 64, 128],
        aspect_ratios=ratios,
        angles=angles,
        strides=[8, 16, 32]
    )
    
    # Create dummy feature maps
    feature_maps = [
        torch.zeros(1, 256, image_h // 8, image_w // 8),
        torch.zeros(1, 256, image_h // 16, image_w // 16),
        torch.zeros(1, 256, image_h // 32, image_w // 32),
    ]
    
    anchors = generator(feature_maps, (image_h, image_w))
    
    # Check shape
    assert anchors.shape[1] == 5, f"Expected 5 columns, got {anchors.shape[1]}"
    
    # Check all anchors have positive width and height
    widths = anchors[:, 2]
    heights = anchors[:, 3]
    
    assert torch.all(widths > 0), f"Found non-positive widths: {widths[widths <= 0]}"
    assert torch.all(heights > 0), f"Found non-positive heights: {heights[heights <= 0]}"
    
    # Check theta is in valid range [-90, 90)
    thetas = anchors[:, 4]
    assert torch.all(thetas >= -90), f"Found theta < -90: {thetas[thetas < -90]}"
    assert torch.all(thetas < 90), f"Found theta >= 90: {thetas[thetas >= 90]}"
    
    # Check centers are within reasonable bounds (can extend slightly beyond image)
    x_centers = anchors[:, 0]
    y_centers = anchors[:, 1]
    
    # Centers should be positive (anchors start from stride/2)
    assert torch.all(x_centers > 0), f"Found non-positive x_center"
    assert torch.all(y_centers > 0), f"Found non-positive y_center"


# Additional property: Correct number of anchors
@settings(max_examples=50)
@given(
    feat_h=st.integers(min_value=4, max_value=32),
    feat_w=st.integers(min_value=4, max_value=32)
)
def test_anchor_count(feat_h, feat_w):
    """Test that the correct number of anchors is generated."""
    ratios = [0.5, 1.0, 2.0]
    angles = [-60, -30, 0, 30, 60]
    
    generator = RotatedAnchorGenerator(
        sizes=[64],
        aspect_ratios=ratios,
        angles=angles,
        strides=[16]
    )
    
    feature_maps = [torch.zeros(1, 256, feat_h, feat_w)]
    anchors = generator(feature_maps, (feat_h * 16, feat_w * 16))
    
    expected_count = feat_h * feat_w * len(ratios) * len(angles)
    
    assert len(anchors) == expected_count, (
        f"Expected {expected_count} anchors, got {len(anchors)}"
    )


# Additional property: Anchors are evenly spaced
@settings(max_examples=50)
@given(
    feat_size=st.integers(min_value=4, max_value=16),
    stride=st.sampled_from([8, 16, 32])
)
def test_anchor_spacing(feat_size, stride):
    """Test that anchor centers are evenly spaced by stride."""
    generator = RotatedAnchorGenerator(
        sizes=[64],
        aspect_ratios=[1.0],
        angles=[0],
        strides=[stride]
    )
    
    feature_maps = [torch.zeros(1, 256, feat_size, feat_size)]
    anchors = generator(feature_maps, (feat_size * stride, feat_size * stride))
    
    # Get unique x and y centers
    x_centers = torch.unique(anchors[:, 0])
    y_centers = torch.unique(anchors[:, 1])
    
    # Check spacing
    if len(x_centers) > 1:
        x_diffs = x_centers[1:] - x_centers[:-1]
        assert torch.allclose(x_diffs, torch.tensor(float(stride))), (
            f"X spacing not equal to stride {stride}: {x_diffs}"
        )
    
    if len(y_centers) > 1:
        y_diffs = y_centers[1:] - y_centers[:-1]
        assert torch.allclose(y_diffs, torch.tensor(float(stride))), (
            f"Y spacing not equal to stride {stride}: {y_diffs}"
        )


# Additional property: First anchor center is at stride/2
@settings(max_examples=20)
@given(stride=st.sampled_from([8, 16, 32, 64]))
def test_anchor_first_center(stride):
    """Test that first anchor center is at (stride/2, stride/2)."""
    generator = RotatedAnchorGenerator(
        sizes=[64],
        aspect_ratios=[1.0],
        angles=[0],
        strides=[stride]
    )
    
    feature_maps = [torch.zeros(1, 256, 4, 4)]
    anchors = generator(feature_maps, (4 * stride, 4 * stride))
    
    # First anchor should be at (stride/2, stride/2)
    expected_center = stride / 2 + stride * 0  # First cell
    
    first_x = anchors[0, 0].item()
    first_y = anchors[0, 1].item()
    
    assert np.isclose(first_x, expected_center), (
        f"First x_center {first_x} != expected {expected_center}"
    )
    assert np.isclose(first_y, expected_center), (
        f"First y_center {first_y} != expected {expected_center}"
    )
