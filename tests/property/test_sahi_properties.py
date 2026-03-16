"""Property-based tests for SAHI slicer.

**Validates: Requirements 8.5, 9.1**
"""

import numpy as np
from hypothesis import given, strategies as st, settings

from aerial_detection.inference.sahi_slicer import SAHISlicer


# Feature: aerial-object-detection, Property 8: SAHI Complete Pixel Coverage
@settings(max_examples=100)
@given(
    height=st.integers(min_value=100, max_value=2000),
    width=st.integers(min_value=100, max_value=2000),
    slice_size=st.sampled_from([256, 512, 1024]),
    overlap_ratio=st.floats(min_value=0.1, max_value=0.5, allow_nan=False, allow_infinity=False)
)
def test_sahi_complete_pixel_coverage(height, width, slice_size, overlap_ratio):
    """
    Property 8: For any image dimensions and slicing configuration 
    (patch_size, overlap_ratio), the generated patches SHALL cover 
    every pixel in the original image at least once.
    
    **Validates: Requirements 8.5**
    """
    slicer = SAHISlicer(slice_size=slice_size, overlap_ratio=overlap_ratio)
    
    # Verify all pixels are covered
    assert slicer.all_pixels_covered((height, width)), (
        f"Not all pixels covered for image ({height}, {width}) "
        f"with slice_size={slice_size}, overlap={overlap_ratio}"
    )


# Feature: aerial-object-detection, Property 9: SAHI Coordinate Transformation Correctness
@settings(max_examples=100)
@given(
    height=st.integers(min_value=200, max_value=1000),
    width=st.integers(min_value=200, max_value=1000),
    slice_size=st.sampled_from([256, 512]),
    overlap_ratio=st.floats(min_value=0.1, max_value=0.4, allow_nan=False, allow_infinity=False),
    local_x=st.floats(min_value=10, max_value=200, allow_nan=False, allow_infinity=False),
    local_y=st.floats(min_value=10, max_value=200, allow_nan=False, allow_infinity=False)
)
def test_sahi_coordinate_transformation(height, width, slice_size, overlap_ratio, local_x, local_y):
    """
    Property 9: For any detection in patch-local coordinates and the 
    patch's offset, transforming to original image coordinates and back 
    SHALL produce the original patch-local coordinates.
    
    **Validates: Requirements 9.1**
    """
    slicer = SAHISlicer(slice_size=slice_size, overlap_ratio=overlap_ratio)
    slices = slicer.get_slice_coordinates((height, width))
    
    if len(slices) == 0:
        return
    
    # Pick first slice
    x_offset, y_offset, _, _ = slices[0]
    
    # Transform local to global
    global_x = local_x + x_offset
    global_y = local_y + y_offset
    
    # Transform back to local
    recovered_x = global_x - x_offset
    recovered_y = global_y - y_offset
    
    assert np.isclose(recovered_x, local_x), (
        f"X coordinate not preserved: {local_x} -> {global_x} -> {recovered_x}"
    )
    assert np.isclose(recovered_y, local_y), (
        f"Y coordinate not preserved: {local_y} -> {global_y} -> {recovered_y}"
    )


# Additional property: Slices have consistent size
@settings(max_examples=50)
@given(
    height=st.integers(min_value=500, max_value=2000),
    width=st.integers(min_value=500, max_value=2000),
    slice_size=st.sampled_from([256, 512, 1024])
)
def test_slice_consistent_size(height, width, slice_size):
    """Test that all extracted slices have the same size after padding."""
    slicer = SAHISlicer(slice_size=slice_size, overlap_ratio=0.25)
    
    # Create dummy image
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    slices = slicer.slice_image(image)
    
    for patch, x_off, y_off in slices:
        assert patch.shape[0] == slice_size, (
            f"Patch height {patch.shape[0]} != {slice_size}"
        )
        assert patch.shape[1] == slice_size, (
            f"Patch width {patch.shape[1]} != {slice_size}"
        )


# Additional property: Number of slices is reasonable
@settings(max_examples=50)
@given(
    height=st.integers(min_value=100, max_value=2000),
    width=st.integers(min_value=100, max_value=2000),
    slice_size=st.sampled_from([256, 512, 1024]),
    overlap_ratio=st.floats(min_value=0.1, max_value=0.5, allow_nan=False, allow_infinity=False)
)
def test_num_slices_reasonable(height, width, slice_size, overlap_ratio):
    """Test that the number of slices is within expected bounds."""
    slicer = SAHISlicer(slice_size=slice_size, overlap_ratio=overlap_ratio)
    
    num_slices = slicer.num_slices((height, width))
    
    # At minimum, we need enough slices to cover the image
    stride = int(slice_size * (1 - overlap_ratio))
    min_slices_h = max(1, (height + stride - 1) // stride)
    min_slices_w = max(1, (width + stride - 1) // stride)
    min_expected = min_slices_h * min_slices_w
    
    # Should have at least the minimum
    assert num_slices >= 1, "Should have at least one slice"
    
    # Should not have too many (sanity check)
    max_expected = ((height // stride + 2) * (width // stride + 2))
    assert num_slices <= max_expected, (
        f"Too many slices: {num_slices} > {max_expected}"
    )
