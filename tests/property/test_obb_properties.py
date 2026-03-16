"""Property-based tests for OBB operations.
Summary of What These Tests Guarantee:

Geometry conversions are stable
Angles are normalized
Area is preserved
Polygon format is correct
No invalid numbers

**Validates: Requirements 2.3, 2.5**
"""

import numpy as np
from hypothesis import given, strategies as st, settings, assume

from aerial_detection.geometry import OBB, obb_equivalent


# Strategy for generating valid OBB parameters
obb_strategy = st.fixed_dictionaries({
    'x': st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False),
    'y': st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False),
    'w': st.floats(min_value=1, max_value=500, allow_nan=False, allow_infinity=False),
    'h': st.floats(min_value=1, max_value=500, allow_nan=False, allow_infinity=False),
    'theta': st.floats(min_value=-89.99, max_value=89.99, allow_nan=False, allow_infinity=False)
})


# Feature: aerial-object-detection, Property 1: OBB Polygon Round-Trip
@settings(max_examples=100)
@given(params=obb_strategy)
def test_obb_polygon_round_trip(params):
    """
    Property 1: OBB → polygon → OBB produces equivalent OBB.
    
    For any valid OBB (x_center, y_center, width, height, theta), 
    converting to polygon format and back to OBB SHALL produce 
    an equivalent OBB (within floating-point tolerance).
    
    **Validates: Requirements 2.3**
    """
    original = OBB(
        x_center=params['x'],
        y_center=params['y'],
        width=params['w'],
        height=params['h'],
        theta=params['theta']
    )
    
    # Convert to polygon
    polygon = original.to_polygon()
    
    # Convert back to OBB
    recovered = OBB.from_polygon(polygon)
    
    # Check equivalence
    assert obb_equivalent(original, recovered, tolerance=1e-4), (
        f"Round-trip failed:\n"
        f"  Original: {original}\n"
        f"  Recovered: {recovered}\n"
        f"  Original polygon: {polygon}\n"
    )


# Feature: aerial-object-detection, Property 2: OBB Angle Normalization
@settings(max_examples=100)
@given(params=obb_strategy)
def test_obb_angle_normalization(params):
    """
    Property 2: For any polygon converted to OBB, the resulting theta 
    SHALL be in the range [-90°, 90°).
    
    **Validates: Requirements 2.5**
    """
    original = OBB(
        x_center=params['x'],
        y_center=params['y'],
        width=params['w'],
        height=params['h'],
        theta=params['theta']
    )
    
    # Convert to polygon and back
    polygon = original.to_polygon()
    recovered = OBB.from_polygon(polygon)
    
    # Check angle is in valid range
    assert -90 <= recovered.theta < 90, (
        f"Angle not normalized: theta={recovered.theta}\n"
        f"  Original: {original}\n"
        f"  Recovered: {recovered}"
    )


# Additional property: Angle normalization for arbitrary angles
@settings(max_examples=100)
@given(
    x=st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False),
    y=st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False),
    w=st.floats(min_value=1, max_value=500, allow_nan=False, allow_infinity=False),
    h=st.floats(min_value=1, max_value=500, allow_nan=False, allow_infinity=False),
    theta=st.floats(min_value=-360, max_value=360, allow_nan=False, allow_infinity=False)
)
def test_normalize_angle_range(x, y, w, h, theta):
    """
    Test that normalize_angle always produces theta in [-90, 90).
    
    **Validates: Requirements 2.5**
    """
    # Create OBB with arbitrary angle (bypass validation by creating directly)
    obb = OBB.__new__(OBB)
    obb.x_center = x
    obb.y_center = y
    obb.width = w
    obb.height = h
    obb.theta = theta
    
    # Normalize
    normalized = obb.normalize_angle()
    
    # Check range
    assert -90 <= normalized.theta < 90, (
        f"Normalized angle out of range: {normalized.theta} (original: {theta})"
    )
    
    # Check that dimensions are still positive
    assert normalized.width > 0
    assert normalized.height > 0


# Property: Polygon has correct number of points
@settings(max_examples=100)
@given(params=obb_strategy)
def test_polygon_shape(params):
    """Test that to_polygon returns exactly 4 corner points."""
    obb = OBB(
        x_center=params['x'],
        y_center=params['y'],
        width=params['w'],
        height=params['h'],
        theta=params['theta']
    )
    
    polygon = obb.to_polygon()
    
    assert polygon.shape == (4, 2), f"Expected shape (4, 2), got {polygon.shape}"
    assert np.all(np.isfinite(polygon)), "Polygon contains non-finite values"


# Property: Area preservation through round-trip
@settings(max_examples=100)
@given(params=obb_strategy)
def test_area_preservation(params):
    """Test that area is preserved through polygon round-trip."""
    original = OBB(
        x_center=params['x'],
        y_center=params['y'],
        width=params['w'],
        height=params['h'],
        theta=params['theta']
    )
    
    polygon = original.to_polygon()
    recovered = OBB.from_polygon(polygon)
    
    assert np.isclose(original.area(), recovered.area(), rtol=1e-4), (
        f"Area not preserved: original={original.area()}, recovered={recovered.area()}"
    )
