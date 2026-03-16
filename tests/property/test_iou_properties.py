"""Property-based tests for Rotated IoU.

**Validates: Requirements 3.2, 3.4, 3.5**
"""

import numpy as np
from hypothesis import given, strategies as st, settings

from aerial_detection.geometry import OBB, rotated_iou


# Strategy for generating valid OBB parameters
obb_strategy = st.fixed_dictionaries({
    'x': st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False),
    'y': st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False),
    'w': st.floats(min_value=1, max_value=500, allow_nan=False, allow_infinity=False),
    'h': st.floats(min_value=1, max_value=500, allow_nan=False, allow_infinity=False),
    'theta': st.floats(min_value=-89.99, max_value=89.99, allow_nan=False, allow_infinity=False)
})


def make_obb(params: dict) -> OBB:
    """Helper to create OBB from strategy params."""
    return OBB(
        x_center=params['x'],
        y_center=params['y'],
        width=params['w'],
        height=params['h'],
        theta=params['theta']
    )


# Feature: aerial-object-detection, Property 3: Rotated IoU Range Invariant
@settings(max_examples=100)
@given(params1=obb_strategy, params2=obb_strategy)
def test_rotated_iou_range_invariant(params1, params2):
    """
    Property 3: For any two valid OBBs, the computed Rotated IoU 
    SHALL be in the range [0.0, 1.0].
    
    **Validates: Requirements 3.2**
    """
    obb1 = make_obb(params1)
    obb2 = make_obb(params2)
    
    iou = rotated_iou(obb1, obb2)
    
    assert 0.0 <= iou <= 1.0, (
        f"IoU out of range: {iou}\n"
        f"  OBB1: {obb1}\n"
        f"  OBB2: {obb2}"
    )


# Feature: aerial-object-detection, Property 4: Rotated IoU Identity
@settings(max_examples=100)
@given(params=obb_strategy)
def test_rotated_iou_identity(params):
    """
    Property 4: For any valid OBB, the Rotated IoU of the OBB 
    with itself SHALL equal 1.0.
    
    **Validates: Requirements 3.4**
    """
    obb = make_obb(params)
    
    iou = rotated_iou(obb, obb)
    
    assert np.isclose(iou, 1.0, rtol=1e-5), (
        f"Self-IoU not 1.0: {iou}\n"
        f"  OBB: {obb}"
    )


# Feature: aerial-object-detection, Property 5: Rotated IoU Symmetry
@settings(max_examples=100)
@given(params1=obb_strategy, params2=obb_strategy)
def test_rotated_iou_symmetry(params1, params2):
    """
    Property 5: For any two valid OBBs A and B, 
    Rotated_IoU(A, B) SHALL equal Rotated_IoU(B, A).
    
    **Validates: Requirements 3.5**
    """
    obb1 = make_obb(params1)
    obb2 = make_obb(params2)
    
    iou_ab = rotated_iou(obb1, obb2)
    iou_ba = rotated_iou(obb2, obb1)
    
    assert np.isclose(iou_ab, iou_ba, rtol=1e-5), (
        f"IoU not symmetric: IoU(A,B)={iou_ab}, IoU(B,A)={iou_ba}\n"
        f"  OBB1: {obb1}\n"
        f"  OBB2: {obb2}"
    )


# Additional property: Non-overlapping boxes have IoU = 0
@settings(max_examples=100)
@given(
    x1=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
    y1=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
    x2=st.floats(min_value=500, max_value=600, allow_nan=False, allow_infinity=False),
    y2=st.floats(min_value=500, max_value=600, allow_nan=False, allow_infinity=False),
    w=st.floats(min_value=10, max_value=50, allow_nan=False, allow_infinity=False),
    h=st.floats(min_value=10, max_value=50, allow_nan=False, allow_infinity=False),
    theta=st.floats(min_value=-45, max_value=45, allow_nan=False, allow_infinity=False)
)
def test_non_overlapping_iou_zero(x1, y1, x2, y2, w, h, theta):
    """
    Test that non-overlapping boxes have IoU = 0.
    
    **Validates: Requirements 3.3**
    """
    # Create two boxes far apart (guaranteed non-overlapping)
    obb1 = OBB(x_center=x1, y_center=y1, width=w, height=h, theta=theta)
    obb2 = OBB(x_center=x2, y_center=y2, width=w, height=h, theta=theta)
    
    iou = rotated_iou(obb1, obb2)
    
    assert iou == 0.0, (
        f"Non-overlapping boxes have non-zero IoU: {iou}\n"
        f"  OBB1: {obb1}\n"
        f"  OBB2: {obb2}"
    )
