"""Property-based tests for Rotated NMS.

**Validates: Requirements 4.3, 4.5, 4.1**
"""

import numpy as np
from hypothesis import given, strategies as st, settings, assume

from aerial_detection.geometry import OBB, rotated_nms, rotated_iou


# Strategy for generating detection lists
@st.composite
def detection_list_strategy(draw, min_size=0, max_size=20):
    """Generate a list of detections (boxes and scores)."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    
    boxes = []
    scores = []
    
    for _ in range(n):
        x = draw(st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False))
        y = draw(st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False))
        w = draw(st.floats(min_value=10, max_value=200, allow_nan=False, allow_infinity=False))
        h = draw(st.floats(min_value=10, max_value=200, allow_nan=False, allow_infinity=False))
        theta = draw(st.floats(min_value=-89, max_value=89, allow_nan=False, allow_infinity=False))
        score = draw(st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False))
        
        boxes.append([x, y, w, h, theta])
        scores.append(score)
    
    return np.array(boxes, dtype=np.float32), np.array(scores, dtype=np.float32)


# Feature: aerial-object-detection, Property 6: Rotated NMS Subset Invariant
@settings(max_examples=100)
@given(
    detections=detection_list_strategy(min_size=0, max_size=15),
    iou_threshold=st.floats(min_value=0.1, max_value=0.9, allow_nan=False, allow_infinity=False)
)
def test_rotated_nms_subset_invariant(detections, iou_threshold):
    """
    Property 6: For any list of detections, applying Rotated NMS SHALL 
    return a subset of the original detection indices (no indices outside 
    the original range, no duplicates).
    
    **Validates: Requirements 4.3, 4.5**
    """
    boxes, scores = detections
    n = len(boxes)
    
    keep = rotated_nms(boxes, scores, iou_threshold)
    
    # Check that keep is a subset of valid indices
    assert all(0 <= idx < n for idx in keep), (
        f"Keep indices out of range: {keep}, n={n}"
    )
    
    # Check no duplicates
    assert len(keep) == len(set(keep)), (
        f"Duplicate indices in keep: {keep}"
    )
    
    # Check that kept count <= original count
    assert len(keep) <= n, (
        f"More kept than original: {len(keep)} > {n}"
    )


# Feature: aerial-object-detection, Property 7: Rotated NMS No High-Overlap Pairs
@settings(max_examples=100)
@given(
    detections=detection_list_strategy(min_size=2, max_size=10),
    iou_threshold=st.floats(min_value=0.3, max_value=0.7, allow_nan=False, allow_infinity=False)
)
def test_rotated_nms_no_high_overlap_pairs(detections, iou_threshold):
    """
    Property 7: For any list of detections and IoU threshold, after 
    applying Rotated NMS, no two kept detections SHALL have Rotated IoU 
    greater than the threshold.
    
    **Validates: Requirements 4.1**
    """
    boxes, scores = detections
    
    keep = rotated_nms(boxes, scores, iou_threshold)
    
    if len(keep) < 2:
        return  # Nothing to check with 0 or 1 detection
    
    # Check all pairs of kept detections
    for i in range(len(keep)):
        for j in range(i + 1, len(keep)):
            idx_i = keep[i]
            idx_j = keep[j]
            
            obb_i = OBB.from_array(boxes[idx_i])
            obb_j = OBB.from_array(boxes[idx_j])
            
            iou = rotated_iou(obb_i, obb_j)
            
            assert iou <= iou_threshold, (
                f"Kept detections have IoU > threshold: "
                f"IoU({idx_i}, {idx_j}) = {iou} > {iou_threshold}\n"
                f"  Box {idx_i}: {boxes[idx_i]}, score={scores[idx_i]}\n"
                f"  Box {idx_j}: {boxes[idx_j]}, score={scores[idx_j]}"
            )


# Additional property: Empty input returns empty output
@settings(max_examples=10)
@given(iou_threshold=st.floats(min_value=0.1, max_value=0.9, allow_nan=False, allow_infinity=False))
def test_empty_input_returns_empty(iou_threshold):
    """
    Test that empty detection list returns empty result.
    
    **Validates: Requirements 4.4**
    """
    boxes = np.array([], dtype=np.float32).reshape(0, 5)
    scores = np.array([], dtype=np.float32)
    
    keep = rotated_nms(boxes, scores, iou_threshold)
    
    assert len(keep) == 0, f"Expected empty result, got {keep}"


# Additional property: Single detection is always kept
@settings(max_examples=100)
@given(
    x=st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False),
    y=st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False),
    w=st.floats(min_value=10, max_value=200, allow_nan=False, allow_infinity=False),
    h=st.floats(min_value=10, max_value=200, allow_nan=False, allow_infinity=False),
    theta=st.floats(min_value=-89, max_value=89, allow_nan=False, allow_infinity=False),
    score=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
    iou_threshold=st.floats(min_value=0.1, max_value=0.9, allow_nan=False, allow_infinity=False)
)
def test_single_detection_kept(x, y, w, h, theta, score, iou_threshold):
    """Test that a single detection is always kept."""
    boxes = np.array([[x, y, w, h, theta]], dtype=np.float32)
    scores = np.array([score], dtype=np.float32)
    
    keep = rotated_nms(boxes, scores, iou_threshold)
    
    assert len(keep) == 1, f"Expected 1 kept, got {len(keep)}"
    assert keep[0] == 0, f"Expected index 0, got {keep[0]}"


# Additional property: Highest score is always kept (when unique)
@settings(max_examples=100)
@given(detections=detection_list_strategy(min_size=1, max_size=10))
def test_highest_score_kept(detections):
    """Test that the highest scoring detection is always kept (when score is unique)."""
    boxes, scores = detections
    
    if len(boxes) == 0:
        return
    
    # Find highest score and check if it's unique
    max_score = np.max(scores)
    max_score_indices = np.where(scores == max_score)[0]
    
    keep = rotated_nms(boxes, scores, iou_threshold=0.5)
    
    # At least one of the highest-scoring detections should be kept
    assert any(idx in keep for idx in max_score_indices), (
        f"No highest score detection in keep: {keep}\n"
        f"Max score indices: {max_score_indices}\n"
        f"Scores: {scores}"
    )
