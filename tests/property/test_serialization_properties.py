"""Property-based tests for serialization.

**Validates: Requirements 13.3**
"""

import numpy as np
from hypothesis import given, strategies as st, settings

from aerial_detection.utils import Detection, ImagePredictions


# Strategy for generating valid Detection
detection_strategy = st.fixed_dictionaries({
    'class_name': st.sampled_from(['plane', 'ship', 'vehicle', 'bridge', 'harbor']),
    'confidence': st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    'x_center': st.floats(min_value=0, max_value=4000, allow_nan=False, allow_infinity=False),
    'y_center': st.floats(min_value=0, max_value=4000, allow_nan=False, allow_infinity=False),
    'width': st.floats(min_value=1, max_value=500, allow_nan=False, allow_infinity=False),
    'height': st.floats(min_value=1, max_value=500, allow_nan=False, allow_infinity=False),
    'angle': st.floats(min_value=-89.99, max_value=89.99, allow_nan=False, allow_infinity=False)
})


def make_detection(params: dict) -> Detection:
    """Create Detection from strategy params."""
    return Detection(**params)


# Feature: aerial-object-detection, Property 10: Detection JSON Round-Trip
@settings(max_examples=100)
@given(params=detection_strategy)
def test_detection_json_round_trip(params):
    """
    Property 10: For any valid Detection object, serializing to JSON 
    and deserializing back SHALL produce an equivalent Detection object.
    
    **Validates: Requirements 13.3**
    """
    original = make_detection(params)
    
    # Serialize to dict and back
    serialized = original.to_dict()
    recovered = Detection.from_dict(serialized)
    
    # Check all fields match
    assert recovered.class_name == original.class_name
    assert np.isclose(recovered.confidence, original.confidence, rtol=1e-6)
    assert np.isclose(recovered.x_center, original.x_center, rtol=1e-6)
    assert np.isclose(recovered.y_center, original.y_center, rtol=1e-6)
    assert np.isclose(recovered.width, original.width, rtol=1e-6)
    assert np.isclose(recovered.height, original.height, rtol=1e-6)
    assert np.isclose(recovered.angle, original.angle, rtol=1e-6)


# Additional property: ImagePredictions JSON round-trip
@settings(max_examples=50)
@given(
    image_path=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N'))),
    width=st.integers(min_value=100, max_value=10000),
    height=st.integers(min_value=100, max_value=10000),
    num_detections=st.integers(min_value=0, max_value=10)
)
def test_image_predictions_json_round_trip(image_path, width, height, num_detections):
    """Test ImagePredictions JSON serialization round-trip."""
    # Create detections
    detections = []
    for i in range(num_detections):
        detections.append(Detection(
            class_name=f'class_{i}',
            confidence=0.5 + i * 0.05,
            x_center=100.0 + i * 10,
            y_center=100.0 + i * 10,
            width=50.0,
            height=30.0,
            angle=i * 10.0 - 45
        ))
    
    original = ImagePredictions(
        image_path=image_path,
        image_width=width,
        image_height=height,
        detections=detections
    )
    
    # Serialize and deserialize
    json_str = original.to_json()
    recovered = ImagePredictions.from_json(json_str)
    
    # Check fields
    assert recovered.image_path == original.image_path
    assert recovered.image_width == original.image_width
    assert recovered.image_height == original.image_height
    assert len(recovered.detections) == len(original.detections)
    
    for orig_det, rec_det in zip(original.detections, recovered.detections):
        assert rec_det.class_name == orig_det.class_name
        assert np.isclose(rec_det.confidence, orig_det.confidence, rtol=1e-6)


# Additional property: Detection to_polygon produces 8 values
@settings(max_examples=100)
@given(params=detection_strategy)
def test_detection_to_polygon_format(params):
    """Test that to_polygon produces correct format."""
    det = make_detection(params)
    polygon = det.to_polygon()
    
    assert len(polygon) == 8, f"Expected 8 values, got {len(polygon)}"
    assert all(isinstance(v, float) for v in polygon)


# Additional property: Filter by confidence preserves valid detections
@settings(max_examples=50)
@given(
    threshold=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
)
def test_filter_by_confidence(threshold):
    """Test that filter_by_confidence works correctly."""
    detections = [
        Detection('a', 0.9, 100, 100, 50, 30, 0),
        Detection('b', 0.5, 200, 200, 50, 30, 0),
        Detection('c', 0.3, 300, 300, 50, 30, 0),
    ]
    
    preds = ImagePredictions('test.png', 1000, 1000, detections)
    filtered = preds.filter_by_confidence(threshold)
    
    # All kept detections should be above threshold
    for det in filtered.detections:
        assert det.confidence >= threshold
    
    # Count should match
    expected_count = sum(1 for d in detections if d.confidence >= threshold)
    assert len(filtered.detections) == expected_count
