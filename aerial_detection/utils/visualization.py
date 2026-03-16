"""Visualization utilities for rotated object detection."""

from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from ..geometry import OBB
from .io import Detection, ImagePredictions


# Default color palette for DOTA classes
DEFAULT_COLORS = {
    'plane': (255, 0, 0),
    'ship': (0, 255, 0),
    'storage-tank': (0, 0, 255),
    'baseball-diamond': (255, 255, 0),
    'tennis-court': (255, 0, 255),
    'basketball-court': (0, 255, 255),
    'ground-track-field': (128, 0, 0),
    'harbor': (0, 128, 0),
    'bridge': (0, 0, 128),
    'large-vehicle': (128, 128, 0),
    'small-vehicle': (128, 0, 128),
    'helicopter': (0, 128, 128),
    'roundabout': (64, 0, 0),
    'soccer-ball-field': (0, 64, 0),
    'swimming-pool': (0, 0, 64),
    'container-crane': (64, 64, 0),
    'airport': (64, 0, 64),
    'helipad': (0, 64, 64),
}


def get_color(class_name: str, color_map: Optional[Dict[str, Tuple[int, int, int]]] = None) -> Tuple[int, int, int]:
    """Get color for a class name."""
    if color_map and class_name in color_map:
        return color_map[class_name]
    if class_name in DEFAULT_COLORS:
        return DEFAULT_COLORS[class_name]
    # Generate deterministic color from class name
    hash_val = hash(class_name)
    return (
        (hash_val & 0xFF),
        ((hash_val >> 8) & 0xFF),
        ((hash_val >> 16) & 0xFF)
    )


def draw_obb(
    image: np.ndarray,
    obb: OBB,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    label: Optional[str] = None,
    font_scale: float = 0.5,
    font_thickness: int = 1
) -> np.ndarray:
    """
    Draw an oriented bounding box on an image.
    
    Args:
        image: Input image (BGR or RGB)
        obb: Oriented bounding box to draw
        color: Line color (BGR)
        thickness: Line thickness
        label: Optional label to display
        font_scale: Font scale for label
        font_thickness: Font thickness for label
        
    Returns:
        Image with OBB drawn
    """
    # Get polygon corners
    polygon = obb.to_polygon()  # Shape: (4, 2)
    pts = polygon.astype(np.int32)
    
    # Draw polygon
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)
    
    # Draw label if provided
    if label:
        # Find top-left corner for label placement
        min_y_idx = np.argmin(pts[:, 1])
        label_pos = (int(pts[min_y_idx, 0]), int(pts[min_y_idx, 1]) - 5)
        
        # Draw background rectangle for label
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        cv2.rectangle(
            image,
            (label_pos[0], label_pos[1] - text_h - baseline),
            (label_pos[0] + text_w, label_pos[1]),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            image, label, label_pos,
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            (255, 255, 255), font_thickness
        )
    
    return image


def draw_detection(
    image: np.ndarray,
    detection: Detection,
    color: Optional[Tuple[int, int, int]] = None,
    thickness: int = 2,
    show_confidence: bool = True,
    font_scale: float = 0.5,
    font_thickness: int = 1,
    color_map: Optional[Dict[str, Tuple[int, int, int]]] = None
) -> np.ndarray:
    """
    Draw a detection on an image.
    
    Args:
        image: Input image
        detection: Detection to draw
        color: Optional override color
        thickness: Line thickness
        show_confidence: Whether to show confidence score
        font_scale: Font scale for label
        font_thickness: Font thickness for label
        color_map: Optional class name to color mapping
        
    Returns:
        Image with detection drawn
    """
    if color is None:
        color = get_color(detection.class_name, color_map)
    
    obb = detection.to_obb()
    
    if show_confidence:
        label = f'{detection.class_name}: {detection.confidence:.2f}'
    else:
        label = detection.class_name
    
    return draw_obb(
        image, obb, color, thickness, label, font_scale, font_thickness
    )


def visualize_detections(
    image: np.ndarray,
    detections: Union[List[Detection], ImagePredictions],
    thickness: int = 2,
    show_confidence: bool = True,
    font_scale: float = 0.5,
    font_thickness: int = 1,
    color_map: Optional[Dict[str, Tuple[int, int, int]]] = None,
    confidence_threshold: float = 0.0
) -> np.ndarray:
    """
    Visualize all detections on an image.
    
    Args:
        image: Input image (will be copied)
        detections: List of Detection objects or ImagePredictions
        thickness: Line thickness
        show_confidence: Whether to show confidence scores
        font_scale: Font scale for labels
        font_thickness: Font thickness for labels
        color_map: Optional class name to color mapping
        confidence_threshold: Minimum confidence to display
        
    Returns:
        Image with all detections drawn
    """
    # Make a copy to avoid modifying original
    vis_image = image.copy()
    
    # Handle ImagePredictions
    if isinstance(detections, ImagePredictions):
        det_list = detections.detections
    else:
        det_list = detections
    
    # Filter by confidence
    det_list = [d for d in det_list if d.confidence >= confidence_threshold]
    
    # Sort by confidence (draw lower confidence first so higher ones are on top)
    det_list = sorted(det_list, key=lambda x: x.confidence)
    
    for det in det_list:
        vis_image = draw_detection(
            vis_image, det,
            thickness=thickness,
            show_confidence=show_confidence,
            font_scale=font_scale,
            font_thickness=font_thickness,
            color_map=color_map
        )
    
    return vis_image


def visualize_ground_truth(
    image: np.ndarray,
    boxes: np.ndarray,
    class_names: List[str],
    thickness: int = 2,
    font_scale: float = 0.5,
    font_thickness: int = 1,
    color_map: Optional[Dict[str, Tuple[int, int, int]]] = None
) -> np.ndarray:
    """
    Visualize ground truth annotations.
    
    Args:
        image: Input image (will be copied)
        boxes: Array of shape (N, 5) with [x, y, w, h, theta]
        class_names: List of class names
        thickness: Line thickness
        font_scale: Font scale for labels
        font_thickness: Font thickness for labels
        color_map: Optional class name to color mapping
        
    Returns:
        Image with ground truth drawn
    """
    vis_image = image.copy()
    
    for box, cls_name in zip(boxes, class_names):
        obb = OBB(box[0], box[1], box[2], box[3], box[4])
        color = get_color(cls_name, color_map)
        vis_image = draw_obb(
            vis_image, obb, color, thickness, cls_name, font_scale, font_thickness
        )
    
    return vis_image


def visualize_comparison(
    image: np.ndarray,
    predictions: Union[List[Detection], ImagePredictions],
    ground_truths: List[Tuple[np.ndarray, str]],
    pred_thickness: int = 2,
    gt_thickness: int = 2,
    show_confidence: bool = True,
    font_scale: float = 0.5,
    confidence_threshold: float = 0.0
) -> np.ndarray:
    """
    Visualize predictions vs ground truth side by side.
    
    Args:
        image: Input image
        predictions: Predicted detections
        ground_truths: List of (box, class_name) tuples
        pred_thickness: Prediction line thickness
        gt_thickness: Ground truth line thickness
        show_confidence: Whether to show confidence scores
        font_scale: Font scale for labels
        confidence_threshold: Minimum confidence to display
        
    Returns:
        Side-by-side comparison image
    """
    # Create prediction visualization
    pred_image = visualize_detections(
        image, predictions,
        thickness=pred_thickness,
        show_confidence=show_confidence,
        font_scale=font_scale,
        confidence_threshold=confidence_threshold
    )
    
    # Create ground truth visualization
    gt_image = image.copy()
    for box, cls_name in ground_truths:
        obb = OBB(box[0], box[1], box[2], box[3], box[4])
        color = get_color(cls_name)
        gt_image = draw_obb(gt_image, obb, color, gt_thickness, cls_name, font_scale)
    
    # Add labels
    cv2.putText(pred_image, 'Predictions', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(gt_image, 'Ground Truth', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Concatenate horizontally
    return np.concatenate([pred_image, gt_image], axis=1)


def save_visualization(
    image: np.ndarray,
    output_path: str,
    detections: Optional[Union[List[Detection], ImagePredictions]] = None,
    **kwargs
):
    """
    Save visualization to file.
    
    Args:
        image: Input image
        output_path: Output file path
        detections: Optional detections to visualize
        **kwargs: Additional arguments for visualize_detections
    """
    if detections is not None:
        vis_image = visualize_detections(image, detections, **kwargs)
    else:
        vis_image = image
    
    # Convert RGB to BGR for OpenCV
    if len(vis_image.shape) == 3 and vis_image.shape[2] == 3:
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(output_path, vis_image)


def create_legend(
    class_names: List[str],
    width: int = 200,
    height_per_class: int = 30,
    color_map: Optional[Dict[str, Tuple[int, int, int]]] = None
) -> np.ndarray:
    """
    Create a legend image showing class colors.
    
    Args:
        class_names: List of class names
        width: Legend width
        height_per_class: Height per class entry
        color_map: Optional class name to color mapping
        
    Returns:
        Legend image
    """
    height = len(class_names) * height_per_class + 20
    legend = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    for i, cls_name in enumerate(class_names):
        y = 10 + i * height_per_class
        color = get_color(cls_name, color_map)
        
        # Draw color box
        cv2.rectangle(legend, (10, y), (30, y + 20), color, -1)
        cv2.rectangle(legend, (10, y), (30, y + 20), (0, 0, 0), 1)
        
        # Draw class name
        cv2.putText(legend, cls_name, (40, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    return legend
