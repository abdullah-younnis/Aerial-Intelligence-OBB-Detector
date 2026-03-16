"""Utility modules for aerial object detection."""

from .io import (
    Detection,
    ImagePredictions,
    save_predictions_batch,
    load_predictions_batch
)

from .visualization import (
    draw_obb,
    draw_detection,
    visualize_detections,
    visualize_ground_truth,
    visualize_comparison,
    save_visualization,
    create_legend,
    get_color,
    DEFAULT_COLORS
)

__all__ = [
    # I/O
    'Detection',
    'ImagePredictions',
    'save_predictions_batch',
    'load_predictions_batch',
    # Visualization
    'draw_obb',
    'draw_detection',
    'visualize_detections',
    'visualize_ground_truth',
    'visualize_comparison',
    'save_visualization',
    'create_legend',
    'get_color',
    'DEFAULT_COLORS'
]
