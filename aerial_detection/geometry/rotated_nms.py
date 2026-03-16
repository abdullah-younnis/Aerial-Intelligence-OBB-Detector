"""Rotated Non-Maximum Suppression for oriented bounding boxes."""

import numpy as np

from .obb import OBB
from .rotated_iou import rotated_iou


def rotated_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5
) -> np.ndarray:
    """
    Perform Non-Maximum Suppression on rotated bounding boxes.
    
    Processes detections in descending order of confidence score,
    suppressing boxes that have IoU above threshold with higher-scoring boxes.
    
    Args:
        boxes: Array of shape (N, 5) with OBB parameters [x, y, w, h, theta]
        scores: Array of shape (N,) with confidence scores
        iou_threshold: IoU threshold for suppression (default 0.5)
        
    Returns:
        Array of indices of kept detections, sorted by descending score
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    
    if len(boxes) != len(scores):
        raise ValueError(
            f"boxes and scores must have same length. "
            f"Got {len(boxes)} boxes and {len(scores)} scores"
        )
    
    # Sort by score in descending order
    order = np.argsort(scores)[::-1]
    
    keep = []
    suppressed = set()
    
    for idx in order:
        if idx in suppressed:
            continue
        
        keep.append(idx)
        
        # Get current box as OBB
        current_obb = OBB.from_array(boxes[idx])
        
        # Check remaining boxes for suppression
        for other_idx in order:
            if other_idx in suppressed or other_idx == idx:
                continue
            if other_idx in keep:
                continue
            
            other_obb = OBB.from_array(boxes[other_idx])
            iou = rotated_iou(current_obb, other_obb)
            
            if iou > iou_threshold:
                suppressed.add(other_idx)
    
    return np.array(keep, dtype=np.int64)


def rotated_nms_per_class(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    iou_threshold: float = 0.5
) -> np.ndarray:
    """
    Perform class-wise Rotated NMS.
    
    NMS is applied separately for each class, then results are combined.
    
    Args:
        boxes: Array of shape (N, 5) with OBB parameters
        scores: Array of shape (N,) with confidence scores
        labels: Array of shape (N,) with class labels
        iou_threshold: IoU threshold for suppression
        
    Returns:
        Array of indices of kept detections
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    
    unique_labels = np.unique(labels)
    keep_all = []
    
    for label in unique_labels:
        mask = labels == label
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            continue
        
        class_boxes = boxes[mask]
        class_scores = scores[mask]
        
        keep_relative = rotated_nms(class_boxes, class_scores, iou_threshold)
        keep_absolute = indices[keep_relative]
        keep_all.extend(keep_absolute.tolist())
    
    # Sort by score
    keep_all = np.array(keep_all, dtype=np.int64)
    if len(keep_all) > 0:
        keep_all = keep_all[np.argsort(scores[keep_all])[::-1]]
    
    return keep_all


def batched_rotated_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    batch_indices: np.ndarray,
    iou_threshold: float = 0.5
) -> np.ndarray:
    """
    Perform Rotated NMS separately for each batch.
    
    Args:
        boxes: Array of shape (N, 5) with OBB parameters
        scores: Array of shape (N,) with confidence scores
        batch_indices: Array of shape (N,) with batch indices
        iou_threshold: IoU threshold for suppression
        
    Returns:
        Array of indices of kept detections
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    
    unique_batches = np.unique(batch_indices)
    keep_all = []
    
    for batch_idx in unique_batches:
        mask = batch_indices == batch_idx
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            continue
        
        batch_boxes = boxes[mask]
        batch_scores = scores[mask]
        
        keep_relative = rotated_nms(batch_boxes, batch_scores, iou_threshold)
        keep_absolute = indices[keep_relative]
        keep_all.extend(keep_absolute.tolist())
    
    return np.array(keep_all, dtype=np.int64)
