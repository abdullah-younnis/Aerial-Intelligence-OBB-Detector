"""Rotated IoU calculation for oriented bounding boxes."""

import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid
from typing import Union

from .obb import OBB


def rotated_iou(obb1: OBB, obb2: OBB) -> float:
    """
    Calculate Intersection over Union for two oriented bounding boxes.
    
    Uses Shapely polygon operations to compute intersection area.
    
    Args:
        obb1: First oriented bounding box
        obb2: Second oriented bounding box
        
    Returns:
        IoU value in range [0.0, 1.0]
    """
    # Convert OBBs to polygons
    poly1_coords = obb1.to_polygon()
    poly2_coords = obb2.to_polygon()
    
    # Create Shapely polygons
    poly1 = Polygon(poly1_coords)
    poly2 = Polygon(poly2_coords)
    
    # Ensure polygons are valid
    if not poly1.is_valid:
        poly1 = make_valid(poly1)
    if not poly2.is_valid:
        poly2 = make_valid(poly2)
    
    # Calculate intersection
    try:
        intersection = poly1.intersection(poly2)
        intersection_area = intersection.area
    except Exception:
        # Handle edge cases where intersection fails
        intersection_area = 0.0
    
    # Calculate union
    area1 = poly1.area
    area2 = poly2.area
    union_area = area1 + area2 - intersection_area
    
    # Avoid division by zero
    if union_area <= 0:
        return 0.0
    
    iou = intersection_area / union_area
    
    # Clamp to [0, 1] to handle floating point errors
    return float(np.clip(iou, 0.0, 1.0))


def rotated_iou_batch(
    obbs1: np.ndarray, 
    obbs2: np.ndarray
) -> np.ndarray:
    """
    Calculate pairwise Rotated IoU for batches of OBBs.
    
    Args:
        obbs1: Array of shape (N, 5) with OBB parameters [x, y, w, h, theta]
        obbs2: Array of shape (M, 5) with OBB parameters [x, y, w, h, theta]
        
    Returns:
        IoU matrix of shape (N, M) where result[i, j] = IoU(obbs1[i], obbs2[j])
    """
    n = len(obbs1)
    m = len(obbs2)
    
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float32)
    
    # Convert arrays to OBB objects
    obb_list1 = [OBB.from_array(obbs1[i]) for i in range(n)]
    obb_list2 = [OBB.from_array(obbs2[j]) for j in range(m)]
    
    # Compute pairwise IoU
    iou_matrix = np.zeros((n, m), dtype=np.float32)
    for i, obb1 in enumerate(obb_list1):
        for j, obb2 in enumerate(obb_list2):
            iou_matrix[i, j] = rotated_iou(obb1, obb2)
    
    return iou_matrix


def rotated_iou_single_vs_batch(
    obb: Union[OBB, np.ndarray],
    obbs: np.ndarray
) -> np.ndarray:
    """
    Calculate IoU between a single OBB and a batch of OBBs.
    
    Args:
        obb: Single OBB or array of shape (5,)
        obbs: Array of shape (N, 5) with OBB parameters
        
    Returns:
        Array of shape (N,) with IoU values
    """
    if isinstance(obb, np.ndarray):
        obb = OBB.from_array(obb)
    
    n = len(obbs)
    if n == 0:
        return np.array([], dtype=np.float32)
    
    ious = np.zeros(n, dtype=np.float32)
    for i in range(n):
        obb2 = OBB.from_array(obbs[i])
        ious[i] = rotated_iou(obb, obb2)
    
    return ious
