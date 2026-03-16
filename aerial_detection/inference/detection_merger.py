"""Detection merger for combining results from multiple patches."""

from typing import List, Tuple
import numpy as np

from ..geometry import rotated_nms


class DetectionMerger:
    """
    Merges detections from multiple patches into final predictions.
    
    Transforms patch-relative coordinates to original image coordinates
    and applies NMS to eliminate duplicates from overlapping patches.
    """
    
    def __init__(
        self,
        nms_threshold: float = 0.5,
        score_threshold: float = 0.05
    ):
        """
        Initialize detection merger.
        
        Args:
            nms_threshold: IoU threshold for NMS
            score_threshold: Minimum score to keep detection
        """
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
    
    def merge(
        self,
        patch_detections: List[Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]],
        image_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Merge detections from all patches.
        
        Args:
            patch_detections: List of (boxes, scores, labels, x_offset, y_offset)
                boxes: (N, 5) OBB parameters in patch-local coordinates
                scores: (N,) confidence scores
                labels: (N,) class labels
                x_offset, y_offset: Patch offset in original image
            image_size: Original image (height, width)
            
        Returns:
            merged_boxes: (N, 5) OBB parameters in original image coordinates
            merged_scores: (N,) confidence scores
            merged_labels: (N,) class labels
        """
        if len(patch_detections) == 0:
            return (
                np.zeros((0, 5), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64)
            )
        
        all_boxes = []
        all_scores = []
        all_labels = []
        
        height, width = image_size
        
        for boxes, scores, labels, x_offset, y_offset in patch_detections:
            if len(boxes) == 0:
                continue
            
            # Transform to original image coordinates
            transformed_boxes = self.transform_to_original(
                boxes, x_offset, y_offset
            )
            
            # Filter by score threshold
            mask = scores >= self.score_threshold
            if mask.sum() == 0:
                continue
            
            # Clip boxes to image bounds (optional, for center coordinates)
            clipped_boxes = self.clip_boxes(transformed_boxes[mask], height, width)
            
            all_boxes.append(clipped_boxes)
            all_scores.append(scores[mask])
            all_labels.append(labels[mask])
        
        if len(all_boxes) == 0:
            return (
                np.zeros((0, 5), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64)
            )
        
        # Concatenate all detections
        merged_boxes = np.concatenate(all_boxes, axis=0)
        merged_scores = np.concatenate(all_scores, axis=0)
        merged_labels = np.concatenate(all_labels, axis=0)
        
        # Apply class-wise NMS
        keep_indices = self._class_wise_nms(
            merged_boxes, merged_scores, merged_labels
        )
        
        return (
            merged_boxes[keep_indices],
            merged_scores[keep_indices],
            merged_labels[keep_indices]
        )
    
    def transform_to_original(
        self,
        boxes: np.ndarray,
        x_offset: int,
        y_offset: int
    ) -> np.ndarray:
        """
        Transform boxes from patch-local to original image coordinates.
        
        Args:
            boxes: (N, 5) boxes in patch coordinates [x, y, w, h, theta]
            x_offset: X offset of patch
            y_offset: Y offset of patch
            
        Returns:
            (N, 5) boxes in original image coordinates
        """
        transformed = boxes.copy()
        transformed[:, 0] += x_offset  # x_center
        transformed[:, 1] += y_offset  # y_center
        return transformed
    
    def transform_to_patch(
        self,
        boxes: np.ndarray,
        x_offset: int,
        y_offset: int
    ) -> np.ndarray:
        """
        Transform boxes from original image to patch-local coordinates.
        
        Args:
            boxes: (N, 5) boxes in original coordinates
            x_offset: X offset of patch
            y_offset: Y offset of patch
            
        Returns:
            (N, 5) boxes in patch-local coordinates
        """
        transformed = boxes.copy()
        transformed[:, 0] -= x_offset
        transformed[:, 1] -= y_offset
        return transformed
    
    def clip_boxes(
        self,
        boxes: np.ndarray,
        height: int,
        width: int
    ) -> np.ndarray:
        """
        Clip box centers to image bounds.
        
        Args:
            boxes: (N, 5) boxes
            height, width: Image dimensions
            
        Returns:
            (N, 5) clipped boxes
        """
        clipped = boxes.copy()
        clipped[:, 0] = np.clip(clipped[:, 0], 0, width)
        clipped[:, 1] = np.clip(clipped[:, 1], 0, height)
        return clipped
    
    def _class_wise_nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        """
        Apply NMS separately for each class.
        
        Args:
            boxes: (N, 5) boxes
            scores: (N,) scores
            labels: (N,) class labels
            
        Returns:
            Indices of kept detections
        """
        unique_labels = np.unique(labels)
        keep_all = []
        
        for label in unique_labels:
            mask = labels == label
            indices = np.where(mask)[0]
            
            if len(indices) == 0:
                continue
            
            class_boxes = boxes[mask]
            class_scores = scores[mask]
            
            keep_relative = rotated_nms(
                class_boxes, class_scores, self.nms_threshold
            )
            keep_absolute = indices[keep_relative]
            keep_all.extend(keep_absolute.tolist())
        
        # Sort by score
        keep_all = np.array(keep_all, dtype=np.int64)
        if len(keep_all) > 0:
            keep_all = keep_all[np.argsort(scores[keep_all])[::-1]]
        
        return keep_all
