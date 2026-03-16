"""Geometry module for OBB operations, Rotated IoU, and Rotated NMS."""

from .obb import OBB, DegeneratePolygonError, InvalidOBBError, obb_equivalent
from .rotated_iou import rotated_iou, rotated_iou_batch, rotated_iou_single_vs_batch
from .rotated_nms import rotated_nms, rotated_nms_per_class, batched_rotated_nms

__all__ = [
    "OBB",
    "DegeneratePolygonError",
    "InvalidOBBError",
    "obb_equivalent",
    "rotated_iou",
    "rotated_iou_batch",
    "rotated_iou_single_vs_batch",
    "rotated_nms",
    "rotated_nms_per_class",
    "batched_rotated_nms",
]
