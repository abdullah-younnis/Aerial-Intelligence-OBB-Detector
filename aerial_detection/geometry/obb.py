"""Oriented Bounding Box (OBB) class and conversion utilities."""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import cv2


class DegeneratePolygonError(Exception):
    """Raised when polygon has collinear or degenerate points."""
    pass


class InvalidOBBError(Exception):
    """Raised when OBB has invalid parameters."""
    pass


@dataclass
class OBB:
    """
    Oriented Bounding Box representation.
    
    Attributes:
        x_center: X coordinate of box center
        y_center: Y coordinate of box center
        width: Width of the box (always positive)
        height: Height of the box (always positive)
        theta: Rotation angle in degrees, range [-90, 90)
    """
    x_center: float
    y_center: float
    width: float
    height: float
    theta: float  # Angle in degrees, range [-90, 90)
    
    def __post_init__(self):
        """Validate OBB parameters after initialization."""
        if self.width <= 0 or self.height <= 0:
            raise InvalidOBBError(
                f"Width and height must be positive. Got width={self.width}, height={self.height}"
            )
        if not np.isfinite(self.x_center) or not np.isfinite(self.y_center):
            raise InvalidOBBError(
                f"Center coordinates must be finite. Got x={self.x_center}, y={self.y_center}"
            )
        if not np.isfinite(self.theta):
            raise InvalidOBBError(f"Theta must be finite. Got theta={self.theta}")
    
    def to_polygon(self) -> np.ndarray:
        """
        Convert OBB to 4 corner points.
        
        Returns:
            np.ndarray: Array of shape (4, 2) with corner coordinates.
                        Points are ordered: top-left, top-right, bottom-right, bottom-left
                        relative to the rotated box.
        """
        # Convert angle to radians
        theta_rad = np.deg2rad(self.theta)
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)
        
        # Half dimensions
        hw = self.width / 2
        hh = self.height / 2
        
        # Corner offsets before rotation (relative to center)
        # Order: top-left, top-right, bottom-right, bottom-left
        corners_local = np.array([
            [-hw, -hh],
            [hw, -hh],
            [hw, hh],
            [-hw, hh]
        ])
        
        # Rotation matrix
        rotation_matrix = np.array([
            [cos_t, -sin_t],
            [sin_t, cos_t]
        ])
        
        # Rotate corners and translate to center
        corners = corners_local @ rotation_matrix.T + np.array([self.x_center, self.y_center])
        
        return corners
    
    @classmethod
    def from_polygon(cls, polygon: np.ndarray) -> 'OBB':
        """
        Create OBB from polygon points using minimum-area enclosing rectangle.
        
        Args:
            polygon: Array of shape (4, 2) or (N, 2) with corner coordinates,
                    or flattened array of shape (8,) for 4 points.
        
        Returns:
            OBB: Oriented bounding box with normalized angle.
        
        Raises:
            DegeneratePolygonError: If polygon is degenerate (collinear points).
        """
        # Handle flattened input
        polygon = np.asarray(polygon, dtype=np.float32)
        if polygon.ndim == 1:
            if len(polygon) != 8:
                raise ValueError(f"Flattened polygon must have 8 values, got {len(polygon)}")
            polygon = polygon.reshape(4, 2)
        
        if polygon.shape[0] < 3:
            raise DegeneratePolygonError("Polygon must have at least 3 points")
        
        # Check for NaN/Inf
        if not np.all(np.isfinite(polygon)):
            raise DegeneratePolygonError("Polygon contains NaN or Inf values")
        
        # Use OpenCV's minAreaRect which returns ((cx, cy), (w, h), angle)
        # OpenCV angle is in range [-90, 0) for minAreaRect
        rect = cv2.minAreaRect(polygon.astype(np.float32))
        
        (cx, cy), (w, h), angle = rect
        
        # Handle degenerate case (zero area)
        if w <= 0 or h <= 0:
            raise DegeneratePolygonError(
                f"Polygon is degenerate (zero area). Got width={w}, height={h}"
            )
        
        # OpenCV's minAreaRect returns angle in [-90, 0) range
        # and width/height may be swapped depending on orientation
        # Normalize to our convention: theta in [-90, 90), width >= height
        obb = cls(
            x_center=float(cx),
            y_center=float(cy),
            width=float(w),
            height=float(h),
            theta=float(angle)
        )
        
        return obb.normalize_angle()
    
    def normalize_angle(self) -> 'OBB':
        """
        Normalize theta to [-90, 90) range.
        
        This ensures consistent representation by swapping width/height
        and adjusting angle when necessary.
        
        Returns:
            OBB: New OBB with normalized angle.
        """
        theta = self.theta
        width = self.width
        height = self.height
        
        # Normalize angle to [-180, 180) first
        while theta >= 180:
            theta -= 360
        while theta < -180:
            theta += 360
        
        # Now normalize to [-90, 90)
        while theta >= 90:
            theta -= 180
            width, height = height, width
        while theta < -90:
            theta += 180
            width, height = height, width
        
        return OBB(
            x_center=self.x_center,
            y_center=self.y_center,
            width=width,
            height=height,
            theta=theta
        )
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array [x_center, y_center, width, height, theta]."""
        return np.array([self.x_center, self.y_center, self.width, self.height, self.theta])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'OBB':
        """Create OBB from numpy array [x_center, y_center, width, height, theta]."""
        if len(arr) != 5:
            raise ValueError(f"Array must have 5 elements, got {len(arr)}")
        return cls(
            x_center=float(arr[0]),
            y_center=float(arr[1]),
            width=float(arr[2]),
            height=float(arr[3]),
            theta=float(arr[4])
        )
    
    def area(self) -> float:
        """Calculate the area of the OBB."""
        return self.width * self.height
    
    def __eq__(self, other: object) -> bool:
        """Check equality with tolerance for floating point comparison."""
        if not isinstance(other, OBB):
            return False
        return np.allclose(self.to_array(), other.to_array(), rtol=1e-5, atol=1e-8)


def obb_equivalent(obb1: OBB, obb2: OBB, tolerance: float = 1e-5) -> bool:
    """
    Check if two OBBs are equivalent (represent the same rectangle).
    
    Two OBBs are equivalent if they have the same center, area, and
    their polygons overlap completely.
    
    Args:
        obb1: First OBB
        obb2: Second OBB
        tolerance: Tolerance for floating point comparison
    
    Returns:
        bool: True if OBBs are equivalent
    """
    # Check centers match
    if not np.isclose(obb1.x_center, obb2.x_center, rtol=tolerance, atol=tolerance):
        return False
    if not np.isclose(obb1.y_center, obb2.y_center, rtol=tolerance, atol=tolerance):
        return False
    
    # Check areas match
    if not np.isclose(obb1.area(), obb2.area(), rtol=tolerance, atol=tolerance):
        return False
    
    # Check dimensions match (accounting for possible width/height swap)
    dims1 = sorted([obb1.width, obb1.height])
    dims2 = sorted([obb2.width, obb2.height])
    if not np.allclose(dims1, dims2, rtol=tolerance, atol=tolerance):
        return False
    
    # Check polygon corners match (order-independent)
    poly1 = obb1.to_polygon()
    poly2 = obb2.to_polygon()
    
    # Sort corners by angle from center for comparison
    def sort_corners(poly: np.ndarray) -> np.ndarray:
        center = poly.mean(axis=0)
        angles = np.arctan2(poly[:, 1] - center[1], poly[:, 0] - center[0])
        return poly[np.argsort(angles)]
    
    poly1_sorted = sort_corners(poly1)
    poly2_sorted = sort_corners(poly2)
    
    return np.allclose(poly1_sorted, poly2_sorted, rtol=tolerance, atol=tolerance)
