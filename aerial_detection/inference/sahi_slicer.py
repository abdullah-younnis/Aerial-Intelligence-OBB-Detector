"""SAHI (Slicing Aided Hyper Inference) slicer for large images."""

from typing import List, Tuple, Generator
import numpy as np


class SAHISlicer:
    """
    Slices large images into overlapping patches for inference.
    
    Implements SAHI-style slicing to process images larger than
    what can fit in GPU memory.
    """
    
    def __init__(
        self,
        slice_size: int = 1024,
        overlap_ratio: float = 0.25,
        pad_value: int = 0
    ):
        """
        Initialize SAHI slicer.
        
        Args:
            slice_size: Size of square slices
            overlap_ratio: Overlap ratio between adjacent slices (0-1)
            pad_value: Pixel value for padding edge slices
        """
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        self.pad_value = pad_value
        
        # Calculate stride
        self.stride = int(slice_size * (1 - overlap_ratio))
    
    def get_slice_coordinates(
        self,
        image_size: Tuple[int, int]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Get slice coordinates without loading image.
        
        Args:
            image_size: (height, width) of the image
            
        Returns:
            List of (x_start, y_start, x_end, y_end) tuples
        """
        height, width = image_size
        slices = []
        
        y = 0
        while y < height:
            x = 0
            while x < width:
                x_end = min(x + self.slice_size, width)
                y_end = min(y + self.slice_size, height)
                
                slices.append((x, y, x_end, y_end))
                
                x += self.stride
                if x_end == width:
                    break
            
            y += self.stride
            if y_end == height:
                break
        
        return slices
    
    def slice_image(
        self,
        image: np.ndarray
    ) -> List[Tuple[np.ndarray, int, int]]:
        """
        Slice image into overlapping patches.
        
        Args:
            image: Input image array (H, W, C) or (H, W)
            
        Returns:
            List of (patch, x_offset, y_offset) tuples
        """
        height, width = image.shape[:2]
        slice_coords = self.get_slice_coordinates((height, width))
        
        slices = []
        for x_start, y_start, x_end, y_end in slice_coords:
            patch = self._extract_patch(image, x_start, y_start, x_end, y_end)
            slices.append((patch, x_start, y_start))
        
        return slices
    
    def slice_image_lazy(
        self,
        image: np.ndarray
    ) -> Generator[Tuple[np.ndarray, int, int], None, None]:
        """
        Generator version of slice_image for memory efficiency.
        
        Yields:
            (patch, x_offset, y_offset) tuples
        """
        height, width = image.shape[:2]
        slice_coords = self.get_slice_coordinates((height, width))
        
        for x_start, y_start, x_end, y_end in slice_coords:
            patch = self._extract_patch(image, x_start, y_start, x_end, y_end)
            yield (patch, x_start, y_start)
    
    def _extract_patch(
        self,
        image: np.ndarray,
        x_start: int,
        y_start: int,
        x_end: int,
        y_end: int
    ) -> np.ndarray:
        """
        Extract a patch from image, padding if necessary.
        
        Args:
            image: Input image
            x_start, y_start: Top-left corner
            x_end, y_end: Bottom-right corner
            
        Returns:
            Patch of shape (slice_size, slice_size, C) or (slice_size, slice_size)
        """
        patch = image[y_start:y_end, x_start:x_end]
        
        # Pad if necessary
        pad_h = self.slice_size - (y_end - y_start)
        pad_w = self.slice_size - (x_end - x_start)
        
        if pad_h > 0 or pad_w > 0:
            if image.ndim == 3:
                pad_width = ((0, pad_h), (0, pad_w), (0, 0))
            else:
                pad_width = ((0, pad_h), (0, pad_w))
            
            patch = np.pad(
                patch,
                pad_width,
                mode='constant',
                constant_values=self.pad_value
            )
        
        return patch
    
    def num_slices(self, image_size: Tuple[int, int]) -> int:
        """Get the number of slices for an image size."""
        return len(self.get_slice_coordinates(image_size))
    
    def covers_pixel(
        self,
        image_size: Tuple[int, int],
        pixel: Tuple[int, int]
    ) -> bool:
        """
        Check if a pixel is covered by at least one slice.
        
        Args:
            image_size: (height, width)
            pixel: (y, x) pixel coordinates
            
        Returns:
            True if pixel is covered
        """
        py, px = pixel
        height, width = image_size
        
        if py < 0 or py >= height or px < 0 or px >= width:
            return False
        
        for x_start, y_start, x_end, y_end in self.get_slice_coordinates(image_size):
            if x_start <= px < x_end and y_start <= py < y_end:
                return True
        
        return False
    
    def all_pixels_covered(self, image_size: Tuple[int, int]) -> bool:
        """
        Verify that all pixels in the image are covered by at least one slice.
        
        Args:
            image_size: (height, width)
            
        Returns:
            True if all pixels are covered
        """
        height, width = image_size
        covered = np.zeros((height, width), dtype=bool)
        
        for x_start, y_start, x_end, y_end in self.get_slice_coordinates(image_size):
            covered[y_start:y_end, x_start:x_end] = True
        
        return covered.all()
