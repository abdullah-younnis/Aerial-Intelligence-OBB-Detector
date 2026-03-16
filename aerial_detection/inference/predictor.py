"""High-level predictor for inference on images."""

from typing import Dict, List, Optional, Union
from pathlib import Path
import numpy as np
import cv2
import torch

from .sahi_slicer import SAHISlicer
from .detection_merger import DetectionMerger
from ..models import RotatedRetinaNet
from ..config import IDX_TO_CLASS


class Predictor:
    """
    High-level predictor for running inference on images.
    
    Automatically uses SAHI slicing for large images.
    """
    
    def __init__(
        self,
        model: RotatedRetinaNet,
        device: str = 'cuda',
        score_threshold: float = 0.5,
        nms_threshold: float = 0.5,
        sahi_slice_size: int = 1024,
        sahi_overlap: float = 0.25,
        auto_sahi_threshold: int = 2048,
        class_names: Optional[Dict[int, str]] = None
    ):
        """
        Initialize predictor.
        
        Args:
            model: Trained RotatedRetinaNet model
            device: Device to run inference on
            score_threshold: Minimum confidence score
            nms_threshold: NMS IoU threshold
            sahi_slice_size: Slice size for SAHI
            sahi_overlap: Overlap ratio for SAHI
            auto_sahi_threshold: Use SAHI if image dimension exceeds this
            class_names: Optional mapping from class index to name
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.auto_sahi_threshold = auto_sahi_threshold
        
        self.slicer = SAHISlicer(
            slice_size=sahi_slice_size,
            overlap_ratio=sahi_overlap
        )
        self.merger = DetectionMerger(
            nms_threshold=nms_threshold,
            score_threshold=score_threshold
        )
        
        self.class_names = class_names or IDX_TO_CLASS
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        num_classes: int = 15,
        device: str = 'cuda',
        **kwargs
    ) -> 'Predictor':
        """
        Load predictor from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            num_classes: Number of classes
            device: Device to use
            **kwargs: Additional predictor arguments
            
        Returns:
            Predictor instance
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Create model
        model = RotatedRetinaNet(num_classes=num_classes, pretrained=False)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return cls(model, device=device, **kwargs)
    
    def predict(
        self,
        image: Union[np.ndarray, str, Path],
        use_sahi: Optional[bool] = None
    ) -> Dict:
        """
        Run inference on an image.
        
        Args:
            image: Image array (H, W, C) BGR or path to image
            use_sahi: Force SAHI on/off, or None for auto
            
        Returns:
            Dict with 'boxes', 'scores', 'labels', 'class_names'
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise ValueError(f"Failed to load image: {image}")
        
        # Convert BGR to RGB
        if image.ndim == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        h, w = image_rgb.shape[:2]
        
        # Decide whether to use SAHI
        if use_sahi is None:
            use_sahi = max(h, w) > self.auto_sahi_threshold
        
        if use_sahi:
            return self._predict_with_sahi(image_rgb)
        else:
            return self._predict_single(image_rgb)
    
    def _predict_single(self, image: np.ndarray) -> Dict:
        """Run inference on a single image without slicing."""
        # Preprocess
        tensor = self._preprocess(image)
        
        # Run model
        with torch.no_grad():
            outputs = self.model(tensor)
        
        # Extract results
        boxes = outputs['boxes'][0].cpu().numpy()
        scores = outputs['scores'][0].cpu().numpy()
        labels = outputs['labels'][0].cpu().numpy()
        
        # Filter by score
        mask = scores >= self.score_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        # Get class names
        class_names = [self.class_names.get(int(lbl), f'class_{lbl}') for lbl in labels]
        
        return {
            'boxes': boxes,
            'scores': scores,
            'labels': labels,
            'class_names': class_names,
            'image_size': image.shape[:2]
        }
    
    def _predict_with_sahi(self, image: np.ndarray) -> Dict:
        """Run inference with SAHI slicing."""
        h, w = image.shape[:2]
        
        patch_detections = []
        
        for patch, x_offset, y_offset in self.slicer.slice_image_lazy(image):
            # Preprocess patch
            tensor = self._preprocess(patch)
            
            # Run model
            with torch.no_grad():
                outputs = self.model(tensor)
            
            boxes = outputs['boxes'][0].cpu().numpy()
            scores = outputs['scores'][0].cpu().numpy()
            labels = outputs['labels'][0].cpu().numpy()
            
            if len(boxes) > 0:
                patch_detections.append((boxes, scores, labels, x_offset, y_offset))
        
        # Merge detections
        boxes, scores, labels = self.merger.merge(patch_detections, (h, w))
        
        # Get class names
        class_names = [self.class_names.get(int(lbl), f'class_{lbl}') for lbl in labels]
        
        return {
            'boxes': boxes,
            'scores': scores,
            'labels': labels,
            'class_names': class_names,
            'image_size': (h, w)
        }
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        # Normalize to [0, 1]
        tensor = torch.from_numpy(image).float() / 255.0
        
        # HWC -> CHW
        tensor = tensor.permute(2, 0, 1)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        
        return tensor.to(self.device)
    
    def predict_batch(
        self,
        images: List[Union[np.ndarray, str, Path]],
        use_sahi: Optional[bool] = None
    ) -> List[Dict]:
        """
        Run inference on multiple images.
        
        Args:
            images: List of images or paths
            use_sahi: Force SAHI on/off
            
        Returns:
            List of prediction dicts
        """
        return [self.predict(img, use_sahi) for img in images]
