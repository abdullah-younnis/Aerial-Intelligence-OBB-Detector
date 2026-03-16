"""I/O utilities for detection results."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json

from ..geometry import OBB


@dataclass
class Detection:
    """Single detection result."""
    class_name: str
    confidence: float
    x_center: float
    y_center: float
    width: float
    height: float
    angle: float  # degrees
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "class": self.class_name,
            "confidence": float(self.confidence),
            "x_center": float(self.x_center),
            "y_center": float(self.y_center),
            "width": float(self.width),
            "height": float(self.height),
            "angle": float(self.angle)
        }
    
    def to_polygon(self) -> List[float]:
        """Convert to polygon format [x1,y1, x2,y2, x3,y3, x4,y4]."""
        obb = OBB(self.x_center, self.y_center, self.width, self.height, self.angle)
        return obb.to_polygon().flatten().tolist()
    
    def to_obb(self) -> OBB:
        """Convert to OBB object."""
        return OBB(self.x_center, self.y_center, self.width, self.height, self.angle)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Detection':
        """Create Detection from dictionary."""
        return cls(
            class_name=data["class"],
            confidence=data["confidence"],
            x_center=data["x_center"],
            y_center=data["y_center"],
            width=data["width"],
            height=data["height"],
            angle=data["angle"]
        )
    
    @classmethod
    def from_obb(
        cls, 
        obb: OBB, 
        class_name: str, 
        confidence: float
    ) -> 'Detection':
        """Create Detection from OBB."""
        return cls(
            class_name=class_name,
            confidence=confidence,
            x_center=obb.x_center,
            y_center=obb.y_center,
            width=obb.width,
            height=obb.height,
            angle=obb.theta
        )


@dataclass
class ImagePredictions:
    """All detections for a single image."""
    image_path: str
    image_width: int
    image_height: int
    detections: List[Detection] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "image": self.image_path,
            "width": self.image_width,
            "height": self.image_height,
            "detections": [d.to_dict() for d in self.detections]
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, path: str):
        """Save to JSON file."""
        with open(path, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ImagePredictions':
        """Create from dictionary."""
        return cls(
            image_path=data["image"],
            image_width=data["width"],
            image_height=data["height"],
            detections=[Detection.from_dict(d) for d in data["detections"]]
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ImagePredictions':
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    @classmethod
    def load(cls, path: str) -> 'ImagePredictions':
        """Load from JSON file."""
        with open(path, 'r') as f:
            return cls.from_json(f.read())
    
    @classmethod
    def from_prediction_dict(
        cls,
        pred_dict: Dict,
        image_path: str,
        class_names: Optional[Dict[int, str]] = None
    ) -> 'ImagePredictions':
        """
        Create from predictor output dict.
        
        Args:
            pred_dict: Dict with 'boxes', 'scores', 'labels', 'image_size'
            image_path: Path to the image
            class_names: Optional mapping from label to class name
        """
        h, w = pred_dict['image_size']
        
        detections = []
        boxes = pred_dict['boxes']
        scores = pred_dict['scores']
        labels = pred_dict['labels']
        
        for i in range(len(boxes)):
            box = boxes[i]
            score = scores[i]
            label = int(labels[i])
            
            if class_names:
                class_name = class_names.get(label, f'class_{label}')
            else:
                class_name = pred_dict.get('class_names', [f'class_{label}'])[i] if 'class_names' in pred_dict else f'class_{label}'
            
            detections.append(Detection(
                class_name=class_name,
                confidence=float(score),
                x_center=float(box[0]),
                y_center=float(box[1]),
                width=float(box[2]),
                height=float(box[3]),
                angle=float(box[4])
            ))
        
        return cls(
            image_path=image_path,
            image_width=w,
            image_height=h,
            detections=detections
        )
    
    def to_polygon_format(self) -> List[Dict]:
        """Convert all detections to polygon format."""
        results = []
        for det in self.detections:
            polygon = det.to_polygon()
            results.append({
                "class": det.class_name,
                "confidence": det.confidence,
                "polygon": polygon
            })
        return results
    
    def filter_by_confidence(self, threshold: float) -> 'ImagePredictions':
        """Return new ImagePredictions with detections above threshold."""
        filtered = [d for d in self.detections if d.confidence >= threshold]
        return ImagePredictions(
            image_path=self.image_path,
            image_width=self.image_width,
            image_height=self.image_height,
            detections=filtered
        )
    
    def filter_by_class(self, class_names: List[str]) -> 'ImagePredictions':
        """Return new ImagePredictions with only specified classes."""
        filtered = [d for d in self.detections if d.class_name in class_names]
        return ImagePredictions(
            image_path=self.image_path,
            image_width=self.image_width,
            image_height=self.image_height,
            detections=filtered
        )


def save_predictions_batch(
    predictions: List[ImagePredictions],
    output_path: str
):
    """Save multiple image predictions to a single JSON file."""
    data = {
        "predictions": [p.to_dict() for p in predictions]
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_predictions_batch(path: str) -> List[ImagePredictions]:
    """Load multiple image predictions from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    return [ImagePredictions.from_dict(p) for p in data["predictions"]]
