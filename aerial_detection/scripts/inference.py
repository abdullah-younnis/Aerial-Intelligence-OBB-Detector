"""Inference script for Rotated RetinaNet.

Usage:
    python -m aerial_detection.scripts.inference --checkpoint model.pth --input image.png --output results.json
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import cv2
import torch

from ..config import DOTA_CLASSES
from ..inference import Predictor
from ..utils import ImagePredictions, Detection


def run_inference(
    checkpoint_path: str,
    input_path: str,
    output_path: str,
    confidence_threshold: float = 0.5,
    nms_threshold: float = 0.5,
    use_sahi: bool = True,
    slice_size: int = 1024,
    overlap_ratio: float = 0.25,
    polygon_format: bool = False,
    device: Optional[str] = None
):
    """
    Run inference on image(s).
    
    Args:
        checkpoint_path: Path to model checkpoint
        input_path: Path to image or directory
        output_path: Path to output JSON file
        confidence_threshold: Confidence threshold for detections
        nms_threshold: NMS IoU threshold
        use_sahi: Whether to use SAHI for large images
        slice_size: SAHI slice size
        overlap_ratio: SAHI overlap ratio
        polygon_format: Output in polygon format
        device: Device to use (cuda/cpu)
    """
    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f'Using device: {device}')
    print(f'Loading model from: {checkpoint_path}')
    
    # Create predictor
    predictor = Predictor(
        checkpoint_path=checkpoint_path,
        num_classes=len(DOTA_CLASSES),
        class_names=DOTA_CLASSES,
        device=device,
        confidence_threshold=confidence_threshold,
        nms_threshold=nms_threshold,
        use_sahi=use_sahi,
        sahi_slice_size=slice_size,
        sahi_overlap_ratio=overlap_ratio
    )
    
    # Get input files
    input_path = Path(input_path)
    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
        image_paths = [p for p in input_path.iterdir() 
                      if p.suffix.lower() in extensions]
        print(f'Found {len(image_paths)} images in directory')
    else:
        raise ValueError(f'Input path does not exist: {input_path}')
    
    # Run inference
    all_predictions: List[ImagePredictions] = []
    
    for img_path in image_paths:
        print(f'Processing: {img_path.name}')
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f'  Warning: Could not load {img_path}')
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run prediction
        result = predictor.predict(image)
        
        # Convert to ImagePredictions
        h, w = result['image_size']
        detections = []
        
        for i in range(len(result['boxes'])):
            box = result['boxes'][i]
            score = result['scores'][i]
            label = int(result['labels'][i])
            class_name = DOTA_CLASSES[label] if label < len(DOTA_CLASSES) else f'class_{label}'
            
            detections.append(Detection(
                class_name=class_name,
                confidence=float(score),
                x_center=float(box[0]),
                y_center=float(box[1]),
                width=float(box[2]),
                height=float(box[3]),
                angle=float(box[4])
            ))
        
        img_pred = ImagePredictions(
            image_path=str(img_path),
            image_width=w,
            image_height=h,
            detections=detections
        )
        all_predictions.append(img_pred)
        
        print(f'  Found {len(detections)} detections')
    
    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if polygon_format:
        # Output in polygon format
        output_data = {
            'predictions': []
        }
        for pred in all_predictions:
            output_data['predictions'].append({
                'image': pred.image_path,
                'width': pred.image_width,
                'height': pred.image_height,
                'detections': pred.to_polygon_format()
            })
    else:
        # Standard format
        output_data = {
            'predictions': [p.to_dict() for p in all_predictions]
        }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f'Results saved to: {output_path}')
    print(f'Total images: {len(all_predictions)}')
    print(f'Total detections: {sum(len(p.detections) for p in all_predictions)}')
    
    return all_predictions


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference with Rotated RetinaNet')
    
    # Required
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image or directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output JSON file')
    
    # Detection parameters
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--nms_threshold', type=float, default=0.5,
                        help='NMS IoU threshold')
    
    # SAHI parameters
    parser.add_argument('--no_sahi', action='store_true',
                        help='Disable SAHI slicing')
    parser.add_argument('--slice_size', type=int, default=1024,
                        help='SAHI slice size')
    parser.add_argument('--overlap', type=float, default=0.25,
                        help='SAHI overlap ratio')
    
    # Output format
    parser.add_argument('--polygon', action='store_true',
                        help='Output in polygon format')
    
    # Device
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    run_inference(
        checkpoint_path=args.checkpoint,
        input_path=args.input,
        output_path=args.output,
        confidence_threshold=args.confidence,
        nms_threshold=args.nms_threshold,
        use_sahi=not args.no_sahi,
        slice_size=args.slice_size,
        overlap_ratio=args.overlap,
        polygon_format=args.polygon,
        device=args.device
    )
