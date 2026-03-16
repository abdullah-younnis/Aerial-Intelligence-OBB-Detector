"""Evaluation script for Rotated RetinaNet.

Usage:
    python -m aerial_detection.scripts.evaluate --checkpoint model.pth --data_root data/dota --split val
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from ..config import DOTA_CLASSES
from ..evaluation import DOTAEvaluator
from ..geometry import OBB
from ..inference import Predictor


def evaluate(
    checkpoint_path: str,
    data_root: str,
    split: str = 'val',
    confidence_threshold: float = 0.3,
    nms_threshold: float = 0.5,
    use_sahi: bool = True,
    slice_size: int = 1024,
    overlap_ratio: float = 0.25,
    output_path: Optional[str] = None,
    device: Optional[str] = None,
    max_images: Optional[int] = None
):
    """
    Evaluate model on DOTA dataset.
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_root: Path to DOTA dataset root
        split: Dataset split (train/val/test)
        confidence_threshold: Confidence threshold for detections
        nms_threshold: NMS IoU threshold
        use_sahi: Whether to use SAHI for large images
        slice_size: SAHI slice size
        overlap_ratio: SAHI overlap ratio
        output_path: Optional path to save detailed results
        device: Device to use (cuda/cpu)
        max_images: Maximum number of images to evaluate (for debugging)
    """
    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f'Using device: {device}')
    print(f'Loading model from: {checkpoint_path}')
    print(f'Evaluating on {split} split')
    
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
    
    # Get image and annotation paths
    data_root = Path(data_root)
    images_dir = data_root / split / 'images'
    labels_dir = data_root / split / 'labelTxt'
    
    if not images_dir.exists():
        raise ValueError(f'Images directory not found: {images_dir}')
    if not labels_dir.exists():
        raise ValueError(f'Labels directory not found: {labels_dir}')
    
    # Get image files
    extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    image_files = sorted([p for p in images_dir.iterdir() 
                         if p.suffix.lower() in extensions])
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f'Found {len(image_files)} images')
    
    # Create evaluator
    evaluator = DOTAEvaluator(iou_threshold=0.5)
    
    # Process each image
    for idx, img_path in enumerate(image_files):
        image_id = img_path.stem
        print(f'[{idx + 1}/{len(image_files)}] Processing: {image_id}')
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f'  Warning: Could not load {img_path}')
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load ground truth
        label_path = labels_dir / f'{image_id}.txt'
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 9:
                        continue
                    
                    # Parse polygon coordinates
                    try:
                        coords = [float(x) for x in parts[:8]]
                        cls_name = parts[8]
                        difficult = int(parts[9]) if len(parts) > 9 else 0
                    except (ValueError, IndexError):
                        continue
                    
                    # Convert polygon to OBB
                    polygon = np.array(coords).reshape(4, 2)
                    obb = OBB.from_polygon(polygon)
                    
                    evaluator.add_ground_truth(image_id, cls_name, obb, bool(difficult))
        
        # Run prediction
        result = predictor.predict(image)
        
        # Add predictions
        for i in range(len(result['boxes'])):
            box = result['boxes'][i]
            score = result['scores'][i]
            label = int(result['labels'][i])
            class_name = DOTA_CLASSES[label] if label < len(DOTA_CLASSES) else f'class_{label}'
            
            obb = OBB(box[0], box[1], box[2], box[3], box[4])
            evaluator.add_prediction(image_id, class_name, float(score), obb)
        
        print(f'  GT: {len([gt for gt in evaluator.ground_truths if gt.image_id == image_id])}, '
              f'Pred: {len([p for p in evaluator.predictions if p.image_id == image_id])}')
    
    # Compute metrics
    print('\nComputing metrics...')
    result = evaluator.evaluate()
    
    # Print results
    print(result.summary())
    
    # Save detailed results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        print(f'\nDetailed results saved to: {output_path}')
    
    return result


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Rotated RetinaNet on DOTA')
    
    # Required
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to DOTA dataset root')
    
    # Dataset
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate')
    
    # Detection parameters
    parser.add_argument('--confidence', type=float, default=0.3,
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
    
    # Output
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save detailed results JSON')
    
    # Device
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    
    # Debug
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum images to evaluate (for debugging)')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    evaluate(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        split=args.split,
        confidence_threshold=args.confidence,
        nms_threshold=args.nms_threshold,
        use_sahi=not args.no_sahi,
        slice_size=args.slice_size,
        overlap_ratio=args.overlap,
        output_path=args.output,
        device=args.device,
        max_images=args.max_images
    )
