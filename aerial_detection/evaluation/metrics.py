"""Evaluation metrics for rotated object detection.

Implements DOTA evaluation protocol with Rotated IoU.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict

from ..geometry import OBB, rotated_iou


@dataclass
class EvaluationResult:
    """Evaluation results container."""
    mAP: float
    per_class_ap: Dict[str, float]
    small_object_recall: float
    precision_at_50: float
    recall_at_50: float
    num_predictions: int
    num_ground_truths: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "mAP@0.5": self.mAP,
            "per_class_AP": self.per_class_ap,
            "small_object_recall": self.small_object_recall,
            "precision@0.5": self.precision_at_50,
            "recall@0.5": self.recall_at_50,
            "num_predictions": self.num_predictions,
            "num_ground_truths": self.num_ground_truths
        }
    
    def summary(self) -> str:
        """Generate summary report."""
        lines = [
            "=" * 50,
            "DOTA Evaluation Results",
            "=" * 50,
            f"mAP@0.5: {self.mAP:.4f}",
            f"Small Object Recall: {self.small_object_recall:.4f}",
            f"Precision@0.5: {self.precision_at_50:.4f}",
            f"Recall@0.5: {self.recall_at_50:.4f}",
            f"Total Predictions: {self.num_predictions}",
            f"Total Ground Truths: {self.num_ground_truths}",
            "-" * 50,
            "Per-Class AP:",
        ]
        for cls_name, ap in sorted(self.per_class_ap.items()):
            lines.append(f"  {cls_name}: {ap:.4f}")
        lines.append("=" * 50)
        return "\n".join(lines)


@dataclass
class GroundTruth:
    """Ground truth annotation."""
    image_id: str
    class_name: str
    obb: OBB
    difficult: bool = False  # DOTA difficult flag
    
    @property
    def is_small(self) -> bool:
        """Check if object is small (width or height < 30 pixels)."""
        return self.obb.width < 30 or self.obb.height < 30


@dataclass
class Prediction:
    """Detection prediction."""
    image_id: str
    class_name: str
    confidence: float
    obb: OBB


class DOTAEvaluator:
    """
    DOTA-style evaluator for rotated object detection.
    
    Computes mAP@0.5 using Rotated IoU following DOTA evaluation protocol.
    """
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize evaluator.
        
        Args:
            iou_threshold: IoU threshold for positive match (default 0.5)
        """
        self.iou_threshold = iou_threshold
        self.ground_truths: List[GroundTruth] = []
        self.predictions: List[Prediction] = []
        self._gt_by_image_class: Dict[str, List[GroundTruth]] = defaultdict(list)
        
    def reset(self):
        """Clear all stored data."""
        self.ground_truths = []
        self.predictions = []
        self._gt_by_image_class = defaultdict(list)
    
    def add_ground_truth(
        self,
        image_id: str,
        class_name: str,
        obb: OBB,
        difficult: bool = False
    ):
        """
        Add a ground truth annotation.
        
        Args:
            image_id: Image identifier
            class_name: Object class name
            obb: Oriented bounding box
            difficult: Whether this is a difficult object (ignored in eval)
        """
        gt = GroundTruth(image_id, class_name, obb, difficult)
        self.ground_truths.append(gt)
        key = f"{image_id}_{class_name}"
        self._gt_by_image_class[key].append(gt)
    
    def add_ground_truths_batch(
        self,
        image_id: str,
        boxes: np.ndarray,
        class_names: List[str],
        difficult: Optional[List[bool]] = None
    ):
        """
        Add multiple ground truths for an image.
        
        Args:
            image_id: Image identifier
            boxes: Array of shape (N, 5) with [x, y, w, h, theta]
            class_names: List of class names
            difficult: Optional list of difficult flags
        """
        if difficult is None:
            difficult = [False] * len(boxes)
        
        for i, (box, cls_name, diff) in enumerate(zip(boxes, class_names, difficult)):
            obb = OBB(box[0], box[1], box[2], box[3], box[4])
            self.add_ground_truth(image_id, cls_name, obb, diff)
    
    def add_prediction(
        self,
        image_id: str,
        class_name: str,
        confidence: float,
        obb: OBB
    ):
        """
        Add a detection prediction.
        
        Args:
            image_id: Image identifier
            class_name: Predicted class name
            confidence: Confidence score
            obb: Predicted oriented bounding box
        """
        pred = Prediction(image_id, class_name, confidence, obb)
        self.predictions.append(pred)
    
    def add_predictions_batch(
        self,
        image_id: str,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_names: List[str]
    ):
        """
        Add multiple predictions for an image.
        
        Args:
            image_id: Image identifier
            boxes: Array of shape (N, 5) with [x, y, w, h, theta]
            scores: Array of confidence scores
            class_names: List of class names
        """
        for box, score, cls_name in zip(boxes, scores, class_names):
            obb = OBB(box[0], box[1], box[2], box[3], box[4])
            self.add_prediction(image_id, cls_name, float(score), obb)

    
    def _compute_ap_for_class(
        self,
        class_name: str
    ) -> Tuple[float, int, int, int]:
        """
        Compute Average Precision for a single class.
        
        Returns:
            Tuple of (AP, num_tp, num_fp, num_gt)
        """
        # Get all predictions for this class, sorted by confidence
        class_preds = [p for p in self.predictions if p.class_name == class_name]
        class_preds.sort(key=lambda x: x.confidence, reverse=True)
        
        # Get all ground truths for this class
        class_gts = [gt for gt in self.ground_truths if gt.class_name == class_name]
        num_gt = sum(1 for gt in class_gts if not gt.difficult)
        
        if num_gt == 0:
            return 0.0, 0, len(class_preds), 0
        
        # Track which GTs have been matched
        gt_matched = {id(gt): False for gt in class_gts}
        
        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))
        
        for i, pred in enumerate(class_preds):
            # Get GTs for this image and class
            key = f"{pred.image_id}_{class_name}"
            image_gts = self._gt_by_image_class.get(key, [])
            
            best_iou = 0.0
            best_gt = None
            
            for gt in image_gts:
                if gt_matched[id(gt)]:
                    continue
                
                iou = rotated_iou(pred.obb, gt.obb)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt
            
            if best_iou >= self.iou_threshold and best_gt is not None:
                if not best_gt.difficult:
                    tp[i] = 1
                    gt_matched[id(best_gt)] = True
            else:
                fp[i] = 1
        
        # Compute precision-recall curve
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recall = tp_cumsum / num_gt
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        
        # Compute AP using 11-point interpolation (PASCAL VOC style)
        ap = self._compute_ap_11point(recall, precision)
        
        return ap, int(tp.sum()), int(fp.sum()), num_gt
    
    def _compute_ap_11point(
        self,
        recall: np.ndarray,
        precision: np.ndarray
    ) -> float:
        """
        Compute AP using 11-point interpolation.
        
        Args:
            recall: Recall values at each threshold
            precision: Precision values at each threshold
            
        Returns:
            Average Precision
        """
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            # Find precision at recall >= t
            mask = recall >= t
            if mask.any():
                p = precision[mask].max()
            else:
                p = 0.0
            ap += p / 11.0
        return ap
    
    def _compute_small_object_recall(self) -> float:
        """Compute recall for small objects (width or height < 30 pixels)."""
        small_gts = [gt for gt in self.ground_truths 
                     if gt.is_small and not gt.difficult]
        
        if len(small_gts) == 0:
            return 1.0  # No small objects to detect
        
        # Track matched small GTs
        matched = set()
        
        # Sort predictions by confidence
        sorted_preds = sorted(self.predictions, key=lambda x: x.confidence, reverse=True)
        
        for pred in sorted_preds:
            for gt in small_gts:
                if id(gt) in matched:
                    continue
                if pred.image_id != gt.image_id:
                    continue
                if pred.class_name != gt.class_name:
                    continue
                
                iou = rotated_iou(pred.obb, gt.obb)
                if iou >= self.iou_threshold:
                    matched.add(id(gt))
                    break
        
        return len(matched) / len(small_gts)

    
    def evaluate(self) -> EvaluationResult:
        """
        Run evaluation and compute all metrics.
        
        Returns:
            EvaluationResult with mAP, per-class AP, and other metrics
        """
        # Get all unique classes
        all_classes = set()
        for gt in self.ground_truths:
            all_classes.add(gt.class_name)
        for pred in self.predictions:
            all_classes.add(pred.class_name)
        
        # Compute AP for each class
        per_class_ap = {}
        total_tp = 0
        total_fp = 0
        total_gt = 0
        
        for class_name in sorted(all_classes):
            ap, tp, fp, num_gt = self._compute_ap_for_class(class_name)
            per_class_ap[class_name] = ap
            total_tp += tp
            total_fp += fp
            total_gt += num_gt
        
        # Compute mAP (mean over classes with ground truths)
        classes_with_gt = [c for c in all_classes 
                          if any(gt.class_name == c for gt in self.ground_truths)]
        if classes_with_gt:
            mAP = np.mean([per_class_ap[c] for c in classes_with_gt])
        else:
            mAP = 0.0
        
        # Compute overall precision and recall
        precision_at_50 = total_tp / (total_tp + total_fp + 1e-10)
        recall_at_50 = total_tp / (total_gt + 1e-10) if total_gt > 0 else 0.0
        
        # Compute small object recall
        small_recall = self._compute_small_object_recall()
        
        return EvaluationResult(
            mAP=float(mAP),
            per_class_ap=per_class_ap,
            small_object_recall=small_recall,
            precision_at_50=precision_at_50,
            recall_at_50=recall_at_50,
            num_predictions=len(self.predictions),
            num_ground_truths=len(self.ground_truths)
        )
    
    def evaluate_from_files(
        self,
        predictions_file: str,
        ground_truth_dir: str,
        class_names: Optional[List[str]] = None
    ) -> EvaluationResult:
        """
        Evaluate from prediction JSON and DOTA annotation files.
        
        Args:
            predictions_file: Path to predictions JSON file
            ground_truth_dir: Directory containing DOTA annotation files
            class_names: Optional list of class names to evaluate
            
        Returns:
            EvaluationResult
        """
        import json
        import os
        
        # Load predictions
        with open(predictions_file, 'r') as f:
            pred_data = json.load(f)
        
        # Add predictions
        for img_pred in pred_data.get('predictions', [pred_data]):
            image_id = os.path.basename(img_pred['image']).split('.')[0]
            for det in img_pred['detections']:
                if class_names and det['class'] not in class_names:
                    continue
                obb = OBB(
                    det['x_center'], det['y_center'],
                    det['width'], det['height'], det['angle']
                )
                self.add_prediction(image_id, det['class'], det['confidence'], obb)
        
        # Load ground truths from DOTA format
        for filename in os.listdir(ground_truth_dir):
            if not filename.endswith('.txt'):
                continue
            
            image_id = filename[:-4]
            filepath = os.path.join(ground_truth_dir, filename)
            
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 9:
                        continue
                    
                    # Parse polygon coordinates
                    coords = [float(x) for x in parts[:8]]
                    cls_name = parts[8]
                    difficult = int(parts[9]) if len(parts) > 9 else 0
                    
                    if class_names and cls_name not in class_names:
                        continue
                    
                    # Convert polygon to OBB
                    polygon = np.array(coords).reshape(4, 2)
                    obb = OBB.from_polygon(polygon)
                    
                    self.add_ground_truth(image_id, cls_name, obb, bool(difficult))
        
        return self.evaluate()


def compute_map(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5
) -> float:
    """
    Convenience function to compute mAP.
    
    Args:
        predictions: List of dicts with keys: image_id, class_name, confidence, box (5-tuple)
        ground_truths: List of dicts with keys: image_id, class_name, box (5-tuple)
        iou_threshold: IoU threshold for positive match
        
    Returns:
        mAP@iou_threshold
    """
    evaluator = DOTAEvaluator(iou_threshold)
    
    for gt in ground_truths:
        box = gt['box']
        obb = OBB(box[0], box[1], box[2], box[3], box[4])
        evaluator.add_ground_truth(
            gt['image_id'],
            gt['class_name'],
            obb,
            gt.get('difficult', False)
        )
    
    for pred in predictions:
        box = pred['box']
        obb = OBB(box[0], box[1], box[2], box[3], box[4])
        evaluator.add_prediction(
            pred['image_id'],
            pred['class_name'],
            pred['confidence'],
            obb
        )
    
    result = evaluator.evaluate()
    return result.mAP
