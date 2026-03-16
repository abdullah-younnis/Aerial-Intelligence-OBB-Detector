"""Loss functions for rotated object detection."""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..geometry import rotated_iou, OBB


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in object detection.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for positive examples
            gamma: Focusing parameter
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predicted logits (N, C) or (N,)
            targets: Ground truth labels (N,) with class indices
            
        Returns:
            Focal loss value
        """
        p = torch.sigmoid(inputs)
        
        # For binary case
        if inputs.dim() == 1 or inputs.shape[1] == 1:
            p = p.view(-1)
            targets = targets.view(-1).float()
            
            p_t = p * targets + (1 - p) * (1 - targets)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            
            focal_weight = (1 - p_t) ** self.gamma
            loss = -alpha_t * focal_weight * torch.log(p_t.clamp(min=1e-8))
        else:
            # Multi-class case
            num_classes = inputs.shape[1]
            
            # Create one-hot targets
            targets_one_hot = F.one_hot(targets, num_classes).float()
            
            p_t = p * targets_one_hot + (1 - p) * (1 - targets_one_hot)
            alpha_t = self.alpha * targets_one_hot + (1 - self.alpha) * (1 - targets_one_hot)
            
            focal_weight = (1 - p_t) ** self.gamma
            bce = F.binary_cross_entropy_with_logits(
                inputs, targets_one_hot, reduction='none'
            )
            loss = alpha_t * focal_weight * bce
            loss = loss.sum(dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class SmoothL1Loss(nn.Module):
    """Smooth L1 loss for box regression."""
    
    def __init__(self, beta: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute smooth L1 loss.
        
        Args:
            inputs: Predicted values
            targets: Target values
            
        Returns:
            Loss value
        """
        diff = torch.abs(inputs - targets)
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class AngleAwareSmoothL1Loss(nn.Module):
    """
    Smooth L1 loss with angle periodicity handling.
    
    Handles wrap-around at ±90° for angle regression.
    """
    
    def __init__(self, beta: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction
    
    def forward(
        self,
        pred_angles: torch.Tensor,
        target_angles: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute angle-aware smooth L1 loss.
        
        Args:
            pred_angles: Predicted angles in degrees
            target_angles: Target angles in degrees
            
        Returns:
            Loss value
        """
        # Compute angle difference with wrap-around
        diff = pred_angles - target_angles
        
        # Normalize to [-90, 90)
        diff = torch.remainder(diff + 90, 180) - 90
        
        # Apply smooth L1
        abs_diff = torch.abs(diff)
        loss = torch.where(
            abs_diff < self.beta,
            0.5 * abs_diff ** 2 / self.beta,
            abs_diff - 0.5 * self.beta
        )
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def assign_targets_to_anchors(
    anchors: torch.Tensor,
    targets: List[Dict],
    positive_iou_threshold: float = 0.5,
    negative_iou_threshold: float = 0.4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Assign ground truth targets to anchors using Rotated IoU.
    
    Uses approximate axis-aligned IoU for speed, with exact rotated IoU
    only for candidates above a threshold.
    
    Args:
        anchors: Tensor of shape (N, 5) with anchor parameters
        targets: List of target dicts with 'boxes' and 'labels'
        positive_iou_threshold: IoU threshold for positive assignment
        negative_iou_threshold: IoU threshold below which is negative
        
    Returns:
        matched_gt_boxes: (N, 5) matched ground truth boxes
        matched_labels: (N,) matched class labels (-1 for ignore, 0 for background)
        matched_ious: (N,) IoU with matched ground truth
    """
    device = anchors.device
    num_anchors = len(anchors)
    
    # Initialize outputs
    matched_gt_boxes = torch.zeros(num_anchors, 5, device=device)
    matched_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
    matched_ious = torch.zeros(num_anchors, device=device)
    
    # Process each image in batch (assuming single image for now)
    if len(targets) == 0:
        return matched_gt_boxes, matched_labels, matched_ious
    
    target = targets[0]
    gt_boxes = target['boxes']
    gt_labels = target['labels']
    
    if len(gt_boxes) == 0:
        return matched_gt_boxes, matched_labels, matched_ious
    
    # Fast approximate IoU using axis-aligned bounding boxes
    # This filters out most anchors quickly
    def fast_aabb_iou(anchors, gt_boxes):
        """Compute approximate IoU using axis-aligned boxes (vectorized)."""
        # Get max dimension as approximate AABB size
        a_cx, a_cy = anchors[:, 0], anchors[:, 1]
        a_size = torch.max(anchors[:, 2], anchors[:, 3])
        
        g_cx, g_cy = gt_boxes[:, 0], gt_boxes[:, 1]
        g_size = torch.max(gt_boxes[:, 2], gt_boxes[:, 3])
        
        # Compute center distances
        # Shape: (num_anchors, num_gt)
        dx = a_cx.unsqueeze(1) - g_cx.unsqueeze(0)
        dy = a_cy.unsqueeze(1) - g_cy.unsqueeze(0)
        center_dist = torch.sqrt(dx**2 + dy**2)
        
        # Approximate overlap threshold
        max_dist = (a_size.unsqueeze(1) + g_size.unsqueeze(0)) * 0.7
        
        # Return mask of potential matches
        return center_dist < max_dist
    
    # Get candidate matches using fast filtering
    candidate_mask = fast_aabb_iou(anchors, gt_boxes)
    
    # For candidates, compute exact rotated IoU
    candidate_indices = torch.where(candidate_mask.any(dim=1))[0]
    
    if len(candidate_indices) == 0:
        return matched_gt_boxes, matched_labels, matched_ious
    
    # Compute exact IoU only for candidates
    candidate_anchors = anchors[candidate_indices].cpu().numpy()
    gt_boxes_np = gt_boxes.cpu().numpy()
    
    iou_matrix = np.zeros((len(candidate_indices), len(gt_boxes)), dtype=np.float32)
    
    for i, anchor_params in enumerate(candidate_anchors):
        anchor_obb = OBB.from_array(anchor_params)
        for j, gt_params in enumerate(gt_boxes_np):
            if candidate_mask[candidate_indices[i], j]:
                gt_obb = OBB.from_array(gt_params)
                iou_matrix[i, j] = rotated_iou(anchor_obb, gt_obb)
    
    iou_tensor = torch.from_numpy(iou_matrix).to(device)
    
    # For each candidate anchor, find best matching GT
    max_iou, max_idx = iou_tensor.max(dim=1)
    
    # Map back to full anchor indices
    full_max_iou = torch.zeros(num_anchors, device=device)
    full_max_idx = torch.zeros(num_anchors, dtype=torch.long, device=device)
    full_max_iou[candidate_indices] = max_iou
    full_max_idx[candidate_indices] = max_idx
    
    # Assign positives (IoU >= positive_threshold)
    positive_mask = full_max_iou >= positive_iou_threshold
    matched_gt_boxes[positive_mask] = gt_boxes[full_max_idx[positive_mask]]
    matched_labels[positive_mask] = gt_labels[full_max_idx[positive_mask]] + 1  # +1 for background=0
    matched_ious[positive_mask] = full_max_iou[positive_mask]
    
    # Assign negatives (IoU < negative_threshold) - includes non-candidates
    negative_mask = full_max_iou < negative_iou_threshold
    matched_labels[negative_mask] = 0  # Background
    
    # Ignore anchors in between thresholds
    ignore_mask = ~positive_mask & ~negative_mask
    matched_labels[ignore_mask] = -1  # Ignore
    
    # Ensure each GT has at least one positive anchor
    for j in range(len(gt_boxes)):
        # Find best anchor among candidates for this GT
        gt_ious = iou_tensor[:, j]
        if gt_ious.max() > 0:
            best_candidate_idx = gt_ious.argmax()
            best_anchor_idx = candidate_indices[best_candidate_idx]
            if matched_labels[best_anchor_idx] != gt_labels[j] + 1:
                matched_gt_boxes[best_anchor_idx] = gt_boxes[j]
                matched_labels[best_anchor_idx] = gt_labels[j] + 1
                matched_ious[best_anchor_idx] = gt_ious[best_candidate_idx]
    
    return matched_gt_boxes, matched_labels, matched_ious


def encode_boxes(
    anchors: torch.Tensor,
    gt_boxes: torch.Tensor
) -> torch.Tensor:
    """
    Encode ground truth boxes relative to anchors.
    
    Args:
        anchors: (N, 5) anchor boxes [x, y, w, h, theta]
        gt_boxes: (N, 5) ground truth boxes
        
    Returns:
        (N, 5) encoded deltas [dx, dy, dw, dh, dtheta]
    """
    # Extract components
    ax, ay, aw, ah, atheta = anchors.unbind(dim=1)
    gx, gy, gw, gh, gtheta = gt_boxes.unbind(dim=1)
    
    # Encode center offsets
    dx = (gx - ax) / aw
    dy = (gy - ay) / ah
    
    # Encode size (log scale)
    dw = torch.log(gw / aw)
    dh = torch.log(gh / ah)
    
    # Encode angle difference
    dtheta = gtheta - atheta
    # Normalize to [-90, 90)
    dtheta = torch.remainder(dtheta + 90, 180) - 90
    
    return torch.stack([dx, dy, dw, dh, dtheta], dim=1)


def decode_boxes(
    anchors: torch.Tensor,
    deltas: torch.Tensor
) -> torch.Tensor:
    """
    Decode predicted deltas to boxes.
    
    Args:
        anchors: (N, 5) anchor boxes
        deltas: (N, 5) predicted deltas
        
    Returns:
        (N, 5) decoded boxes
    """
    ax, ay, aw, ah, atheta = anchors.unbind(dim=1)
    dx, dy, dw, dh, dtheta = deltas.unbind(dim=1)
    
    # Decode center
    px = dx * aw + ax
    py = dy * ah + ay
    
    # Decode size
    pw = torch.exp(dw) * aw
    ph = torch.exp(dh) * ah
    
    # Decode angle
    ptheta = atheta + dtheta
    # Normalize to [-90, 90)
    ptheta = torch.remainder(ptheta + 90, 180) - 90
    
    return torch.stack([px, py, pw, ph, ptheta], dim=1)


class RotatedRetinaNetLoss(nn.Module):
    """
    Combined loss for Rotated RetinaNet.
    """
    
    def __init__(
        self,
        num_classes: int,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        positive_iou_threshold: float = 0.5,
        negative_iou_threshold: float = 0.4,
        reg_weight: float = 1.0
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.positive_iou_threshold = positive_iou_threshold
        self.negative_iou_threshold = negative_iou_threshold
        self.reg_weight = reg_weight
        
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='sum')
        self.smooth_l1 = SmoothL1Loss(beta=1.0, reduction='sum')
        self.angle_loss = AngleAwareSmoothL1Loss(beta=1.0, reduction='sum')
    
    def forward(
        self,
        cls_logits: torch.Tensor,
        box_regression: torch.Tensor,
        anchors: torch.Tensor,
        targets: List[Dict]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            cls_logits: (B, N, num_classes) classification logits
            box_regression: (B, N, 5) box deltas
            anchors: (N, 5) anchors
            targets: List of target dicts
            
        Returns:
            Dict with 'cls_loss', 'reg_loss', 'total_loss'
        """
        device = cls_logits.device
        batch_size = cls_logits.shape[0]
        
        total_cls_loss = torch.tensor(0.0, device=device)
        total_reg_loss = torch.tensor(0.0, device=device)
        num_positives = 0
        
        for b in range(batch_size):
            # Get targets for this image
            target = targets[b] if b < len(targets) else {'boxes': torch.zeros(0, 5), 'labels': torch.zeros(0)}
            
            # Assign targets to anchors
            matched_gt_boxes, matched_labels, _ = assign_targets_to_anchors(
                anchors,
                [target],
                self.positive_iou_threshold,
                self.negative_iou_threshold
            )
            
            # Get positive and valid masks
            positive_mask = matched_labels > 0
            valid_mask = matched_labels >= 0  # Exclude ignored
            
            num_pos = positive_mask.sum().item()
            num_positives += max(num_pos, 1)
            
            # Classification loss (on valid anchors)
            if valid_mask.sum() > 0:
                # Convert labels: background=0, classes=1..C -> 0..C-1 for focal loss
                cls_targets = matched_labels[valid_mask].clone()
                cls_targets[cls_targets > 0] -= 1  # Shift class indices
                
                cls_loss = self.focal_loss(
                    cls_logits[b, valid_mask],
                    cls_targets
                )
                total_cls_loss = total_cls_loss + cls_loss
            
            # Regression loss (on positive anchors only)
            if positive_mask.sum() > 0:
                # Encode ground truth
                gt_deltas = encode_boxes(
                    anchors[positive_mask],
                    matched_gt_boxes[positive_mask]
                )
                
                pred_deltas = box_regression[b, positive_mask]
                
                # Box regression loss (dx, dy, dw, dh)
                reg_loss = self.smooth_l1(pred_deltas[:, :4], gt_deltas[:, :4])
                
                # Angle loss
                angle_loss = self.angle_loss(pred_deltas[:, 4], gt_deltas[:, 4])
                
                total_reg_loss = total_reg_loss + reg_loss + angle_loss
        
        # Normalize by number of positives
        cls_loss = total_cls_loss / num_positives
        reg_loss = total_reg_loss / num_positives * self.reg_weight
        
        return {
            'cls_loss': cls_loss,
            'reg_loss': reg_loss,
            'total_loss': cls_loss + reg_loss
        }
