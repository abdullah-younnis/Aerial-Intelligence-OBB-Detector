"""Default configuration values for aerial detection."""

from dataclasses import dataclass, field
from typing import List

# DOTA dataset class definitions
DOTA_CLASSES = [
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field',
    'roundabout', 'harbor', 'swimming-pool', 'helicopter'
]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(DOTA_CLASSES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(DOTA_CLASSES)}
NUM_CLASSES = len(DOTA_CLASSES)


@dataclass
class TrainingConfig:
    """Training hyperparameters and settings."""
    # Data
    data_root: str = "data/dota"
    patch_size: int = 1024
    overlap: float = 0.25
    
    # Model
    backbone: str = 'resnet50'
    num_classes: int = NUM_CLASSES
    anchor_scales: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 512])
    anchor_ratios: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    anchor_angles: List[float] = field(default_factory=lambda: [-90, -60, -30, 0, 30, 60])
    
    # Training
    batch_size: int = 8
    learning_rate: float = 0.01
    weight_decay: float = 0.0001
    num_epochs: int = 12
    lr_scheduler: str = 'step'
    lr_steps: List[int] = field(default_factory=lambda: [8, 11])
    
    # Loss
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    positive_iou_threshold: float = 0.5
    negative_iou_threshold: float = 0.4
    
    # Checkpointing
    checkpoint_dir: str = 'checkpoints'
    save_interval: int = 1


# Default inference settings
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_NMS_THRESHOLD = 0.5
DEFAULT_SAHI_SLICE_SIZE = 1024
DEFAULT_SAHI_OVERLAP = 0.25
