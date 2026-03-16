"""Training script for Rotated RetinaNet.

Usage:
    python -m aerial_detection.scripts.train --data_root data/dota --epochs 50
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import DOTA_CLASSES
from ..data import DOTADataset, get_train_transforms, get_val_transforms
from ..models import RotatedRetinaNet


def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging to console and file."""
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    
    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(console)
    
    # File handler
    log_file = os.path.join(output_dir, 'train.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    return logger


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    loss: float,
    output_dir: str,
    is_best: bool = False
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Save latest checkpoint
    path = os.path.join(output_dir, 'checkpoint_latest.pth')
    torch.save(checkpoint, path)
    
    # Save epoch checkpoint
    epoch_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, epoch_path)
    
    if is_best:
        best_path = os.path.join(output_dir, 'checkpoint_best.pth')
        torch.save(checkpoint, best_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> int:
    """Load checkpoint and return starting epoch."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint.get('epoch', 0) + 1


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    logger: logging.Logger
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        
        # Forward pass
        loss_dict = model(images, targets)
        
        # Total loss
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        losses.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        optimizer.step()
        
        total_loss += losses.item()
        
        # Log progress
        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            avg_loss = total_loss / (batch_idx + 1)
            msg = (f'Epoch [{epoch}] Batch [{batch_idx + 1}/{num_batches}] '
                   f'Loss: {losses.item():.4f} Avg: {avg_loss:.4f}')
            logger.info(msg)
            print(msg, flush=True)
    
    return total_loss / num_batches


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> float:
    """Validate model and return average loss."""
    model.train()  # Keep in train mode to compute losses
    total_loss = 0.0
    
    for images, targets in dataloader:
        images = images.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()
    
    return total_loss / len(dataloader) if len(dataloader) > 0 else 0.0


def collate_fn(batch):
    """Custom collate function for detection."""
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    images = torch.stack(images, dim=0)
    return images, targets


def train(args):
    """Main training function."""
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info(f'Training config: {vars(args)}')
    
    # Save config
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Create datasets with fixed size for batching
    image_size = (args.patch_size, args.patch_size)
    train_transforms = get_train_transforms(size=image_size)
    val_transforms = get_val_transforms(size=image_size)
    
    # Check if split files exist
    train_split_file = os.path.join(args.data_root, 'splits', 'train.txt')
    val_split_file = os.path.join(args.data_root, 'splits', 'val.txt')
    
    if os.path.exists(train_split_file) and os.path.exists(val_split_file):
        # Use split files (train data split into train/val)
        logger.info('Using split files for train/val')
        train_dataset = DOTADataset(
            root_dir=args.data_root,
            split='train',
            transforms=train_transforms,
            split_file='splits/train.txt'
        )
        val_dataset = DOTADataset(
            root_dir=args.data_root,
            split='train',  # Use train dir but filter by val.txt
            transforms=val_transforms,
            split_file='splits/val.txt'
        )
    else:
        # Use separate train/val directories
        train_dataset = DOTADataset(
            root_dir=args.data_root,
            split='train',
            transforms=train_transforms
        )
        val_dataset = DOTADataset(
            root_dir=args.data_root,
            split='val',
            transforms=val_transforms
        )
    
    logger.info(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')
    
    # On Windows, num_workers > 0 can cause hangs, so use 0 by default
    import platform
    num_workers = 0 if platform.system() == 'Windows' else args.num_workers
    logger.info(f'Using {num_workers} data loading workers')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create model
    model = RotatedRetinaNet(
        num_classes=len(DOTA_CLASSES),
        backbone=args.backbone,
        pretrained=args.pretrained
    )
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler (step decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step,
        gamma=args.lr_gamma
    )
    
    # Resume from checkpoint
    start_epoch = 1
    if args.resume:
        logger.info(f'Resuming from checkpoint: {args.resume}')
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler)
        logger.info(f'Resuming from epoch {start_epoch}')
    
    # Training loop
    best_val_loss = float('inf')
    
    print(f'\n{"="*60}')
    print(f'Starting training for {args.epochs} epochs')
    print(f'Train batches: {len(train_loader)}, Val batches: {len(val_loader)}')
    print(f'{"="*60}\n', flush=True)
    
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        
        print(f'Epoch {epoch}/{args.epochs} starting...', flush=True)
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, logger
        )
        
        # Validate
        val_loss = validate(model, val_loader, device)
        
        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        epoch_time = time.time() - epoch_start
        
        # Log epoch summary
        logger.info(
            f'Epoch [{epoch}/{args.epochs}] '
            f'Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} '
            f'LR: {current_lr:.6f} Time: {epoch_time:.1f}s'
        )
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        if epoch % args.save_interval == 0 or is_best:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                output_dir, is_best
            )
            logger.info(f'Saved checkpoint at epoch {epoch}')
    
    # Save final model
    final_path = os.path.join(output_dir, 'model_final.pth')
    torch.save(model.state_dict(), final_path)
    logger.info(f'Training complete. Final model saved to {final_path}')
    
    return output_dir


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Rotated RetinaNet')
    
    # Data
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to DOTA dataset root')
    parser.add_argument('--patch_size', type=int, default=1024,
                        help='Training patch size')
    parser.add_argument('--overlap', type=float, default=0.2,
                        help='Patch overlap ratio')
    
    # Model
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101'],
                        help='Backbone architecture')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained backbone')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--lr_step', type=int, default=20,
                        help='LR scheduler step size')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='LR scheduler gamma')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Checkpointing
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
