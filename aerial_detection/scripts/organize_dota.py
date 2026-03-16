"""Script to organize DOTA dataset into the expected structure.

This script consolidates the downloaded DOTA data (which comes in multiple
nested folders) into a clean structure:

data/dota/
├── train/
│   ├── images/
│   └── labelTxt/
├── val/
│   ├── images/
│   └── labelTxt/
└── test/
    ├── images/
    └── labelTxt/

Usage:
    python -m aerial_detection.scripts.organize_dota --source aerial_detection/data/dota --dest data/dota
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import List, Tuple


def find_image_folders(root: Path) -> List[Path]:
    """Find all folders containing images."""
    image_folders = []
    for path in root.rglob('*'):
        if path.is_dir() and path.name == 'images':
            # Check if it contains image files
            has_images = any(
                f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
                for f in path.iterdir() if f.is_file()
            )
            if has_images:
                image_folders.append(path)
        # Also check for part folders with images directly inside
        elif path.is_dir() and path.name.startswith('part'):
            images_subdir = path / 'images'
            if images_subdir.exists():
                image_folders.append(images_subdir)
    return image_folders


def find_label_folders(root: Path) -> List[Path]:
    """Find all folders containing label txt files."""
    label_folders = []
    for path in root.rglob('*'):
        if path.is_dir() and path.name == 'labelTxt':
            # Check if it contains txt files
            has_labels = any(f.suffix == '.txt' for f in path.iterdir() if f.is_file())
            if has_labels:
                label_folders.append(path)
    return label_folders


def copy_files(src_folders: List[Path], dest_folder: Path, extensions: List[str]):
    """Copy files from multiple source folders to destination."""
    dest_folder.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    
    for src_folder in src_folders:
        for file in src_folder.iterdir():
            if file.is_file() and file.suffix.lower() in extensions:
                dest_file = dest_folder / file.name
                if dest_file.exists():
                    skipped += 1
                else:
                    shutil.copy2(file, dest_file)
                    copied += 1
    
    return copied, skipped


def organize_split(source_split: Path, dest_split: Path, split_name: str):
    """Organize a single split (train/val/test)."""
    print(f"\n{'='*50}")
    print(f"Processing {split_name} split...")
    print(f"{'='*50}")
    
    # Find image folders
    image_folders = find_image_folders(source_split)
    print(f"Found {len(image_folders)} image folder(s):")
    for f in image_folders:
        print(f"  - {f}")
    
    # Find label folders
    label_folders = find_label_folders(source_split)
    print(f"Found {len(label_folders)} label folder(s):")
    for f in label_folders:
        print(f"  - {f}")
    
    # Copy images
    if image_folders:
        dest_images = dest_split / 'images'
        copied, skipped = copy_files(
            image_folders, dest_images,
            ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
        )
        print(f"Images: copied {copied}, skipped {skipped} (already exist)")
    else:
        print("WARNING: No image folders found!")
    
    # Copy labels
    if label_folders:
        dest_labels = dest_split / 'labelTxt'
        copied, skipped = copy_files(label_folders, dest_labels, ['.txt'])
        print(f"Labels: copied {copied}, skipped {skipped} (already exist)")
    else:
        print("WARNING: No label folders found!")


def verify_dataset(dest_root: Path):
    """Verify the organized dataset."""
    print(f"\n{'='*50}")
    print("Dataset Verification")
    print(f"{'='*50}")
    
    for split in ['train', 'val', 'test']:
        split_dir = dest_root / split
        if not split_dir.exists():
            print(f"{split}: NOT FOUND")
            continue
        
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labelTxt'
        
        num_images = len(list(images_dir.glob('*.*'))) if images_dir.exists() else 0
        num_labels = len(list(labels_dir.glob('*.txt'))) if labels_dir.exists() else 0
        
        # Check matching
        if images_dir.exists() and labels_dir.exists():
            image_stems = {f.stem for f in images_dir.iterdir() if f.is_file()}
            label_stems = {f.stem for f in labels_dir.iterdir() if f.suffix == '.txt'}
            matched = len(image_stems & label_stems)
            images_only = len(image_stems - label_stems)
            labels_only = len(label_stems - image_stems)
        else:
            matched = images_only = labels_only = 0
        
        print(f"\n{split}:")
        print(f"  Images: {num_images}")
        print(f"  Labels: {num_labels}")
        print(f"  Matched pairs: {matched}")
        if images_only > 0:
            print(f"  Images without labels: {images_only}")
        if labels_only > 0:
            print(f"  Labels without images: {labels_only}")


def main():
    parser = argparse.ArgumentParser(description='Organize DOTA dataset')
    parser.add_argument('--source', type=str, required=True,
                        help='Source directory with downloaded DOTA data')
    parser.add_argument('--dest', type=str, required=True,
                        help='Destination directory for organized data')
    parser.add_argument('--copy', action='store_true',
                        help='Copy files (default: move)')
    parser.add_argument('--verify-only', action='store_true',
                        help='Only verify existing dataset')
    args = parser.parse_args()
    
    source_root = Path(args.source)
    dest_root = Path(args.dest)
    
    if not source_root.exists():
        print(f"ERROR: Source directory does not exist: {source_root}")
        return
    
    if args.verify_only:
        verify_dataset(dest_root)
        return
    
    print(f"Source: {source_root}")
    print(f"Destination: {dest_root}")
    
    # Process each split
    for split in ['train', 'val', 'test']:
        source_split = source_root / split
        if source_split.exists():
            dest_split = dest_root / split
            organize_split(source_split, dest_split, split)
        else:
            print(f"\nSkipping {split}: source folder not found")
    
    # Verify
    verify_dataset(dest_root)
    
    print(f"\n{'='*50}")
    print("Organization complete!")
    print(f"{'='*50}")
    print(f"\nYou can now use the dataset with:")
    print(f"  python -m aerial_detection.scripts.train --data_root {dest_root}")


if __name__ == '__main__':
    main()


def create_train_val_split(
    data_root: Path,
    val_ratio: float = 0.15,
    seed: int = 42
):
    """
    Create a train/val split from the train set when val labels are missing.
    
    This creates symlinks or copies to organize the data.
    """
    import random
    
    train_images = data_root / 'train' / 'images'
    train_labels = data_root / 'train' / 'labelTxt'
    
    # Get matched pairs
    image_stems = {f.stem for f in train_images.iterdir() if f.is_file()}
    label_stems = {f.stem for f in train_labels.iterdir() if f.suffix == '.txt'}
    matched = sorted(image_stems & label_stems)
    
    print(f"Total matched pairs: {len(matched)}")
    
    # Shuffle and split
    random.seed(seed)
    random.shuffle(matched)
    
    val_count = int(len(matched) * val_ratio)
    val_stems = set(matched[:val_count])
    train_stems = set(matched[val_count:])
    
    print(f"Train: {len(train_stems)}, Val: {len(val_stems)}")
    
    # Create split files
    splits_dir = data_root / 'splits'
    splits_dir.mkdir(exist_ok=True)
    
    with open(splits_dir / 'train.txt', 'w') as f:
        for stem in sorted(train_stems):
            f.write(f"{stem}\n")
    
    with open(splits_dir / 'val.txt', 'w') as f:
        for stem in sorted(val_stems):
            f.write(f"{stem}\n")
    
    print(f"Split files saved to {splits_dir}")
    print("Use these with DOTADataset by passing split_file parameter")
    
    return train_stems, val_stems
