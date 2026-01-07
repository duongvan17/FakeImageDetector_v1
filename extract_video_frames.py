"""
Extract frames from videos in the Political Deepfakes Benchmark dataset.

This script extracts frames from video files and saves them as images,
using the video labels to categorize them as real/fake.

Usage:
    python extract_video_frames.py --input_dir dataset --output_dir political_data --frames_per_video 10
"""

import os
import argparse
import csv
import random
from pathlib import Path
from collections import Counter
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_video_labels(labels_path):
    """Parse video labels CSV file."""
    labels = {}
    
    with open(labels_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        
        for row in reader:
            if len(row) >= 2:
                filename = row[0].strip()
                label_val = row[1].strip()
                
                if label_val == '0':
                    labels[filename] = 'real'
                elif label_val == '1':
                    labels[filename] = 'fake'
                    
    return labels


def extract_frames(video_path, output_dir, num_frames=10, label='fake'):
    """Extract frames from a video file."""
    import cv2
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return 0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            logger.error(f"No frames in video: {video_path}")
            return 0
        
        # Calculate frame indices to extract (evenly distributed)
        if total_frames <= num_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        
        video_name = Path(video_path).stem
        extracted = 0
        
        for idx, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Save frame
                output_path = output_dir / f"{video_name}_frame{idx:03d}.jpg"
                cv2.imwrite(str(output_path), frame)
                extracted += 1
        
        cap.release()
        return extracted
        
    except Exception as e:
        logger.error(f"Error extracting frames from {video_path}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description='Extract frames from videos')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to dataset directory containing video folder')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output directory (will add to existing dataset)')
    parser.add_argument('--frames_per_video', type=int, default=10,
                        help='Number of frames to extract per video')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of videos for training')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    
    # Find video directory
    video_dir = input_path / 'video'
    if not video_dir.exists():
        logger.error(f"Video directory not found: {video_dir}")
        return
    
    # Find labels file
    labels_path = input_path / 'label' / 'video_verified_label.csv'
    if not labels_path.exists():
        logger.error(f"Labels file not found: {labels_path}")
        return
    
    # Parse labels
    labels = parse_video_labels(labels_path)
    logger.info(f"Loaded {len(labels)} video labels")
    
    # Count labels
    real_count = sum(1 for v in labels.values() if v == 'real')
    fake_count = sum(1 for v in labels.values() if v == 'fake')
    logger.info(f"Videos - Real: {real_count}, Fake: {fake_count}")
    
    # Find all videos
    video_files = list(video_dir.glob('*.mp4')) + list(video_dir.glob('*.MP4')) + \
                  list(video_dir.glob('*.mov')) + list(video_dir.glob('*.MOV'))
    
    logger.info(f"Found {len(video_files)} video files")
    
    # Separate by label
    real_videos = [v for v in video_files if labels.get(v.name) == 'real']
    fake_videos = [v for v in video_files if labels.get(v.name) == 'fake']
    
    logger.info(f"Real videos: {len(real_videos)}, Fake videos: {len(fake_videos)}")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(real_videos)
    random.shuffle(fake_videos)
    
    real_train_count = int(len(real_videos) * args.train_ratio)
    fake_train_count = int(len(fake_videos) * args.train_ratio)
    
    splits = {
        'train': {
            'real': real_videos[:real_train_count],
            'fake': fake_videos[:fake_train_count]
        },
        'test': {
            'real': real_videos[real_train_count:],
            'fake': fake_videos[fake_train_count:]
        }
    }
    
    # Create output directories
    category = 'political'
    for split in ['train', 'test']:
        for label in ['real', 'fake']:
            (output_path / split / category / label).mkdir(parents=True, exist_ok=True)
    
    # Extract frames
    stats = Counter()
    
    for split in ['train', 'test']:
        for label in ['real', 'fake']:
            videos = splits[split][label]
            target_dir = output_path / split / category / label
            
            logger.info(f"\nProcessing {split}/{label}: {len(videos)} videos")
            
            for video in videos:
                frames = extract_frames(video, target_dir, args.frames_per_video, label)
                stats[(split, label)] += frames
                if frames > 0:
                    logger.info(f"  Extracted {frames} frames from {video.name}")
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("Frame Extraction Complete!")
    logger.info("="*50)
    
    for (split, label), count in sorted(stats.items()):
        logger.info(f"  {split}/{category}/{label}: +{count} frames")
    
    total_train = stats[('train', 'real')] + stats[('train', 'fake')]
    total_test = stats[('test', 'real')] + stats[('test', 'fake')]
    logger.info(f"\n  Total new train frames: {total_train}")
    logger.info(f"  Total new test frames: {total_test}")


if __name__ == "__main__":
    main()
