#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MP4 File Resolution Statistics Tool
Uses multi-threading with progress bar
"""

import os
import sys
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import cv2
from tqdm import tqdm


def get_video_resolution(video_path):
    """
    Get video file resolution - optimized version
    
    Args:
        video_path (str): Video file path
        
    Returns:
        tuple: (width, height) or None (if failed)
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        if width > 0 and height > 0:
            return (width, height)
        else:
            return None
            
    except Exception:
        return None


def find_mp4_files(folder_path):
    """
    Find all MP4 files in folder
    
    Args:
        folder_path (str): Folder path
        
    Returns:
        list: List of MP4 file paths
    """
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Error: Folder {folder_path} does not exist")
        return []
    
    mp4_files = []
    for file_path in folder.rglob("*.mp4"):
        if file_path.is_file():
            mp4_files.append(file_path)
    
    for file_path in folder.rglob("*.MP4"):
        if file_path.is_file():
            mp4_files.append(file_path)
    
    return mp4_files


def process_video_file(video_path):
    """
    Process a single video file and return resolution
    
    Args:
        video_path (Path): Video file path
        
    Returns:
        tuple: Resolution (width, height) or None
    """
    resolution = get_video_resolution(video_path)
    return resolution


def main():
    parser = argparse.ArgumentParser(description="Statistics of MP4 file resolutions in folder")
    parser.add_argument("folder", help="Folder path to scan")
    parser.add_argument("--threads", "-t", type=int, default=32, help="Number of threads (default: 32)")
    parser.add_argument("--output", "-o", default="resolution_stats.txt", help="Output results to file (default: resolution_stats.txt)")
    
    args = parser.parse_args()
    
    print(f"Scanning folder: {args.folder}")
    mp4_files = find_mp4_files(args.folder)
    
    if not mp4_files:
        print("No MP4 files found")
        return
    
    print(f"Found {len(mp4_files)} MP4 files")
    print(f"Using {args.threads} threads for processing")
    
    resolution_stats = defaultdict(int)
    failed_count = 0
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        future_to_file = {
            executor.submit(process_video_file, video_path): video_path 
            for video_path in mp4_files
        }
        
        with tqdm(total=len(mp4_files), desc="Processing video files", unit="file", 
                  smoothing=0.1, mininterval=0.5) as pbar:
            for future in as_completed(future_to_file):
                video_path = future_to_file[future]
                try:
                    resolution = future.result()
                    
                    if resolution:
                        resolution_key = f"{resolution[0]}x{resolution[1]}"
                        resolution_stats[resolution_key] += 1
                    else:
                        failed_count += 1
                        
                except Exception:
                    failed_count += 1
                
                pbar.update(1)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    output_lines = []
    output_lines.append("MP4 File Resolution Statistics Results")
    output_lines.append("=" * 40)
    output_lines.append(f"Scanned folder: {args.folder}")
    output_lines.append(f"Total files: {len(mp4_files)}")
    output_lines.append(f"Successfully processed: {len(mp4_files) - failed_count}")
    output_lines.append(f"Failed: {failed_count}")
    output_lines.append(f"Resolution types found: {len(resolution_stats)}")
    output_lines.append(f"Processing time: {total_time:.2f} seconds")
    output_lines.append(f"Processing speed: {len(mp4_files)/total_time:.1f} files/second")
    output_lines.append("")
    
    if resolution_stats:
        output_lines.append("Resolution Statistics:")
        output_lines.append("-" * 20)
        
        sorted_resolutions = sorted(resolution_stats.items(), 
                                  key=lambda x: (int(x[0].split('x')[0]), int(x[0].split('x')[1])))
        
        for resolution, count in sorted_resolutions:
            output_lines.append(f"{resolution}: {count} files")
    
    for line in output_lines:
        print(line)
    
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        print(f"\nResults saved to: {args.output}")
    except Exception as e:
        print(f"Error saving file: {e}")


if __name__ == "__main__":
    main()
