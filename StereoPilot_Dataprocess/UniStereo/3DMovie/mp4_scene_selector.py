"""
MP4 Scene Selector
Read mp4 file paths from txt file, parse Scene and part numbers from filenames,
select odd parts (1,3,5,7 etc.) of each Scene and copy to new folder
"""

import os
import shutil
import re
import argparse
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
from tqdm import tqdm


def parse_filename(filepath):
    """
    Parse mp4 filename to extract movie name, Scene number and part number
    
    Args:
        filepath: Full path to mp4 file
        
    Returns:
        tuple: (movie_prefix, scene_number, part_number) or (None, None, None) if parsing fails
        
    Example:
        AlphaandOmega_16fps-Scene-0008_001.mp4 -> ("AlphaandOmega_16fps", 8, 1)
    """
    filename = os.path.basename(filepath)
    
    pattern = r'^(.+)-Scene-(\d+)_(\d+)\.mp4$'
    match = re.search(pattern, filename)
    
    if match:
        movie_prefix = match.group(1)
        scene_num = int(match.group(2))
        part_num = int(match.group(3))
        return movie_prefix, scene_num, part_num
    
    return None, None, None


def read_file_paths(txt_file):
    """
    Read mp4 file paths from txt file
    
    Args:
        txt_file: txt file containing mp4 file paths
        
    Returns:
        list: list of mp4 file paths
    """
    file_paths = []
    
    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and line.endswith('.mp4'):
                    file_paths.append(line)
    except Exception as e:
        print(f"Error reading file {txt_file}: {e}")
        return []
    
    return file_paths


def group_files_by_scene(file_paths):
    """
    Group files by movie name and Scene number
    
    Args:
        file_paths: list of mp4 file paths
        
    Returns:
        dict: {(movie_prefix, scene_number): [(filepath, part_number), ...]}
    """
    scenes = defaultdict(list)
    
    for filepath in file_paths:
        movie_prefix, scene_num, part_num = parse_filename(filepath)
        
        if movie_prefix is not None and scene_num is not None and part_num is not None:
            key = (movie_prefix, scene_num)
            scenes[key].append((filepath, part_num))
        else:
            print(f"Unable to parse filename: {filepath}")
    
    return scenes


def select_odd_parts(scenes):
    """
    Select odd parts for each Scene of each movie
    
    Args:
        scenes: {(movie_prefix, scene_number): [(filepath, part_number), ...]}
        
    Returns:
        list: list of selected file paths
    """
    selected_files = []
    
    for (movie_prefix, scene_num), files in scenes.items():
        files.sort(key=lambda x: x[1])
        
        odd_parts = [(filepath, part_num) for filepath, part_num in files if part_num % 2 == 1]
        
        if odd_parts:
            print(f"{movie_prefix} Scene {scene_num:04d}: Found {len(files)} parts, selected {len(odd_parts)} odd parts")
            selected_files.extend([filepath for filepath, _ in odd_parts])
        else:
            print(f"{movie_prefix} Scene {scene_num:04d}: No odd parts found")
    
    return selected_files


def copy_single_file(filepath, output_dir, pbar, lock, stats, verbose=False):
    """
    Copy a single file, used for multi-threading
    
    Args:
        filepath: Source file path
        output_dir: Output directory path
        pbar: Progress bar object
        lock: Thread lock
        stats: Statistics dictionary
        verbose: Whether to show detailed information

    Returns:
        tuple: (success, message)
    """
    try:
        if os.path.exists(filepath):
            filename = os.path.basename(filepath)
            dest_path = os.path.join(output_dir, filename)
            
            if os.path.exists(dest_path):
                with lock:
                    stats['skipped'] += 1
                    pbar.set_postfix_str(f"Success: {stats['success']}, Skipped: {stats['skipped']}, Failed: {stats['failed']}")
                    pbar.update(1)
                    if verbose:
                        tqdm.write(f"Skipped existing file: {filename}")
                return True, f"Skipped existing file: {filename}"
            
            shutil.copy2(filepath, dest_path)
            
            with lock:
                stats['success'] += 1
                pbar.set_postfix_str(f"Success: {stats['success']}, Skipped: {stats['skipped']}, Failed: {stats['failed']}")
                pbar.update(1)
                if verbose:
                    tqdm.write(f"Copied: {filename}")
            
            return True, f"Copied: {filename}"
        else:
            with lock:
                stats['failed'] += 1
                pbar.set_postfix_str(f"Success: {stats['success']}, Skipped: {stats['skipped']}, Failed: {stats['failed']}")
                pbar.update(1)
                if verbose:
                    tqdm.write(f"Source file does not exist: {filepath}")
            return False, f"Source file does not exist: {filepath}"
            
    except Exception as e:
        with lock:
            stats['failed'] += 1
            pbar.set_postfix_str(f"Success: {stats['success']}, Skipped: {stats['skipped']}, Failed: {stats['failed']}")
            pbar.update(1)
            if verbose:
                tqdm.write(f"Error copying file {filepath}: {e}")
        return False, f"Error copying file {filepath}: {e}"


def copy_files(selected_files, output_dir, max_workers=4, verbose=False):
    """
    Copy selected files to output directory using multi-threading
    
    Args:
        selected_files: List of selected file paths
        output_dir: Output directory path
        max_workers: Maximum number of threads
        verbose: Whether to show detailed information
    """
    os.makedirs(output_dir, exist_ok=True)
    
    stats = {'success': 0, 'skipped': 0, 'failed': 0}
    lock = Lock()
    
    with tqdm(total=len(selected_files), desc="Copying files", unit="file") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(copy_single_file, filepath, output_dir, pbar, lock, stats, verbose): filepath
                for filepath in selected_files
            }
            
            for future in as_completed(future_to_file):
                filepath = future_to_file[future]
                try:
                    success, message = future.result()
                except Exception as e:
                    with lock:
                        stats['failed'] += 1
                        pbar.set_postfix_str(f"Success: {stats['success']}, Skipped: {stats['skipped']}, Failed: {stats['failed']}")
                        pbar.update(1)
                        if verbose:
                            tqdm.write(f"Exception occurred while processing file {filepath}: {e}")
    
    print(f"\nCopy complete! Success: {stats['success']} files, Skipped: {stats['skipped']} files, Failed: {stats['failed']} files")


def main():
    parser = argparse.ArgumentParser(description='MP4 Scene Selector - Select odd parts of each Scene')
    parser.add_argument('input_txt', help='txt file containing mp4 file paths')
    parser.add_argument('output_dir', help='Output directory path')
    parser.add_argument('--dry-run', action='store_true', help='Only show files that would be selected, do not actually copy')
    parser.add_argument('--threads', type=int, default=16, help='Number of threads to use for copying files (default: 16)')
    parser.add_argument('--verbose', action='store_true', help='Show detailed copy information')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_txt):
        print(f"Error: Input file does not exist: {args.input_txt}")
        return
    
    print(f"Reading file paths from: {args.input_txt}")
    
    with tqdm(desc="Reading file paths", unit="line") as pbar:
        file_paths = []
        try:
            with open(args.input_txt, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and line.endswith('.mp4'):
                        file_paths.append(line)
                    pbar.update(1)
        except Exception as e:
            print(f"Error reading file {args.input_txt}: {e}")
            return
    
    if not file_paths:
        print("No valid mp4 file paths found")
        return
    
    print(f"Total found {len(file_paths)} mp4 files")
    
    print("Grouping by movie and Scene...")
    scenes = defaultdict(list)
    
    with tqdm(file_paths, desc="Parsing filenames", unit="file") as pbar:
        for filepath in pbar:
            movie_prefix, scene_num, part_num = parse_filename(filepath)
            
            if movie_prefix is not None and scene_num is not None and part_num is not None:
                key = (movie_prefix, scene_num)
                scenes[key].append((filepath, part_num))
            else:
                if args.verbose:
                    print(f"Unable to parse filename: {filepath}")
    
    print(f"Found {len(scenes)} different movie-Scene combinations")
    
    selected_files = select_odd_parts(scenes)
    print(f"\nTotal selected {len(selected_files)} files")
    
    if args.dry_run:
        print("\n--- Dry run mode, the following files would be selected ---")
        for filepath in selected_files:
            print(os.path.basename(filepath))
    else:
        print(f"\nStarting to copy files to: {args.output_dir}")
        print(f"Using {args.threads} threads for copying")
        copy_files(selected_files, args.output_dir, max_workers=args.threads, verbose=args.verbose)


if __name__ == "__main__":
    main() 