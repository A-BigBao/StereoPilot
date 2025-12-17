"""
Distributed batch process rectify_832.py automation script
Features:
1. Scan all npz files in npz_folder
2. Find corresponding raw video files based on file names
3. Split the files to be processed into multiple shards, support multi-machine parallel processing
4. Batch execute rectify_832.py to process the specified shards
5. Statistics success/failure numbers and record to file
"""

import os
import os.path as osp
import subprocess
import argparse
from pathlib import Path
import glob
from tqdm import tqdm
import time 
import math
from concurrent.futures import ThreadPoolExecutor
import threading


def load_processed_files(processed_file):
    """Load processed file list"""
    if osp.exists(processed_file):
        with open(processed_file, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def save_processed_file(processed_file, vid):
    """Save processed file"""
    with open(processed_file, 'a') as f:
        f.write(f"{vid}\n")


def get_npz_files_list(npz_folder, npz_list_file):
    """Get npz file list, read from cache file first"""
    if osp.exists(npz_list_file):
        print(f"Read npz list from cache file: {npz_list_file}")
        with open(npz_list_file, 'r') as f:
            npz_files = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(npz_files)} npz files from cache")
        return npz_files
    
    print(f"Scan npz folder: {npz_folder}")
    npz_pattern = osp.join(npz_folder, "*.npz")
    npz_paths = glob.glob(npz_pattern)
    npz_files = [osp.basename(path) for path in npz_paths]
    
    with open(npz_list_file, 'w') as f:
        for npz_file in npz_files:
            f.write(f"{npz_file}\n")
    
    print(f"Found {len(npz_files)} npz files, saved to {npz_list_file}")
    return npz_files


def find_matching_video_files(npz_files, raw_video_folder):
    """Find corresponding video files based on npz file names"""
    matching_pairs = []
    # video_extensions = ['.webm', '.mp4']  # Supported video formats
    # we only download webm format.
    video_extensions = ['.webm']  # Supported video formats
    
    print(f"Find corresponding video files in video folder: {raw_video_folder}")
    
    for npz_file in tqdm(npz_files, desc="Find matching video files"):
        vid = osp.splitext(npz_file)[0]
        
        if '_' in vid:
            raw_video_id = vid.rsplit('_', 1)[0]
        else:
            raw_video_id = vid
        
        video_found = False
        for ext in video_extensions:
            video_path = osp.join(raw_video_folder, f"{raw_video_id}{ext}")
            if osp.exists(video_path):
                matching_pairs.append((vid, npz_file, video_path))
                video_found = True
                break
        
        # if not video_found:
        #     print(f"Warning: No corresponding video file found for {vid} (raw_video_id: {raw_video_id})")
    
    print(f"Found {len(matching_pairs)} matching npz-video pairs")
    return matching_pairs


def split_files_into_shards(file_list, total_shards, current_shard):
    """Split file list into specified number of shards, and return the files in the current shard"""
    if not (0 <= current_shard < total_shards):
        raise ValueError(f"Shard number {current_shard} out of range [0, {total_shards-1}]")
    
    total_files = len(file_list)
    # Use integer division and modulo to ensure the shard size is as uniform as possible
    base_shard_size = total_files // total_shards
    remainder = total_files % total_shards
    
    # The first `remainder` shards will have one more file
    start_idx = current_shard * base_shard_size + min(current_shard, remainder)
    
    # The size of the current shard
    current_shard_size = base_shard_size + (1 if current_shard < remainder else 0)
    end_idx = start_idx + current_shard_size
    
    return file_list[start_idx:end_idx]


def save_shard_info(shard_info_file, total_shards, current_shard, total_shard_files, pending_shard_files):
    """Save shard information to file"""
    with open(shard_info_file, 'w') as f:
        f.write(f"Total shards number: {total_shards}\n")
        f.write(f"Current shard: {current_shard}\n")
        f.write(f"Generation time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total files in this shard: {len(total_shard_files)}\n")
        f.write(f"Pending files in this shard: {len(pending_shard_files)}\n\n")
        
        f.write(f"--- Pending files list ({len(pending_shard_files)}) ---\n")
        if not pending_shard_files:
            f.write("None\n")
        else:
            for i, (vid, _, _) in enumerate(pending_shard_files):
                f.write(f"{i+1:4d}. {vid}\n")
        
        f.write(f"\n--- All files in this shard ({len(total_shard_files)}) ---\n")
        for i, (vid, _, _) in enumerate(total_shard_files):
            f.write(f"{i+1:4d}. {vid}\n")


# Multi-threading synchronization lock
log_lock = threading.Lock()
processed_file_lock = threading.Lock()
stats_lock = threading.Lock()


def safe_log_write(log_f, message):
    """Thread-safe log writing"""
    with log_lock:
        log_f.write(message)
        log_f.flush()


def safe_save_processed_file(processed_file, vid):
    """Thread-safe processed file saving"""
    with processed_file_lock:
        with open(processed_file, 'a') as f:
            f.write(f"{vid}\n")


def process_single_video(args_tuple, log_f, processed_file, stats):
    """Function to process a single video, for multi-threading call"""
    vid, npz_file, video_path, npz_folder, raw_video_folder, output_folder, output_hfov = args_tuple
    
    print(f"\nProcessing: {vid}")
    safe_log_write(log_f, f"Processing: {vid} - ")
    
    success, message = run_rectify(
        vid, 
        npz_folder, 
        raw_video_folder, 
        output_folder, 
        output_hfov
    )
    
    if success:
        with stats_lock:
            stats['success_count'] += 1
        safe_save_processed_file(processed_file, vid)
        print(f"✓ Success: {vid}")
        safe_log_write(log_f, f"SUCCESS\n")
    else:
        with stats_lock:
            stats['failed_count'] += 1
        print(f"✗ Failed: {vid} - {message}")
        # Format multi-line error information into log-friendly format
        log_message = message.replace('\n', '\n    ')  # Add indentation to multi-line information
        safe_log_write(log_f, f"FAILED:\n    {log_message}\n")
    
    # Display current statistics
    with stats_lock:
        total_processed = stats['success_count'] + stats['failed_count']
        print(f"Total progress: {total_processed}/{stats['total_files']} - Success: {stats['success_count']}, Failed: {stats['failed_count']}")
    
    return success


def run_rectify(vid, npz_folder, raw_video_folder, output_folder, output_hfov):
    """Execute a single rectify_832.py task"""
    cmd = [
        "python", "./rectify_832.py",
        f"--vid={vid}",
        f"--npz_folder={npz_folder}",
        f"--raw_video_folder={raw_video_folder}",
        f"--output_folder={output_folder}",
        f"--output_hfov={output_hfov}"
    ]
    
    env = os.environ.copy()
    env['JAX_PLATFORMS'] = 'cpu'
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600 
        )
        
        if result.returncode == 0:
            return True, "SUCCESS"
        else:
            error_msg = f"FAILED - Return code: {result.returncode}"
            if result.stderr:
                error_msg += f"\nSTDERR:\n{result.stderr}"
            if result.stdout:
                error_msg += f"\nSTDOUT:\n{result.stdout}"
            return False, error_msg
            
    except subprocess.TimeoutExpired:
        return False, "FAILED - Timeout (1 hour)"
    except Exception as e:
        return False, f"FAILED - Exception: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="Distributed batch process rectify_832.py")
    # Put rectify_832.py in the same folder with this script
    parser.add_argument('--npz_folder', type=str, 
                       default='./stereo4d-code/stereo4d/train',
                       help='npz folder path')
    parser.add_argument('--raw_video_folder', type=str,
                       default='./stereo4d-code/stereo4d_raw',
                       help='raw video folder path')
    parser.add_argument('--output_folder', type=str,
                       default='./stereo4d_processed',
                       help='output folder path')
    parser.add_argument('--output_hfov', type=float, default=90.0,
                       help='output horizontal fov')
    parser.add_argument('--total_shards', type=int, required=True,
                       help='total shards number (required)')
    parser.add_argument('--current_shard', type=int, required=True,
                       help='current shard number, from 0 (required)')
    parser.add_argument('--dry_run', action='store_true',
                       help='only show shard information, not process')

    args = parser.parse_args()
    
    if args.current_shard < 0 or args.current_shard >= args.total_shards:
        print(f"Error: Shard number {args.current_shard} out of range [0, {args.total_shards-1}]")
        return
    
    shard_suffix = f"shard_{args.current_shard}_of_{args.total_shards}"
    npz_list_file = "../../npz_train_files_list.txt"
    processed_file = f"processed_train_files_480p_{shard_suffix}.txt"
    log_file = f"batch_rectify_train_480p_{shard_suffix}.log"
    shard_info_file = f"shard_info_{shard_suffix}.txt"
    
    print("=== Distributed batch process rectify_832.py script ===")
    print(f"NPZ folder: {args.npz_folder}")
    print(f"Raw video folder: {args.raw_video_folder}")
    print(f"Output folder: {args.output_folder}")
    print(f"Output HFOV: {args.output_hfov}")
    print(f"Total shards number: {args.total_shards}")
    print(f"Current shard number: {args.current_shard}")
    print(f"Only preview mode: {args.dry_run}")
    print()

    if not osp.exists(args.npz_folder):
        print(f"Error: NPZ folder does not exist: {args.npz_folder}")
        return
    
    if not osp.exists(args.raw_video_folder):
        print(f"Error: Raw video folder does not exist: {args.raw_video_folder}")
        return
    
    os.makedirs(args.output_folder, exist_ok=True)
    
    npz_files = get_npz_files_list(args.npz_folder, npz_list_file)
    if not npz_files:
        print("Error: No npz files found")
        return
    
    matching_pairs = find_matching_video_files(npz_files, args.raw_video_folder)
    if not matching_pairs:
        print("Error: No matching video files found")
        return
        
    sorted_matching_pairs = sorted(matching_pairs, key=lambda p: p[0])

    try:
        this_shard_total_files = split_files_into_shards(sorted_matching_pairs, args.total_shards, args.current_shard)
    except ValueError as e:
        print(f"Error: {e}")
        return

    print(f"\nShard information:")
    print(f"  Total files to match: {len(sorted_matching_pairs)}")
    print(f"  Total files assigned to this shard: {len(this_shard_total_files)}")

    processed_vids = set()
    if osp.exists(processed_file):
        print(f"Load shard {args.current_shard} processed files: {processed_file}")
        shard_processed_vids = load_processed_files(processed_file)
        processed_vids.update(shard_processed_vids)
        print(f"Loaded {len(shard_processed_vids)} shard processed items")
    
    print(f"Total processed file IDs (after deduplication): {len(processed_vids)}")

    shard_pending_pairs = [(vid, npz, path) for vid, npz, path in this_shard_total_files
                           if vid not in processed_vids]
    
    print(f"Current shard pending files number: {len(shard_pending_pairs)}")
    print()

    save_shard_info(shard_info_file, args.total_shards, args.current_shard, this_shard_total_files, shard_pending_pairs)
    print(f"Shard detailed information saved to: {shard_info_file}")
    
    if not shard_pending_pairs:
        print("Current shard has no files to process!")
        return
    
    if args.dry_run:
        print("\n=== Preview mode - Current shard pending files list ===")
        for i, (vid, npz_file, video_path) in enumerate(shard_pending_pairs[:10]):
            print(f"{i+1:4d}. {vid}")
        if len(shard_pending_pairs) > 10:
            print(f"     ... there are {len(shard_pending_pairs)-10} more files")
        print(f"\nDetailed list please see: {shard_info_file}")
        return
    
    stats = {
        'success_count': 0,
        'failed_count': 0,
        'total_files': len(shard_pending_pairs)
    }
    
    task_args = [
        (vid, npz_file, video_path, args.npz_folder, args.raw_video_folder, args.output_folder, args.output_hfov)
        for vid, npz_file, video_path in shard_pending_pairs
    ]

    with open(log_file, 'a') as log_f:
        log_f.write(f"\n=== Shard {args.current_shard}/{args.total_shards} processing started: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        log_f.write(f"Current shard pending files number: {len(shard_pending_pairs)}\n")
        log_f.write(f"Using 4 threads to process\n")
        
        print(f"Start using 4 threads to process {len(shard_pending_pairs)} files...")
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for args_tuple in task_args:
                future = executor.submit(process_single_video, args_tuple, log_f, processed_file, stats)
                futures.append(future)
            
            with tqdm(total=len(futures), desc=f"Processing shard {args.current_shard}") as pbar:
                for future in futures:
                    try:
                        future.result()
                        pbar.update(1)
                    except Exception as e:
                        print(f"Exception occurred during thread processing: {e}")
                        pbar.update(1)
    
    success_count = stats['success_count']
    failed_count = stats['failed_count']
    
    print(f"\n=== Shard {args.current_shard}/{args.total_shards} processing completed ===")
    print(f"Number of files processed in this shard: {len(shard_pending_pairs)}")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Success rate: {success_count/(success_count+failed_count)*100:.1f}%" if (success_count+failed_count) > 0 else "No data")
    print(f"Shard processed files recorded in: {processed_file}")
    print(f"Detailed log recorded in: {log_file}")
    print(f"Shard information recorded in: {shard_info_file}")


if __name__ == "__main__":
    main() 