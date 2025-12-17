"""
batch conversion tool for video FPS
batch convert the video files in the input txt file to 16fps
without preserving audio and subtitle streams, using multi-threading, displaying progress and recording errors
"""

import os
import sys
import subprocess
import logging
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fps_conversion.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

error_lock = threading.Lock()
error_files = []

def check_ffmpeg():
    """check if ffmpeg is available"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("ffmpeg not found, please ensure it is installed and added to PATH")
        return False

def get_video_info(video_path):
    """get the video information"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', 
            '-show_streams', '-select_streams', 'v:0', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        import json
        data = json.loads(result.stdout)
        if 'streams' in data and len(data['streams']) > 0:
            stream = data['streams'][0]
            fps = stream.get('r_frame_rate', '0/1')
            if '/' in fps:
                num, den = map(float, fps.split('/'))
                current_fps = num / den if den != 0 else 0
            else:
                current_fps = float(fps)
            return current_fps
        return None
    except Exception as e:
        logger.error(f"Failed to get video information for {video_path}: {e}")
        return None

def convert_video_fps(video_path, output_dir=None):
    """
    convert the video to 16fps
    
    Args:
        video_path: the input video path
        output_dir: the output directory, if None, create a new directory next to the original file
    
    Returns:
        tuple: (whether the conversion is successful, error message)
    """
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        current_fps = get_video_info(video_path)
        if current_fps is None:
            raise ValueError("Failed to get video fps information")
        
        if abs(current_fps - 16.0) < 0.0001:
            logger.info(f"Video is already 16fps, skipping: {video_path}")
            return True, "Already 16fps"
        
        if current_fps < 16.0:
            logger.info(f"Video fps is less than 16({current_fps:.2f}), skipping (avoid interpolation): {video_path}")
            return True, "fps is less than 16, skipping"
        
        input_path = Path(video_path)
        if output_dir:
            output_path = Path(output_dir) / f"{input_path.stem}_16fps{input_path.suffix}"
        else:
            output_path = input_path.parent / f"{input_path.stem}_16fps{input_path.suffix}"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-vf', 'fps=16',
            '-c:v', 'libx264', 
            '-crf', '18', 
            '-preset', 'medium',
            '-pix_fmt', 'yuv420p',
            '-an',
            '-sn',
            '-y',
            str(output_path)
        ]
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise ValueError("Failed to create output file or it is empty")
        
        output_fps = get_video_info(str(output_path))
        if output_fps is None or abs(output_fps - 16.0) > 0.0001:
            logger.warning(f"Failed to validate output video fps: expected 16.0, actual {output_fps}")
        
        logger.info(f"Conversion successful: {video_path} -> {output_path} (fps: {current_fps:.2f} -> 16.0)")
        return True, None
        
    except subprocess.TimeoutExpired:
        error_msg = f"Conversion timed out: {video_path}"
        logger.error(error_msg)
        return False, error_msg
    except subprocess.CalledProcessError as e:
        error_msg = f"ffmpeg error for {video_path}: {e.stderr}"
        logger.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Conversion failed for {video_path}: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

def process_single_video(args):
    """function to process a single video"""
    video_path, output_dir = args
    success, error_msg = convert_video_fps(video_path, output_dir)
    
    if not success:
        with error_lock:
            error_files.append({
                'path': video_path,
                'error': error_msg
            })
    
    return success, video_path, error_msg

def read_video_list(txt_file):
    """read the video paths from the txt file"""
    video_paths = []
    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    if os.path.exists(line):
                        video_paths.append(line)
                    else:
                        logger.warning(f"Line {line_num} does not exist: {line}")
        
        logger.info(f"Read {len(video_paths)} valid video files from {txt_file}")
        return video_paths
    except Exception as e:
        logger.error(f"Failed to read video list: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description='Batch convert videos to 16fps')
    parser.add_argument('input_txt', help='txt file containing video paths')
    parser.add_argument('-o', '--output_dir', type=str, default='./3DMovie_fps16_nonnative', help='Output directory (optional, default next to original file)')
    parser.add_argument('-t', '--threads', type=int, default=8, help='Thread number (default 8)')
    parser.add_argument('--dry-run', action='store_true', help='Only check files, no actual conversion')
    
    args = parser.parse_args()
    
    if not check_ffmpeg():
        return 1
    
    video_paths = read_video_list(args.input_txt)
    if not video_paths:
        logger.error("No valid video files found")
        return 1
    
    logger.info(f"Preparing to process {len(video_paths)} video files, using {args.threads} threads")
    
    if args.dry_run:
        logger.info("Dry run mode, only checking files:")
        for i, path in enumerate(video_paths, 1):
            fps = get_video_info(path)
            print(f"{i:3d}. {path} (current fps: {fps:.2f})")
        return 0
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        future_to_path = {
            executor.submit(process_single_video, (path, args.output_dir)): path 
            for path in video_paths
        }
        
        with tqdm(total=len(video_paths), desc="Conversion progress", unit="video") as pbar:
            for future in as_completed(future_to_path):
                success, video_path, error_msg = future.result()
                if success:
                    success_count += 1
                pbar.update(1)
                pbar.set_postfix({
                    'Success': success_count,
                    'Failed': len(error_files)
                })
    
    total_files = len(video_paths)
    failed_count = len(error_files)
    
    logger.info(f"\n=== Processing completed ===")
    logger.info(f"Total files: {total_files}")
    logger.info(f"Successfully converted: {success_count}")
    logger.info(f"Failed files: {failed_count}")
    
    if error_files:
        error_report_path = 'step1_fps_conversion_errors.txt'
        with open(error_report_path, 'w', encoding='utf-8') as f:
            f.write("=== FPS conversion error report ===\n\n")
            for error_info in error_files:
                f.write(f"File: {error_info['path']}\n")
                f.write(f"Error: {error_info['error']}\n")
                f.write("-" * 50 + "\n")
        
        logger.info(f"Error details saved to: {error_report_path}")
    
    return 0 if failed_count == 0 else 1

if __name__ == '__main__':
    sys.exit(main()) 