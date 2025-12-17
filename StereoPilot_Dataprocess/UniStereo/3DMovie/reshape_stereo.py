#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stereo Video Reshaping Tool

This script reads a list of left eye videos, automatically infers corresponding right eye videos,
and uses FFmpeg to uniformly resize left and right eye videos to 832×480 resolution,
ensuring identical geometric transformations, no distortion, and pixel alignment.

Usage:
    python reshape_stereo.py --list left_list.txt --out_dir ./output/

Input file format:
    One absolute path per line to MP4 files starting with 'left'

Output:
    - Left and right eye videos resized to 832×480 resolution
    - Filenames appended with _832x480.mp4 suffix
    - Saved to specified output directory
"""

import os
import sys
import argparse
import logging
import subprocess
import shutil
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reshape_stereo.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class StereoVideoReshaper:
    """Stereo Video Reshaper"""
    
    def __init__(self, list_file, out_dir=None, max_workers=4):
        """
        Initialize reshaper
        
        Args:
            list_file (str): Left eye video list file path
            out_dir (str): Output directory, defaults to same level as source files
            max_workers (int): Maximum number of concurrent processes
        """
        self.list_file = list_file
        self.out_dir = out_dir
        self.max_workers = max_workers
        self.left_video_paths = []
        self.video_pairs = []
        self.processed_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        
        self.target_width = 832
        self.target_height = 480
        self.video_filter = f"scale={self.target_width}:{self.target_height}:force_original_aspect_ratio=increase:flags=lanczos,crop={self.target_width}:{self.target_height},setsar=1:1"
        self.codec_params = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium"]
    
    def check_ffmpeg(self):
        """Check if FFmpeg is available"""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"], 
                capture_output=True, 
                text=True, 
                timeout=1000
            )
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                logger.info(f"FFmpeg available: {version_line}")
                return True
            else:
                logger.error("FFmpeg not available")
                return False
        except FileNotFoundError:
            logger.error("FFmpeg not installed or not in PATH")
            return False
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg version check timeout")
            return False
        except Exception as e:
            logger.error(f"Error checking FFmpeg: {e}")
            return False
    
    def load_video_list(self):
        """Load left eye video paths from txt file"""
        try:
            with open(self.list_file, 'r', encoding='utf-8') as f:
                self.left_video_paths = [line.strip() for line in f if line.strip()]
            logger.info(f"Successfully loaded {len(self.left_video_paths)} left eye video file paths")
            return True
        except Exception as e:
            logger.error(f"Failed to load video paths: {e}")
            return False
    
    def get_video_resolution(self, video_path):
        """
        Get video resolution
        
        Args:
            video_path (str): Video file path
            
        Returns:
            tuple: (width, height) or (None, None) if failed
        """
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json", 
                "-show_streams", "-select_streams", "v:0", video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                stream = data['streams'][0]
                width = stream.get('width', 0)
                height = stream.get('height', 0)
                return width, height
            else:
                logger.error(f"Unable to get video resolution: {video_path}")
                return None, None
                
        except Exception as e:
            logger.error(f"Error getting video resolution: {video_path} - {e}")
            return None, None
    
    def check_video_requirements(self, video_path):
        """
        Check if video meets processing requirements
        
        Args:
            video_path (str): Video file path
            
        Returns:
            tuple: (is_valid, width, height, aspect_ratio, message)
        """
        width, height = self.get_video_resolution(video_path)
        
        if width is None or height is None:
            return False, None, None, None, "Unable to get video resolution"
        
        if width < 1776:
            return False, width, height, None, f"Width {width} is less than minimum requirement 1776"
        
        if height < 480:
            return False, width, height, None, f"Height {height} is less than minimum requirement 480"
        
        aspect_ratio = width / height
        
        return True, width, height, aspect_ratio, "Check passed"

    def find_right_video(self, left_path):
        """
        Infer right eye video path from left eye video path
        
        Args:
            left_path (str): Left eye video file path
            
        Returns:
            str: Right eye video path, or None if not found
        """
        try:
            dir_path = os.path.dirname(left_path)
            filename = os.path.basename(left_path)
            
            if filename.startswith('left'):
                right_filename = 'right' + filename[4:]
                right_path = os.path.join(dir_path, right_filename)
            else:
                logger.warning(f"Left eye video filename does not start with 'left': {left_path}")
                return None
            
            if os.path.exists(right_path):
                return right_path
            else:
                logger.warning(f"Right eye video does not exist: {right_path}")
                return None
        except Exception as e:
            logger.error(f"Error inferring right eye video path: {left_path} - {e}")
            return None
    
    def prepare_video_pairs(self):
        """Prepare video pair list"""
        logger.info("Preparing video pair list...")
        
        for left_path in self.left_video_paths:
            if not os.path.exists(left_path):
                logger.warning(f"Left eye video does not exist: {left_path}")
                self.skipped_count += 1
                continue
            
            is_valid, width, height, aspect_ratio, message = self.check_video_requirements(left_path)
            if not is_valid:
                logger.warning(f"Left eye video does not meet requirements: {os.path.basename(left_path)} - {message}")
                self.skipped_count += 1
                continue
            
            logger.info(f"Left eye video check passed: {os.path.basename(left_path)} ({width}×{height})")
            
            right_path = self.find_right_video(left_path)
            if right_path is None:
                self.skipped_count += 1
                continue
            
            _, width_right, height_right, _, _ = self.check_video_requirements(right_path)
                
            if width != width_right or height != height_right:
                logger.warning(f"Left and right eye video resolutions do not match: Left {width}×{height} vs Right {width_right}×{height_right} - {os.path.basename(left_path)}")
                self.skipped_count += 1
                continue
            
            logger.info(f"Left and right eye video resolutions match: {width}×{height} - {os.path.basename(left_path)}")
            
            if self.out_dir:
                output_dir = self.out_dir
            else:
                output_dir = os.path.dirname(left_path)
            
            left_basename = os.path.splitext(os.path.basename(left_path))[0]
            right_basename = os.path.splitext(os.path.basename(right_path))[0]
            
            left_output = os.path.join(output_dir, f"{left_basename}_832x480.mp4")
            right_output = os.path.join(output_dir, f"{right_basename}_832x480.mp4")
            
            self.video_pairs.append({
                'left_input': left_path,
                'right_input': right_path,
                'left_output': left_output,
                'right_output': right_output,
                'output_dir': output_dir
            })
        
        logger.info(f"Preparation complete: {len(self.video_pairs)} video pairs to process, {self.skipped_count} pairs skipped")
    
    def process_video_pair(self, video_pair):
        """
        Process a single video pair
        
        Args:
            video_pair (dict): Video pair information
            
        Returns:
            tuple: (success, error_message, process_time)
        """
        try:
            left_input = video_pair['left_input']
            right_input = video_pair['right_input']
            left_output = video_pair['left_output']
            right_output = video_pair['right_output']
            output_dir = video_pair['output_dir']
            
            os.makedirs(output_dir, exist_ok=True)
            
            cmd = [
                "ffmpeg", "-y",
                "-i", left_input,
                "-i", right_input,
                "-filter_complex", 
                f"[0:v]{self.video_filter}[left_out];[1:v]{self.video_filter}[right_out]",
                "-map", "[left_out]",
                *self.codec_params,
                left_output,
                "-map", "[right_out]",
                *self.codec_params,
                right_output
            ]
            
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3000  
            )
            
            process_time = time.time() - start_time
            
            if result.returncode == 0:
                if (os.path.exists(left_output) and os.path.exists(right_output) and
                    os.path.getsize(left_output) > 1000 and os.path.getsize(right_output) > 1000):
                    return True, None, process_time
                else:
                    error_msg = "Output files do not exist or have abnormal size"
                    logger.error(f"Processing failed {left_input}: {error_msg}")
                    return False, error_msg, process_time
            else:
                error_msg = f"FFmpeg return code: {result.returncode}, Error: {result.stderr}"
                logger.error(f"Processing failed {left_input}: {error_msg}")
                return False, error_msg, process_time
                
        except subprocess.TimeoutExpired:
            error_msg = "FFmpeg processing timeout"
            logger.error(f"Processing timeout {left_input}: {error_msg}")
            return False, error_msg, 0
        except Exception as e:
            error_msg = f"Processing exception: {str(e)}"
            logger.error(f"Processing exception {left_input}: {error_msg}")
            return False, error_msg, 0
    
    def process_all_videos(self):
        """Process all video pairs"""
        if not self.video_pairs:
            logger.warning("No video pairs to process")
            return
        
        logger.info(f"Starting to process {len(self.video_pairs)} video pairs using {self.max_workers} processes")
        
        total_process_time = 0
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_pair = {
                executor.submit(self.process_video_pair, pair): pair 
                for pair in self.video_pairs
            }
            
            with tqdm(total=len(self.video_pairs), desc="Processing progress") as pbar:
                for future in as_completed(future_to_pair):
                    video_pair = future_to_pair[future]
                    try:
                        success, error_msg, process_time = future.result()
                        total_process_time += process_time
                        
                        if success:
                            self.processed_count += 1
                            logger.info(f"Processing successful: {os.path.basename(video_pair['left_input'])} ({process_time:.1f}s)")
                        else:
                            self.failed_count += 1
                            logger.error(f"Processing failed: {os.path.basename(video_pair['left_input'])} - {error_msg}")
                    except Exception as e:
                        self.failed_count += 1
                        logger.error(f"Task exception: {os.path.basename(video_pair['left_input'])} - {str(e)}")
                    
                    pbar.update(1)
        
        total_pairs = len(self.video_pairs)
        avg_time = total_process_time / total_pairs if total_pairs > 0 else 0
        
        logger.info("=" * 60)
        logger.info("Processing complete statistics:")
        logger.info(f"Total video pairs: {len(self.left_video_paths)}")
        logger.info(f"Skipped: {self.skipped_count}")
        logger.info(f"Attempted: {total_pairs}")
        logger.info(f"Successful: {self.processed_count}")
        logger.info(f"Failed: {self.failed_count}")
        logger.info(f"Success rate: {self.processed_count/total_pairs*100:.1f}%" if total_pairs > 0 else "Success rate: 0%")
        logger.info(f"Total processing time: {total_process_time:.1f}s")
        logger.info(f"Average processing time: {avg_time:.1f}s/pair")
        logger.info("=" * 60)
    
    def run(self):
        """Run complete processing workflow"""
        logger.info("Starting stereo video reshaping processing...")
        
        if not self.check_ffmpeg():
            logger.error("FFmpeg check failed, cannot continue")
            return False
        
        if not self.load_video_list():
            logger.error("Failed to load video list")
            return False
        
        self.prepare_video_pairs()
        
        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)
            logger.info(f"Output directory: {self.out_dir}")
        
        self.process_all_videos()
        
        return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Stereo Video Reshaping Tool - Uniformly resize left and right eye videos to 832×480 resolution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python reshape_stereo.py --list left_videos.txt
    python reshape_stereo.py --list left_videos.txt --out_dir ./output/
    python reshape_stereo.py --list left_videos.txt --out_dir ./output/ --workers 8
        """
    )
    
    parser.add_argument('--list', required=True, 
                       help='txt file containing left eye MP4 file paths (required)')
    parser.add_argument('--out_dir', default=None,
                       help='Output directory (optional, defaults to same level as source files)')
    parser.add_argument('--workers', type=int, default=16,
                       help='Number of concurrent processes (default: 16)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.list):
        print(f"Error: Input file does not exist: {args.list}")
        sys.exit(1)
    
    reshaper = StereoVideoReshaper(args.list, args.out_dir, args.workers)
    
    try:
        success = reshaper.run()
        if success:
            logger.info("Stereo video reshaping processing complete!")
        else:
            logger.error("Stereo video reshaping processing failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("User interrupted program execution")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Program execution error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 