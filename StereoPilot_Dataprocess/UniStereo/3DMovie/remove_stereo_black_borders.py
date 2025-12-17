import os
import sys
import cv2
import numpy as np
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import shutil
from tqdm import tqdm

class StereoVideoProcessor:
    def __init__(self, output_dir, max_workers=16):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.lock = Lock()
        self.stats = {'processed': 0, 'cropped': 0, 'copied': 0, 'errors': 0}
        
        log_file = "remove_stereo_black_borders.log"
        self.log_file = open(log_file, 'w', encoding='utf-8')
    
    def __del__(self):
        if hasattr(self, 'log_file'):
            self.log_file.close()
    
    def log(self, message):
        with self.lock:
            self.log_file.write(f"{message}\n")
            self.log_file.flush()
    
    def detect_borders(self, video_path, sample_frames=5):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0, 0, 0, 0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if total_frames <= sample_frames:
            indices = list(range(total_frames))
        else:
            step = total_frames // sample_frames
            indices = list(range(0, total_frames, step))[:sample_frames]
        
        borders = {'top': [], 'bottom': [], 'left': [], 'right': []}
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            threshold = 10
            
            # Top border
            top = 0
            for i in range(height):
                if np.max(gray[i, :]) > threshold:
                    break
                top = i + 1
            
            # Bottom border
            bottom = 0
            for i in range(height - 1, -1, -1):
                if np.max(gray[i, :]) > threshold:
                    break
                bottom = height - i
            
            # Left border
            left = 0
            for i in range(width):
                if np.max(gray[:, i]) > threshold:
                    break
                left = i + 1
            
            # Right border
            right = 0
            for i in range(width - 1, -1, -1):
                if np.max(gray[:, i]) > threshold:
                    break
                right = width - i
            
            borders['top'].append(top)
            borders['bottom'].append(bottom)
            borders['left'].append(left)
            borders['right'].append(right)
        
        cap.release()
        
        if not borders['top']:
            return 0, 0, 0, 0
        
        return (int(np.median(borders['top'])), int(np.median(borders['bottom'])),
                int(np.median(borders['left'])), int(np.median(borders['right'])))
    
    def crop_video(self, input_path, output_path, top, bottom, left, right):
        try:
            cap = cv2.VideoCapture(input_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            new_width = width - left - right
            new_height = height - top - bottom
            
            if new_width <= 0 or new_height <= 0:
                return False
            
            cmd = [
                'ffmpeg', '-i', input_path,
                '-filter:v', f'crop={new_width}:{new_height}:{left}:{top}',
                '-c:a', 'copy', '-y', output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            return result.returncode == 0
            
        except Exception:
            return False
    
    def process_video_pair(self, left_path):
        left_path = Path(left_path)
        if not left_path.exists():
            with self.lock:
                self.stats['errors'] += 1
            self.log(f"ERROR: Left video not found: {left_path}")
            return False
        
        right_path = Path(str(left_path).replace('left_', 'right_'))
        if not right_path.exists():
            with self.lock:
                self.stats['errors'] += 1
            self.log(f"ERROR: Right video not found: {right_path}")
            return False
        
        left_output = self.output_dir / left_path.name
        right_output = self.output_dir / right_path.name
        
        try:
            top, bottom, left_crop, right_crop = self.detect_borders(str(left_path))
            
            # Add safety margin to ensure complete border removal
            safety_margin = 10
            top = top + safety_margin if top > 0 else 0
            bottom = bottom + safety_margin if bottom > 0 else 0
            left_crop = left_crop + safety_margin if left_crop > 0 else 0
            right_crop = right_crop + safety_margin if right_crop > 0 else 0
            
            has_borders = any([top > 0, bottom > 0, left_crop > 0, right_crop > 0])
            
            if has_borders:
                # Left video: normal cropping
                left_success = self.crop_video(str(left_path), str(left_output), 
                                             top, bottom, left_crop, right_crop)
                
                # Right video: mirror left/right cropping
                right_success = self.crop_video(str(right_path), str(right_output),
                                              top, bottom, right_crop, left_crop)
                
                if left_success and right_success:
                    with self.lock:
                        self.stats['cropped'] += 2
                        self.stats['processed'] += 2
                    self.log(f"CROPPED: {left_path.name} & {right_path.name} - borders: t{top} b{bottom} l{left_crop} r{right_crop}")
                    return True
                else:
                    with self.lock:
                        self.stats['errors'] += 2
                    self.log(f"ERROR: Crop failed for {left_path.name} & {right_path.name}")
                    return False
            else:
                shutil.copy2(left_path, left_output)
                shutil.copy2(right_path, right_output)
                
                with self.lock:
                    self.stats['copied'] += 2
                    self.stats['processed'] += 2
                self.log(f"COPIED: {left_path.name} & {right_path.name} - no borders detected")
                return True
                
        except Exception as e:
            with self.lock:
                self.stats['errors'] += 2
            self.log(f"ERROR: Processing failed for {left_path.name}: {str(e)}")
            return False
    
    def process_from_txt(self, txt_file):
        txt_path = Path(txt_file)
        if not txt_path.exists():
            raise FileNotFoundError(f"File not found: {txt_file}")
        
        video_paths = []
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and line.startswith('/') and Path(line).name.startswith('left_'):
                    video_paths.append(line)
        
        if not video_paths:
            print("No valid left_ video paths found in txt file")
            return
        
        print(f"Found {len(video_paths)} video pairs")
        print(f"Output directory: {self.output_dir}")
        print(f"Using {self.max_workers} threads")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.process_video_pair, path) for path in video_paths]
            
            with tqdm(total=len(video_paths), desc="Processing video pairs", unit="pair") as pbar:
                for future in as_completed(futures):
                    future.result()
                    pbar.update(1)
                    pbar.set_postfix({
                        'Processed': self.stats['processed'],
                        'Cropped': self.stats['cropped'],
                        'Copied': self.stats['copied'],
                        'Errors': self.stats['errors']
                    })
        
        print(f"\nProcessing completed:")
        print(f"Total processed: {self.stats['processed']} videos")
        print(f"Cropped: {self.stats['cropped']} videos")
        print(f"Copied: {self.stats['copied']} videos")
        print(f"Errors: {self.stats['errors']} videos")

def check_dependencies():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("ERROR: FFmpeg not found. Please install FFmpeg first.")
        sys.exit(1)
    
    try:
        import cv2
        import tqdm
    except ImportError as e:
        print(f"ERROR: Missing Python package: {e}")
        print("Install with: pip install opencv-python tqdm")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Remove black borders from stereo video pairs")
    parser.add_argument("txt_file", help="Text file containing left_ video paths")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--threads", type=int, default=16, help="Number of threads (default: 16)")
    
    args = parser.parse_args()
    
    check_dependencies()
    
    if args.threads <= 0:
        args.threads = 1
    
    try:
        processor = StereoVideoProcessor(args.output_dir, args.threads)
        processor.process_from_txt(args.txt_file)
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
