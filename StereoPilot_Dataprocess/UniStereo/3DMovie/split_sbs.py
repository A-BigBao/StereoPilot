import os
import subprocess
import json
import argparse
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

class SBSProcessor:
    def __init__(self, max_workers=4, log_path=None, error_log_path=None):
        self.max_workers = max_workers
        self.log_path = log_path or "split_sbs.log"
        self.error_log_path = error_log_path or "split_sbs_errors.log"
        self.lock = threading.Lock()
        self.failed_files = []
        
    def write_log(self, msg, is_error=False):
        """Write log in a thread-safe manner"""
        log_file_path = self.error_log_path if is_error else self.log_path
        with self.lock:
            with open(log_file_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

    def get_video_info(self, video_path):
        """Get video information"""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_streams", str(video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return None, None
                
            info = json.loads(result.stdout)
            
            video_stream = next(
                (stream for stream in info["streams"] if stream["codec_type"] == "video"),
                None
            )
            
            if video_stream:
                width = int(video_stream["width"])
                height = int(video_stream["height"])
                return width, height
        except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError, ValueError) as e:
            self.write_log(f"❌ Failed to get video info {video_path}: {str(e)}", is_error=True)
            return None, None
        except Exception as e:
            self.write_log(f"❌ Exception getting video info {video_path}: {str(e)}", is_error=True)
            return None, None
        
        return None, None

    def detect_sbs_type(self, video_path):
        """Detect Full-SBS or Half-SBS format"""
        width, height = self.get_video_info(video_path)
        if not width or not height:
            return None
        
        ratio = width / height
        if ratio >= 3.0:
            return True  # Full-SBS
        else:
            return False  # Half-SBS

    def process_eye_video(self, video_path, output_dir, eye_type, is_full_sbs):
        """Process single eye video (left or right)"""
        video_basename = Path(video_path).stem
        output_filename = f"{eye_type}_{video_basename}.mp4"
        output_path = Path(output_dir) / output_filename
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if eye_type == "left":
            crop_filter = "crop=iw/2:ih:0:0"
        else:
            crop_filter = "crop=iw/2:ih:iw/2:0"
        
        if is_full_sbs:
            filter_v = crop_filter
            codec_args = ["-c:v", "copy"]
        else:
            filter_v = f"{crop_filter},scale=2*iw:ih,setsar=1"
            codec_args = ["-c:v", "libx264", "-crf", "18", "-preset", "medium"]
        
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", filter_v,
            *codec_args,
            "-an", "-sn",
            str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                error_msg = f"❌ {eye_type} eye processing failed: {video_path}\nError: {result.stderr}"
                self.write_log(error_msg, is_error=True)
                return None
            
            self.write_log(f"✅ {eye_type} eye processing completed: {output_path}")
            return str(output_path)
            
        except subprocess.TimeoutExpired:
            error_msg = f"❌ {eye_type} eye processing timeout: {video_path}"
            self.write_log(error_msg, is_error=True)
            return None
        except Exception as e:
            error_msg = f"❌ {eye_type} eye processing exception: {video_path} - {str(e)}"
            self.write_log(error_msg, is_error=True)
            return None

    def process_video(self, video_path, output_dir):
        """Process a single video file"""
        video_path = Path(video_path)
        
        if not video_path.exists():
            error_msg = f"⚠️ File does not exist: {video_path}"
            self.write_log(error_msg, is_error=True)
            self.failed_files.append(str(video_path))
            return {"success": False, "video_path": str(video_path), "error": "File does not exist"}
        
        is_full_sbs = self.detect_sbs_type(video_path)
        if is_full_sbs is None:
            error_msg = f"❌ Unable to get video information: {video_path}"
            self.write_log(error_msg, is_error=True)
            self.failed_files.append(str(video_path))
            return {"success": False, "video_path": str(video_path), "error": "Unable to get video information"}
        
        sbs_type = "Full" if is_full_sbs else "Half"
        self.write_log(f"🚀 Start processing: {video_path} ({sbs_type}-SBS)")
        
        results = {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            left_future = executor.submit(
                self.process_eye_video, video_path, output_dir, "left", is_full_sbs
            )
            right_future = executor.submit(
                self.process_eye_video, video_path, output_dir, "right", is_full_sbs
            )
            
            results["left"] = left_future.result()
            results["right"] = right_future.result()
        
        success = results["left"] is not None and results["right"] is not None
        
        if not success:
            self.failed_files.append(str(video_path))
            
        return {
            "success": success,
            "video_path": str(video_path),
            "left_output": results["left"],
            "right_output": results["right"]
        }

    def process_video_list(self, video_list, output_dir):
        """Process video list"""
        results = []
        successful_count = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_video = {
                executor.submit(self.process_video, video_path, output_dir): video_path 
                for video_path in video_list
            }
            
            with tqdm(total=len(video_list), desc="Processing videos", unit="video") as pbar:
                for future in as_completed(future_to_video):
                    video_path = future_to_video[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result["success"]:
                            successful_count += 1
                            
                        pbar.set_postfix({"Success": successful_count, "Failed": len(results) - successful_count})
                            
                    except Exception as e:
                        error_msg = f"❌ Exception occurred while processing video: {video_path} - {str(e)}"
                        self.write_log(error_msg, is_error=True)
                        self.failed_files.append(video_path)
                        results.append({
                            "success": False, 
                            "video_path": video_path, 
                            "error": str(e)
                        })
                    
                    pbar.update(1)
        
        return results, successful_count

def read_video_list(input_file):
    """Read video list file"""
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            videos = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        return videos
    except FileNotFoundError:
        print(f"❌ Input file does not exist: {input_file}")
        return None
    except Exception as e:
        print(f"❌ Error reading input file: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="SBS (Side-by-Side) video splitting tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python split_sbs.py -i video_list.txt -o output_dir
  python split_sbs.py -i video_list.txt -o output_dir -t 8
  python split_sbs.py --input video_list.txt --output output_dir --threads 4
        """
    )
    
    parser.add_argument(
        "-i", "--input", 
        required=True,
        help="Input video list file path"
    )
    
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output directory path"
    )
    
    parser.add_argument(
        "-t", "--threads",
        type=int,
        default=16,
        help="Number of parallel processing threads (default: 16)"
    )
    
    parser.add_argument(
        "--log-prefix",
        default="split_sbs",
        help="Log file name prefix (default: split_sbs)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"{args.log_prefix}.log"
    error_log_path = f"{args.log_prefix}_errors.log"
    
    video_list = read_video_list(args.input)
    if video_list is None:
        return 1
    
    if not video_list:
        print("⚠️ Input file is empty or has no valid video paths")
        return 1
    
    print(f"📋 Found {len(video_list)} video files")
    print(f"📁 Output directory: {output_dir}")
    print(f"🧵 Using {args.threads} threads")
    print(f"📝 Log file: {log_path}")
    print(f"📝 Error log: {error_log_path}")
    
    processor = SBSProcessor(
        max_workers=args.threads,
        log_path=log_path,
        error_log_path=error_log_path
    )
    
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"🎬 SBS processing started: {datetime.now()}\n\n")
    with open(error_log_path, "w", encoding="utf-8") as f:
        f.write(f"🎬 SBS error log started: {datetime.now()}\n\n")
    
    processor.write_log(f"📋 Input file: {args.input}")
    processor.write_log(f"📁 Output directory: {output_dir}")
    processor.write_log(f"🧵 Threads: {args.threads}")
    processor.write_log(f"📦 Videos to process: {len(video_list)}")
    
    try:
        results, successful_count = processor.process_video_list(video_list, output_dir)
        
        total_count = len(video_list)
        failed_count = total_count - successful_count
        
        print(f"\n📊 Processing complete statistics:")
        print(f"   Total: {total_count} videos")
        print(f"   Success: {successful_count} videos")
        print(f"   Failed: {failed_count} videos")
        
        processor.write_log(f"🏁 Processing complete - Success: {successful_count}/{total_count}")
        
        if processor.failed_files:
            processor.write_log("❌ Failed files list:", is_error=True)
            for failed_file in processor.failed_files:
                processor.write_log(f"   - {failed_file}", is_error=True)
            
            print(f"\n⚠️ {len(processor.failed_files)} files failed to process, see error log for details:")
            print(f"   {error_log_path}")
        
        return 0 if failed_count == 0 else 1
        
    except KeyboardInterrupt:
        print("\n⏹️ User interrupted processing")
        processor.write_log("⏹️ User interrupted processing")
        return 1
    except Exception as e:
        error_msg = f"❌ Unexpected error during processing: {str(e)}"
        print(error_msg)
        processor.write_log(error_msg, is_error=True)
        return 1

if __name__ == "__main__":
    exit(main())
