import argparse
import os
import subprocess
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

lock = threading.Lock()

def get_video_frame_count(video_path):
    """Get the total frame count of a video"""
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-select_streams', 'v:0',
            '-count_frames',
            '-show_entries', 'stream=nb_frames',
            '-of', 'json',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if 'streams' in data and len(data['streams']) > 0:
                frames = data['streams'][0].get('nb_frames')
                if frames:
                    return int(frames), None
                else:
                    return None, "Video file has no nb_frames information"
            else:
                return None, "Unable to get video stream information"
        else:
            return None, f"ffprobe execution failed: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return None, "ffprobe execution timeout"
    except Exception as e:
        return None, f"Error getting frame count: {str(e)}"


def process_single_video(video_path, min_frames, pbar=None):
    """Process a single video file"""
    if not os.path.exists(video_path):
        with lock:
            if pbar:
                pbar.write(f"❌ File does not exist: {video_path}")
        return {"valid": False, "video_path": video_path, "error": "File does not exist", "frame_count": 0}
    
    frame_count, error = get_video_frame_count(video_path)
    
    if error:
        with lock:
            if pbar:
                pbar.write(f"❌ {os.path.basename(video_path)}: {error}")
        return {"valid": False, "video_path": video_path, "error": error, "frame_count": 0}
    
    if frame_count >= min_frames:
        with lock:
            if pbar:
                pbar.write(f"✅ {os.path.basename(video_path)}: {frame_count} frames (kept)")
        return {"valid": True, "video_path": video_path, "frame_count": frame_count}
    else:
        with lock:
            if pbar:
                pbar.write(f"🚫 {os.path.basename(video_path)}: {frame_count} frames (filtered)")
        return {"valid": False, "video_path": video_path, "frame_count": frame_count, "error": f"Insufficient frames ({frame_count}<{min_frames})"}

def main(input_txt, output_txt=None, min_frames=65, max_workers=16):
    """Main function"""
    if not os.path.exists(input_txt):
        print(f"❌ Input file does not exist: {input_txt}")
        return
    
    with open(input_txt, 'r', encoding='utf-8') as f:
        video_paths = [line.strip() for line in f.readlines() if line.strip()]
    
    if not video_paths:
        print("❌ No video paths found in input file")
        return
    
    print(f"📝 Read {len(video_paths)} video files from {input_txt}")
    print(f"🔍 Filter criteria: ≥{min_frames} frames")
    print(f"🚀 Using {max_workers} threads for parallel processing...")
    
    valid_videos = []
    invalid_videos = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(video_paths), desc="Checking videos", unit="video") as pbar:
            future_to_path = {
                executor.submit(process_single_video, video_path, min_frames, pbar): video_path 
                for video_path in video_paths
            }
            
            for future in as_completed(future_to_path):
                try:
                    result = future.result()
                    if result["valid"]:
                        valid_videos.append(result)
                    else:
                        invalid_videos.append(result)
                except Exception as e:
                    video_path = future_to_path[future]
                    pbar.write(f"❌ Error processing {video_path}: {str(e)}")
                    invalid_videos.append({"valid": False, "video_path": video_path, "error": str(e)})
                
                pbar.update(1)
    
    if not output_txt:
        output_txt = input_txt.replace('.txt', f'_filtered_{min_frames}frames.txt')
    
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(f"# Filtered video file list (≥{min_frames} frames)\n")
        f.write(f"# Original files: {len(video_paths)}, Valid files: {len(valid_videos)}\n\n")
        for item in valid_videos:
            f.write(f"{item['video_path']}\n")
    
    if invalid_videos:
        error_file = output_txt.replace('.txt', '_removed.txt')
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write(f"# List of filtered out video files\n")
            f.write(f"# Total {len(invalid_videos)} files filtered\n\n")
            for item in invalid_videos:
                frame_info = f"({item['frame_count']} frames)" if item.get('frame_count', 0) > 0 else ""
                f.write(f"{item['video_path']}\t# {item.get('error', 'Unknown error')} {frame_info}\n")
        print(f"📝 Filtered files list saved to: {error_file}")
    
    print(f"\n🎉 Processing complete!")
    print(f"✅ Valid videos: {len(valid_videos)}/{len(video_paths)}")
    print(f"🚫 Filtered out: {len(invalid_videos)}")
    print(f"📄 Results saved to: {output_txt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter video file list by frame count")
    parser.add_argument("input_txt", type=str, help="Input txt file with one video file path per line")
    parser.add_argument("--output", "-o", type=str, help="Output txt file path (default: input_filename_filtered_[min_frames]frames.txt)")
    parser.add_argument("--min-frames", "-f", type=int, default=81, help="Minimum frame count threshold (default: 81)")
    parser.add_argument("--workers", "-w", type=int, default=16, help="Number of parallel processing threads (default: 16)")
    
    args = parser.parse_args()
    main(args.input_txt, args.output, args.min_frames, args.workers) 