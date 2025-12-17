import argparse
import os
import subprocess
import json
import math
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

lock = threading.Lock()

def get_video_info(video_path):
    """Get video frame count and frame rate information"""
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-select_streams', 'v:0',
            '-count_frames',
            '-show_entries', 'stream=nb_frames,r_frame_rate',
            '-of', 'json',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if 'streams' in data and len(data['streams']) > 0:
                stream = data['streams'][0]
                frames = stream.get('nb_frames')
                fps_str = stream.get('r_frame_rate', '25/1')
                
                if frames:
                    if '/' in fps_str:
                        num, den = fps_str.split('/')
                        fps = float(num) / float(den) if float(den) != 0 else 25
                    else:
                        fps = float(fps_str)
                    
                    return int(frames), fps, None
                else:
                    return None, None, "Video file has no nb_frames information"
            else:
                return None, None, "Unable to get video stream information"
        else:
            return None, None, f"ffprobe execution failed: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return None, None, "ffprobe execution timeout"
    except Exception as e:
        return None, None, f"Error getting video information: {str(e)}"

def split_video_by_frames(video_path, total_frames, fps, frames_per_segment=81, output_dir=None):
    """Split video by frame count precisely"""
    try:
        video_name = os.path.basename(video_path)
        video_stem = os.path.splitext(video_name)[0]
        video_ext = os.path.splitext(video_name)[1]
        
        if output_dir is None:
            output_dir = os.path.dirname(video_path)
        
        os.makedirs(output_dir, exist_ok=True)
        
        num_segments = total_frames // frames_per_segment
        
        if num_segments == 0:
            return False, f"Insufficient frames ({total_frames}<{frames_per_segment})", []
        
        output_files = []
        failed_segments = []
        
        for i in range(num_segments):
            start_frame = i * frames_per_segment
            end_frame = start_frame + frames_per_segment - 1
            output_file = os.path.join(output_dir, f"{video_stem}_{i+1:03d}{video_ext}")
            
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vf', f'select=between(n\\,{start_frame}\\,{end_frame}),setpts=N/{fps}/TB,fps={fps}',
                '-vsync', '0',
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '18',
                '-an',
                '-pix_fmt', 'yuv420p',
                '-y',
                output_file
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if result.returncode == 0:
                    output_files.append(output_file)
                else:
                    failed_segments.append(f"Segment {i+1}: {result.stderr}")
            except subprocess.TimeoutExpired:
                failed_segments.append(f"Segment {i+1}: ffmpeg execution timeout")
            except Exception as e:
                failed_segments.append(f"Segment {i+1}: {str(e)}")
        
        if failed_segments:
            error_msg = "; ".join(failed_segments)
            return len(output_files) > 0, f"Some segments failed: {error_msg}", output_files
        else:
            return True, f"Successfully split into {len(output_files)} segments", output_files
            
    except Exception as e:
        return False, f"Error during splitting: {str(e)}", []

def process_single_video(video_path, frames_per_segment, output_dir=None, pbar=None):
    """Process a single video file"""
    if not os.path.exists(video_path):
        with lock:
            if pbar:
                pbar.write(f"❌ File does not exist: {video_path}")
        return {
            "success": False, 
            "video_path": video_path, 
            "error": "File does not exist", 
            "output_files": []
        }
    
    video_name = os.path.basename(video_path)
    
    total_frames, fps, error = get_video_info(video_path)
    
    if error:
        with lock:
            if pbar:
                pbar.write(f"❌ {video_name}: {error}")
        return {
            "success": False, 
            "video_path": video_path, 
            "error": error, 
            "output_files": []
        }
    
    if total_frames < frames_per_segment:
        with lock:
            if pbar:
                pbar.write(f"🚫 {video_name}: {total_frames} frames < {frames_per_segment} frames (discarded)")
        return {
            "success": False, 
            "video_path": video_path, 
            "error": f"Insufficient frames ({total_frames}<{frames_per_segment})", 
            "output_files": []
        }
    
    with lock:
        if pbar:
            pbar.write(f"🎬 {video_name}: {total_frames} frames, {fps:.2f}fps - Starting split...")
    
    success, message, output_files = split_video_by_frames(video_path, total_frames, fps, frames_per_segment, output_dir)
    
    with lock:
        if pbar:
            if success:
                pbar.write(f"✅ {video_name}: {message}")
            else:
                pbar.write(f"❌ {video_name}: {message}")
    
    return {
        "success": success,
        "video_path": video_path,
        "error": message if not success else None,
        "output_files": output_files,
        "total_frames": total_frames,
        "segments_created": len(output_files)
    }

def main(input_txt, frames_per_segment=81, max_workers=16, output_dir=None):
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
    print(f"✂️ Split criteria: {frames_per_segment} frames per segment")
    print(f"📁 Output directory: {output_dir}")
    print(f"🚀 Using {max_workers} threads for parallel processing...")
    
    successful_videos = []
    failed_videos = []
    total_segments_created = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(video_paths), desc="Processing videos", unit="video") as pbar:
            future_to_path = {
                executor.submit(process_single_video, video_path, frames_per_segment, output_dir, pbar): video_path 
                for video_path in video_paths
            }
            
            for future in as_completed(future_to_path):
                try:
                    result = future.result()
                    if result["success"]:
                        successful_videos.append(result)
                        total_segments_created += result["segments_created"]
                    else:
                        failed_videos.append(result)
                except Exception as e:
                    video_path = future_to_path[future]
                    pbar.write(f"❌ Error processing {video_path}: {str(e)}")
                    failed_videos.append({
                        "success": False, 
                        "video_path": video_path, 
                        "error": str(e), 
                        "output_files": []
                    })
                
                pbar.update(1)
    
    report_file = input_txt.replace('.txt', f'_split_{frames_per_segment}frames_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# Video Splitting Processing Report\n")
        f.write(f"# Processing: {frames_per_segment} frames/segment\n")
        f.write(f"# Original videos: {len(video_paths)}\n")
        f.write(f"# Successfully processed: {len(successful_videos)}\n")
        f.write(f"# Failed/Discarded: {len(failed_videos)}\n")
        f.write(f"# Total segments created: {total_segments_created}\n\n")
        
        if successful_videos:
            f.write("## Successfully Processed Videos\n")
            for item in successful_videos:
                f.write(f"{item['video_path']}\t# {item['total_frames']} frames -> {item['segments_created']} segments\n")
            f.write("\n")
        
        if failed_videos:
            f.write("## Failed/Discarded Videos\n")
            for item in failed_videos:
                f.write(f"{item['video_path']}\t# {item.get('error', 'Unknown error')}\n")
    
    print(f"\n🎉 Processing complete!")
    print(f"✅ Successfully processed: {len(successful_videos)}/{len(video_paths)} videos")
    print(f"🚫 Failed/Discarded: {len(failed_videos)} videos")
    print(f"✂️ Total segments created: {total_segments_created}")
    print(f"📄 Detailed report saved to: {report_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split video files by fixed frame count")
    parser.add_argument("input_txt", type=str, help="Input txt file with one video file path per line")
    parser.add_argument("--frames", "-f", type=int, default=81, help="Number of frames per segment (default: 81)")
    parser.add_argument("--workers", "-w", type=int, default=16, help="Number of parallel processing threads (default: 16)")
    parser.add_argument("--output-dir", "-o", type=str, default="./3DMovie_fps16_scene_81frames", help="Output directory (default: ./3DMovie_fps16_scene_81frames)")
    
    args = parser.parse_args()
    main(args.input_txt, args.frames, args.workers, args.output_dir) 