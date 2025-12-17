import argparse
import os
import logging
from scenedetect import detect, ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logging.basicConfig(
    filename='split_scenes.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

lock = threading.Lock()

def process_single_video(video_path, output_base_dir, pbar=None):
    """Process a single video file"""
    try:
        if not os.path.exists(video_path):
            with lock:
                if pbar:
                    pbar.write(f"❌ Video file does not exist: {video_path}")
            return {"success": False, "video_path": video_path, "error": "File does not exist"}
            
        video_name = os.path.basename(video_path)
        video_dir = os.path.dirname(video_path)
        video_stem = os.path.splitext(video_name)[0]

        output_dir = os.path.join(output_base_dir, f"{video_stem}_scenes")

        with lock:
            if pbar:
                pbar.write(f"🎬 Processing video: {video_name}")
        
        scene_list = detect(video_path, ContentDetector())

        filtered_scenes = [
            scene for scene in scene_list
        ]

        with lock:
            if pbar:
                pbar.write(f"📊 Detected {len(filtered_scenes)} valid scenes")

        split_video_ffmpeg(video_path, filtered_scenes, output_dir=output_dir)

        with lock:
            if pbar:
                pbar.write(f"✅ Output directory: {output_dir}")
        
        return {"success": True, "video_path": video_path, "output_dir": output_dir}

    except Exception as e:
        error_msg = str(e)
        logging.error(f"Error processing video {video_path}", exc_info=True)
        with lock:
            if pbar:
                pbar.write(f"❌ Error processing video {video_name}: {error_msg}")
        return {"success": False, "video_path": video_path, "error": error_msg}

def save_error_report(failed_videos, output_base_dir):
    """Save error report"""
    if not failed_videos:
        return
    
    error_file = os.path.join(output_base_dir, "failed_videos.txt")
    with open(error_file, 'w', encoding='utf-8') as f:
        f.write("# List of failed video files\n")
        f.write(f"# Total {len(failed_videos)} files failed to process\n\n")
        for item in failed_videos:
            f.write(f"{item['video_path']}\t# Error: {item['error']}\n")
    
    print(f"📝 Error report saved to: {error_file}")

def main(txt_file_path, output_base_dir, max_workers=6):
    """Main function to process all videos in the txt file"""
    try:
        if not os.path.exists(txt_file_path):
            raise FileNotFoundError(f"txt file does not exist: {txt_file_path}")
        
        os.makedirs(output_base_dir, exist_ok=True)
            
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            video_paths = [line.strip() for line in f.readlines() if line.strip()]
        
        if not video_paths:
            print("❌ No video paths found in txt file")
            return
            
        print(f"📝 Read {len(video_paths)} video files from {txt_file_path}")
        print(f"📁 Output directory: {output_base_dir}")
        print(f"🚀 Using {max_workers} threads for parallel processing...")
        
        success_count = 0
        failed_videos = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            with tqdm(total=len(video_paths), desc="Processing videos", unit="video") as pbar:
                future_to_path = {
                    executor.submit(process_single_video, video_path, output_base_dir, pbar): video_path 
                    for video_path in video_paths
                }
                
                for future in as_completed(future_to_path):
                    video_path = future_to_path[future]
                    try:
                        result = future.result()
                        if result["success"]:
                            success_count += 1
                        else:
                            failed_videos.append(result)
                    except Exception as e:
                        error_msg = str(e)
                        logging.error(f"Thread error processing {video_path}", exc_info=True)
                        failed_videos.append({"video_path": video_path, "error": error_msg})
                    
                    pbar.update(1)
        
        if failed_videos:
            save_error_report(failed_videos, output_base_dir)
        
        print(f"\n🎉 Processing complete!")
        print(f"✅ Successfully processed: {success_count}/{len(video_paths)} video files")
        if failed_videos:
            print(f"❌ Failed to process: {len(failed_videos)} video files")
            print(f"📄 See failed_videos.txt and step2_split_scenes.log for detailed error information")

    except Exception as e:
        logging.error("Program execution error", exc_info=True)
        print("❌ Program execution error, see split_scenes.log for details")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch video scene splitting tool")
    parser.add_argument("txt_file", type=str, help="Text file containing video file paths, one absolute path per line")
    parser.add_argument("output_dir", type=str, help="Base output directory for split scenes")
    parser.add_argument("--workers", "-w", type=int, default=6, help="Number of parallel processing threads (default: 6)")
    args = parser.parse_args()
    main(args.txt_file, args.output_dir, args.workers)