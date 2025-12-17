import os
import argparse

def collect_video_files(root_dir, output_file):
    video_paths = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".mp4") and filename.lower().startswith("left"):
                full_path = os.path.abspath(os.path.join(dirpath, filename))
                video_paths.append(full_path)

    with open(output_file, "w", encoding="utf-8") as f:
        for path in video_paths:
            f.write(path + "\n")

    print(f"Found {len(video_paths)} video files, saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursively collect all MP4 files starting with 'left'")
    parser.add_argument("folder", type=str, help="The folder path to traverse")
    parser.add_argument("--output", type=str, default="leftvideo_paths.txt", help="The text file name to save the video paths")

    args = parser.parse_args()
    collect_video_files(args.folder, args.output)