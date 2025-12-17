import os
import argparse

def collect_video_files(root_dir, output_file):
    video_extensions = {".mp4", ".mkv"}
    video_paths = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in video_extensions:
                full_path = os.path.abspath(os.path.join(dirpath, filename))
                video_paths.append(full_path)

    with open(output_file, "w", encoding="utf-8") as f:
        for path in video_paths:
            f.write(path + "\n")

    print(f"Found {len(video_paths)} video files, saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="get all video paths from the folder")
    parser.add_argument("folder", type=str, help="the folder path containing the video files")
    parser.add_argument("--output", type=str, default="3DMovie.txt", help="the text file name to save the video paths")

    args = parser.parse_args()
    collect_video_files(args.folder, args.output)