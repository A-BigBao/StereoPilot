"""
Stereo video generation utilities.
Generate Side-by-Side (SBS) and Red-Cyan anaglyph stereo videos from left/right eye videos.
"""
import subprocess
import sys
import argparse
from pathlib import Path


def check_and_install_dependencies():
    """Check and install required dependencies."""
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy'
    }
    
    missing_packages = []
    for module_name, pip_name in required_packages.items():
        try:
            __import__(module_name)
        except ImportError:
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"Installing missing packages: {', '.join(missing_packages)}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
        print("Installation complete.")


# Check dependencies before importing
check_and_install_dependencies()

import cv2
import numpy as np


def convert_to_h264(input_path: Path, output_path: Path = None) -> Path:
    """
    Convert video to H.264 codec using FFmpeg.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video (default: replace input)
    
    Returns:
        Path to converted video
    """
    input_path = Path(input_path)
    if output_path is None:
        temp_path = input_path.parent / f"{input_path.stem}_h264_temp.mp4"
        output_path = input_path
        replace_original = True
    else:
        temp_path = Path(output_path)
        replace_original = False

    cmd = [
        'ffmpeg', '-y',
        '-i', str(input_path),
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-an',  
        str(temp_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        if replace_original:
            # Replace original with converted file
            import shutil
            shutil.move(str(temp_path), str(output_path))
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Warning: FFmpeg conversion failed: {e.stderr.decode() if e.stderr else 'Unknown error'}")
        # Clean up temp file if exists
        if temp_path.exists() and replace_original:
            temp_path.unlink()
        return input_path
    except FileNotFoundError:
        print("Warning: FFmpeg not found. Video will remain in mp4v codec.")
        return input_path


def compose_anaglyph(left_bgr: np.ndarray, right_bgr: np.ndarray, mode: str = "color") -> np.ndarray:
    """
    Generate red-cyan anaglyph image (BGR space).
    
    Modes:
      - color: R=left.R, G=right.G, B=right.B (common color anaglyph)
      - halfcolor: R=left grayscale, G=right.G, B=right.B
      - gray: R=left grayscale, G=right grayscale, B=right grayscale
    
    Args:
        left_bgr: Left eye frame in BGR format
        right_bgr: Right eye frame in BGR format
        mode: Anaglyph mode ('color', 'halfcolor', 'gray')
    
    Returns:
        Anaglyph frame in BGR format
    """
    if left_bgr.shape != right_bgr.shape:
        raise ValueError("Left and right frame dimensions do not match")

    h, w, _ = left_bgr.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    if mode == "color":
        out[:, :, 2] = left_bgr[:, :, 2]   # R from left
        out[:, :, 1] = right_bgr[:, :, 1]  # G from right
        out[:, :, 0] = right_bgr[:, :, 0]  # B from right
    elif mode == "halfcolor":
        left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
        out[:, :, 2] = left_gray
        out[:, :, 1] = right_bgr[:, :, 1]
        out[:, :, 0] = right_bgr[:, :, 0]
    elif mode == "gray":
        left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
        out[:, :, 2] = left_gray
        out[:, :, 1] = right_gray
        out[:, :, 0] = right_gray
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return out


def create_sbs_video(left_path: Path, right_path: Path, output_path: Path) -> int:
    """
    Create Side-by-Side (SBS) stereo video.
    Places left and right videos side by side in the same frame.
    
    Args:
        left_path: Path to left eye video
        right_path: Path to right eye video
        output_path: Path to output SBS video
    
    Returns:
        Number of frames processed
    """
    left_cap = cv2.VideoCapture(str(left_path))
    right_cap = cv2.VideoCapture(str(right_path))
    
    if not left_cap.isOpened():
        raise RuntimeError(f"Cannot open left video: {left_path}")
    if not right_cap.isOpened():
        raise RuntimeError(f"Cannot open right video: {right_path}")
    
    lw = int(left_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    lh = int(left_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(left_cap.get(cv2.CAP_PROP_FPS) or 16.0)

    out_w = lw * 2
    out_h = lh
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))
    
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {output_path}")
    
    frame_idx = 0
    while True:
        ok_l, left_frame = left_cap.read()
        ok_r, right_frame = right_cap.read()
        
        if not ok_l or not ok_r:
            break
        
        # Create SBS frame
        sbs_frame = np.zeros((lh, out_w, 3), dtype=np.uint8)
        sbs_frame[:, :lw, :] = left_frame
        sbs_frame[:, lw:, :] = right_frame
        
        writer.write(sbs_frame)
        frame_idx += 1
    
    left_cap.release()
    right_cap.release()
    writer.release()
    return frame_idx


def create_anaglyph_video(left_path: Path, right_path: Path, output_path: Path, mode: str = "color") -> int:
    """
    Create red-cyan anaglyph stereo video.
    
    Args:
        left_path: Path to left eye video
        right_path: Path to right eye video
        output_path: Path to output anaglyph video
        mode: Anaglyph mode ('color', 'halfcolor', 'gray')
    
    Returns:
        Number of frames processed
    """
    left_cap = cv2.VideoCapture(str(left_path))
    right_cap = cv2.VideoCapture(str(right_path))
    
    if not left_cap.isOpened():
        raise RuntimeError(f"Cannot open left video: {left_path}")
    if not right_cap.isOpened():
        raise RuntimeError(f"Cannot open right video: {right_path}")
    
    lw = int(left_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    lh = int(left_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(left_cap.get(cv2.CAP_PROP_FPS) or 16.0)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (lw, lh))
    
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {output_path}")
    
    frame_idx = 0
    while True:
        ok_l, left_frame = left_cap.read()
        ok_r, right_frame = right_cap.read()
        
        if not ok_l or not ok_r:
            break
        
        anaglyph = compose_anaglyph(left_frame, right_frame, mode)
        writer.write(anaglyph)
        frame_idx += 1
    
    left_cap.release()
    right_cap.release()
    writer.release()
    return frame_idx


def generate_stereo_videos(
    left_video: str,
    right_video: str,
    output_dir: str = "./stereo_output",
    output_name: str = None
) -> dict:
    """
    Generate both SBS and anaglyph stereo videos from left/right eye videos.
    
    Args:
        left_video: Path to left eye video
        right_video: Path to right eye video
        output_dir: Output directory (default: ./stereo_output)
        output_name: Base name for output files (default: derived from left video)
    
    Returns:
        Dictionary with paths to generated videos:
        {
            'sbs': Path to SBS video,
            'anaglyph': Path to anaglyph video,
            'frames': Number of frames processed
        }
    """
    left_path = Path(left_video)
    right_path = Path(right_video)
    output_path = Path(output_dir)
    
    if not left_path.exists():
        raise FileNotFoundError(f"Left video not found: {left_path}")
    if not right_path.exists():
        raise FileNotFoundError(f"Right video not found: {right_path}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine output name
    if output_name is None:
        output_name = left_path.stem
    
    # Output paths
    sbs_output = output_path / f"{output_name}_sbs.mp4"
    anaglyph_output = output_path / f"{output_name}_anaglyph.mp4"
    
    print(f"Generating stereo videos...")
    print(f"  Left:  {left_path}")
    print(f"  Right: {right_path}")
    print(f"  Output: {output_path}")
    
    # Generate SBS video
    print(f"  Creating SBS video...")
    sbs_frames = create_sbs_video(left_path, right_path, sbs_output)
    print(f"    -> {sbs_output} ({sbs_frames} frames)")
    
    # Generate anaglyph video
    print(f"  Creating anaglyph video...")
    anaglyph_frames = create_anaglyph_video(left_path, right_path, anaglyph_output, mode="color")
    print(f"    -> {anaglyph_output} ({anaglyph_frames} frames)")
    
    # Convert to H.264 codec
    print(f"  Converting to H.264 codec...")
    convert_to_h264(sbs_output)
    print(f"    -> {sbs_output} (H.264)")
    convert_to_h264(anaglyph_output)
    print(f"    -> {anaglyph_output} (H.264)")
    
    print(f"Done!")
    
    return {
        'sbs': str(sbs_output),
        'anaglyph': str(anaglyph_output),
        'frames': sbs_frames
    }


def main():
    """Command line interface for stereo video generation."""
    parser = argparse.ArgumentParser(
        description="Generate SBS and anaglyph stereo videos from left/right eye videos"
    )
    parser.add_argument(
        "--left", "-l",
        type=str,
        required=True,
        help="Path to left eye video"
    )
    parser.add_argument(
        "--right", "-r",
        type=str,
        required=True,
        help="Path to right eye video"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="./stereo_output",
        help="Output directory (default: ./stereo_output)"
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        default=None,
        help="Base name for output files (default: derived from left video)"
    )
    
    args = parser.parse_args()
    
    result = generate_stereo_videos(
        left_video=args.left,
        right_video=args.right,
        output_dir=args.output_dir,
        output_name=args.name
    )
    
    print(f"\nOutput files:")
    print(f"  SBS:      {result['sbs']}")
    print(f"  Anaglyph: {result['anaglyph']}")


if __name__ == "__main__":
    main()

