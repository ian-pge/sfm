import argparse
import subprocess
import sys
from pathlib import Path
import json

def get_video_duration(video_path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Error getting duration: {e}")
        sys.exit(1)

def process_video(video_path, num_frames, downscale_factor, output_dir=None):
    video_path = Path(video_path).resolve()
    
    if not video_path.exists():
        print(f"Error: Video file not found at {video_path}")
        sys.exit(1)

    if output_dir is None:
        # Default to datasets/<video_name_no_ext>
        root_dir = Path(__file__).parent.parent
        output_dir = root_dir / "datasets" / video_path.stem
    else:
        output_dir = Path(output_dir).resolve()

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {video_path}...")
    print(f"Output directory: {output_dir}")
    
    duration = get_video_duration(video_path)
    print(f"Video Duration: {duration}s")
    
    # Calculate FPS to get approx num_frames
    # num_frames = duration * fps  =>  fps = num_frames / duration
    if duration <= 0:
        print("Error: Invalid video duration.")
        sys.exit(1)
        
    target_fps = num_frames / duration
    print(f"Target FPS: {target_fps:.4f} to extract ~{num_frames} frames")
    
    # Build filter chain
    filters = [f"fps={target_fps}"]
    if downscale_factor > 1:
        # Scale width by 1/factor, keep aspect ratio (-1)
        filters.append(f"scale=iw/{downscale_factor}:-1")
    
    filter_str = ",".join(filters)
    
    output_pattern = images_dir / "frame_%05d.png"
    
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", filter_str,
        "-vsync", "0", # Prevent duplicating frames
        "-q:v", "1",   # High quality for jpg/png (ignored for png usually but good practice)
        str(output_pattern)
    ]
    
    print("Running ffmpeg...")
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully extracted frames to {images_dir}")
        print(f"Total frames extracted: {len(list(images_dir.glob('*.png')))}")
    except subprocess.CalledProcessError as e:
        print(f"Error running ffmpeg: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--num_frames", type=int, required=True, help="Number of frames to extract")
    parser.add_argument("--downscale", type=int, default=1, help="Downscale factor (e.g. 2 for half size). Default 1 (no downscale).")
    parser.add_argument("--output", help="Output dataset directory. Defaults to datasets/<video_name>")
    
    args = parser.parse_args()
    
    process_video(args.video, args.num_frames, args.downscale, args.output)

if __name__ == "__main__":
    main()
