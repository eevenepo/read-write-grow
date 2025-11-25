import subprocess
from pathlib import Path
import math


def get_video_duration(path: str) -> float:
    """
    Return duration in seconds using ffprobe.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    out = subprocess.check_output(cmd)
    duration_str = out.decode().strip()
    return float(duration_str)


def segment_video(
    compressed_path: str,
    out_dir: str,
    segment_seconds: int = 2,
) -> list[str]:
    """
    Split compressed grayscale video into time-based segments.

    - compressed_path: input preprocessed (grayscale) video
    - out_dir: directory to store segments
    - segment_seconds: length of each segment in seconds

    Returns: list of segment file paths in order.
    """
    compressed_path = str(compressed_path)
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    duration = get_video_duration(compressed_path)
    n_segments = math.ceil(duration / segment_seconds)

    segment_paths: list[str] = []

    for i in range(n_segments):
        start = i * segment_seconds
        out_path = out_dir_path / f"segment_{i:04d}.mp4"

        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start),
            "-t", str(segment_seconds),
            "-i", compressed_path,
            "-c", "copy",
            str(out_path),
        ]
        print("Running command:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        segment_paths.append(str(out_path))

    return segment_paths

def concat_segments(segment_paths: list[str], out_path: str) -> str:
    """
    Concatenate segment video files back into a single video using ffmpeg concat demuxer.

    - segment_paths: list of segment file paths in order
    - out_path: output merged video file

    Returns: out_path
    """
    out_path = str(out_path)
    out_path_path = Path(out_path)
    out_path_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_list = out_path_path.with_suffix(".txt")

    # Use absolute paths so ffmpeg doesn't get confused
    lines = [f"file '{Path(p).resolve().as_posix()}'\n" for p in segment_paths]
    tmp_list.write_text("".join(lines), encoding="utf-8")

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(tmp_list),
        "-c", "copy",
        out_path,
    ]
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    tmp_list.unlink(missing_ok=True)

    return out_path
