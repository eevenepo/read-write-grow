import subprocess
from pathlib import Path


def compress_to_grayscale(
    input_path: str,
    output_path: str,
    width: int = 160,
    height: int = 90,
    crf: int = 40,
    fps: int | None = None,
    gop_size: int | None = None,
) -> str:
    """
    Convert the input video to a tiny grayscale, downscaled video using x265.

    - input_path: path to original video
    - output_path: where to save the compressed grayscale video
    - width, height: target resolution (e.g. 160x90)
    - crf: quality factor (35–45 is strong compression, higher = smaller file)
    - fps: optional; if set, forces a target fps (e.g. 12)
    - gop_size: optional; if set, forces a fixed Group of Pictures size (keyframe interval).
                Crucial for segmenting video without re-encoding.

    Returns: output_path
    """
    input_path = str(input_path)
    output_path = str(output_path)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Build the video filter string: scale + grayscale
    scale_filter = f"scale={width}:{height},format=gray"
    if fps is not None:
        vf = f"{scale_filter},fps={fps}"
    else:
        vf = scale_filter

    cmd = [
        "ffmpeg",
        "-y",                 # overwrite output
        "-i", input_path,
        "-vf", vf,
        "-c:v", "libx265",    # HEVC encoder
        "-preset", "slow",    # better compression (you can set "medium" if needed)
        "-crf", str(crf),
        "-an",                # no audio
        "-pix_fmt", "yuv420p", # Ensure compatibility
    ]

    # Force fixed GOP if requested (essential for clean segmentation)
    if gop_size is not None:
        cmd.extend([
            "-g", str(gop_size),
            "-keyint_min", str(gop_size),
            "-sc_threshold", "0",  # Disable scene change detection to enforce strict GOP
        ])

    cmd.append(output_path)

    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return output_path
