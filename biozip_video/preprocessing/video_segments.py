from pathlib import Path
from typing import Dict, List, Tuple

from biozip_video.preprocessing.compress_video import compress_to_grayscale
from biozip_video.preprocessing.segment_video import segment_video, concat_segments
# DNA codec imports (already there, just leaving them for later use)
from biozip_video.video_dna import (
    encode_segment_bytes_to_oligos,
    decode_oligos_to_segment_bytes,
)


def prepare_grayscale_segments(
    input_video: str,
    work_dir: str = "preprocessing_out",
    width: int = 160,
    height: int = 90,
    crf: int = 40,
    fps: int = 12,
    segment_seconds: int = 2,
) -> Tuple[str, List[str]]:
    """
    Full preprocessing pipeline:
      - take an input video
      - convert to grayscale + downscale + compress
      - segment into time-based segments

    Returns:
      compressed_path: path to grayscale compressed video
      segment_paths: list of segment file paths in order
    """
    work_dir_path = Path(work_dir)
    work_dir_path.mkdir(parents=True, exist_ok=True)

    compressed_path = work_dir_path / "compressed_gray.mp4"

    compress_to_grayscale(
        input_path=input_video,
        output_path=str(compressed_path),
        width=width,
        height=height,
        crf=crf,
        fps=fps,
    )

    segments_dir = work_dir_path / "segments"
    segment_paths = segment_video(
        compressed_path=str(compressed_path),
        out_dir=str(segments_dir),
        segment_seconds=segment_seconds,
    )

    return str(compressed_path), segment_paths


def load_segment_bytes(segment_paths: List[str]) -> Dict[int, bytes]:
    """
    Read each segment file into raw bytes.

    Returns:
      dict mapping segment_id -> bytes
      where segment_id is the index in the sorted segment_paths.
    """
    segment_bytes: Dict[int, bytes] = {}

    for idx, path in enumerate(sorted(segment_paths)):
        with open(path, "rb") as f:
            data = f.read()
        segment_bytes[idx] = data

    return segment_bytes


def rebuild_video_from_segment_bytes(
    segment_bytes: Dict[int, bytes],
    out_video_path: str,
    temp_dir: str = "preprocessing_rebuild_tmp",
) -> str:
    """
    Take a mapping {segment_id: bytes}, write them to temp files in order,
    and concat back into a single video.

    This is exactly what we'll use AFTER DNA decoding:
      DNA -> trits -> bytes -> this function -> playable video.
    """
    temp_dir_path = Path(temp_dir)
    temp_dir_path.mkdir(parents=True, exist_ok=True)

    # 1. Write each segment's bytes to a file with the proper order
    segment_paths: List[str] = []
    for segment_id in sorted(segment_bytes.keys()):
        segment_path = temp_dir_path / f"segment_{segment_id:04d}.mp4"
        with open(segment_path, "wb") as f:
            f.write(segment_bytes[segment_id])
        segment_paths.append(str(segment_path))

    # 2. Concat them into one video
    concat_segments(segment_paths, out_video_path)

    return out_video_path


# ------------------------------
# Backwards-compatible aliases
# ------------------------------

# If some of your existing code/tests still use "chunk" naming, these aliases
# make sure they keep working without changes.

def prepare_grayscale_chunks(*args, **kwargs):
    return prepare_grayscale_segments(*args, **kwargs)


def load_chunk_bytes(segment_paths: List[str]) -> Dict[int, bytes]:
    return load_segment_bytes(segment_paths)


def rebuild_video_from_chunk_bytes(
    chunk_bytes: Dict[int, bytes],
    out_video_path: str,
    temp_dir: str = "preprocessing_rebuild_tmp",
) -> str:
    return rebuild_video_from_segment_bytes(chunk_bytes, out_video_path, temp_dir=temp_dir)
