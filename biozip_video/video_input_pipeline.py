"""Video encoding pipeline: video -> segments -> DNA oligos."""
from pathlib import Path
from typing import Dict, List, Any
import io

from oligos.oligos import Oligo
from biozip_video.preprocessing.video_segments import (
    prepare_grayscale_segments,
    load_segment_bytes,
)
from biozip_video.video_dna import encode_segment_bytes_to_oligos


def encode_video_to_oligos(
    input_video: str,
    huffman_dict_path: Path,
    file_id: int = 0,
    work_dir: str = "preprocessing_out",
    width: int = 160,
    height: int = 90,
    crf: int = 40,
    fps: int = 12,
    segment_seconds: int = 2,
    payload_len: int = 100,
    overlap: int = 20,
) -> Dict[str, Any]:
    """
    High-level video pipeline (encoder):
      input_video -> grayscale + compress -> segments -> DNA oligos
    
    Args:
        input_video: Path or file-like object containing video data
        huffman_dict_path: Path to Huffman dictionary
        file_id: File ID for encoding
        work_dir: Temporary directory for preprocessing
        width, height: Target video dimensions
        crf: Compression factor (higher = more compressed)
        fps: Target frames per second
        segment_seconds: Length of each segment in seconds
        payload_len: DNA payload length
        overlap: DNA overlap length
    
    Returns:
        Dictionary with compressed video path, segments, oligos, and sequences
    """
    # Handle both file paths and uploaded file objects
    if isinstance(input_video, str):
        input_path = input_video
    else:
        # For Streamlit uploaded files
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(input_video.getbuffer() if hasattr(input_video, 'getbuffer') else input_video.read())
            input_path = tmp.name

    input_path_obj = Path(input_path)
    if not input_path_obj.exists():
        raise FileNotFoundError(f"Input video not found: {input_path_obj}")

    # 1) Preprocess: grayscale + compress + segment
    compressed_path, segment_paths = prepare_grayscale_segments(
        input_video=input_path,
        work_dir=work_dir,
        width=width,
        height=height,
        crf=crf,
        fps=fps,
        segment_seconds=segment_seconds,
    )

    # 2) Load all segments as raw bytes
    segment_bytes: Dict[int, bytes] = load_segment_bytes(segment_paths)

    # 3) Build DNA codec
    codec = Oligo(huffman_dict_path)

    # 4) Encode each segment into oligos
    segment_oligos: Dict[int, List[Dict[str, Any]]] = {}
    all_oligo_sequences: List[str] = []

    for segment_id, seg_bytes in segment_bytes.items():
        oligos = encode_segment_bytes_to_oligos(
            segment_bytes=seg_bytes,
            file_id=file_id,
            segment_id=segment_id,
            codec=codec,
            payload_len=payload_len,
            overlap=overlap,
        )
        segment_oligos[segment_id] = oligos

        # flatten DNA sequences (bag of oligos)
        all_oligo_sequences.extend(f["sequence"] for f in oligos)

    return {
        "compressed_video_path": str(compressed_path),
        "segment_paths": segment_paths,
        "segment_bytes": segment_bytes,
        "segment_oligos": segment_oligos,
        "oligo_sequences": all_oligo_sequences,
    }


if __name__ == "__main__":
    # Simple local smoke test
    result = encode_video_to_oligos(
        input_video="example_input.mp4",
        huffman_dict_path=Path("oligos/huffman_bytes_dict.json"),
    )

    print("Compressed video:", result["compressed_video_path"])
    print("Num segments:", len(result["segment_paths"]))
    print("Total oligos:", len(result["oligo_sequences"]))
