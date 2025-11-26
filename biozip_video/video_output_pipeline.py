from pathlib import Path
from typing import Dict, List, Any

from oligos.oligos import (
    Oligo,
    orient_oligo,
)
from biozip_video.preprocessing.video_segments import concat_segments
from biozip_video.video_dna import (
    decode_video_index_trits,
    decode_oligos_to_segment_bytes,
)

# Use the robust enhancement function
from biozip_video.enhancement.video_enhance import enhance_video


# -------------------------------------------------------------------------
# Simple logging helper
# -------------------------------------------------------------------------

def log(msg: str) -> None:
    """Lightweight progress logger."""
    print(f"[BioZip decode] {msg}", flush=True)


# -------------------------------------------------------------------------
# Grouping & segment decoding
# -------------------------------------------------------------------------

def group_oligos_by_segment(
    oligo_seqs: List[str],
    codec: Oligo,
    payload_len: int = 100,
) -> Dict[int, List[str]]:
    """
    Given a flat pool of oligo sequences (any order),
    group them by segment_id using the 20-nt index block.
    """
    log(f"Grouping {len(oligo_seqs)} oligos by segment...")
    segment_groups: Dict[int, List[str]] = {}

    for i, seq in enumerate(oligo_seqs, start=1):
        oriented = orient_oligo(seq)

        # Extract payload + index region
        payload = oriented[1 : 1 + payload_len]
        index_dna = oriented[1 + payload_len : 1 + payload_len + 20]  # 20 nt index

        # Decode index
        index_trits = codec.goldman_to_ternary(index_dna, start=payload[-1])
        file_id, segment_id, fragment_id, parity_ok = decode_video_index_trits(index_trits)

        if not parity_ok:
            log(f"WARNING: parity check failed for segment={segment_id}, fragment={fragment_id}")

        segment_groups.setdefault(segment_id, []).append(oriented)

        if i % 100 == 0:
            log(f"  Processed {i} / {len(oligo_seqs)} oligos...")

    log(f"Done grouping. Found {len(segment_groups)} segments.")
    return segment_groups


def decode_oligo_pool_to_segments(
    oligo_seqs: List[str],
    huffman_dict_path: Path,
    payload_len: int = 100,
    overlap: int = 20,
) -> Dict[int, bytes]:
    """Pool-of-oligos → per-segment bytes."""
    log("Initializing Oligo codec...")
    codec = Oligo(huffman_dict_path)

    # 1) Group by segment
    segment_groups = group_oligos_by_segment(
        oligo_seqs=oligo_seqs,
        codec=codec,
        payload_len=payload_len,
    )

    # 2) Decode each segment independently
    segment_bytes: Dict[int, bytes] = {}

    log("Decoding each segment to raw bytes...")
    for idx, (segment_id, seg_oligos) in enumerate(sorted(segment_groups.items()), start=1):
        log(f"  [{idx}/{len(segment_groups)}] Decoding segment {segment_id} "
            f"from {len(seg_oligos)} oligos...")
        seg_bytes = decode_oligos_to_segment_bytes(
            oligos=seg_oligos,
            codec=codec,
            payload_len=payload_len,
            overlap=overlap,
        )
        segment_bytes[segment_id] = seg_bytes

    return segment_bytes


def rebuild_video_from_segments(
    segment_bytes: Dict[int, bytes],
    out_video_path: str,
    temp_dir: str = "video_rebuild_tmp",
) -> str:
    """Write segment bytes to temp .mp4 files and concat back into a single video."""
    temp_dir_path = Path(temp_dir)
    temp_dir_path.mkdir(parents=True, exist_ok=True)

    log(f"Rebuilding video from {len(segment_bytes)} segments...")
    segment_paths: List[str] = []

    for seg_id in sorted(segment_bytes.keys()):
        seg_path = temp_dir_path / f"segment_{seg_id:04d}.mp4"
        seg_path.write_bytes(segment_bytes[seg_id])
        segment_paths.append(str(seg_path))

    log(f"Concatenating {len(segment_paths)} segments into {out_video_path}...")
    concat_segments(segment_paths, out_video_path)

    return out_video_path


# -------------------------------------------------------------------------
# High-level pipeline: oligo pool → video (+ optional color)
# -------------------------------------------------------------------------

def decode_oligo_pool_to_video(
    oligo_seqs: List[str],
    huffman_dict_path: Path,
    out_video_path: str = "reconstructed_video.mp4",
    payload_len: int = 100,
    overlap: int = 20,
    # New arguments to match Streamlit UI (prevent crashes)
    width: int = 160,
    height: int = 90,
    fps: int = 12,
    *,
    do_colorize: bool = False,
    do_upscale: bool = False,   # Added to support Streamlit UI
    target_fps: int = 24,       # Added to support Streamlit UI
    color_model_dir: str | Path = "models",
    esrgan_model_path: str | Path | None = None, # Added
) -> Dict[str, Any]:
    """
    High-level video pipeline (decoder).
    """
    log("=== BioZip DNA → Video decode start ===")
    log(f"Target Geometry: {width}x{height} @ {fps}fps")
    log(f"Enhancements: Color={do_colorize}, Upscale={do_upscale}")

    # 1) Decode oligos → raw segment bytes
    segment_bytes = decode_oligo_pool_to_segments(
        oligo_seqs=oligo_seqs,
        huffman_dict_path=huffman_dict_path,
        payload_len=payload_len,
        overlap=overlap,
    )

    # 2) Rebuild base (grayscale) video
    grayscale_out_path = rebuild_video_from_segments(
        segment_bytes=segment_bytes,
        out_video_path=out_video_path,
    )

    final_path = grayscale_out_path
    colorized = False

    # 3) Optional Enhancement (Color + Upscale + Smooth)
    # Even if colorize is False, we might want upscaling or smoothing.
    # We trigger this if ANY enhancement flag is set.
    if do_colorize or do_upscale:
        log("Starting AI enhancement step...")
        grayscale_path = Path(grayscale_out_path)
        enhanced_path = grayscale_path.with_name(grayscale_path.stem + "_enhanced.mp4")

        try:
            enhance_video(
                input_path=str(grayscale_path),
                output_path=str(enhanced_path),
                do_colorize=do_colorize,
                do_upscale=do_upscale,
                do_smooth=True,  # Always smooth if we are enhancing
                color_model_dir=color_model_dir,
                esrgan_model_path=esrgan_model_path,
                target_fps=target_fps
            )
            final_path = str(enhanced_path)
            colorized = do_colorize
            log("Enhancement complete.")
        except Exception as e:
            log(f"Enhancement failed: {e}. Returning base video.")

    return {
        "segment_bytes": segment_bytes,
        "out_video_path": final_path,
        "grayscale_out_video_path": grayscale_out_path,
        "colorized": colorized,
    }