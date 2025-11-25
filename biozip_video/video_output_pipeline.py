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


def group_oligos_by_segment(
    oligo_seqs: List[str],
    codec: Oligo,
    payload_len: int = 100,
) -> Dict[int, List[str]]:
    """
    Given a flat pool of oligo sequences (any order),
    group them by segment_id using the 20-nt index block.

    Returns:
        segment_id -> list of oligo sequences (still raw strings).
    """
    from biozip_video.video_dna import encode_video_index_trits  # for consistency (not strictly needed)

    segment_groups: Dict[int, List[str]] = {}

    for seq in oligo_seqs:
        oriented = orient_oligo(seq)

        # Extract payload + index region
        payload = oriented[1 : 1 + payload_len]
        index_dna = oriented[1 + payload_len : 1 + payload_len + 20]  # 20 nt index

        # Decode index: must use the same prev base (payload[-1]) we used in the encoder
        index_trits = codec.goldman_to_ternary(index_dna, start=payload[-1])
        file_id, segment_id, fragment_id, parity_ok = decode_video_index_trits(index_trits)

        # You *could* check parity_ok here and optionally drop bad ones;
        # for now just warn.
        if not parity_ok:
            # non-fatal warning
            print(f"WARNING: parity check failed for segment={segment_id}, fragment={fragment_id}")

        segment_groups.setdefault(segment_id, []).append(oriented)

    return segment_groups


def decode_oligo_pool_to_segments(
    oligo_seqs: List[str],
    huffman_dict_path: Path,
    payload_len: int = 100,
    overlap: int = 20,
) -> Dict[int, bytes]:
    """
    Pool-of-oligos → per-segment bytes.

    Steps:
      - group oligos by segment_id using index block
      - for each segment, reuse decode_oligos_to_segment_bytes()

    Returns:
        segment_id -> bytes
    """
    codec = Oligo(huffman_dict_path)

    # 1) Group by segment
    segment_groups = group_oligos_by_segment(
        oligo_seqs=oligo_seqs,
        codec=codec,
        payload_len=payload_len,
    )

    # 2) Decode each segment independently
    segment_bytes: Dict[int, bytes] = {}

    for segment_id, seg_oligos in segment_groups.items():
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
    """
    Write segment bytes to temp .mp4 files and concat back into a single video.
    """
    temp_dir_path = Path(temp_dir)
    temp_dir_path.mkdir(parents=True, exist_ok=True)

    segment_paths: List[str] = []

    for seg_id in sorted(segment_bytes.keys()):
        seg_path = temp_dir_path / f"segment_{seg_id:04d}.mp4"
        seg_path.write_bytes(segment_bytes[seg_id])
        segment_paths.append(str(seg_path))

    # Use the existing concat helper
    concat_segments(segment_paths, out_video_path)

    return out_video_path


def decode_oligo_pool_to_video(
    oligo_seqs: List[str],
    huffman_dict_path: Path,
    out_video_path: str = "reconstructed_video.mp4",
    payload_len: int = 100,
    overlap: int = 20,
) -> Dict[str, Any]:
    """
    High-level video pipeline (decoder):

      oligo pool (unordered DNA sequences)
        -> per-segment bytes
        -> reconstructed .mp4 video

    Returns:
        {
          "segment_bytes": Dict[int, bytes],
          "out_video_path": str,
        }
    """
    segment_bytes = decode_oligo_pool_to_segments(
        oligo_seqs=oligo_seqs,
        huffman_dict_path=huffman_dict_path,
        payload_len=payload_len,
        overlap=overlap,
    )

    out_path = rebuild_video_from_segments(
        segment_bytes=segment_bytes,
        out_video_path=out_video_path,
    )

    return {
        "segment_bytes": segment_bytes,
        "out_video_path": out_path,
    }


if __name__ == "__main__":
    # Example: round-trip using encode_video_to_oligos + this decoder
    from biozip_video.video_input_pipeline import encode_video_to_oligos

    enc = encode_video_to_oligos(
        input_video="example_input.mp4",
        huffman_dict_path=Path("oligos/huffman_bytes_dict.json"),
    )

    pool = enc["oligo_sequences"]

    dec = decode_oligo_pool_to_video(
        oligo_seqs=pool,
        huffman_dict_path=Path("oligos/huffman_bytes_dict.json"),
        out_video_path="reconstructed_from_pool.mp4",
    )

    print("Reconstructed video:", dec["out_video_path"])
