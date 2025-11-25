from pathlib import Path

from oligos.oligos import Oligo
from biozip_video.preprocessing.video_segments import (
    prepare_grayscale_segments,
    load_segment_bytes,
)
from biozip_video.video_dna import (
    encode_segment_bytes_to_oligos,
    decode_oligos_to_segment_bytes,
)
from biozip_video.preprocessing.compress_video import compress_to_grayscale


def main():
    input_video = "example_input.mp4"
    if not Path(input_video).exists():
        raise FileNotFoundError(f"{input_video} not found")

    compressed_path, segment_paths = prepare_grayscale_segments(
        input_video=input_video,
        work_dir="preprocessing_out",
        width=160,
        height=90,
        crf=40,
        fps=12,
        segment_seconds=2,
    )

    print(f"Compressed video: {compressed_path}")
    print(f"Segments: {segment_paths}")

    # Load segment bytes
    segment_bytes_map = load_segment_bytes(segment_paths)

    seg_id = 0
    seg_bytes = segment_bytes_map[seg_id]
    print(f"Loaded segment {seg_id}: {len(seg_bytes)} bytes")

    codec = Oligo(Path("oligos/huffman_bytes_dict.json"))

    oligos = encode_segment_bytes_to_oligos(
        segment_bytes=seg_bytes,
        file_id=0,
        segment_id=seg_id,
        codec=codec,
    )

    print(f"Encoded into {len(oligos)} oligos")

    oligo_seqs = [frag["sequence"] for frag in oligos]

    recovered_bytes = decode_oligos_to_segment_bytes(
        oligos=oligo_seqs,
        codec=codec,
    )

    if recovered_bytes == seg_bytes:
        print("🎉 SUCCESS: Segment DNA round-trip is PERFECT and lossless!")
    else:
        print("❌ ERROR: Recovered bytes DO NOT match original segment bytes")
        Path("debug_original.bin").write_bytes(seg_bytes)
        Path("debug_recovered.bin").write_bytes(recovered_bytes)
        print("Wrote debug_original.bin and debug_recovered.bin")


if __name__ == "__main__":
    main()