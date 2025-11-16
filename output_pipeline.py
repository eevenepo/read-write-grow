# output_pipeline.py
from pathlib import Path
from typing import List, Dict, Any
import sys
import os
from text_reconstruction import reconstruct_text_with_gemini


from oligos.oligos import (
    Oligo,
    orient_oligo,
    decode_index_trits,
    assemble_master_dna,
    trits_to_int,
)

# Must match input_pipeline constants
HEADER_TRITS_LEN = 25
PAYLOAD_LEN = 100
OVERLAP = 20


def strip_header_and_unpad(
    ternary_padded: str,
    header_len: int = HEADER_TRITS_LEN,
) -> str:
    """
    Given a padded ternary stream with a fixed-length header,
    recover the original Huffman payload trits using the header length.
    Any extra trits after the payload are ignored as padding.
    """
    if len(ternary_padded) < header_len:
        raise ValueError("Ternary stream too short to contain header")

    header_trits = ternary_padded[:header_len]
    payload_len = trits_to_int(header_trits)

    remaining = ternary_padded[header_len:]
    if payload_len > len(remaining):
        raise ValueError(
            f"Header says payload_len={payload_len}, "
            f"but only {len(remaining)} trits available"
        )

    payload_trits = remaining[:payload_len]
    # anything after payload_trits is padding
    return payload_trits


def decode_single_oligo(seq: str, oligo_codec: Oligo) -> Dict[str, Any]:
    """
    Decode one full oligo sequence into:
        file_id, fragment_index, payload, index_trits, etc.
    """
    oriented = orient_oligo(seq)
    if len(oriented) != 117:
        raise ValueError(f"Expected oligo length 117, got {len(oriented)}")

    orientation_start = oriented[0]
    orientation_end = oriented[-1]

    payload = oriented[1:101]  # 100 nt
    index_dna = oriented[101:116]  # 15 nt

    index_trits = oligo_codec.goldman_to_ternary(index_dna, start=payload[-1])
    file_id, fragment_index, parity_ok = decode_index_trits(index_trits)

    return {
        "file_id": file_id,
        "fragment_index": fragment_index,
        "parity_ok": parity_ok,
        "payload": payload,
        "index_trits": index_trits,
        "index_dna": index_dna,
        "orientation_start": orientation_start,
        "orientation_end": orientation_end,
        "sequence_oriented": oriented,
    }


# ---------- Byte parsing back to skeleton ----------


def parse_dictionary_bytes(dict_bytes: bytes) -> Dict[int, str]:
    """
    Inverse of serialize_dictionary_to_bytes, but tolerant to truncation at the end:
    - Only parses complete [id:4][len:1][word:len] entries.
    - If the last entry is incomplete, it is ignored with a warning.
    """
    id_to_word: Dict[int, str] = {}
    i = 0
    n = len(dict_bytes)

    while i + 5 <= n:
        token_id = int.from_bytes(dict_bytes[i : i + 4], "big")
        length = dict_bytes[i + 4]
        i += 5

        if i + length > n:
            missing = i + length - n
            print(
                f"WARNING: truncated dictionary word for token_id {token_id} "
                f"(missing {missing} byte(s)); ignoring this entry and any trailing bytes",
                file=sys.stderr,
            )
            i = n
            break

        word_bytes = dict_bytes[i : i + length]
        i += length

        word = word_bytes.decode("ascii")
        id_to_word[token_id] = word

    leftover = n - i
    if leftover > 0:
        print(
            f"WARNING: ignoring {leftover} trailing byte(s) in dictionary stream",
            file=sys.stderr,
        )

    return id_to_word


def parse_relational_bytes_to_skeleton(
    rel_bytes: bytes,
    id_to_word: Dict[int, str],
) -> Dict[str, Any]:
    """
    Inverse of serialize_relational_to_bytes, tolerant to truncation and missing dictionary IDs.

    Encoding side format:

        header:
            [total_tokens: 4 bytes]
            [first_position: 2 bytes]

        then, for each important token i (in order):
            [gap: 1 byte][token_id: 4 bytes]

        convention:
            - token 0: gap = first_position  (distance from position 0)
            - token i>0: gap = positions[i] - positions[i-1]
    """
    if len(rel_bytes) < 6:
        raise ValueError("Relational bytes too short")

    total_tokens = int.from_bytes(rel_bytes[0:4], "big")
    header_first_position = int.from_bytes(rel_bytes[4:6], "big")

    # 1) Parse only full [gap:1][token_id:4] entries
    entries: List[tuple[int, int]] = []  # (gap, token_id)

    i = 6
    n = len(rel_bytes)
    while i + 5 <= n:
        gap_val = rel_bytes[i]
        token_id = int.from_bytes(rel_bytes[i + 1 : i + 5], "big")
        i += 5
        entries.append((gap_val, token_id))

    leftover = n - i
    if leftover > 0:
        print(
            f"WARNING: ignoring {leftover} trailing byte(s) in relational stream",
            file=sys.stderr,
        )

    if not entries:
        return {
            "total_tokens": total_tokens,
            "num_important_tokens": 0,
            "first_position": header_first_position,
            "gaps": [],
            "tokens": [],
            "debug_positions": [],
        }

    gaps_all = [g for (g, _) in entries]
    token_ids_all = [tid for (_, tid) in entries]

    # 2) Reconstruct all positions from gaps
    positions_all: List[int] = []
    pos = gaps_all[0]
    positions_all.append(pos)
    for g in gaps_all[1:]:
        pos = pos + g
        positions_all.append(pos)

    if header_first_position != gaps_all[0]:
        print(
            f"WARNING: header first_position ({header_first_position}) "
            f"!= first gap ({gaps_all[0]})",
            file=sys.stderr,
        )

    # 3) Filter out tokens whose IDs are missing from the dictionary
    used_tokens: List[str] = []
    used_positions: List[int] = []

    for pos, tid in zip(positions_all, token_ids_all):
        if tid not in id_to_word:
            print(
                f"WARNING: token_id {tid} missing in dictionary; "
                f"dropping token at position {pos}",
                file=sys.stderr,
            )
            continue
        used_tokens.append(id_to_word[tid])
        used_positions.append(pos)

    num_important = len(used_tokens)

    if num_important == 0:
        return {
            "total_tokens": total_tokens,
            "num_important_tokens": 0,
            "first_position": 0,
            "gaps": [],
            "tokens": [],
            "debug_positions": [],
        }

    # 4) Recompute gaps based on surviving positions
    first_position = used_positions[0]
    gaps_used: List[int] = []
    for j in range(1, num_important):
        gaps_used.append(used_positions[j] - used_positions[j - 1])

    gap_skeleton: Dict[str, Any] = {
        "total_tokens": total_tokens,
        "num_important_tokens": num_important,
        "first_position": first_position,
        "gaps": gaps_used,
        "tokens": used_tokens,
        "debug_positions": used_positions,
    }
    return gap_skeleton


def decode_oligo_pool_to_skeleton(
    oligo_seqs: List[str],
    huffman_dict_path: Path,
    dict_file_id: int = 0,
    rel_file_id: int = 1,
    payload_len: int = PAYLOAD_LEN,
    overlap: int = OVERLAP,
) -> Dict[str, Any]:
    """
    OUTPUT PIPELINE:
      oligo pool
        -> dict/rel fragments
        -> master DNA
        -> ternary (with header/padding)
        -> strip header/padding
        -> Huffman bytes
        -> gap skeleton
    """
    oligo_codec = Oligo(huffman_dict_path)
    fragments_by_file: Dict[int, List[Dict[str, Any]]] = {}

    # 1) Decode each oligo
    for seq in oligo_seqs:
        frag = decode_single_oligo(seq, oligo_codec)
        if not frag["parity_ok"]:
            print(
                f"WARNING: parity mismatch in fragment {frag['fragment_index']} "
                f"(file_id={frag['file_id']})",
                file=sys.stderr,
            )
        fragments_by_file.setdefault(frag["file_id"], []).append(frag)

    if dict_file_id not in fragments_by_file:
        raise ValueError("No dictionary fragments found in pool")
    if rel_file_id not in fragments_by_file:
        raise ValueError("No relational fragments found in pool")

    # 2) Reassemble master DNA
    dna_dict_master = assemble_master_dna(
        fragments_by_file[dict_file_id],
        overlap=overlap,
        payload_len=payload_len,
    )
    dna_rel_master = assemble_master_dna(
        fragments_by_file[rel_file_id],
        overlap=overlap,
        payload_len=payload_len,
    )

    # 3) DNA -> ternary (Goldman inverse)
    dict_ternary_padded = oligo_codec.goldman_to_ternary(dna_dict_master, start="A")
    rel_ternary_padded = oligo_codec.goldman_to_ternary(dna_rel_master, start="C")

    # 3b) Strip header + padding so we recover exact Huffman payload lengths
    dict_ternary = strip_header_and_unpad(dict_ternary_padded)
    rel_ternary = strip_header_and_unpad(rel_ternary_padded)

    # 4) ternary -> bytes (Huffman inverse)
    dict_bytes = oligo_codec.ternary_to_bytes(dict_ternary)
    rel_bytes = oligo_codec.ternary_to_bytes(rel_ternary)

    # 5) bytes -> dictionary + skeleton
    id_to_word = parse_dictionary_bytes(dict_bytes)
    gap_skeleton = parse_relational_bytes_to_skeleton(rel_bytes, id_to_word)

    return gap_skeleton


if __name__ == "__main__":
    from input_pipeline import encode_text_to_dna
    from oligos.oligos import fragment_master_dna

    text = """
        DNA-based data storage has emerged as a powerful concept for archiving massive amounts
        of information in a compact, durable medium. Modern techniques allow digital files to
        be translated into sequences of nucleotides, synthesized as DNA, and later recovered
        through sequencing and decoding pipelines.

        As research advances, scientists are exploring new coding schemes, error-correction
        strategies, and biochemical methods to make DNA storage cheaper and more reliable.
        These efforts aim to bridge the gap between proof-of-concept experiments and practical
        systems that can store real-world datasets over long time scales.

        At the same time, semantic compression and machine learning are opening the door to
        storing not just raw bits, but compressed representations of meaning. By keeping only
        the most important tokens or concepts, and reconstructing the rest with language models,
        future DNA archives could capture the essence of documents using far fewer bases.
        """

    enc_result = encode_text_to_dna(
        text=text,
        masking_ratio=0.3,
        huffman_dict_path=Path("oligos/huffman_bytes_dict.json"),
    )

    dict_frags = fragment_master_dna(enc_result["dna_dict_master"], file_id=0)
    rel_frags = fragment_master_dna(enc_result["dna_rel_master"], file_id=1)

    pool = [f["sequence"] for f in dict_frags + rel_frags]

    decoded_skeleton = decode_oligo_pool_to_skeleton(
        pool,
        huffman_dict_path=Path("oligos/huffman_bytes_dict.json"),
    )

    print("ENCODED skeleton:", enc_result["gap_skeleton"])
    print("DECODED skeleton:", decoded_skeleton)

    # ---- LLM reconstruction with Gemini ----
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set GEMINI_API_KEY environment variable")

    reconstructed = reconstruct_text_with_gemini(
        decoded_skeleton,
        api_key=api_key,
        model_name="gemini-2.5-flash",
    )

    print("\nOriginal text:")
    print(text)
    print("\nReconstructed text (from decoded skeleton):")
    print(reconstructed)
