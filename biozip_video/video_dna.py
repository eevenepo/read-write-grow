from typing import List, Dict, Any, Tuple

from oligos.oligos import (
    Oligo,
    reverse_complement,
    orient_oligo,
    trits_to_dna_with_prev,
    int_to_fixed_trits,
    trits_to_int,
)

HEADER_TRITS_LEN = 25     # length of header that stores payload trit length
PAYLOAD_LEN = 100         # nt per payload in each oligo
OVERLAP = 20              # nt overlap between consecutive payloads
INDEX_TRITS_LEN = 20      # 2 (file) + 6 (segment) + 11 (fragment) + 1 (parity)


# -------------------------------------------------------------------
# Header + padding helpers (same idea as biozip_text)
# -------------------------------------------------------------------

def add_header_and_pad(
    ternary: str,
    header_len: int = HEADER_TRITS_LEN,
    payload_len: int = PAYLOAD_LEN,
    overlap: int = OVERLAP,
) -> str:
    """
    Add a fixed-length ternary header that encodes the original ternary
    length, then pad with '0' trits so that fragmentation never cuts 
    through real payload trits.

    We want final DNA length L such that:
        L = payload_len + k * (payload_len - overlap)
    """
    original_len = len(ternary)
    header = int_to_fixed_trits(original_len, header_len)  # base-3, fixed length

    full = header + ternary
    L = len(full)
    step = payload_len - overlap

    if L <= payload_len:
        desired_len = payload_len
    else:
        # smallest k >= 0 with payload_len + k*step >= L
        k = (L - payload_len + step - 1) // step
        desired_len = payload_len + k * step

    pad_len = desired_len - L
    if pad_len < 0:
        raise RuntimeError("Padding logic error: pad_len < 0")

    padding = "0" * pad_len
    return full + padding


def strip_header_and_unpad(
    ternary_padded: str,
    header_len: int = HEADER_TRITS_LEN,
) -> str:
    """
    Inverse of add_header_and_pad:
      - read header_len trits as an integer = original payload length
      - return exactly that many following trits.
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

    return remaining[:payload_len]  # ignore padding after payload


# -------------------------------------------------------------------
# Video-specific index encoding
# -------------------------------------------------------------------

def encode_video_index_trits(file_id: int, segment_id: int, fragment_index: int) -> str:
    """
    20-trit index block for video-aware DNA:
        - file_id:        2 trits
        - segment_id:     6 trits
        - fragment_index: 11 trits
        - parity:         1 trit (sum of first 19 trits mod 3)
    """
    file_trits = int_to_fixed_trits(file_id, 2)
    segment_trits = int_to_fixed_trits(segment_id, 6)
    frag_trits = int_to_fixed_trits(fragment_index, 11)

    head = file_trits + segment_trits + frag_trits
    total = sum(int(t) for t in head)
    parity = str(total % 3)

    return head + parity  # total length = 20


def decode_video_index_trits(trits: str) -> Tuple[int, int, int, bool]:
    """
    Inverse of encode_video_index_trits.
    Returns:
        file_id, segment_id, fragment_index, parity_ok
    """
    if len(trits) != INDEX_TRITS_LEN:
        raise ValueError(f"Expected {INDEX_TRITS_LEN} trits, got {len(trits)}")

    file_id = trits_to_int(trits[:2])
    segment_id = trits_to_int(trits[2:8])
    fragment_index = trits_to_int(trits[8:19])
    parity = int(trits[19])

    total = sum(int(x) for x in trits[:19])
    parity_ok = (total % 3) == parity

    return file_id, segment_id, fragment_index, parity_ok

def fragment_dna_for_segment(
    dna: str,
    file_id: int,
    segment_id: int,
    codec: Oligo,
    payload_len: int = PAYLOAD_LEN,
    overlap: int = OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Fragment a master DNA string into overlapping payloads with index + orientation:
      - payload length = payload_len
      - overlap between successive payloads = overlap
      - step = payload_len - overlap
      - odd fragments are reverse-complemented
      - each fragment gets a 20-nt index block (Goldman from last payload base)
      - 1-nt orientation bases at start (A/T) and end (C/G)
    """
    fragments: List[Dict[str, Any]] = []

    step = payload_len - overlap
    fragment_index = 0

    for start in range(0, max(1, len(dna) - payload_len + 1), step):
        payload = dna[start : start + payload_len]
        if len(payload) < payload_len:
            break

        # Reverse-complement odd fragments
        if fragment_index % 2 == 1:
            payload = reverse_complement(payload)

        # Encode index trits and map to DNA, continuing from last payload base
        index_trits = encode_video_index_trits(file_id, segment_id, fragment_index)
        index_dna = trits_to_dna_with_prev(payload[-1], index_trits, codec)

        # Orientation bases
        orientation_start = "A" if payload[0] != "A" else "T"
        orientation_end = "C" if index_dna[-1] != "C" else "G"

        full_seq = orientation_start + payload + index_dna + orientation_end

        fragments.append(
            {
                "file_id": file_id,
                "segment_id": segment_id,
                "fragment_index": fragment_index,
                "payload": payload,
                "index_trits": index_trits,
                "index_dna": index_dna,
                "sequence": full_seq,
            }
        )

        fragment_index += 1

    return fragments


def assemble_dna_from_fragments(
    fragments: List[Dict[str, Any]],
    overlap: int = OVERLAP,
    payload_len: int = PAYLOAD_LEN,
) -> str:
    """
    Reassemble master DNA string from a list of fragments.

    Each fragment dict must have:
        - "fragment_index": int
        - "payload": str (DNA, already orientation-corrected and RC-fixed)
    """
    if not fragments:
        raise ValueError("No fragments provided to assembler")

    frags = sorted(fragments, key=lambda f: f["fragment_index"])

    master = frags[0]["payload"]
    for frag in frags[1:]:
        payload = frag["payload"]
        # We expect last 'overlap' nt of master == first 'overlap' of payload,
        # but for now just stitch with overlap trimming.
        master += payload[overlap:]

    return master

def encode_segment_bytes_to_oligos(
    segment_bytes: bytes,
    file_id: int,
    segment_id: int,
    codec: Oligo,
    payload_len: int = PAYLOAD_LEN,
    overlap: int = OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Encode a single video segment (binary data) into a list of oligo dicts.
    Pipeline:
        bytes -> ternary (Huffman)
              -> header + pad (safe for fragmenting)
              -> DNA (Goldman)
              -> overlapping payloads + index + orientation
    """
    # 1. bytes -> ternary
    ternary_raw = codec.bytes_to_ternary(segment_bytes)

    # 2. header + padding to align with fragment pattern
    ternary_padded = add_header_and_pad(
        ternary_raw,
        header_len=HEADER_TRITS_LEN,
        payload_len=payload_len,
        overlap=overlap,
    )

    # 3. ternary -> DNA
    dna_master = codec.ternary_to_goldman(ternary_padded, start="A")

    # 4. DNA -> oligo fragments
    fragments = fragment_dna_for_segment(
        dna=dna_master,
        file_id=file_id,
        segment_id=segment_id,
        codec=codec,
        payload_len=payload_len,
        overlap=overlap,
    )

    return fragments


def decode_oligos_to_segment_bytes(
    oligos: List[str],
    codec: Oligo,
    payload_len: int = PAYLOAD_LEN,
    overlap: int = OVERLAP,
) -> bytes:
    """
    Given a list of oligo sequences (strings) for ONE segment,
    reconstruct the original segment bytes.
    """
    parsed: List[Dict[str, Any]] = []

    for seq in oligos:
        # 1. Normalize orientation (forward: start in {A,T}, end in {C,G})
        seq = orient_oligo(seq)

        # 2. Extract payload and index
        payload = seq[1 : 1 + payload_len]
        index_dna = seq[1 + payload_len : 1 + payload_len + INDEX_TRITS_LEN]

        # 3. Decode index trits (using correct starting base)
        index_trits = codec.goldman_to_ternary(index_dna, start=payload[-1])
        file_id, segment_id, fragment_index, parity_ok = decode_video_index_trits(index_trits)

        if not parity_ok:
            # For now, just warn; in real system you'd drop or repair this fragment
            print(f"WARNING: parity check failed for fragment_index={fragment_index}")

        # 4. Undo reverse complement on odd fragments
        if fragment_index % 2 == 1:
            payload = reverse_complement(payload)

        parsed.append(
            {
                "file_id": file_id,
                "segment_id": segment_id,
                "fragment_index": fragment_index,
                "payload": payload,
            }
        )

    # 5. Reassemble master DNA
    dna_master = assemble_dna_from_fragments(
        parsed,
        overlap=overlap,
        payload_len=payload_len,
    )

    # Optional sanity check: Goldman never produces same base twice in a row
    # If you want to debug further, uncomment:s
    # for i in range(1, len(dna_master)):
    #     if dna_master[i] == dna_master[i - 1]:
    #         print("DEBUG: repeat at", i, dna_master[i-5:i+5])
    #         raise ValueError(f"Invalid Goldman DNA: repeated base {dna_master[i]} at {i}")

    # 6. DNA -> ternary (still includes header + padding)
    trits_padded = codec.goldman_to_ternary(dna_master, start="A")

    # 7. Strip header + padding to recover exact Huffman payload
    trits_raw = strip_header_and_unpad(trits_padded, header_len=HEADER_TRITS_LEN)

    # 8. ternary -> bytes
    recovered_bytes = codec.ternary_to_bytes(trits_raw)
    return recovered_bytes
