# input_pipeline.py
from pathlib import Path

from input_encoder import InputEncoder
from oligos.oligos import Oligo, fragment_master_dna, int_to_fixed_trits

# Header + padding constants
HEADER_TRITS_LEN = 25
PAYLOAD_LEN = 100
OVERLAP = 20


def add_header_and_pad(
    ternary: str,
    header_len: int = HEADER_TRITS_LEN,
    payload_len: int = PAYLOAD_LEN,
    overlap: int = OVERLAP,
) -> str:
    """
    Add a fixed-length trit header encoding the original Huffman length,
    then pad with dummy trits so that the total length in trits
    (which equals DNA length) fits the fragmentation pattern:

        len(dna) = payload_len + k * (payload_len - overlap)

    This ensures fragmentation never chops real Huffman payload; only padding.
    """
    original_len = len(ternary)  # Huffman payload length in trits
    header = int_to_fixed_trits(original_len, header_len)  # base-3, fixed 25 trits

    full = header + ternary
    L = len(full)
    step = payload_len - overlap  # e.g., 80

    if L <= payload_len:
        desired_len = payload_len
    else:
        # smallest k >= 0 such that payload_len + k * step >= L
        k = (L - payload_len + step - 1) // step
        desired_len = payload_len + k * step

    pad_len = desired_len - L
    if pad_len < 0:
        raise RuntimeError("Padding logic error: pad_len < 0")

    # '0' trits used as padding; they are ignored via the header on decode
    padding = "0" * pad_len
    return full + padding


def encode_text_to_dna(
    text: str,
    masking_ratio: float,
    huffman_dict_path: Path,
):
    """
    TEXT -> semantic gap skeleton -> (dict/rel bytes) -> ternary (Huffman) ->
    header+padding -> DNA (Goldman)
    """
    # 1) Semantic stage
    encoder = InputEncoder()
    gap_skeleton = encoder.encode_to_gap_skeleton(text, masking_ratio=masking_ratio)
    token_to_id = encoder.build_word_dictionary(gap_skeleton)

    dict_bytes = encoder.serialize_dictionary_to_bytes(token_to_id)
    rel_bytes = encoder.serialize_relational_to_bytes(gap_skeleton, token_to_id)

    # 2) Huffman + header + Goldman stage
    oligo = Oligo(huffman_dict_path)

    dict_ternary = oligo.bytes_to_ternary(dict_bytes)
    rel_ternary = oligo.bytes_to_ternary(rel_bytes)

    # Add header + padding so fragmentation can't truncate real data
    dict_ternary_padded = add_header_and_pad(dict_ternary)
    rel_ternary_padded = add_header_and_pad(rel_ternary)

    dna_dict_master = oligo.ternary_to_goldman(dict_ternary_padded, start="A")
    dna_rel_master = oligo.ternary_to_goldman(rel_ternary_padded, start="C")

    return {
        "gap_skeleton": gap_skeleton,
        "token_to_id": token_to_id,
        "dictionary_bytes": dict_bytes,
        "relational_bytes": rel_bytes,
        "dict_ternary": dict_ternary,
        "rel_ternary": rel_ternary,
        "dict_ternary_padded": dict_ternary_padded,
        "rel_ternary_padded": rel_ternary_padded,
        "dna_dict_master": dna_dict_master,
        "dna_rel_master": dna_rel_master,
    }


if __name__ == "__main__":
    text = (
        "DNA storage is a promising technology for future data systems. "
        "Instead of storing information in silicon chips or magnetic disks, "
        "data is encoded into sequences of nucleotides. "
        "This approach could dramatically increase storage density and durability."
    )

    result = encode_text_to_dna(
        text=text,
        masking_ratio=0.3,
        huffman_dict_path=Path("oligos/huffman_bytes_dict.json"),
    )

    dict_frags = fragment_master_dna(result["dna_dict_master"], file_id=0)
    rel_frags = fragment_master_dna(result["dna_rel_master"], file_id=1)

    print("total_tokens:", result["gap_skeleton"]["total_tokens"])
    print("num_important_tokens:", result["gap_skeleton"]["num_important_tokens"])
    print("dna_dict_master length:", len(result["dna_dict_master"]))
    print("dna_rel_master  length:", len(result["dna_rel_master"]))
    print("dict_frags count:", len(dict_frags))
    print("rel_frags count:", len(rel_frags))
    print("one dict frag length:", len(dict_frags[0]["sequence"]))
