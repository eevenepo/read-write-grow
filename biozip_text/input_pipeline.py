"""Text encoding pipeline: text -> semantic skeleton -> DNA oligos."""
from pathlib import Path
from typing import Dict, Any

from biozip_text.input_encoder import InputEncoder
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
    then pad with dummy trits so fragmentation never chops real payload.

    Final DNA length L must satisfy: L = payload_len + k * (payload_len - overlap)
    """
    original_len = len(ternary)
    header = int_to_fixed_trits(original_len, header_len)

    full = header + ternary
    L = len(full)
    step = payload_len - overlap

    if L <= payload_len:
        desired_len = payload_len
    else:
        k = (L - payload_len + step - 1) // step
        desired_len = payload_len + k * step

    pad_len = desired_len - L
    if pad_len < 0:
        raise RuntimeError("Padding logic error: pad_len < 0")

    padding = "0" * pad_len
    return full + padding


def encode_text_to_dna(
    text: str,
    masking_ratio: float,
    huffman_dict_path: Path,
) -> Dict[str, Any]:
    """
    Complete text-to-DNA pipeline:
    text -> semantic skeleton -> dictionary/relational bytes -> ternary -> DNA
    
    Args:
        text: Input text to encode
        masking_ratio: Fraction of tokens to mask (0.0-1.0)
        huffman_dict_path: Path to Huffman dictionary JSON
    
    Returns:
        Dictionary with DNA sequences, skeleton, and metadata
    """
    # 1) Semantic encoding: extract important tokens
    encoder = InputEncoder()
    enc_result = encoder.encode(text, masking_ratio=masking_ratio)
    gap_skeleton = enc_result["gap_skeleton"]
    token_to_id = encoder.build_word_dictionary(gap_skeleton)

    # 2) Serialize dictionary and relational data
    dict_bytes = encoder.serialize_dictionary_to_bytes(token_to_id)
    rel_bytes = encoder.serialize_relational_to_bytes(gap_skeleton, token_to_id)

    # 3) Huffman + header + Goldman encoding
    oligo = Oligo(huffman_dict_path)

    dict_ternary = oligo.bytes_to_ternary(dict_bytes)
    rel_ternary = oligo.bytes_to_ternary(rel_bytes)

    dict_ternary_padded = add_header_and_pad(dict_ternary)
    rel_ternary_padded = add_header_and_pad(rel_ternary)

    dna_dict_master = oligo.ternary_to_goldman(dict_ternary_padded, start="A")
    dna_rel_master = oligo.ternary_to_goldman(rel_ternary_padded, start="C")

    return {
        "gap_skeleton": gap_skeleton,
        "token_to_id": token_to_id,
        "dna_dict_master": dna_dict_master,
        "dna_rel_master": dna_rel_master,
        "metadata": enc_result["metadata"],
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
