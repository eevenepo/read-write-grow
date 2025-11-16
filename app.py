# app.py
import os
from pathlib import Path

import streamlit as st

from input_pipeline import encode_text_to_dna
from output_pipeline import decode_oligo_pool_to_skeleton
from oligos.oligos import fragment_master_dna
from text_reconstruction import reconstruct_text_with_gemini

st.set_page_config(page_title="DNA Semantic Storage", layout="centered")

# ---------------------- API KEY / MODEL ----------------------
MODEL_NAME_DEFAULT = "gemini-2.5-flash"

def get_gemini_api_key() -> str:
    """
    Get the Gemini API key from Streamlit secrets.
    Works both locally (via .streamlit/secrets.toml)
    and on Streamlit Cloud (via app secrets).
    """
    try:
        return st.secrets["api_keys"]["GEMINI"]
    except Exception:
        st.error(
            "Gemini API key not found in Streamlit secrets.\n\n"
            "Add it under `[api_keys] GEMINI = \"...\"` in your secrets."
        )
        st.stop()

# ---------------------- PAGE LAYOUT ----------------------
st.title("DNA Semantic Storage Demo")
st.caption("Two actions: Text → DNA file, DNA file → Text")

# Shared setting: Huffman dictionary path
st.sidebar.header("Settings")
huffman_path_str = st.sidebar.text_input(
    "Huffman dictionary path",
    value="oligos/huffman_bytes_dict.json",
)

# ---------------------- 1. TEXT → DNA FILE ----------------------
st.header("1. Text → DNA oligo file")

default_text = (
    "DNA-based data storage has emerged as a powerful concept for archiving massive amounts "
    "of information in a compact, durable medium."
)

input_text = st.text_area(
    "Input text",
    value=default_text,
    height=150,
)

masking_ratio = st.slider(
    "Masking ratio",
    min_value=0.1,
    max_value=0.9,
    value=0.3,
    step=0.05,
    help="Higher = more aggressive compression (fewer tokens kept).",
)

if st.button("Encode text → DNA file"):
    huffman_path = Path(huffman_path_str)
    if not huffman_path.exists():
        st.error(f"Huffman dictionary not found at {huffman_path}")
    elif not input_text.strip():
        st.error("Please enter some text.")
    else:
        with st.spinner("Encoding text to DNA and fragmenting into oligos..."):
            try:
                enc_result = encode_text_to_dna(
                    text=input_text,
                    masking_ratio=masking_ratio,
                    huffman_dict_path=huffman_path,
                )

                dict_frags = fragment_master_dna(enc_result["dna_dict_master"], file_id=0)
                rel_frags = fragment_master_dna(enc_result["dna_rel_master"], file_id=1)

                all_frags = dict_frags + rel_frags

                # Simple text format: one oligo sequence per line
                file_text = "\n".join(f["sequence"] for f in all_frags)
                bytes_data = file_text.encode("ascii")

            except Exception as e:
                st.error(f"Error during encoding: {e}")
            else:
                total_bases = ''.join([x['payload'] for x in all_frags])
                st.success(
                    f"Generated {len(all_frags)} oligos. (dict: {len(dict_frags)}, rel: {len(rel_frags)})."
                )
                st.success( f"**Estimated price (0.05$/nt):** {0.05 * len(total_bases):.1f} Euro")
                st.download_button(
                    label="Download DNA oligo file",
                    data=bytes_data,
                    file_name="dna_oligos.txt",
                    mime="text/plain",
                )
# ---------------------- 2. DNA FILE → TEXT ----------------------
st.header("2. DNA oligo file → reconstructed text")

uploaded_file = st.file_uploader(
    "Upload DNA oligo file (one sequence per line, as produced above)",
    type=["txt", "fa", "fasta"],
)

# Gemini settings
model_name = st.text_input(
    "Gemini model name",
    value=MODEL_NAME_DEFAULT,
)

if st.button("Decode DNA file → text"):
    huffman_path = Path(huffman_path_str)

    if not huffman_path.exists():
        st.error(f"Huffman dictionary not found at {huffman_path}")
    elif uploaded_file is None:
        st.error("Please upload a DNA oligo file.")
    else:
        try:
            raw = uploaded_file.read().decode("ascii", errors="ignore")
            lines = [ln.strip() for ln in raw.splitlines()]
            pool = [ln for ln in lines if ln and not ln.startswith(">")]

            if not pool:
                st.error("No valid sequences found in file.")
                st.stop()

            with st.spinner("Decoding oligos back to semantic skeleton..."):
                gap_skeleton = decode_oligo_pool_to_skeleton(
                    pool,
                    huffman_dict_path=huffman_path,
                )

            # 🔐 Get the key from Streamlit secrets
            api_key = get_gemini_api_key()

            with st.spinner("Reconstructing readable text with Gemini..."):
                reconstructed = reconstruct_text_with_gemini(
                    gap_skeleton,
                    api_key=api_key,
                    model_name=model_name,  # use the value from the text input
                )

            st.success("Decoding and reconstruction complete.")
            st.text_area(
                "Reconstructed text",
                value=reconstructed,
                height=250,
            )

        except Exception as e:
            st.error(f"Error during decoding or reconstruction: {e}")
