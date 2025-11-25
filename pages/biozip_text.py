from pathlib import Path
import streamlit as st
import logging

from biozip_text.input_pipeline import encode_text_to_dna
from biozip_text.output_pipeline import decode_oligo_pool_to_skeleton
from oligos.oligos import fragment_master_dna
from biozip_text.text_reconstruction import reconstruct_text_with_gemini
from config import Config
from utilities import get_dna_statistics, format_cost_estimate, log_encoding_stats

# ---------------------- LOGGING ----------------------
logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)

# ---------------------- INITIALIZE CONFIG ----------------------
try:
    Config.initialize()
except Exception as e:
    st.error(f"Configuration error: {e}")
    st.stop()

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="BioZip – DNA Semantic Storage Demo",
    page_icon="🧬",
    layout="centered",
)

# ---------------------- GLOBAL STYLES ----------------------
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Alexandria:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"] {
            font-family: 'Alexandria', sans-serif !important;
        }

        .main .block-container {
            max-width: 750px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        /* Buttons */
        div.stButton > button {
            background: #0b3d91;
            color: white;
            padding: 0.55rem 1.3rem;
            border-radius: 999px;
            border: none;
            font-weight: 600;
        }

        div.stButton > button:hover {
            background: #1053c4;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------- API KEY LOADING ----------------------
MODEL_NAME_DEFAULT = Config.GEMINI_MODEL

def get_gemini_api_key() -> str:
    """Fetch Gemini API key from secrets or environment."""
    try:
        return st.secrets["api_keys"]["GEMINI"]
    except Exception:
        try:
            return Config.validate_api_key()
        except ValueError:
            st.error(
                "Gemini API key missing.\n\nSet GEMINI_API_KEY environment variable "
                "or add it to `.streamlit/secrets.toml`:\n"
                "[api_keys]\nGEMINI = \"...\"\n\nAnd also in Streamlit Cloud."
            )
            st.stop()

# ---------------------- HEADER ----------------------
if st.button("← Back to overview"):
    st.switch_page("app.py")

st.markdown("## DNA Semantic Storage Demo")
st.caption("Convert text → DNA oligo file and back again using semantic reconstruction.")
st.markdown("<br>", unsafe_allow_html=True)

# ---------------------- SIDEBAR ----------------------
st.sidebar.header("Settings")
st.sidebar.info(
    f"**Environment**: {Config.ENV.value}\n\n"
    f"**Gemini Model**: {Config.GEMINI_MODEL}"
)

huffman_path = Config.HUFFMAN_DICT_PATH

# ---------------------- 1. TEXT → DNA ----------------------
st.subheader("1. Text → DNA oligo file")
st.caption("Start with text and generate synthetic DNA oligos.")

default_text = (
    "DNA-based data storage has emerged as a powerful concept for archiving massive amounts "
    "of information in a compact, durable medium."
)

input_text = st.text_area("Input text", value=default_text, height=150)

masking_ratio = st.slider(
    "Masking ratio",
    min_value=0.1,
    max_value=0.9,
    value=0.3,
    step=0.05,
    help="Higher = more compression (fewer tokens kept).",
)

if st.button("Encode text → DNA file"):
    if not huffman_path.exists():
        st.error("Huffman dictionary not found.")
    elif not input_text.strip():
        st.error("Text cannot be empty.")
    elif len(input_text) > Config.MAX_TEXT_SIZE_CHARS:
        st.error(
            f"Text too large. Maximum {Config.MAX_TEXT_SIZE_CHARS} characters."
        )
    else:
        with st.spinner("Encoding text into DNA and fragmenting into oligos..."):
            try:
                enc_result = encode_text_to_dna(
                    text=input_text,
                    masking_ratio=masking_ratio,
                    huffman_dict_path=huffman_path,
                )

                dict_frags = fragment_master_dna(enc_result["dna_dict_master"], file_id=0)
                rel_frags = fragment_master_dna(enc_result["dna_rel_master"], file_id=1)
                all_frags = dict_frags + rel_frags

                file_text = "\n".join(f["sequence"] for f in all_frags)
                bytes_data = file_text.encode("ascii")
                total_bases = sum(len(f["payload"]) for f in all_frags)

                # Log statistics
                metadata = enc_result["metadata"]
                log_encoding_stats(
                    total_tokens=metadata["num_tokens"],
                    important_tokens=metadata["num_important_tokens"],
                    masking_ratio=masking_ratio,
                    dna_length=len(file_text),
                    num_oligos=len(all_frags),
                )

            except Exception as e:
                logger.error(f"Encoding error: {e}")
                st.error(f"Encoding error: {e}")
            else:
                st.success(
                    f"Generated {len(all_frags)} oligos "
                    f"(dict: {len(dict_frags)}, rel: {len(rel_frags)})."
                )

                stats = get_dna_statistics([f["sequence"] for f in all_frags])
                
                cost = Config.DEFAULT_COST_PER_NT * total_bases
                st.markdown(
                    f"""
                    ### 💰 Estimated synthesis cost  
                    <span style="font-size:1.5rem; font-weight:700;">{format_cost_estimate(Config.DEFAULT_COST_PER_NT, total_bases)}</span>
                    <div style="opacity:0.7;">@ {Config.DEFAULT_COST_PER_NT} €/nt (payload only)</div>
                    
                    **DNA Statistics:**
                    - Total oligos: {stats['total_oligos']}
                    - Total bases: {stats['total_bases']}
                    - Avg length: {stats['average_length']:.0f} nt
                    """,
                    unsafe_allow_html=True,
                )

                st.download_button(
                    "Download DNA oligo file",
                    data=bytes_data,
                    file_name="dna_oligos.txt",
                    mime="text/plain",
                )

st.markdown("---")

# ---------------------- 2. DNA → TEXT ----------------------
st.subheader("2. DNA oligo file → reconstructed text")
st.caption("Upload oligos to reconstruct readable text using Gemini.")

uploaded_file = st.file_uploader(
    "DNA oligo file",
    type=["txt", "fa", "fasta"],
)

model_name = st.text_input("Gemini model name", value=MODEL_NAME_DEFAULT)

if st.button("Decode DNA file → text"):
    huffman_path = Path(Config.HUFFMAN_DICT_PATH)
    if not huffman_path.exists():
        st.error("Huffman dictionary not found.")
    elif uploaded_file is None:
        st.error("Upload a DNA file first.")
    else:
        try:
            raw = uploaded_file.read().decode("ascii", errors="ignore")
            lines = [ln.strip() for ln in raw.splitlines()]
            pool = [ln for ln in lines if ln and not ln.startswith(">")]

            if not pool:
                st.error("No valid sequences found.")
                st.stop()

            with st.spinner("Decoding oligos → semantic structure..."):
                skeleton = decode_oligo_pool_to_skeleton(
                    pool,
                    huffman_dict_path=huffman_path,
                )

            api_key = get_gemini_api_key()

            with st.spinner("Reconstructing readable text..."):
                text = reconstruct_text_with_gemini(
                    skeleton,
                    api_key=api_key,
                    model_name=model_name,
                )

            st.success("Decoding complete.")
            st.text_area("Reconstructed text", value=text, height=260)

        except Exception as e:
            logger.error(f"Decoding error: {e}")
            st.error(f"Decoding error: {e}")

st.caption("📧 support@biozip.com · © 2025 BioZip")
