from pathlib import Path
import streamlit as st
import logging
from PIL import Image

from biozip_text.input_pipeline import encode_text_to_dna
from biozip_text.output_pipeline import decode_oligo_pool_to_skeleton
from oligos.oligos import fragment_master_dna
from biozip_text.text_reconstruction import reconstruct_text_with_gemini
from config import Config
from utilities import (
    get_dna_statistics,
    format_cost_estimate,
    log_encoding_stats,
    apply_theme,
    validate_text_input,
    validate_dna_file,
    calculate_compression_metrics,
    display_metrics_card,
    save_encoding_result,
    get_encoding_result,
    display_error_with_help,
)

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
    layout="wide",
)

# ---------------------- GLOBAL STYLES ----------------------
apply_theme()

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

# Hero Section
col_hero_text, col_hero_img = st.columns([2, 1])
with col_hero_text:
    st.title("Text Pipeline")
    st.markdown(
        """
        <div style="font-size: 1.1rem; margin-bottom: 1rem;">
        Experience the power of <b>Semantic Compression</b>. We don't just store bits; we store meaning.
        By stripping away non-essential words and using AI to reconstruct them later, we achieve 
        unprecedented density for DNA storage.
        </div>
        """, 
        unsafe_allow_html=True
    )

with col_hero_img:
    st.markdown(
        """
        <div style="text-align: center; font-size: 4rem;">
        📝🧬
        </div>
        """, 
        unsafe_allow_html=True
    )

st.markdown("---")

# ---------------------- SIDEBAR ----------------------
with st.sidebar:
    try:
        logo = Image.open("assets/logo.png")
        st.image(logo, width="stretch")
    except FileNotFoundError:
        pass
    
    st.markdown("---")
    st.markdown("### Navigation")
    st.page_link("app.py", label="Home", icon="🏠")
    st.page_link("pages/biozip_text.py", label="Text Pipeline", icon="📝")
    st.page_link("pages/biozip_video.py", label="Video Pipeline", icon="🎬")
    st.page_link("pages/technical_details.py", label="Architecture", icon="⚙️")
    st.markdown("---")

st.sidebar.header("Settings")
if Config.IS_STREAMLIT_CLOUD:
    st.sidebar.info(
        "**Running on**: Streamlit Cloud\n\n"
        f"**Gemini Model**: {Config.GEMINI_MODEL}\n\n"
        "Full text pipeline available."
    )
else:
    st.sidebar.info(
        "**Running on**: Local\n\n"
        f"**Gemini Model**: {Config.GEMINI_MODEL}\n\n"
        "All features available."
    )

huffman_path = Config.HUFFMAN_DICT_PATH

# ---------------------- MAIN CONTENT ----------------------
col_encode, col_decode = st.columns(2, gap="large")

# --- LEFT COLUMN: ENCODE ---
with col_encode:
    st.markdown("### 1. Encode (Text → DNA)")
    st.info("Start here: Convert plain text into a synthetic DNA oligo file.")
    
    default_text = (
        "DNA-based data storage has emerged as a powerful concept for archiving massive amounts "
        "of information in a compact, durable medium."
    )

    input_text = st.text_area("Input text", value=default_text, height=200)

    masking_ratio = st.slider(
        "Masking ratio",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.05,
        help="Higher = more compression (fewer tokens kept).",
    )

    if st.button("🧬 Encode to DNA"):
        # Validate input
        is_valid, error_msg = validate_text_input(input_text, Config.MAX_TEXT_SIZE_CHARS)
        
        if not huffman_path.exists():
            st.error("Huffman dictionary not found. Please check your installation.")
        elif not is_valid:
            st.error(error_msg)
        else:
            progress_bar = st.progress(0, text="Starting encoding...")
            
            try:
                progress_bar.progress(10, text="Analyzing text semantics...")
                
                enc_result = encode_text_to_dna(
                    text=input_text,
                    masking_ratio=masking_ratio,
                    huffman_dict_path=huffman_path,
                )
                
                progress_bar.progress(50, text="Fragmenting into DNA oligos...")

                dict_frags = fragment_master_dna(enc_result["dna_dict_master"], file_id=0)
                rel_frags = fragment_master_dna(enc_result["dna_rel_master"], file_id=1)
                all_frags = dict_frags + rel_frags

                file_text = "\n".join(f["sequence"] for f in all_frags)
                bytes_data = file_text.encode("ascii")
                total_bases = sum(len(f["payload"]) for f in all_frags)

                progress_bar.progress(90, text="Finalizing...")

                # Log statistics
                metadata = enc_result["metadata"]
                log_encoding_stats(
                    total_tokens=metadata["num_tokens"],
                    important_tokens=metadata["num_important_tokens"],
                    masking_ratio=masking_ratio,
                    dna_length=len(file_text),
                    num_oligos=len(all_frags),
                )
                
                # Save result to session state for later retrieval
                save_encoding_result("text_encoding", {
                    "bytes_data": bytes_data,
                    "stats": get_dna_statistics([f["sequence"] for f in all_frags]),
                    "total_bases": total_bases,
                    "metadata": metadata,
                })
                
                progress_bar.progress(100, text="Complete!")

            except Exception as e:
                logger.error(f"Encoding error: {e}")
                display_error_with_help(e, "Encoding failed")
            else:
                st.success(
                    f"✅ Generated {len(all_frags)} oligos "
                    f"(dict: {len(dict_frags)}, rel: {len(rel_frags)})."
                )

                stats = get_dna_statistics([f["sequence"] for f in all_frags])
                
                # Calculate and display compression metrics
                metrics = calculate_compression_metrics(
                    original_size=len(input_text),
                    compressed_size=metadata["num_important_tokens"],
                    dna_bases=total_bases,
                )
                display_metrics_card(metrics, "Semantic Compression Results")
                
                st.markdown(
                    f"""
                    <div style="background-color: var(--card-bg); border: 2px solid var(--border-color); padding: 1rem; margin-top: 1rem; box-shadow: 4px 4px 0px var(--shadow-color);">
                        <h4 style="margin:0; color: var(--text-color);">💰 Estimated Cost: {format_cost_estimate(Config.DEFAULT_COST_PER_NT, total_bases)}</h4>
                        <div style="font-size: 0.9rem; opacity: 0.8; color: var(--text-color);">@ {Config.DEFAULT_COST_PER_NT} €/nt</div>
                        <hr style="margin: 0.5rem 0; border-top: 1px solid var(--border-color);">
                        <div style="color: var(--text-color);">
                            <b>DNA Stats:</b><br>
                            • Total oligos: {stats['total_oligos']:,}<br>
                            • Total bases: {stats['total_bases']:,}<br>
                            • Avg length: {stats['average_length']:.0f} nt
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.download_button(
                    "⬇️ Download DNA File",
                    data=bytes_data,
                    file_name="dna_oligos.txt",
                    mime="text/plain",
                )

# --- RIGHT COLUMN: DECODE ---
with col_decode:
    st.markdown("### 2. Decode (DNA → Text)")
    st.info("Reconstruction: Upload oligos and let AI restore the meaning.")

    uploaded_file = st.file_uploader(
        "Upload DNA oligo file",
        type=["txt", "fa", "fasta"],
    )

    model_name = st.text_input("Gemini model name", value=MODEL_NAME_DEFAULT)

    if st.button("✨ Decode & Reconstruct"):
        huffman_path = Path(Config.HUFFMAN_DICT_PATH)
        
        # Validate inputs
        if not huffman_path.exists():
            st.error("Huffman dictionary not found. Please check your installation.")
        else:
            # Use the new validation helper
            is_valid, warning_or_error, pool = validate_dna_file(uploaded_file)
            
            if not is_valid:
                st.error(warning_or_error)
            else:
                if warning_or_error:  # There's a warning about skipped sequences
                    st.warning(warning_or_error)
                
                progress_bar = st.progress(0, text="Starting decoding...")
                
                try:
                    progress_bar.progress(20, text=f"Processing {len(pool)} oligos...")
                    
                    skeleton = decode_oligo_pool_to_skeleton(
                        pool,
                        huffman_dict_path=huffman_path,
                    )
                    
                    progress_bar.progress(50, text="Connecting to Gemini AI...")

                    api_key = get_gemini_api_key()

                    progress_bar.progress(60, text="Reconstructing text with AI...")
                    
                    text = reconstruct_text_with_gemini(
                        skeleton,
                        api_key=api_key,
                        model_name=model_name,
                    )
                    
                    progress_bar.progress(100, text="Complete!")

                    st.success("✅ Decoding complete!")
                    
                    # Show skeleton stats
                    st.markdown(
                        f"""
                        <div style="background-color: var(--card-bg); border: 2px solid var(--border-color); 
                                    padding: 1rem; margin: 1rem 0; box-shadow: 4px 4px 0px var(--shadow-color);">
                            <b>📊 Decoding Stats:</b><br>
                            • Oligos processed: {len(pool):,}<br>
                            • Tokens recovered: {skeleton.get('num_important_tokens', 'N/A')}<br>
                            • Total token positions: {skeleton.get('total_tokens', 'N/A')}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    
                    st.text_area("Reconstructed text", value=text, height=300)
                    
                    # Offer download of reconstructed text
                    st.download_button(
                        "⬇️ Download Reconstructed Text",
                        data=text.encode("utf-8"),
                        file_name="reconstructed_text.txt",
                        mime="text/plain",
                    )

                except Exception as e:
                    logger.error(f"Decoding error: {e}")
                    display_error_with_help(e, "Decoding failed")

st.markdown("---")
st.caption("📧 support@biozip.com · © 2025 BioZip")
