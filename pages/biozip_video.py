import io
from pathlib import Path
import streamlit as st
import logging

from biozip_video.video_input_pipeline import encode_video_to_oligos
from biozip_video.video_output_pipeline import decode_oligo_pool_to_video
from config import Config
from utilities import InMemoryWorkspace, get_dna_statistics, format_cost_estimate, validate_dna_sequence

# ---------------------- LOGGING ----------------------
logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="BioZip – Video DNA Storage Demo",
    page_icon="🎞️",
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

# ---------------------- HEADER ----------------------
if st.button("← Back to overview"):
    st.switch_page("app.py")

st.markdown("## Video → DNA → Video Demo")
st.caption("Compress a video, encode to DNA oligos, and reconstruct it back.")
st.markdown("<br>", unsafe_allow_html=True)

# ---------------------- SIDEBAR ----------------------
st.sidebar.header("Settings")
st.sidebar.info(
    f"**Environment**: {Config.ENV.value}\n\n"
    "All processing is done in-memory. No files are stored locally."
)

huffman_path = Config.HUFFMAN_DICT_PATH

st.sidebar.markdown("---")
st.sidebar.caption("Video preprocessing")

width = st.sidebar.number_input("Width", min_value=64, max_value=640, value=160, step=16)
height = st.sidebar.number_input("Height", min_value=36, max_value=360, value=90, step=18)
fps = st.sidebar.number_input("FPS", min_value=1, max_value=60, value=12, step=1)
crf = st.sidebar.number_input("CRF (quality, higher = more compression)", min_value=10, max_value=51, value=40, step=1)
segment_seconds = st.sidebar.number_input("Segment length (seconds)", min_value=1, max_value=30, value=2, step=1)

st.sidebar.markdown("---")
cost_per_nt = st.sidebar.number_input(
    "Cost per nucleotide (€)",
    min_value=0.0,
    max_value=1.0,
    value=0.05,
    step=0.01,
)

# ---------------------- 1. VIDEO → DNA ----------------------
st.subheader("1. Video → DNA oligo file")
st.caption("Upload a short video and generate synthetic DNA oligos.")

uploaded_video = st.file_uploader(
    "Input video",
    type=["mp4", "mov", "mkv"],
    help="Keep it short for the demo (a few seconds).",
)

encode_btn = st.button("Encode video → DNA file")

if encode_btn:
    if not huffman_path.exists():
        st.error("Huffman dictionary not found.")
    elif uploaded_video is None:
        st.error("Upload a video first.")
    else:
        with st.spinner("Preprocessing video and encoding into DNA oligos..."):
            try:
                with InMemoryWorkspace(prefix="biozip_encode_") as work_dir:
                    enc_result = encode_video_to_oligos(
                        input_video=uploaded_video,
                        huffman_dict_path=huffman_path,
                        file_id=0,
                        work_dir=str(work_dir),
                        width=width,
                        height=height,
                        crf=crf,
                        fps=fps,
                        segment_seconds=segment_seconds,
                        payload_len=100,
                        overlap=20,
                    )
            except Exception as e:
                logger.error(f"Encoding error: {e}")
                st.error(f"Encoding error: {e}")
            else:
                oligo_sequences = enc_result["oligo_sequences"]
                
                st.success(
                    f"Generated {len(oligo_sequences)} oligos across "
                    f"{len(enc_result['segment_paths'])} segments."
                )

                stats = get_dna_statistics(oligo_sequences)
                total_bases = stats['total_bases']
                
                st.markdown(
                    f"""
                    ### 💰 Estimated synthesis cost  
                    <span style="font-size:1.5rem; font-weight:700;">{format_cost_estimate(cost_per_nt, total_bases)}</span>
                    <div style="opacity:0.7;">@ {cost_per_nt:.2f} €/nt (payload only)</div>
                    
                    **DNA Statistics:**
                    - Total oligos: {stats['total_oligos']}
                    - Total bases: {stats['total_bases']}
                    - Avg length: {stats['average_length']:.0f} nt
                    """,
                    unsafe_allow_html=True,
                )

                # Prepare downloadable oligo file
                file_text = "\n".join(oligo_sequences)
                bytes_data = file_text.encode("ascii")

                st.download_button(
                    "Download DNA oligo file",
                    data=bytes_data,
                    file_name="video_dna_oligos.txt",
                    mime="text/plain",
                )

st.markdown("---")

# ---------------------- 2. DNA → VIDEO ----------------------
st.subheader("2. DNA oligo file → reconstructed video")
st.caption("Upload a DNA oligo file and reconstruct the video bytes.")

uploaded_dna_file = st.file_uploader(
    "DNA oligo file",
    type=["txt", "fa", "fasta"],
    key="video_dna_file",
)

out_video_name = st.text_input(
    "Output video filename",
    value="reconstructed_video_from_dna.mp4",
)

decode_btn = st.button("Decode DNA file → video")

if decode_btn:
    if not huffman_path.exists():
        st.error("Huffman dictionary not found.")
    elif uploaded_dna_file is None:
        st.error("Upload a DNA oligo file first.")
    else:
        try:
            raw = uploaded_dna_file.read().decode("ascii", errors="ignore")
            lines = [ln.strip() for ln in raw.splitlines()]
            pool = [ln for ln in lines if ln and not ln.startswith(">")]

            # Validate sequences: ensure they look like DNA (ACGT) and within expected length
            valid_pool = [s for s in pool if validate_dna_sequence(s)]
            invalid_count = len(pool) - len(valid_pool)

            if not valid_pool:
                st.error("No valid DNA sequences found in uploaded file. Make sure the file contains A/C/G/T sequences (one per line).")
                st.stop()

            if invalid_count > 0:
                st.warning(f"Ignored {invalid_count} lines that did not look like valid DNA sequences.")

            pool = valid_pool

            with st.spinner("Decoding oligos → segments → video..."):
                # Create a temporary workspace, run decoding and load bytes while
                # the workspace still exists. Previously the temp dir was deleted
                # before reading the file which caused FileNotFoundError.
                video_bytes = None
                with InMemoryWorkspace(prefix="biozip_decode_") as work_dir:
                    out_video_path = str(work_dir / out_video_name)
                    result = decode_oligo_pool_to_video(
                        oligo_seqs=pool,
                        huffman_dict_path=huffman_path,
                        out_video_path=out_video_path,
                        payload_len=100,
                        overlap=20,
                    )

                    # Read the reconstructed video into memory for download
                    try:
                        with open(result["out_video_path"], "rb") as f:
                            video_bytes = f.read()
                    except Exception as e:
                        logger.error(f"Failed to read reconstructed video: {e}")
                        st.error(f"Failed to read reconstructed video: {e}")

            if video_bytes is None:
                # Already reported to user; stop further processing
                st.stop()

            st.success("Decoding complete. Video reconstructed.")

            st.download_button(
                "Download reconstructed video",
                data=video_bytes,
                file_name=out_video_name,
                mime="video/mp4",
            )

            # Show the video inline if possible
            try:
                st.video(video_bytes)
            except Exception as e:
                logger.warning(f"Could not preview video: {e}")
                st.info("Video reconstructed successfully but could not preview in browser.")

        except Exception as e:
            logger.error(f"Decoding error: {e}")
            st.error(f"Decoding error: {e}")

st.caption("📧 support@biozip.com · © 2025 BioZip")
