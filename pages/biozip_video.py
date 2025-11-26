import io
from pathlib import Path
import streamlit as st
import logging
from PIL import Image

from biozip_video.enhancement.video_enhance import enhance_video
from biozip_video.video_input_pipeline import encode_video_to_oligos
from biozip_video.video_output_pipeline import decode_oligo_pool_to_video
from config import Config
from utilities import (
    InMemoryWorkspace,
    get_dna_statistics,
    format_cost_estimate,
    validate_dna_sequence,
    apply_theme,
    validate_video_file,
    validate_dna_file,
    display_error_with_help,
    format_file_size,
    save_encoding_result,
)

# ---------------------- LOGGING ----------------------
logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="BioZip – Video DNA Storage Demo",
    page_icon="🎞️",
    layout="wide",
)

# ---------------------- GLOBAL STYLES ----------------------
apply_theme()

# ---------------------- HEADER ----------------------
if st.button("← Back to overview"):
    st.switch_page("app.py")

# Hero Section
col_hero_text, col_hero_img = st.columns([2, 1])
with col_hero_text:
    st.title("Video Pipeline")
    st.markdown(
        """
        <div style="font-size: 1.1rem; margin-bottom: 1rem;">
        Video is heavy, but DNA is dense. We use <b>Generative Restoration</b> to bridge the gap.
        By compressing video to a tiny grayscale skeleton and using AI to "dream" the colors and details back,
        we make video storage on DNA a reality.
        </div>
        """, 
        unsafe_allow_html=True
    )

with col_hero_img:
    st.markdown(
        """
        <div style="text-align: center; font-size: 4rem;">
        🎬🧬
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
        "AI enhancement features are disabled. Run locally for full capabilities."
    )
else:
    st.sidebar.info(
        "**Running on**: Local\n\n"
        "All features available including AI enhancement."
    )

huffman_path = Config.HUFFMAN_DICT_PATH

# --- Video preprocessing settings ---
st.sidebar.markdown("### Preprocessing (before DNA encoding)")

width = st.sidebar.number_input("Width", min_value=64, max_value=640, value=160, step=16)
height = st.sidebar.number_input("Height", min_value=36, max_value=360, value=90, step=18)
fps = st.sidebar.number_input("FPS", min_value=1, max_value=60, value=12, step=1)
crf = st.sidebar.number_input(
    "CRF (quality, higher = more compression)",
    min_value=10, max_value=51, value=40, step=1
)
segment_seconds = st.sidebar.number_input(
    "Segment length (seconds)",
    min_value=1, max_value=30, value=2, step=1
)

# --- AI enhancement settings (after reconstruction) ---
st.sidebar.markdown("---")
st.sidebar.markdown("### AI Enhancement (after decoding)")

# Disable heavy AI features on Streamlit Cloud (models too large, no GPU)
is_cloud = Config.IS_STREAMLIT_CLOUD

do_colorize = st.sidebar.checkbox(
    "AI colorization",
    value=False,
    disabled=is_cloud,
    help="Colorize the reconstructed grayscale video using an OpenCV DNN colorizer." + (" (Disabled on Streamlit Cloud)" if is_cloud else ""),
)

# Show saturation slider only when colorization is enabled
saturation_boost = 1.2  # default - slightly boosted for better colors
if do_colorize:
    saturation_boost = st.sidebar.slider(
        "Color saturation",
        min_value=0.5,
        max_value=2.5,
        value=1.2,
        step=0.1,
        help="Adjust color intensity. 1.0 = original model output, higher = more vivid colors",
    )

do_upscale = st.sidebar.checkbox(
    "AI upscaling (ESRGAN x4)",
    value=False,
    disabled=is_cloud,
    help="Upscale the video using ESRGAN. This increases resolution and detail." + (" (Disabled on Streamlit Cloud)" if is_cloud else ""),
)

do_white_balance = st.sidebar.checkbox(
    "Auto white balance",
    value=True,
    disabled=is_cloud or not do_colorize,
    help="Automatically correct color temperature for more natural colors." + (" (Requires colorization)" if not do_colorize else ""),
)

if is_cloud:
    st.sidebar.caption("⚠️ AI enhancement disabled on Streamlit Cloud. Run locally for full features.")

target_fps = st.sidebar.number_input(
    "Smoothed output FPS",
    min_value=12,
    max_value=60,
    value=24,
    step=1,
    help="Output FPS after temporal smoothing. Smoothing is always applied."
)

st.sidebar.markdown("---")
cost_per_nt = st.sidebar.number_input(
    "Cost per nucleotide (€)",
    min_value=0.0,
    max_value=1.0,
    value=0.05,
    step=0.01,
)

# ---------------------- MAIN CONTENT ----------------------
col_encode, col_decode = st.columns(2, gap="large")

# --- LEFT COLUMN: ENCODE ---
with col_encode:
    st.markdown("### 1. Encode (Video → DNA)")
    st.info("Compress & Encode: Upload a short video clip.")

    uploaded_video = st.file_uploader(
        "Input video",
        type=["mp4", "mov", "mkv"],
        help="Keep it short for the demo (a few seconds).",
    )

    encode_btn = st.button("🧬 Encode Video")

    if encode_btn:
        # Validate inputs
        is_valid, error_msg = validate_video_file(uploaded_video, max_size_mb=100)
        
        if not huffman_path.exists():
            st.error("Huffman dictionary not found. Please check your installation.")
        elif not is_valid:
            st.error(error_msg)
        else:
            # Show file info
            st.caption(f"📁 File: {uploaded_video.name} ({format_file_size(uploaded_video.size)})")
            
            progress_bar = st.progress(0, text="Starting video encoding...")
            
            try:
                progress_bar.progress(10, text="Preprocessing video (grayscale, resize, compress)...")
                
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
                    
                    progress_bar.progress(80, text="Encoding segments to DNA...")
                    
                    oligo_sequences = enc_result["oligo_sequences"]
                    
                    # Save to session state
                    file_text = "\n".join(oligo_sequences)
                    bytes_data = file_text.encode("ascii")
                    save_encoding_result("video_encoding", {
                        "bytes_data": bytes_data,
                        "oligo_count": len(oligo_sequences),
                        "segment_count": len(enc_result['segment_paths']),
                    })
                    
                    progress_bar.progress(100, text="Complete!")
                    
            except Exception as e:
                logger.error(f"Encoding error: {e}")
                display_error_with_help(e, "Video encoding failed")
            else:
                st.success(
                    f"✅ Generated {len(oligo_sequences):,} oligos across "
                    f"{len(enc_result['segment_paths'])} segments."
                )

                stats = get_dna_statistics(oligo_sequences)
                total_bases = stats['total_bases']
                
                st.markdown(
                    f"""
                    <div style="background-color: var(--card-bg); border: 2px solid var(--border-color); padding: 1rem; margin-top: 1rem; box-shadow: 4px 4px 0px var(--shadow-color);">
                        <h4 style="margin:0; color: var(--text-color);">💰 Estimated Cost: {format_cost_estimate(cost_per_nt, total_bases)}</h4>
                        <div style="font-size: 0.9rem; opacity: 0.8; color: var(--text-color);">@ {cost_per_nt:.2f} €/nt</div>
                        <hr style="margin: 0.5rem 0; border-top: 1px solid var(--border-color);">
                        <div style="color: var(--text-color);">
                            <b>DNA Stats:</b><br>
                            • Total oligos: {stats['total_oligos']:,}<br>
                            • Total bases: {stats['total_bases']:,}<br>
                            • Avg length: {stats['average_length']:.0f} nt<br>
                            • Segments: {len(enc_result['segment_paths'])}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.download_button(
                    "⬇️ Download DNA File",
                    data=bytes_data,
                    file_name="video_dna_oligos.txt",
                    mime="text/plain",
                )

# --- RIGHT COLUMN: DECODE ---
with col_decode:
    st.markdown("### 2. Decode (DNA → Video)")
    st.info(
        f"**Reconstruction Settings:** {width}x{height} px @ {fps} fps. "
        "Ensure these match the encoding settings!"
    )

    uploaded_dna_file = st.file_uploader(
        "Upload DNA oligo file",
        type=["txt", "fa", "fasta"],
        key="video_dna_file",
    )

    out_video_name = st.text_input(
        "Output filename",
        value="reconstructed_video.mp4",
    )

    decode_btn = st.button("✨ Decode & Enhance")

    if decode_btn:
        # Validate inputs
        if not huffman_path.exists():
            st.error("Huffman dictionary not found. Please check your installation.")
        else:
            is_valid, warning_or_error, valid_pool = validate_dna_file(uploaded_dna_file)
            
            if not is_valid:
                st.error(warning_or_error)
            else:
                if warning_or_error:
                    st.warning(warning_or_error)
                
                # Show what enhancements will be applied
                enhancements = []
                if do_colorize:
                    enhancements.append("🎨 Colorization")
                if do_upscale:
                    enhancements.append("🔍 ESRGAN Upscaling")
                enhancements.append("✨ Temporal Smoothing")
                
                st.caption(f"**Enhancements:** {' → '.join(enhancements)}")
                
                progress_bar = st.progress(0, text="Starting video decoding...")
                
                try:
                    progress_bar.progress(10, text=f"Processing {len(valid_pool):,} oligos...")
                    
                    # Locate ESRGAN model if needed
                    models_dir = Path("models")
                    esrgan_candidates = ["RRDB_ESRGAN_x4.pth", "RealESRGAN_x4plus.pth"]
                    esrgan_model_path = None
                    for c in esrgan_candidates:
                        p = models_dir / c
                        if p.exists():
                            esrgan_model_path = str(p)
                            break

                    progress_bar.progress(20, text="Decoding DNA to video segments...")
                    
                    with InMemoryWorkspace(prefix="biozip_decode_") as work_dir:
                        base_out_path = work_dir / out_video_name

                        progress_bar.progress(40, text="Reconstructing video...")
                        
                        # Pass all parameters to the updated pipeline
                        decode_result = decode_oligo_pool_to_video(
                            oligo_seqs=valid_pool,
                            huffman_dict_path=huffman_path,
                            out_video_path=str(base_out_path),
                            payload_len=100,
                            overlap=20,
                            width=width,
                            height=height,
                            fps=fps,
                            do_colorize=do_colorize,
                            do_upscale=do_upscale,
                            do_white_balance=do_white_balance,
                            saturation_boost=saturation_boost,
                            target_fps=int(target_fps) if target_fps else None,
                            color_model_dir=str(models_dir),
                            esrgan_model_path=esrgan_model_path
                        )
                        
                        progress_bar.progress(90, text="Finalizing video...")

                        final_path = Path(decode_result["out_video_path"])

                        # Read final bytes
                        with open(final_path, "rb") as f:
                            video_bytes = f.read()
                    
                    progress_bar.progress(100, text="Complete!")

                except Exception as e:
                    logger.error(f"Decoding error: {e}")
                    display_error_with_help(e, "Video decoding failed")
                else:
                    st.success("✅ Reconstruction complete!")
                    
                    # Show result stats
                    st.markdown(
                        f"""
                        <div style="background-color: var(--card-bg); border: 2px solid var(--border-color); 
                                    padding: 1rem; margin: 1rem 0; box-shadow: 4px 4px 0px var(--shadow-color);">
                            <b>📊 Reconstruction Stats:</b><br>
                            • Oligos processed: {len(valid_pool):,}<br>
                            • Output size: {format_file_size(len(video_bytes))}<br>
                            • Colorized: {'Yes' if decode_result.get('colorized') else 'No'}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.download_button(
                        "⬇️ Download Video",
                        data=video_bytes,
                        file_name=final_path.name,
                        mime="video/mp4",
                    )

                    try:
                        st.video(video_bytes)
                    except Exception:
                        st.info("Preview unavailable, please download.")
