import streamlit as st
from PIL import Image
from config import Config
from utilities import apply_theme

st.set_page_config(
    page_title="BioZip - Technical Architecture",
    page_icon="🧬",
    layout="wide",
)

# ---------------------- GLOBAL STYLES ----------------------
apply_theme()

if st.button("← Back to Home"):
    st.switch_page("app.py")

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

if Config.IS_STREAMLIT_CLOUD:
    st.sidebar.info(
        "**Running on**: Streamlit Cloud\n\n"
        f"**Gemini Model**: {Config.GEMINI_MODEL}\n\n"
        "AI enhancement features disabled."
    )
else:
    st.sidebar.info(
        "**Running on**: Local\n\n"
        f"**Gemini Model**: {Config.GEMINI_MODEL}\n\n"
        "All features available."
    )

# Hero Section
col_hero_text, col_hero_img = st.columns([2, 1])
with col_hero_text:
    st.title("Technical Architecture")
    st.markdown(
        """
        <div style="font-size: 1.1rem; margin-bottom: 1rem;">
        BioZip uses advanced compression, biological encoding schemes, and AI reconstruction 
        to store data efficiently in synthetic DNA. Below is a deep dive into both pipelines,
        including details on how features adapt between local and cloud deployment.
        </div>
        """, 
        unsafe_allow_html=True
    )

with col_hero_img:
    st.markdown(
        """
        <div style="text-align: center; font-size: 4rem;">
        ⚙️🧬
        </div>
        """, 
        unsafe_allow_html=True
    )

# Deployment mode indicator
if Config.IS_STREAMLIT_CLOUD:
    st.warning("☁️ **Cloud Mode**: You're viewing the Streamlit Cloud version. AI enhancement features (colorization, upscaling) are disabled. Run locally for full capabilities.")
else:
    st.success("💻 **Local Mode**: All features are available including AI colorization and ESRGAN upscaling.")

st.markdown("---")

# ---------------------- TEXT PIPELINE ----------------------
st.header("1. Text Pipeline: Semantic Compression")
st.markdown(
    """
    The text pipeline prioritizes **meaning** over exact bitwise storage. By storing only the 
    "semantic skeleton" of a document and using Large Language Models (LLMs) to hallucinate 
    the missing context back into place, we achieve extreme compression ratios.
    """
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔽 Encoding (Write)")
    st.markdown(
        """
        <div class="tech-card">
        <b>1. Semantic Analysis</b><br>
        Inspired by the "Crossword" compression theory (<i>Language Models as Semantic Compressors</i>, 
        <a href="https://arxiv.org/pdf/2304.01106" target="_blank">arXiv:2304.01106</a>), we treat text 
        reconstruction as a puzzle. We use <code>spaCy</code> for POS tagging and 
        <code>Sentence-Transformers</code> to identify "load-bearing" tokens (entities, nouns) that carry 
        the core meaning.
        <br><br>
        <b>2. Semantic Masking</b><br>
        Tokens with low semantic weight (stopwords, common connectors) are dropped based on a user-defined 
        <code>masking_ratio</code>. This creates a sparse "skeleton" of the text, significantly 
        reducing the token count while preserving the narrative arc.
        <br><br>
        <b>3. Binary Compression</b><br>
        The skeleton structure (gaps + token IDs) and the dictionary of unique words are serialized into bytes. 
        We then apply <code>zlib</code> (DEFLATE) compression to remove statistical redundancy in the binary stream.
        <br><br>
        <b>4. DNA Transcoding</b><br>
        <ul>
            <li><b>Huffman Coding:</b> Maps compressed bytes to Ternary (0, 1, 2) based on frequency.</li>
            <li><b>Goldman Encoding:</b> Converts Ternary to DNA (A, C, G, T) using the rotation scheme 
            proposed by <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC3672958/" target="_blank">Goldman et al. (2013)</a>. 
            This guarantees no homopolymers (e.g., "AAAA") are created, reducing synthesis errors.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown("### 🔼 Decoding (Read)")
    st.markdown(
        """
        <div class="tech-card">
        <b>1. DNA Sequencing & Decoding</b><br>
        The DNA oligos are sequenced and decoded back from Goldman (DNA) → Ternary → Huffman → Bytes.
        <br><br>
        <b>2. Decompression</b><br>
        The bytes are decompressed via <code>zlib</code> to recover the semantic skeleton and dictionary.
        <br><br>
        <b>3. AI Reconstruction</b><br>
        The skeleton (e.g., <i>"DNA ... storage ... powerful ... concept"</i>) is fed into 
        <b>Google Gemini</b>. The LLM uses its world knowledge to reconstruct the original 
        grammatical structure and fill in the missing context, restoring the document's meaning.
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# ---------------------- VIDEO PIPELINE ----------------------
st.header("2. Video Pipeline: Generative Restoration")
st.markdown(
    """
    Video is heavy. To store it in DNA, we strip it down to its bare essentials—low resolution, 
    grayscale, and high compression—and rely on Generative AI to "dream" the details back in.
    """
)

col3, col4 = st.columns(2)

with col3:
    st.markdown("### 🔽 Encoding (Write)")
    st.markdown(
        """
        <div class="tech-card">
        <b>1. Extreme Preprocessing</b><br>
        The video is downscaled to a tiny resolution (e.g., <b>160x90</b>) and converted to <b>Grayscale</b> 
        (Luma channel only). Since the human eye is more sensitive to brightness than color, discarding chroma 
        saves ~66% of the data immediately.
        <br><br>
        <b>2. HEVC Compression</b><br>
        We use <code>ffmpeg</code> with <b>libx265 (HEVC)</b>, a <code>slow</code> preset, and a high CRF 
        (Constant Rate Factor 40+). This removes massive amounts of spatial and temporal redundancy, 
        resulting in a tiny binary payload.
        <br><br>
        <b>3. Segmentation</b><br>
        The video is split into short segments (e.g., 2s) using <code>ffmpeg -f segment</code> with a fixed 
        GOP (Group of Pictures) size. This ensures each DNA oligo pool is a self-contained video chunk 
        that can be decoded independently.
        <br><br>
        <b>4. DNA Encoding</b><br>
        The binary video segments are encoded into DNA using the same Huffman + Goldman scheme as the text pipeline.
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown("### 🔼 Decoding (Read)")
    st.markdown(
        """
        <div class="tech-card">
        <b>1. Reconstruction</b><br>
        DNA is decoded back into binary video segments, which are concatenated into a low-res, 
        grayscale video file. This step works identically on both local and cloud deployments.
        <br><br>
        <b>2. AI Colorization</b> <span style="color: #ff6b6b;">🖥️ Local Only</span><br>
        A CNN (<b>Zhang et al.</b>) analyzes the Luma (L) channel and predicts the missing Chroma (ab) channels. 
        It "hallucinates" plausible colors (e.g., sky is blue, grass is green) based on the semantic content 
        of the grayscale frames. <i>Requires ~100MB colorization model.</i>
        <br><br>
        <b>3. Super-Resolution</b> <span style="color: #ff6b6b;">🖥️ Local Only</span><br>
        <b>Real-ESRGAN</b> (x4) upscales the video (e.g., 160p → 640p). It uses a GAN trained on real-world 
        images to invent realistic high-frequency textures and sharpen edges that were lost during compression.
        <i>Requires ~64MB ESRGAN model + significant compute.</i>
        <br><br>
        <b>4. Temporal Smoothing</b> <span style="color: #4ecdc4;">✓ Always Available</span><br>
        AI models process each frame in isolation, often causing jittery colors or flickering. 
        We apply a <b>stabilization filter</b> that blends the current frame with its neighbors to smooth out this noise. 
        To prevent "ghosting" (double-exposure effects) when the camera cuts to a new scene, our algorithm detects 
        the change and instantly resets the filter, ensuring crisp transitions.
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# ---------------------- DEPLOYMENT MODES ----------------------
st.header("3. Deployment Modes")
st.markdown(
    """
    BioZip automatically detects its runtime environment and adjusts available features accordingly.
    This ensures the app runs smoothly on resource-constrained cloud platforms while providing
    full capabilities for local development.
    """
)

col5, col6 = st.columns(2)

with col5:
    st.markdown("### ☁️ Streamlit Cloud")
    st.markdown(
        """
        <div class="tech-card">
        <b>Detection Method</b><br>
        The app detects cloud deployment by checking:
        <ul>
            <li><code>STREAMLIT_SHARING_MODE</code> environment variable</li>
            <li><code>STREAMLIT_SERVER_HEADLESS</code> = "true"</li>
            <li>Existence of <code>/mount/src</code> directory</li>
        </ul>
        <br>
        <b>Available Features</b><br>
        ✅ Text encoding & decoding (full Gemini integration)<br>
        ✅ Video encoding (FFmpeg preprocessing)<br>
        ✅ Video decoding (grayscale output)<br>
        ✅ Temporal smoothing<br>
        ❌ AI Colorization (model too large)<br>
        ❌ ESRGAN Upscaling (requires GPU/heavy compute)<br>
        </div>
        """,
        unsafe_allow_html=True
    )

with col6:
    st.markdown("### 💻 Local Development")
    st.markdown(
        """
        <div class="tech-card">
        <b>Detection Method</b><br>
        If none of the cloud indicators are present, the app runs in local/development mode.
        You can also force this with <code>ENV=development</code>.
        <br><br>
        <b>Available Features</b><br>
        ✅ Text encoding & decoding (full Gemini integration)<br>
        ✅ Video encoding (FFmpeg preprocessing)<br>
        ✅ Video decoding (grayscale output)<br>
        ✅ Temporal smoothing<br>
        ✅ AI Colorization (Zhang et al. CNN)<br>
        ✅ ESRGAN Upscaling (4x resolution boost)<br>
        <br>
        <b>Requirements for AI Enhancement</b><br>
        Place models in <code>models/</code> directory:
        <ul>
            <li><code>RealESRGAN_x4plus.pth</code></li>
            <li><code>colorization_release_v2.caffemodel</code></li>
            <li><code>colorization_deploy_v2.prototxt</code></li>
            <li><code>pts_in_hull.npy</code></li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    """
    ### 📚 References & Credits
    
    **Research Papers:**
    - **Semantic Compression:** *Language Models as Semantic Compressors* (Delétang et al., 2023) - [arXiv:2304.01106](https://arxiv.org/pdf/2304.01106)
    - **DNA Encoding:** *Towards practical, high-capacity, low-maintenance information storage in synthesized DNA* (Goldman et al., 2013) - [PMC3672958](https://pmc.ncbi.nlm.nih.gov/articles/PMC3672958/)
    - **Colorization:** *Colorful Image Colorization* (Zhang et al., 2016) - [ECCV 2016](https://arxiv.org/abs/1603.08511)
    - **Super-Resolution:** *Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data* (Wang et al., 2021) - [ICCV 2021](https://arxiv.org/abs/2107.10833)

    **Core Technologies:**
    - **Encoding:** `biopython`, `zlib`
    - **AI/ML:** `torch`, `opencv-python`, `google-generativeai`, `spacy`, `sentence-transformers`
    - **Super-Resolution:** `basicsr`, `realesrgan`
    - **App:** `streamlit`
    """
)
