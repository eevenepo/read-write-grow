import streamlit as st
from PIL import Image
from config import Config

st.set_page_config(
    page_title="BioZip - Technical Architecture",
    page_icon="🧬",
    layout="wide",
)

# ---------------------- GLOBAL STYLES ----------------------
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Alexandria:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"] {
            font-family: 'Alexandria', sans-serif !important;
        }

        /* Primary CTA button style (all st.button) */
        div.stButton > button {
            background: #0b3d91;
            color: #ffffff;
            padding: 0.75rem 2rem;
            border-radius: 12px;
            border: none;
            font-weight: 600;
            font-size: 1.1rem;
            cursor: pointer;
            width: 100%;
            transition: all 0.3s ease;
        }

        div.stButton > button:hover {
            background: #1053c4;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(11, 61, 145, 0.2);
        }

        .stMarkdown h2, .stMarkdown h3 {
            color: #0b3d91;
        }
        .tech-card {
            background-color: #f0f2f6;
            color: #31333F; /* Force dark text for readability */
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            border-left: 5px solid #0b3d91;
        }
        .tech-card code {
            color: #ff4b4b; /* Streamlit red for code */
            background-color: #ffffff; /* White bg for code */
        }
    </style>
""", unsafe_allow_html=True)

if st.button("← Back to Home"):
    st.switch_page("app.py")

# ---------------------- SIDEBAR ----------------------
with st.sidebar:
    try:
        logo = Image.open("assets/logo.png")
        st.image(logo, use_container_width=True)
    except FileNotFoundError:
        pass
    
    st.markdown("---")
    st.markdown("### Navigation")
    st.page_link("app.py", label="Home", icon="🏠")
    st.page_link("pages/biozip_text.py", label="Text Pipeline", icon="📝")
    st.page_link("pages/biozip_video.py", label="Video Pipeline", icon="🎬")
    st.page_link("pages/technical_details.py", label="Architecture", icon="⚙️")
    st.markdown("---")

st.sidebar.info(
    f"**Environment**: {Config.ENV.value}\n\n"
    f"**Gemini Model**: {Config.GEMINI_MODEL}"
)

# Hero Section
col_hero_text, col_hero_img = st.columns([2, 1])
with col_hero_text:
    st.title("Technical Architecture")
    st.markdown(
        """
        <div style="font-size: 1.1rem; color: #555; margin-bottom: 1rem;">
        BioZip uses advanced compression, biological encoding schemes, and AI reconstruction 
        to store data efficiently in synthetic DNA. Below is a deep dive into the pipelines.
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
        grayscale video file.
        <br><br>
        <b>2. AI Colorization</b><br>
        A CNN (<b>Zhang et al.</b>) analyzes the Luma (L) channel and predicts the missing Chroma (ab) channels. 
        It "hallucinates" plausible colors (e.g., sky is blue, grass is green) based on the semantic content 
        of the grayscale frames.
        <br><br>
        <b>3. Super-Resolution</b><br>
        <b>Real-ESRGAN</b> (x4) upscales the video (e.g., 160p → 640p). It uses a GAN trained on real-world 
        images to invent realistic high-frequency textures and sharpen edges that were lost during compression.
        <br><br>
        <b>4. Temporal Smoothing</b><br>
        AI models process each frame in isolation, often causing jittery colors or flickering. 
        We apply a <b>stabilization filter</b> that blends the current frame with its neighbors to smooth out this noise. 
        To prevent "ghosting" (double-exposure effects) when the camera cuts to a new scene, our algorithm detects 
        the change and instantly resets the filter, ensuring crisp transitions.
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

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
