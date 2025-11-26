"""BioZip: DNA-based data storage application landing page."""
import streamlit as st
from PIL import Image
from utilities import check_and_download_models, apply_theme

# Check and download models on startup
with st.spinner("Checking and downloading required AI models... (This may take a minute on first run)"):
    try:
        check_and_download_models()
    except Exception as e:
        st.error(f"Failed to download models: {e}")

st.set_page_config(
    page_title="BioZip",
    page_icon="🧬",
    layout="wide",
)

# -------------- Global Styles -----------------
apply_theme()

# -------------- Sidebar -----------------
with st.sidebar:
    try:
        logo = Image.open("assets/logo.png")
        st.image(logo, use_container_width=True)
    except FileNotFoundError:
        st.warning("Logo not found")
    
    st.markdown("---")
    st.markdown("### Navigation")
    st.page_link("app.py", label="Home", icon="🏠")
    st.page_link("pages/biozip_text.py", label="Text Pipeline", icon="📝")
    st.page_link("pages/biozip_video.py", label="Video Pipeline", icon="🎬")
    st.page_link("pages/technical_details.py", label="Architecture", icon="⚙️")

# -------------- Hero Section -----------------
col_hero_text, col_hero_img = st.columns([1.5, 1])

with col_hero_text:
    st.markdown("# Unlock the Future of Data Storage")
    st.markdown(
        """
        <div style="font-size: 1.2rem; line-height: 1.6; color: #555; margin-bottom: 2rem;">
        Harness the power of DNA to store massive amounts of data in a fraction of the space.  
        BioZip offers DNA as a secure and sustainable data medium, built to last for millennia.
        <br><br>
        <b>Store Smarter. Store Forever.</b>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Hero CTAs
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📝 Try Text Demo", key="hero_text_demo"):
            st.switch_page("pages/biozip_text.py")
    with c2:
        if st.button("🎬 Try Video Demo", key="hero_video_demo"):
            st.switch_page("pages/biozip_video.py")

with col_hero_img:
    try:
        # Display logo prominently in hero if available, or a nice graphic
        st.image("assets/logo.png", width=400)
    except:
        st.markdown("🧬")

st.markdown("---")

# -------------- Story Sections -----------------
with st.container():
    txt, img = st.columns([1, 1], gap="large")
    with img:
        st.image("https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExdzhmeHQ3ZXA5dnB6dzF3ZmF0Mm5wenhmemp1ODc1MnRqZzBvbDR5cyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/pOEbLRT4SwD35IELiQ/giphy.gif", use_container_width=True)

    with txt:
        st.markdown("### The Age of Data")
        st.markdown(
            """
            While our need for data storage has grown exponentially, this has come at a cost.  
            Modern data centers consume huge amounts of energy, cause pollution, and fragment habitats.
            
            We need a solution that scales with humanity without destroying our home.
            """
        )

st.markdown("<br>", unsafe_allow_html=True)

with st.container():
    img, txt = st.columns([1, 1], gap="large")
    with img:
        st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExc2p3c2V0dXk5eDc0d2ZjdTZoaHdjY253cGtkcW5lbWF5OWRoMTUweSZlcD12MV9naWZzX3NlYXJjaCZjdD1n/8NdQyUi0C7ug0/giphy.gif", use_container_width=True)
    with txt:
        st.markdown("### Nature's Oldest Database")
        st.markdown(
            """
            Unlike magnetic disks that degrade in decades, DNA boasts extreme resilience.
            
            *   **Density:** Store the world's data in a shoebox.
            *   **Durability:** Lasts for 1,000,000+ years in cold storage.
            *   **Sustainability:** Biodegradable and zero-energy at rest.
            """
        )

# -------------- Features -----------------
st.markdown("---")
st.markdown("<h2 style='text-align: center;'>Why DNA?</h2>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

with st.container():
    resilience, eco, space = st.columns(3)

    with resilience:
        st.markdown(
            """
            <div class="feature-card">
                <h3>💾 Resilience</h3>
                <p>In good conditions, DNA can last over 1,000,000+ years. No more data rot or migration headaches.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with eco:
        st.markdown(
            """
            <div class="feature-card">
                <h3>🌱 Eco-friendly</h3>
                <p>DNA storage requires no power to maintain data. It's the ultimate cold storage solution.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with space:
        st.markdown(
            """
            <div class="feature-card">
                <h3>🚀 Universal</h3>
                <p>As long as there is life, we will know how to read DNA. It is the only truly future-proof format.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("---")

# -------------- Bottom CTA -----------------
st.markdown("<h2 style='text-align: center;'>Ready to Encode?</h2>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

bottom_col1, bottom_col2, bottom_col3 = st.columns([1, 1, 1])

with bottom_col1:
    st.info("**Text Pipeline**\n\nSemantic compression → token skeleton → DNA.")
    if st.button("Go to Text Demo", key="bottom_text_demo"):
        st.switch_page("pages/biozip_text.py")

with bottom_col2:
    st.info("**Video Pipeline**\n\nGrayscale compression → DNA → AI Restoration.")
    if st.button("Go to Video Demo", key="bottom_video_demo"):
        st.switch_page("pages/biozip_video.py")

with bottom_col3:
    st.info("**Technical Details**\n\nDeep dive into our algorithms and architecture.")
    if st.button("View Architecture", key="bottom_tech_demo"):
        st.switch_page("pages/technical_details.py")

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        📧 Contact: support@biozip.com | © 2025 BioZip
    </div>
    """, 
    unsafe_allow_html=True
)
