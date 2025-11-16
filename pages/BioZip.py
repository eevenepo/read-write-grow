import streamlit as st

# -------------- Page Config -----------------
st.set_page_config(
    page_title="BioZip",
    page_icon="pages/icon-192.webp",
)

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Alexandria:wght@300;400;600;700&display=swap" rel="stylesheet">

    <style>
        html, body, [class*="css"]  {
            font-family: 'Alexandria', sans-serif !important;
        }
    </style>
""", unsafe_allow_html=True) # treid using slides font but doesn't work right now

st.image("pages/favicon.ico")
st.markdown("""
    # Unlock the Future of Data Storage

    Harness the power of DNA to store massive amounts of data in a fraction of the space. 
                
    BioZip offers DNA as a secure and sustainable data medium, built to last for millennia.

    Store Smarter. Store Forever.
    """)
st.button("Try BioZip today")

with st.container():
    txt, img = st.columns(2)
    with img:
        st.markdown("![Alt Text](https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExdzhmeHQ3ZXA5dnB6dzF3ZmF0Mm5wenhmemp1ODc1MnRqZzBvbDR5cyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/pOEbLRT4SwD35IELiQ/giphy.gif)")

    with txt:
        st.markdown("""
        ### The age of data
        While our need for data storage has grown exponentially, this has come at a cost. Modern data centers consume huge amounts of energy,
        cause lots of pollution and fragment habitats. 
        """)

with st.container():
    img, txt = st.columns(2)
    with img:
        st.markdown("![Alt Text](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExc2p3c2V0dXk5eDc0d2ZjdTZoaHdjY253cGtkcW5lbWF5OWRoMTUweSZlcD12MV9naWZzX3NlYXJjaCZjdD1n/8NdQyUi0C7ug0/giphy.gif)")
    with txt:
        st.markdown("""
        ### Nature's oldest database
        Unlike magnetic disks, DNA boasts extreme resilience, lasting up to 1,000,000 years.
        """)

st.markdown("## Why DNA?")
with st.container():
    resilience, eco, space = st.columns(3)

    with resilience:
        st.markdown("""
                    ### Resilience 💾
                    In good conditions, DNA can last over 1,000,000+ years(!)""")
    with eco:
        st.markdown("""
                    ### Ecofriendly 🌱
                    DNA storage requires no power, and the sequences are completely biodegradable
                    """)
    with space:
        st.markdown("""
                    ### The data of space 🚀
                    
                    The knowledge of mankind to the stars, in the size of a phone.
                    """)

st.markdown('## Convinced? Try it today! :dna:')
st.button('Take me to the demo')
with st.container():
    st.write("📧 Contact: support@biozip.com | © 2025 BioZip")
