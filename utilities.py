"""
Shared utilities and helper functions for BioZip pipelines.
"""
import io
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import tempfile
import shutil
import requests
import os
import streamlit as st

logger = logging.getLogger(__name__)


class InMemoryWorkspace:
    """
    Context manager for temporary in-memory workspace.
    Used for preprocessing video and other temporary operations.
    """

    def __init__(self, prefix: str = "biozip_"):
        self.prefix = prefix
        self.temp_dir: Path = None

    def __enter__(self) -> Path:
        self.temp_dir = Path(tempfile.mkdtemp(prefix=self.prefix))
        logger.debug(f"Created temporary workspace: {self.temp_dir}")
        return self.temp_dir

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.debug(f"Cleaned up temporary workspace: {self.temp_dir}")


def bytes_to_download_file(data: bytes, filename: str, mime_type: str = "text/plain") -> Tuple[bytes, str, str]:
    """
    Prepare bytes for Streamlit download button.
    
    Args:
        data: Binary data to download
        filename: Download filename
        mime_type: MIME type for the file
    
    Returns:
        Tuple of (data, filename, mime_type)
    """
    return data, filename, mime_type


def format_cost_estimate(cost_per_nt: float, total_bases: int, decimals: int = 2) -> str:
    """
    Format estimated synthesis cost as a readable string.
    
    Args:
        cost_per_nt: Cost per nucleotide in euros
        total_bases: Total number of bases
        decimals: Number of decimal places
    
    Returns:
        Formatted cost string (e.g., "12.50 €")
    """
    total_cost = cost_per_nt * total_bases
    return f"{total_cost:.{decimals}f} €"


def get_dna_statistics(oligo_sequences: List[str]) -> Dict[str, Any]:
    """
    Compute statistics about generated DNA oligos.
    
    Args:
        oligo_sequences: List of DNA sequences
    
    Returns:
        Dictionary with statistics (total oligos, total bases, average length, etc.)
    """
    if not oligo_sequences:
        return {
            "total_oligos": 0,
            "total_bases": 0,
            "average_length": 0,
            "min_length": 0,
            "max_length": 0,
        }

    lengths = [len(seq) for seq in oligo_sequences]
    total_bases = sum(lengths)

    return {
        "total_oligos": len(oligo_sequences),
        "total_bases": total_bases,
        "average_length": total_bases / len(oligo_sequences),
        "min_length": min(lengths),
        "max_length": max(lengths),
    }


def validate_dna_sequence(seq: str, min_length: int = 50, max_length: int = 200) -> bool:
    """
    Validate DNA sequence (contains only ACGT and within length bounds).
    
    Args:
        seq: DNA sequence string
        min_length: Minimum allowed length
        max_length: Maximum allowed length
    
    Returns:
        True if valid, False otherwise
    """
    if not seq:
        return False

    if not (min_length <= len(seq) <= max_length):
        return False

    valid_bases = set("ACGT")
    return all(base in valid_bases for base in seq)


def log_encoding_stats(
    total_tokens: int,
    important_tokens: int,
    masking_ratio: float,
    dna_length: int,
    num_oligos: int,
) -> None:
    """
    Log encoding statistics for monitoring.
    
    Args:
        total_tokens: Total tokens in original text
        important_tokens: Tokens preserved after masking
        masking_ratio: Masking ratio applied
        dna_length: Length of final DNA sequence
        num_oligos: Number of oligos generated
    """
    compression_ratio = 1 - (important_tokens / total_tokens) if total_tokens > 0 else 0
    logger.info(
        f"Encoding complete: "
        f"total_tokens={total_tokens}, "
        f"important_tokens={important_tokens}, "
        f"compression_ratio={compression_ratio:.2%}, "
        f"dna_length={dna_length}nt, "
        f"num_oligos={num_oligos}"
    )


def log_decoding_stats(
    num_oligos: int,
    recovered_tokens: int,
    total_tokens: int,
) -> None:
    """
    Log decoding statistics for monitoring.
    
    Args:
        num_oligos: Number of oligos decoded
        recovered_tokens: Number of tokens recovered
        total_tokens: Expected total tokens
    """
    recovery_ratio = (recovered_tokens / total_tokens * 100) if total_tokens > 0 else 0
    logger.info(
        f"Decoding complete: "
        f"num_oligos={num_oligos}, "
        f"recovered_tokens={recovered_tokens}/{total_tokens} "
        f"({recovery_ratio:.1f}%)"
    )


def download_file_from_url(url: str, dest_path: Path) -> None:
    """
    Download a file from a URL to a destination path.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def check_and_download_models() -> None:
    """
    Check if required model files exist, and download them if missing.
    Only downloads in development environment to prevent deployment timeouts.
    """
    # Import Config here to avoid circular imports
    from config import Config

    if Config.ENV != "development":
        logger.info("Skipping model download in non-development environment.")
        return

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    models = {
        "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "colorization_release_v2.caffemodel": "https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1"
    }

    for model_name, url in models.items():
        model_path = models_dir / model_name
        if not model_path.exists():
            logger.info(f"Downloading {model_name}...")
            print(f"Downloading {model_name}...") # Print to stdout for Streamlit logs
            try:
                download_file_from_url(url, model_path)
                logger.info(f"Successfully downloaded {model_name}")
            except Exception as e:
                logger.error(f"Failed to download {model_name}: {e}")
                # Remove partial file if download failed
                if model_path.exists():
                    model_path.unlink()
                raise e


def apply_theme():
    """
    Apply a high-contrast theme (Light or Dark) based on user selection.
    """
    # Initialize session state for theme if not present
    if "theme" not in st.session_state:
        st.session_state.theme = "Dark"

    # Add theme toggle to sidebar
    with st.sidebar:
        st.markdown("### Theme Settings")
        selected_theme = st.radio(
            "Choose Theme",
            ["Dark", "Light"],
            index=0 if st.session_state.theme == "Dark" else 1,
            key="theme_selector"
        )
        
        # Update session state if changed
        if selected_theme != st.session_state.theme:
            st.session_state.theme = selected_theme
            st.rerun()

    # Define CSS variables based on theme
    if st.session_state.theme == "Dark":
        css_vars = """
            --bg-color: #000000;
            --text-color: #ffffff;
            --accent-color: #ffffff;
            --secondary-bg: #000000;
            --border-color: #ffffff;
            --button-bg: #ffffff;
            --button-text: #000000;
            --card-bg: #000000;
            --sidebar-bg: #000000;
            --code-bg: #1a1a1a;
            --shadow-color: #ffffff;
        """
    else:
        css_vars = """
            --bg-color: #ffffff;
            --text-color: #000000;
            --accent-color: #000000;
            --secondary-bg: #ffffff;
            --border-color: #000000;
            --button-bg: #000000;
            --button-text: #ffffff;
            --card-bg: #ffffff;
            --sidebar-bg: #ffffff;
            --code-bg: #f0f0f0;
            --shadow-color: #000000;
        """

    st.markdown(
        f"""
        <link href="https://fonts.googleapis.com/css2?family=Alexandria:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            /* High Contrast Theme Variables */
            :root {{
                {css_vars}
            }}

            /* Force background and text colors */
            .stApp {{
                background-color: var(--bg-color);
                color: var(--text-color);
            }}
            
            /* Fix white header bar */
            header[data-testid="stHeader"] {{
                background-color: var(--bg-color) !important;
            }}

            html, body, [class*="css"] {{
                font-family: 'Alexandria', sans-serif !important;
                color: var(--text-color) !important;
                background-color: var(--bg-color);
            }}

            /* Primary CTA button style (all st.button) */
            div.stButton > button {{
                background-color: var(--button-bg) !important;
                color: var(--button-text) !important;
                border: 2px solid var(--border-color) !important;
                border-radius: 0px !important; /* Sharp edges for high contrast feel */
                font-weight: 700 !important;
                text-transform: uppercase;
                padding: 0.75rem 2rem;
                font-size: 1.1rem;
                transition: all 0.2s ease;
                box-shadow: 4px 4px 0px var(--shadow-color) !important;
            }}

            div.stButton > button:hover {{
                background-color: var(--bg-color) !important;
                color: var(--text-color) !important;
                border: 2px solid var(--border-color) !important;
                transform: translate(-2px, -2px);
                box-shadow: 6px 6px 0px var(--border-color) !important;
            }}

            /* Headings */
            h1, h2, h3, h4, h5, h6 {{
                color: var(--text-color) !important;
                font-weight: 800 !important;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}

            /* Cards (feature-card, pipeline-card, tech-card) */
            .feature-card, .pipeline-card, .tech-card {{
                background-color: var(--card-bg) !important;
                border: 3px solid var(--border-color) !important;
                border-radius: 0px !important;
                color: var(--text-color) !important;
                box-shadow: 8px 8px 0px var(--shadow-color) !important; /* Hard shadow */
                padding: 2rem;
                margin-bottom: 1rem;
            }}
            
            /* Remove specific tech-card border-left if it conflicts */
            .tech-card {{
                border-left: 3px solid var(--border-color) !important;
            }}

            /* Inputs */
            .stTextInput > div > div > input, .stTextArea > div > div > textarea, .stNumberInput > div > div > input {{
                background-color: var(--bg-color) !important;
                color: var(--text-color) !important;
                border: 2px solid var(--border-color) !important;
                border-radius: 0px !important;
                caret-color: var(--text-color);
            }}
            
            /* Sidebar */
            [data-testid="stSidebar"] {{
                background-color: var(--sidebar-bg) !important;
                border-right: 3px solid var(--border-color) !important;
            }}
            
            /* Force all text in sidebar to be correct color */
            [data-testid="stSidebar"] * {{
                color: var(--text-color) !important;
            }}

            /* Info/Success/Error boxes */
            .stAlert {{
                background-color: var(--bg-color) !important;
                border: 2px solid var(--border-color) !important;
                color: var(--text-color) !important;
                border-radius: 0px !important;
                box-shadow: 4px 4px 0px var(--shadow-color) !important;
            }}
            
            /* Links */
            a {{
                color: var(--text-color) !important;
                text-decoration: underline !important;
                font-weight: bold;
            }}
            
            /* Dividers */
            hr {{
                border-top: 2px solid var(--border-color) !important;
            }}
            
            /* Code blocks */
            code {{
                color: var(--text-color) !important;
                background-color: var(--code-bg) !important;
                border: 1px solid var(--border-color) !important;
                font-weight: bold !important;
            }}
            
            /* Radio buttons in sidebar */
            .stRadio > label {{
                color: var(--text-color) !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------- INPUT VALIDATION ----------------------

def validate_text_input(text: str, max_chars: int = 10000) -> Tuple[bool, str]:
    """
    Validate text input for encoding.
    
    Args:
        text: Input text to validate
        max_chars: Maximum allowed characters
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not text or not text.strip():
        return False, "Text cannot be empty. Please enter some text to encode."
    
    if len(text) > max_chars:
        return False, f"Text too long ({len(text):,} chars). Maximum allowed: {max_chars:,} characters."
    
    # Check for minimum meaningful content
    words = text.split()
    if len(words) < 3:
        return False, "Text too short. Please enter at least a few words for meaningful compression."
    
    return True, ""


def validate_video_file(file, max_size_mb: int = 50) -> Tuple[bool, str]:
    """
    Validate uploaded video file.
    
    Args:
        file: Streamlit uploaded file object
        max_size_mb: Maximum file size in MB
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if file is None:
        return False, "No video file uploaded. Please select a video file."
    
    # Check file size
    file_size_mb = file.size / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return False, f"Video file too large ({file_size_mb:.1f} MB). Maximum allowed: {max_size_mb} MB for demo."
    
    # Check extension
    valid_extensions = [".mp4", ".mov", ".mkv", ".avi", ".webm"]
    file_ext = Path(file.name).suffix.lower()
    if file_ext not in valid_extensions:
        return False, f"Unsupported video format '{file_ext}'. Supported: {', '.join(valid_extensions)}"
    
    return True, ""


def validate_dna_file(file) -> Tuple[bool, str, List[str]]:
    """
    Validate and parse uploaded DNA oligo file.
    
    Args:
        file: Streamlit uploaded file object
    
    Returns:
        Tuple of (is_valid, error_message, oligo_sequences)
    """
    if file is None:
        return False, "No DNA file uploaded. Please select a file.", []
    
    try:
        raw = file.read().decode("ascii", errors="ignore")
        file.seek(0)  # Reset file pointer
    except Exception as e:
        return False, f"Could not read file: {e}", []
    
    if not raw.strip():
        return False, "File is empty. Please upload a valid DNA oligo file.", []
    
    lines = [ln.strip() for ln in raw.splitlines()]
    # Filter out FASTA headers and empty lines
    oligos = [ln for ln in lines if ln and not ln.startswith(">")]
    
    if not oligos:
        return False, "No valid DNA sequences found. Check that the file contains oligo sequences.", []
    
    # Validate sequences contain only valid bases
    valid_bases = set("ACGT")
    invalid_count = 0
    valid_oligos = []
    
    for seq in oligos:
        if all(base in valid_bases for base in seq):
            valid_oligos.append(seq)
        else:
            invalid_count += 1
    
    if not valid_oligos:
        return False, "No valid DNA sequences found. Sequences must contain only A, C, G, T.", []
    
    warning = ""
    if invalid_count > 0:
        warning = f" (Note: {invalid_count} invalid sequences were skipped)"
    
    return True, warning, valid_oligos


# ---------------------- METRICS & ANALYTICS ----------------------

def calculate_compression_metrics(
    original_size: int,
    compressed_size: int,
    dna_bases: int,
) -> Dict[str, Any]:
    """
    Calculate comprehensive compression metrics.
    
    Args:
        original_size: Original data size in bytes/chars
        compressed_size: Compressed size (e.g., important tokens)
        dna_bases: Total DNA bases used
    
    Returns:
        Dictionary with various metrics
    """
    compression_ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
    bits_per_base = (compressed_size * 8) / dna_bases if dna_bases > 0 else 0
    
    # DNA storage density (theoretical)
    # 1 nucleotide = 2 bits of information
    theoretical_bits = dna_bases * 2
    efficiency = (compressed_size * 8) / theoretical_bits * 100 if theoretical_bits > 0 else 0
    
    return {
        "original_size": original_size,
        "compressed_size": compressed_size,
        "compression_ratio_percent": round(compression_ratio, 1),
        "dna_bases": dna_bases,
        "bits_per_base": round(bits_per_base, 2),
        "storage_efficiency_percent": round(efficiency, 1),
        "space_saved_percent": round(compression_ratio, 1),
    }


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def display_metrics_card(metrics: Dict[str, Any], title: str = "Compression Metrics") -> None:
    """
    Display a styled metrics card in Streamlit.
    
    Args:
        metrics: Dictionary of metrics to display
        title: Card title
    """
    st.markdown(
        f"""
        <div style="background-color: var(--card-bg); border: 2px solid var(--border-color); 
                    padding: 1.5rem; margin: 1rem 0; box-shadow: 4px 4px 0px var(--shadow-color);">
            <h4 style="margin: 0 0 1rem 0; color: var(--text-color);">📊 {title}</h4>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem;">
                <div style="color: var(--text-color);">
                    <b>Compression:</b> {metrics.get('compression_ratio_percent', 0)}%
                </div>
                <div style="color: var(--text-color);">
                    <b>DNA Bases:</b> {metrics.get('dna_bases', 0):,}
                </div>
                <div style="color: var(--text-color);">
                    <b>Storage Efficiency:</b> {metrics.get('storage_efficiency_percent', 0)}%
                </div>
                <div style="color: var(--text-color);">
                    <b>Bits/Base:</b> {metrics.get('bits_per_base', 0)}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------- SESSION STATE HELPERS ----------------------

def save_encoding_result(key: str, data: Dict[str, Any]) -> None:
    """
    Save encoding result to session state for later retrieval.
    
    Args:
        key: Unique key for this encoding result
        data: Data to save
    """
    if "encoding_results" not in st.session_state:
        st.session_state.encoding_results = {}
    st.session_state.encoding_results[key] = data


def get_encoding_result(key: str) -> Dict[str, Any] | None:
    """
    Retrieve encoding result from session state.
    
    Args:
        key: Key for the encoding result
    
    Returns:
        Saved data or None if not found
    """
    if "encoding_results" not in st.session_state:
        return None
    return st.session_state.encoding_results.get(key)


def clear_encoding_results() -> None:
    """
    Clear all saved encoding results from session state.
    """
    if "encoding_results" in st.session_state:
        st.session_state.encoding_results = {}


# ---------------------- ERROR HANDLING HELPERS ----------------------

class BioZipError(Exception):
    """Base exception for BioZip errors."""
    
    def __init__(self, message: str, suggestion: str = ""):
        self.message = message
        self.suggestion = suggestion
        super().__init__(self.message)


class EncodingError(BioZipError):
    """Error during encoding process."""
    pass


class DecodingError(BioZipError):
    """Error during decoding process."""
    pass


class ModelNotFoundError(BioZipError):
    """Error when required model files are missing."""
    pass


def display_error_with_help(error: Exception, context: str = "") -> None:
    """
    Display an error message with helpful suggestions.
    
    Args:
        error: The exception that occurred
        context: Additional context about what was being attempted
    """
    error_msg = str(error)
    
    # Common error patterns and suggestions
    suggestions = {
        "ffmpeg": "FFmpeg is not installed. Install it with `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Linux).",
        "GEMINI_API_KEY": "Gemini API key not configured. Set it in environment variables or Streamlit secrets.",
        "huffman": "Huffman dictionary file not found. Ensure the `oligos/` directory contains the required files.",
        "model": "Required model file not found. Run locally to auto-download, or manually place models in the `models/` directory.",
        "memory": "Out of memory. Try with a smaller file or reduce resolution settings.",
        "timeout": "Operation timed out. Try with a smaller file.",
    }
    
    suggestion = ""
    for key, sugg in suggestions.items():
        if key.lower() in error_msg.lower():
            suggestion = sugg
            break
    
    if isinstance(error, BioZipError) and error.suggestion:
        suggestion = error.suggestion
    
    st.error(f"❌ **{context}**: {error_msg}" if context else f"❌ {error_msg}")
    
    if suggestion:
        st.info(f"💡 **Suggestion:** {suggestion}")
