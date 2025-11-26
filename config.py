"""
Centralized configuration for BioZip application.
"""
from pathlib import Path
from enum import Enum
import os
from typing import Optional


class Environment(str, Enum):
    """Application environment."""
    DEVELOPMENT = "development"
    PRODUCTION = "production"


class CloudProvider(str, Enum):
    """Cloud storage provider."""
    LOCAL = "local"
    AWS_S3 = "aws_s3"
    GCS = "gcs"


def is_running_on_streamlit_cloud() -> bool:
    """Detect if running on Streamlit Cloud."""
    # Streamlit Cloud sets specific environment variables
    return (
        os.getenv("STREAMLIT_SHARING_MODE") is not None
        or os.getenv("STREAMLIT_SERVER_HEADLESS") == "true"
        or os.path.exists("/mount/src")  # Streamlit Cloud mounts repo here
    )


class Config:
    """Central configuration management."""

    # Environment - auto-detect Streamlit Cloud
    IS_STREAMLIT_CLOUD = is_running_on_streamlit_cloud()
    ENV = Environment.PRODUCTION if IS_STREAMLIT_CLOUD else Environment(os.getenv("ENV", "development"))

    # API Keys
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # Paths
    BASE_DIR = Path(__file__).parent
    HUFFMAN_DICT_PATH = BASE_DIR / "oligos" / "huffman_bytes_dict.json"
    HUFFMAN_TERNARY_PATH = BASE_DIR / "oligos" / "huffman_dict.json"

    # Text pipeline parameters
    MAX_TEXT_SIZE_CHARS = 10000
    DEFAULT_MASKING_RATIO = 0.3
    PAYLOAD_LEN_TEXT = 100
    OVERLAP_TEXT = 20
    HEADER_TRITS_LEN = 25

    # Video pipeline parameters
    PAYLOAD_LEN_VIDEO = 100
    OVERLAP_VIDEO = 20
    DEFAULT_VIDEO_WIDTH = 160
    DEFAULT_VIDEO_HEIGHT = 90
    DEFAULT_VIDEO_FPS = 12
    DEFAULT_VIDEO_CRF = 40
    DEFAULT_SEGMENT_SECONDS = 2
    DEFAULT_COST_PER_NT = 0.05

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # Cloud storage (not used in deployment-ready version)
    CLOUD_PROVIDER = CloudProvider(os.getenv("CLOUD_PROVIDER", "local"))
    AWS_S3_BUCKET: Optional[str] = os.getenv("AWS_S3_BUCKET")
    AWS_REGION: Optional[str] = os.getenv("AWS_REGION", "us-east-1")
    GCS_BUCKET: Optional[str] = os.getenv("GCS_BUCKET")
    GCS_PROJECT_ID: Optional[str] = os.getenv("GCS_PROJECT_ID")
    LOCAL_UPLOADS_DIR = "uploads"

    @classmethod
    def initialize(cls) -> None:
        """Validate configuration on startup."""
        if not cls.HUFFMAN_DICT_PATH.exists():
            raise FileNotFoundError(
                f"Huffman dictionary not found at {cls.HUFFMAN_DICT_PATH}"
            )

    @classmethod
    def validate_api_key(cls) -> str:
        """Get and validate Gemini API key from environment or secrets."""
        if not cls.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY not set. "
                "Please set it as an environment variable or in Streamlit secrets."
            )
        return cls.GEMINI_API_KEY
