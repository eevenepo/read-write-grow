"""
Shared utilities and helper functions for BioZip pipelines.
"""
import io
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import tempfile
import shutil

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
