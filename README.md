## BioZip - RWG Hackaton 2025

A prototype exploring semantic compression for DNA-based data storage.
The system reduces text to a compact semantic skeleton, encodes that representation into
DNA-like oligos, and reconstructs readable text using a language model (Gemini).

This repository provides the complete end-to-end workflow:

- Semantic analysis + token importance masking (spaCy + Sentence-Transformers)
- Encoding the semantic skeleton into bytes → Huffman ternary → Goldman DNA
- Fragmenting DNA into overlapping 117-nt oligos
- Decoding noisy oligos back into a semantic skeleton
- Reconstructing readable text via Gemini

## Pipeline Overview

Input Text
	↓
Semantic Encoder (spaCy + SBERT)
	↓
Important Tokens + Positions → Gap Skeleton
	↓
Bytes → Huffman Ternary → Goldman DNA
	↓
Fragment into 117-nt Oligos (with 20-nt overlap)
	↓
[Optional Noise Simulation: subs / ins / dels]
	↓
Reassembly → Inverse Goldman → Huffman Decode
	↓
Semantic Skeleton
	↓
Gemini LLM Reconstruction → Readable Text

## Highlights

- Streamlit UI (`app.py`) with two flows:
  - Encode text → DNA oligo file (downloadable)
  - Upload DNA oligos → decode → reconstruct text

- Modular encoder/decoder architecture:
  - `input_encoder.py` — semantic token weighting + skeleton generation
  - `input_pipeline.py` — text → bytes → ternary → DNA
  - `output_pipeline.py` — oligos → DNA → bytes → skeleton

- Low-level utilities:
  - Huffman coding
  - Ternary ↔ DNA (Goldman)
  - Fragmentation, reverse-complementing, and reassembly

## Quick Start

Prerequisites

- Python 3.11+
- Recommended: create a virtual environment (venv or conda)

Install via pip

```bash
pip install -r dependencies.txt
python -m spacy download en_core_web_sm
```

Or using conda

```bash
conda env create -f environment.yml
conda activate rwg-env
pip install -r dependencies.txt
python -m spacy download en_core_web_sm
```

The project expects Huffman dictionary JSONs under `oligos/` (e.g. `oligos/huffman_bytes_dict.json`).

## Running the Streamlit Demo

```bash
streamlit run app.py
```

The UI supports:

1. Text → DNA oligo file

	Paste text → select masking ratio →
	Click “Encode text → DNA file” → download `dna_oligos.txt` (one 117-nt oligo per line).

2. DNA file → reconstructed text

	Upload `dna_oligos.txt` → decode → Gemini reconstructs a readable paragraph.

## Repository Structure

- `app.py` — Streamlit UI demo
- `input_encoder.py` — semantic token importance + gap skeleton
- `input_pipeline.py` — text → bytes → ternary → DNA encoding
- `output_pipeline.py` — oligos → DNA → bytes → skeleton decoding
- `text_reconstruction.py` — Gemini prompt + reconstruction logic
- `oligos/oligos.py` — Huffman coding, ternary/DNA mapping, fragmentation helpers

## Testing & Validation

You can smoke-test the pipeline by running:

```bash
python input_pipeline.py
python output_pipeline.py
```

Then use the Streamlit app for a full end-to-end encode → decode → reconstruct cycle.

## Authors

- Valeria Jackson Sandoval
- Emiel Evenepoel
- Yichen Fu

## License

This project is released under the MIT License. See `LICENSE` for details.
