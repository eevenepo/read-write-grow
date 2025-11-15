import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict, Any
import lzma
import json

class InputEncoder:
    """
    Semantic encoder that compresses text by masking semantically less important tokens.
    Exports the total token number and important tokens with their positions in the paragraph.
    
    Installation:
    pip install sentence-transformers spacy numpy
    python -m spacy download en_core_web_sm
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the encoder with sentence-BERT model and spaCy tokenizer.
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        # Load sentence-BERT model (produces 384-dim vectors like in paper)
        self.bert_model = SentenceTransformer(model_name)
        
        # Load spaCy for tokenization
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Downloading spaCy model...")
            import os
            os.system('python -m spacy download en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
    
    def tokenize_text(self, text: str) -> List[List[Tuple[str, int, int]]]:
        """
        Tokenize text into sentences and tokens using spaCy.
        Args:
            text: Input text string
        Returns:
            List of sentences, where each sentence is a list of (token, start_idx, end_idx) tuples
        """
        doc = self.nlp(text)
        
        sentences = []
        for sent in doc.sents:
            tokens = []
            for token in sent:
                if not token.is_punct and not token.is_stop and not token.is_space:
                    # Store token text and its character position, leave out the punctuation
                    tokens.append((token.text, token.idx, token.idx + len(token.text)))
            sentences.append(tokens)
        
        return sentences
    
    def get_sentence_embedding(self, sentence_tokens: List[Tuple[str, int, int]]) -> np.ndarray:
        """
        Get sentence-BERT embedding for a sentence.
        Args:
            sentence_tokens: List of (token, start_idx, end_idx) tuples        
        Returns:
            384-dimensional sentence embedding
        """
        # Reconstruct sentence from tokens
        sentence_text = ' '.join([token[0] for token in sentence_tokens])
        # Use SentenceTransformer
        embedding = self.bert_model.encode(sentence_text, convert_to_numpy=True)

        return embedding
    
    def cosine_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine distance between two vectors.
        Args:
            vec1, vec2: Input vectors    
        Returns:
            Cosine distance = (1 - cosine similarity)
        """
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return 1.0 - similarity
    
    def compute_token_importance(
    self,
    sentences: List[List[Tuple[str, int, int]]]) -> Dict[str, float]:
        """
        Compute semantic importance (μ_k) for each unique token in the text.

        Args:
            sentences: List of tokenized sentences (each is a list of
                    (token_text, char_start, char_end))

        Returns:
            Dictionary mapping each unique token (string) to its importance weight μ_k
        """
        M = len(sentences)

        # ----------------------------------------------------------------------
        # STEP 1: For each token occurrence, compute normalized semantic loss σ̄_in
        # ----------------------------------------------------------------------

        # (sentence_idx, token_idx) -> normalized loss σ̄_in
        token_losses: Dict[Tuple[int, int], float] = {}

        for i, sentence in enumerate(sentences):
            N_i = len(sentence)
            if N_i == 0:
                continue

            # Original sentence embedding μ_i
            mu_i = self.get_sentence_embedding(sentence)

            # Compute raw semantic loss σ_in for each token via masking
            sigma_values: List[float] = []
            for n, (token_text, _, _) in enumerate(sentence):
                # Mask the n-th token
                masked_sentence = sentence[:n] + [('#', 0, 0)] + sentence[n+1:]

                # Masked sentence embedding μ̂_i
                mu_hat_i = self.get_sentence_embedding(masked_sentence)

                # σ_in = cosine_distance(μ_i, μ̂_i)
                sigma_in = self.cosine_distance(mu_i, mu_hat_i)
                sigma_values.append(sigma_in)

            # Normalize semantic losses within the sentence (equation 13)
            total_sigma = sum(sigma_values)
            if total_sigma > 0:
                normalized_losses = [sigma / total_sigma for sigma in sigma_values]
            else:
                # If all sigma are zero, distribute uniformly
                normalized_losses = [1.0 / N_i] * N_i

            # Store normalized loss by (sentence_idx, token_idx)
            for n in range(N_i):
                token_losses[(i, n)] = normalized_losses[n]

        # ----------------------------------------------------------------------
        # STEP 2: Build vocabulary K and positions Q_ik
        # ----------------------------------------------------------------------
        # vocab[token][i] = list of token indices n where token appears in sentence i
        vocab: Dict[str, Dict[int, List[int]]] = {}

        for i, sentence in enumerate(sentences):
            for n, (token_text, _, _) in enumerate(sentence):
                if token_text not in vocab:
                    vocab[token_text] = {}
                if i not in vocab[token_text]:
                    vocab[token_text][i] = []
                vocab[token_text][i].append(n)

        # Precompute sentence lengths α_i = N_i
        sentence_lengths = [len(s) for s in sentences]

        # ----------------------------------------------------------------------
        # STEP 3: Compute μ_k for each token type (equation 15)
        # ----------------------------------------------------------------------
        token_weights: Dict[str, float] = {}

        for token, sent_positions in vocab.items():
            # Total occurrences across all sentences: Σ_i |Q_ik|
            total_occurrences = sum(len(positions) for positions in sent_positions.values())
            if total_occurrences == 0:
                token_weights[token] = 0.0
                continue

            # Weighted sum Σ_i α_i Σ_{n ∈ Q_ik} σ̄_in
            weighted_sum = 0.0

            for i, positions in sent_positions.items():
                alpha_i = sentence_lengths[i]  # α_i = N_i
                loss_sum = 0.0

                for n in positions:
                    # Look up σ̄_in from token_losses
                    loss_sum += token_losses.get((i, n), 0.0)

                weighted_sum += alpha_i * loss_sum

            # μ_k = (1 / Σ_i |Q_ik|) * weighted_sum
            token_weights[token] = weighted_sum / total_occurrences

        return token_weights
    
    def get_important_tokens_with_positions(self, sentences: List[List[Tuple[str, int, int]]], 
                                           token_weights: Dict[str, float], 
                                           masking_ratio: float) -> List[Dict]:
        """
        Get important tokens (those NOT masked) with their positions.
        Args:
            sentences: List of tokenized sentences
            token_weights: Dictionary of token importance weights
            masking_ratio: Fraction of unique tokens to mask (0 to 1)
        Returns:
            List of dictionaries containing token info for DNA encoding
        """
        # Sort tokens by weight (ascending order - least important first)
        sorted_tokens = sorted(token_weights.items(), key=lambda x: x[1])
        
        # Determine which unique tokens to mask (least important)
        num_to_mask = int(masking_ratio * len(sorted_tokens))
        tokens_to_mask = set([token for token, _ in sorted_tokens[:num_to_mask]])
        
        # Collect all important (non-masked) tokens with their positions
        important_tokens = []
        token_position = 0  # Global token position counter
        
        for sent_idx, sentence in enumerate(sentences):
            for token_idx, (token_text, char_start, char_end) in enumerate(sentence):
                if token_text not in tokens_to_mask:
                    # This token is important - save it
                    important_tokens.append({
                        'token': token_text,
                        'importance_weight': token_weights[token_text],
                        'global_position': token_position,
                        'sentence_index': sent_idx,
                        'token_index_in_sentence': token_idx,
                        'char_start': char_start,
                        'char_end': char_end
                    })
                token_position += 1
        
        return important_tokens
    

    #VALERIAS CODE
    def positions_to_gaps(self, positions: List[int]) -> Dict:
        """
        Convert sorted token positions into gap encoding.
        
        Input:
            [0, 1, 4, 5, 7, 8, 9]

        Output:
            {
            "first_position": 0,
            "gaps": [1, 3, 1, 2, 1, 1]
            }
        """
        if not positions:
            return {"first_position": 0, "gaps": []}

        first_position = positions[0]
        gaps = []

        for i in range(1, len(positions)):
            gaps.append(positions[i] - positions[i - 1])

        return {
            "first_position": first_position,
            "gaps": gaps
        }
    
    def encode_to_gap_skeleton(self, text: str, masking_ratio: float = 0.3) -> Dict[str, Any]:
        """
        High-level helper:
        Given raw text, compute important tokens and return a gap-encoded semantic skeleton.

        Output example for:
            "DNA storage is a promising technology for future data systems."

        {
          "total_tokens": 10,
          "num_important_tokens": 7,
          "first_position": 0,
          "gaps": [1, 3, 1, 2, 1, 1],
          "tokens": [
            "DNA",
            "storage",
            "promising",
            "technology",
            "future",
            "data",
            "systems"
          ],
          "debug_positions": [0, 1, 4, 5, 7, 8, 9]
        }

        Notes:
          - gap[i] = positions[i] - positions[i-1]
          - first_position is stored explicitly (usually 0 for the first kept token)
        """
        print(f"Encoding text to GAP semantic skeleton (masking_ratio={masking_ratio})")

        # 1) Tokenize
        print("  Step 1: Tokenizing...")
        sentences = self.tokenize_text(text)
        print(f"    Found {len(sentences)} sentences")

        # 2) Compute token importance μ_k
        print("  Step 2: Computing token importance (μ_k)...")
        token_weights = self.compute_token_importance(sentences)
        print(f"    Computed weights for {len(token_weights)} unique tokens")

        # 3) Select important tokens + positions
        print("  Step 3: Selecting important tokens...")
        important_tokens = self.get_important_tokens_with_positions(
            sentences,
            token_weights,
            masking_ratio,
        )
        print(f"    Kept {len(important_tokens)} important tokens")

        # Make sure important tokens are sorted by global_position
        important_tokens = sorted(important_tokens, key=lambda t: t["global_position"])

        # Extract positions and token texts
        positions: List[int] = [t["global_position"] for t in important_tokens]
        tokens_only: List[str] = [t["token"] for t in important_tokens]

        # 4) Convert positions → gaps (reuse helper)
        gap_info = self.positions_to_gaps(positions)
        first_position = gap_info["first_position"]
        gaps = gap_info["gaps"]

        # 5) Total tokens = global tokens across all sentences
        total_tokens = sum(len(s) for s in sentences)

        gap_skeleton: Dict[str, Any] = {
            "total_tokens": total_tokens,
            "num_important_tokens": len(tokens_only),
            "first_position": first_position,
            "gaps": gaps,
            "tokens": tokens_only,
            "debug_positions": positions,
        }

        return gap_skeleton
    
    def build_word_dictionary(self, gap_skeleton: Dict[str, Any]) -> Dict[str, int]:
        """
        Build a word -> token_id dictionary from the gap skeleton.

        IDs are assigned in order of first appearance in gap_skeleton["tokens"].

        """
        tokens = gap_skeleton["tokens"]

        unique_tokens: List[str] = []
        seen = set()

        for tok in tokens:
            if tok not in seen:
                seen.add(tok)
                unique_tokens.append(tok)

        token_to_id = {tok: idx for idx, tok in enumerate(unique_tokens)}
        return token_to_id
    
    def serialize_dictionary_to_bytes(self, token_to_id: Dict[str, int]) -> bytes:
        """
        Serialize the dictionary (word -> id) into a byte stream using ASCII encoding.

        Format per entry:
            [token_id: 4 bytes big-endian]
            [word_length: 1 byte]
            [word_ascii: word_length bytes]

        Token IDs must be sequential integers (0..N-1).
        Returns:
            Byte string suitable for Huffman → trits → DNA encoding.
        """

        # Invert mapping: id -> word
        id_to_word = {idx: word for word, idx in token_to_id.items()}

        # Validate ID sequence integrity
        max_id = max(id_to_word.keys(), default=-1)
        for token_id in range(max_id + 1):
            if token_id not in id_to_word:
                raise ValueError(
                    f"Missing token_id {token_id} — IDs must be consecutive from 0..N-1"
                )

        byte_chunks = []

        for token_id in range(max_id + 1):
            word = id_to_word[token_id]

            # ASCII check
            try:
                word_bytes = word.encode("ascii")
            except UnicodeEncodeError:
                raise ValueError(f"Word '{word}' contains non-ASCII characters.")

            length = len(word_bytes)
            if length > 255:
                raise ValueError(f"Word '{word}' too long (max 255 for 1 byte length).")

            # Serialize
            byte_chunks.append(token_id.to_bytes(4, "big"))
            byte_chunks.append(length.to_bytes(1, "big"))
            byte_chunks.append(word_bytes)

        return b"".join(byte_chunks)
    
    def serialize_relational_to_bytes(
        self,
        gap_skeleton: Dict[str, Any],
        token_to_id: Dict[str, int],
    ) -> bytes:
        """
        Conventions:
            - gap for the FIRST important token is set to first_position
              (distance from position 0).
            - For token i > 0, gap = position[i] - position[i-1],
              i.e. the same 'gaps' you already computed.

        Inputs:
            gap_skeleton:
                {
                    "total_tokens": int,
                    "num_important_tokens": int,
                    "first_position": int,
                    "gaps": List[int],          # length = num_important_tokens - 1 (usually)
                    "tokens": List[str],        # important token texts in order
                    "debug_positions": List[int]
                }

            token_to_id: mapping from token string -> token_id (int)

        Returns:
            bytes: relational byte stream ready for Huffman → trits → DNA.
        """

        total_tokens = gap_skeleton["total_tokens"]
        first_position = gap_skeleton["first_position"]
        gaps = gap_skeleton["gaps"]
        tokens = gap_skeleton["tokens"]

        num_important = len(tokens)

        # Basic consistency check:
        if num_important == 0:
            # Header only, no tokens
            return total_tokens.to_bytes(4, "big") + first_position.to_bytes(2, "big")

        if len(gaps) != max(0, num_important - 1):
            raise ValueError(
                f"Gap count ({len(gaps)}) must be num_important_tokens - 1 ({num_important - 1})."
            )

        # Range checks for header fields
        if not (0 <= total_tokens <= 0xFFFFFFFF):
            raise ValueError("total_tokens must fit in 4 bytes (0..2^32-1).")

        if not (0 <= first_position <= 0xFFFF):
            raise ValueError("first_position must fit in 2 bytes (0..65535).")

        byte_chunks = []

        # ------------------------------------------------------------------
        # 1) Header
        # ------------------------------------------------------------------
        byte_chunks.append(total_tokens.to_bytes(4, "big"))
        byte_chunks.append(first_position.to_bytes(2, "big"))

        # ------------------------------------------------------------------
        # 2) Per-token entries: [gap:1 byte][token_id:4 bytes]
        #    Convention:
        #       - token 0: gap = first_position (distance from position 0)
        #       - token i>0: gap = gaps[i-1] (difference to previous important token)
        # ------------------------------------------------------------------
        for i, tok in enumerate(tokens):
            if tok not in token_to_id:
                raise ValueError(f"Token '{tok}' not found in token_to_id dictionary.")

            token_id = token_to_id[tok]

            if i == 0:
                gap_val = first_position
            else:
                gap_val = gaps[i - 1]

            if not (0 <= gap_val <= 0xFF):
                raise ValueError(
                    f"Gap value {gap_val} out of 1-byte range (0..255) for token index {i}."
                )

            if not (0 <= token_id <= 0xFFFFFFFF):
                raise ValueError(
                    f"token_id {token_id} out of 4-byte range (0..2^32-1)."
                )

            # Serialize [gap:1][token_id:4]
            byte_chunks.append(gap_val.to_bytes(1, "big"))
            byte_chunks.append(token_id.to_bytes(4, "big"))

        return b"".join(byte_chunks)

    
    #Do we need mask_tokens?
    
    def mask_tokens(self, sentences: List[List[Tuple[str, int, int]]], 
                   token_weights: Dict[str, float], 
                   masking_ratio: float) -> List[List[str]]:
        """
        Mask the least important tokens according to masking ratio.
        Args:
            sentences: List of tokenized sentences
            token_weights: Dictionary of token importance weights
            masking_ratio: Fraction of unique tokens to mask (0 to 1)      
        Returns:
            List of sentences with tokens masked (as strings)
        """
        # Sort tokens by weight (ascending order - least important first)
        sorted_tokens = sorted(token_weights.items(), key=lambda x: x[1])
        
        # Determine how many unique tokens to mask
        num_to_mask = int(masking_ratio * len(sorted_tokens))
        tokens_to_mask = set([token for token, _ in sorted_tokens[:num_to_mask]])
        
        # Apply masking
        masked_sentences = []
        for sentence in sentences:
            masked_sentence = []
            for token_text, _, _ in sentence:
                if token_text in tokens_to_mask:
                    masked_sentence.append('#')
                else:
                    masked_sentence.append(token_text)
            masked_sentences.append(masked_sentence)
        
        return masked_sentences
    
    #Do we need the LZMA compression?
    
    def lz_encode(self, masked_sentences: List[List[str]]) -> bytes:
        """
        Convert masked text to compressed bit string using LZ compression.
        Args:
            masked_sentences: List of masked sentences
        Returns:
            Compressed byte string
        """
        # Reconstruct text from masked sentences
        text = ''
        for sentence in masked_sentences:
            text += ' '.join(sentence) + ' '
        
        # Apply LZMA compression (Python's implementation of LZ)
        compressed = lzma.compress(text.encode('utf-8'))
        
        return compressed
    
    def save_important_tokens(self, important_tokens: List[Dict], output_file: str, total_tokens: int):
        """
        Save important tokens to a file for DNA encoding.
        
        Args:
            important_tokens: List of token dictionaries
            output_file: Path to output file
            total_tokens: Total number of tokens in the original text
        """
        # JSON format with simplified structure
        json_file = output_file.replace('.txt', '.json')
        
        json_output = { # Structure for JSON output, still modifiable based on needs
            "total_tokens": total_tokens,
            "tokens": [
                {
                    "position": item['global_position'],
                    "text": item['token']
                }
                for item in important_tokens
            ]
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Saved JSON data to: {json_file}")
        
        # Simple token sequence for DNA encoding (just the tokens in order)
        seq_file = output_file.replace('.txt', '_sequence.txt')
        with open(seq_file, 'w', encoding='utf-8') as f:
            tokens_only = [item['token'] for item in important_tokens]
            f.write(' '.join(tokens_only))
        print(f"  ✓ Saved token sequence to: {seq_file}")
    
    #What is this doing?

    def encode(self, text: str, masking_ratio: float = 0.3, output_file: str = 'important_tokens.txt') -> Tuple[bytes, Dict]:
        """
        Full encoding pipeline: tokenize -> compute importance -> mask -> compress.
        Also saves important tokens to file for DNA encoding.
        Args:
            text: Input text to compress
            masking_ratio: Fraction of unique tokens to mask (default 0.3)
            output_file: Path to save important tokens    
        Returns:
            Tuple of (compressed_bytes, metadata_dict)
        """
        print(f"Encoding text with masking ratio: {masking_ratio}")
        
        # Step 1: Tokenize
        print("Step 1: Tokenizing...")
        sentences = self.tokenize_text(text)
        print(f"  Found {len(sentences)} sentences")
        
        # Step 2: Compute token importance
        print("Step 2: Computing token importance...")
        token_weights = self.compute_token_importance(sentences)
        print(f"  Computed weights for {len(token_weights)} unique tokens")
        
        # Step 3: Get important tokens with positions
        print("Step 3: Extracting important tokens for DNA encoding...")
        important_tokens = self.get_important_tokens_with_positions(sentences, token_weights, masking_ratio)
        print(f"  Found {len(important_tokens)} important tokens (kept after masking)")
        
        # Step 4: Save important tokens to file
        print("Step 4: Saving important tokens to file...")
        total_tokens = sum(len(s) for s in sentences)
        self.save_important_tokens(important_tokens, output_file, total_tokens)
        
        # Step 5: Mask tokens
        print("Step 5: Masking tokens...")
        masked_sentences = self.mask_tokens(sentences, token_weights, masking_ratio)
        
        # Step 6: LZ encoding
        print("Step 6: Applying LZ compression...")
        compressed = self.lz_encode(masked_sentences)
        
        # Store metadata, maybe useful for analysis
        metadata = {
            'masking_ratio': masking_ratio,
            'num_sentences': len(sentences),
            'num_tokens': sum(len(s) for s in sentences),
            'num_unique_tokens': len(token_weights),
            'num_important_tokens': len(important_tokens),
            'num_masked_tokens': sum(len(s) for s in sentences) - len(important_tokens),
            'compressed_size': len(compressed),
            'token_weights': token_weights,
            'important_tokens': important_tokens
        }
        
        print(f"\nEncoding complete!")
        print(f"  Total tokens: {metadata['num_tokens']}")
        print(f"  Unique tokens: {metadata['num_unique_tokens']}")
        print(f"  Important tokens saved: {metadata['num_important_tokens']}")
        print(f"  Tokens masked: {metadata['num_masked_tokens']}")
        print(f"  Compressed size: {metadata['compressed_size']} bytes")
        
        return compressed, metadata


# Example usage
if __name__ == "__main__":
    # Sample text, part of the following: https://www.bbc.com/future/article/20220202-floating-homes-the-benefits-of-living-on-water
    text = """When a heavy storm hit in October 2022, residents of the floating community of Schoonschip in Amsterdam had little doubt they could ride it out. They tied up their bikes and outdoor benches, checked in with neighbours to ensure everyone had enough food and water, and hunkered down as their neighborhood slid up and down its steel foundational pillars, rising along with the water and descending to its original position after the rain subsided.
    "We feel safer in a storm because we are floating," says Siti Boelen, a Dutch television producer who moved into Schoonschip two years ago. "I think it's kind of strange that building on water is not a priority worldwide."
    As sea levels rise and supercharged storms cause waters to swell, floating neighbourhoods offer an experiment in flood defence that could allow coastal communities to better withstand climate change. In the land-scarce but densely populated Netherlands, demand for such homes is growing. And, as more people look to build on the water there, officials are working to update zoning laws to make the construction of floating homes easier.
    "The municipality wants to expand the concept of floating because it is multifunctional use of space for housing, and because the sustainable way is the way forward," says Nienke van Renssen, an Amsterdam city councillor from the GreenLeft party.
    """
    
    # ---------------------------------------------------------------
    # 1. Original encoder summary (your existing LZMA/semantic output)
    # ---------------------------------------------------------------
    encoder = InputEncoder()
    compressed, metadata = encoder.encode(text, masking_ratio=0.3, output_file='important_tokens.txt')
    
    print("\n" + "="*70)
    print("COMPRESSION STATISTICS")
    print("="*70)
    original_size = len(text.encode('utf-8'))
    print(f"Original text size:        {original_size} bytes")
    print(f"Total tokens:              {metadata['num_tokens']}")
    print(f"Unique tokens:             {metadata['num_unique_tokens']}")
    print(f"Important tokens (for DNA):{metadata['num_important_tokens']}")
    print(f"Masked tokens:             {metadata['num_masked_tokens']}")
    print(f"Compressed size:           {metadata['compressed_size']} bytes")
    print(f"Compression ratio:         {original_size / metadata['compressed_size']:.2f}x")

    print("\n" + "="*70)
    print("SAMPLE OF IMPORTANT TOKENS (First 10)")
    print("="*70)
    for i, token_info in enumerate(metadata['important_tokens'][:10], 1):
        print(f"{i:2d}. Position {token_info['global_position']:3d}: '{token_info['token']}' "
              f"(weight: {token_info['importance_weight']:.6f}, "
              f"sentence: {token_info['sentence_index']}, "
              f"char_pos: {token_info['char_start']}-{token_info['char_end']})")
    
    print("\n✓ Files created for DNA encoding:")
    print("  - important_tokens.txt")
    print("  - important_tokens_sequence.txt")

    # ---------------------------------------------------------------
    # 2. GAP semantic skeleton (clean output)
    # ---------------------------------------------------------------
    print("\n" + "="*70)
    print("GAP SEMANTIC SKELETON (masking_ratio=0.4)")
    print("="*70)

    gap_skeleton = encoder.encode_to_gap_skeleton(text, masking_ratio=0.4)

    print(f"total_tokens:         {gap_skeleton['total_tokens']}")
    print(f"num_important_tokens: {gap_skeleton['num_important_tokens']}")
    if gap_skeleton["num_important_tokens"] > 0:
        print(
            f"token-level compression: "
            f"{gap_skeleton['total_tokens'] / gap_skeleton['num_important_tokens']:.2f}x"
        )

    # Show the kept semantic skeleton (first 50)
    print("\nIMPORTANT TOKENS IN ORDER (first 50)")
    print("-" * 70)
    for i, (tok, pos) in enumerate(
        zip(gap_skeleton["tokens"], gap_skeleton["debug_positions"])
    ):
        if i >= 50:
            break
        print(f"{i+1:3d}. pos={pos:3d}  token='{tok}'")

    print("\n✓ GAP semantic skeleton ready for dictionary + relational encoding")

