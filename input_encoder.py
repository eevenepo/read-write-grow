import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict, Any, Set, Tuple as TypingTuple
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

    # ------------------------
    # Short, high-impact list
    # ------------------------
    # Words that fundamentally change meaning - NEVER filter these out
    SEMANTIC_CRITICAL_WORDS: Set[str] = {
        # Negation
        "not", "no", "never", "n't",

        # Logical structure
        "because", "if", "but",

        # Modal verbs (minimal)
        "must", "should", "may", "might", "could", "would",

        # Quantifiers (minimal)
        "all", "any", "most", "few",
    }

    # POS tags to keep even if they're stop words
    POS_TO_KEEP: Set[str] = {
        "NOUN", "PROPN", "ADJ", "VERB", "NUM", "AUX", "PRON"
    }

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
    
    # -------------------------
    # tokenize_text with POS + critical words
    # -------------------------
    def tokenize_text(self, text: str) -> List[List[Tuple[str, int, int]]]:
        """
        Tokenize text into sentences and tokens using spaCy, but:
          - drop uninformative stopwords like "a", "the" etc.
          - ALWAYS keep semantic-critical words
          - keep content POS even if they are stopwords (e.g. pronouns, numerals)
        
        Returns:
            List of sentences, where each sentence is a list of
            (token_text, start_idx, end_idx) tuples
        """
        doc = self.nlp(text)

        sentences: List[List[Tuple[str, int, int]]] = []
        for sent in doc.sents:
            tokens: List[Tuple[str, int, int]] = []
            for token in sent:
                # Skip punctuation and whitespace
                if token.is_punct or token.is_space:
                    continue

                token_lower = token.text.lower()

                # Keep token if ANY of these conditions are true:
                # 1. It's semantically critical (negations, modals, etc.)
                # 2. It's not a stop word
                # 3. It's a stop word but has an important POS tag
                if (
                    token_lower in self.SEMANTIC_CRITICAL_WORDS
                    or not token.is_stop
                    or token.pos_ in self.POS_TO_KEEP
                ):
                    tokens.append(
                        (token.text, token.idx, token.idx + len(token.text))
                    )

            if tokens:  # Only add non-empty sentences
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
        sentences: List[List[Tuple[str, int, int]]]
    ) -> Dict[str, float]:
        """
        Compute semantic importance (μ_k) for each unique token in the text.
        """
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

            # Normalize semantic losses within the sentence
            total_sigma = sum(sigma_values)
            if total_sigma > 0:
                normalized_losses = [sigma / total_sigma for sigma in sigma_values]
            else:
                # If all sigma are zero, distribute uniformly
                normalized_losses = [1.0 / N_i] * N_i

            # Store normalized loss by (sentence_idx, token_idx)
            for n in range(N_i):
                token_losses[(i, n)] = normalized_losses[n]

        # Build vocabulary: token -> sentence -> positions
        vocab: Dict[str, Dict[int, List[int]]] = {}

        for i, sentence in enumerate(sentences):
            for n, (token_text, _, _) in enumerate(sentence):
                if token_text not in vocab:
                    vocab[token_text] = {}
                if i not in vocab[token_text]:
                    vocab[token_text][i] = []
                vocab[token_text][i].append(n)

        # Precompute sentence lengths
        sentence_lengths = [len(s) for s in sentences]

        token_weights: Dict[str, float] = {}

        for token, sent_positions in vocab.items():
            total_occurrences = sum(len(positions) for positions in sent_positions.values())
            if total_occurrences == 0:
                token_weights[token] = 0.0
                continue

            weighted_sum = 0.0

            for i, positions in sent_positions.items():
                alpha_i = sentence_lengths[i]
                loss_sum = 0.0

                for n in positions:
                    loss_sum += token_losses.get((i, n), 0.0)

                weighted_sum += alpha_i * loss_sum

            token_weights[token] = weighted_sum / total_occurrences

        return token_weights
    
    def get_important_tokens_with_positions(
        self,
        sentences: List[List[Tuple[str, int, int]]], 
        token_weights: Dict[str, float], 
        masking_ratio: float
    ) -> List[Dict]:
        """
        Get important tokens (those NOT masked) with their positions.
        *Semantic-critical words are never masked, regardless of weight.*
        """
        # Sort tokens by weight (ascending order - least important first)
        sorted_tokens = sorted(token_weights.items(), key=lambda x: x[1])
        
        # Determine how many unique tokens to mask (least important)
        num_to_mask = int(masking_ratio * len(sorted_tokens))
        
        # Build mask set but NEVER include semantic-critical words
        tokens_to_mask: Set[str] = set()
        for token, _ in sorted_tokens:
            if len(tokens_to_mask) >= num_to_mask:
                break
            if token.lower() in self.SEMANTIC_CRITICAL_WORDS:
                continue
            tokens_to_mask.add(token)
        
        # Collect all important (non-masked) tokens with their positions
        important_tokens = []
        token_position = 0  # Global token position counter
        
        for sent_idx, sentence in enumerate(sentences):
            for token_idx, (token_text, char_start, char_end) in enumerate(sentence):
                token_lower = token_text.lower()
                # Never mask semantic-critical words
                if token_lower in self.SEMANTIC_CRITICAL_WORDS or token_text not in tokens_to_mask:
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
    

    def positions_to_gaps(self, positions: List[int]) -> Dict:
        """
        Convert sorted token positions into gap encoding.
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

        # 4) Convert positions → gaps
        gap_info = self.positions_to_gaps(positions)
        first_position = gap_info["first_position"]
        gaps = gap_info["gaps"]

        # 5) Total tokens = tokens after semantic filtering
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
        """
        id_to_word = {idx: word for word, idx in token_to_id.items()}

        max_id = max(id_to_word.keys(), default=-1)
        for token_id in range(max_id + 1):
            if token_id not in id_to_word:
                raise ValueError(
                    f"Missing token_id {token_id} — IDs must be consecutive from 0..N-1"
                )

        byte_chunks = []

        for token_id in range(max_id + 1):
            word = id_to_word[token_id]

            try:
                word_bytes = word.encode("ascii")
            except UnicodeEncodeError:
                raise ValueError(f"Word '{word}' contains non-ASCII characters.")

            length = len(word_bytes)
            if length > 255:
                raise ValueError(f"Word '{word}' too long (max 255 for 1 byte length).")

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
        Serialize the relational (gap skeleton) to bytes.

        Header:
            total_tokens: 4 bytes
            first_position: 2 bytes

        Then for each important token:
            [gap:1 byte][token_id:4 bytes]
        """
        total_tokens = gap_skeleton["total_tokens"]
        first_position = gap_skeleton["first_position"]
        gaps = gap_skeleton["gaps"]
        tokens = gap_skeleton["tokens"]

        num_important = len(tokens)

        if num_important == 0:
            return total_tokens.to_bytes(4, "big") + first_position.to_bytes(2, "big")

        if len(gaps) != max(0, num_important - 1):
            raise ValueError(
                f"Gap count ({len(gaps)}) must be num_important_tokens - 1 ({num_important - 1})."
            )

        if not (0 <= total_tokens <= 0xFFFFFFFF):
            raise ValueError("total_tokens must fit in 4 bytes (0..2^32-1).")

        if not (0 <= first_position <= 0xFFFF):
            raise ValueError("first_position must fit in 2 bytes (0..65535).")

        byte_chunks = []

        # Header
        byte_chunks.append(total_tokens.to_bytes(4, "big"))
        byte_chunks.append(first_position.to_bytes(2, "big"))

        # Per-token entries
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

            byte_chunks.append(gap_val.to_bytes(1, "big"))
            byte_chunks.append(token_id.to_bytes(4, "big"))

        return b"".join(byte_chunks)

    
    def mask_tokens(
        self,
        sentences: List[List[Tuple[str, int, int]]], 
        token_weights: Dict[str, float], 
        masking_ratio: float
    ) -> List[List[str]]:
        """
        Mask the least important tokens according to masking ratio.
        Semantic-critical words are NEVER masked.
        """
        # Sort tokens by weight (ascending order - least important first)
        sorted_tokens = sorted(token_weights.items(), key=lambda x: x[1])
        
        num_to_mask = int(masking_ratio * len(sorted_tokens))
        tokens_to_mask: Set[str] = set()
        for token, _ in sorted_tokens:
            if len(tokens_to_mask) >= num_to_mask:
                break
            if token.lower() in self.SEMANTIC_CRITICAL_WORDS:
                continue
            tokens_to_mask.add(token)
        
        masked_sentences: List[List[str]] = []
        for sentence in sentences:
            masked_sentence: List[str] = []
            for token_text, _, _ in sentence:
                token_lower = token_text.lower()
                if token_lower in self.SEMANTIC_CRITICAL_WORDS:
                    # never mask critical words
                    masked_sentence.append(token_text)
                elif token_text in tokens_to_mask:
                    masked_sentence.append('#')
                else:
                    masked_sentence.append(token_text)
            masked_sentences.append(masked_sentence)
        
        return masked_sentences
    
    def lz_encode(self, masked_sentences: List[List[str]]) -> bytes:
        """
        Convert masked text to compressed bit string using LZ compression.
        """
        text = ''
        for sentence in masked_sentences:
            text += ' '.join(sentence) + ' '
        
        compressed = lzma.compress(text.encode('utf-8'))
        
        return compressed
    
    def save_important_tokens(
        self,
        important_tokens: List[Dict],
        output_file: str,
        total_tokens: int
    ):
        """
        Save important tokens to a file for DNA encoding.
        """
        json_file = output_file.replace('.txt', '.json')
        
        json_output = {
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
        
        seq_file = output_file.replace('.txt', '_sequence.txt')
        with open(seq_file, 'w', encoding='utf-8') as f:
            tokens_only = [item['token'] for item in important_tokens]
            f.write(' '.join(tokens_only))
        print(f"  ✓ Saved token sequence to: {seq_file}")
    
    def encode(
        self,
        text: str,
        masking_ratio: float = 0.3,
        output_file: str = 'important_tokens.txt'
    ) -> TypingTuple[bytes, Dict]:
        """
        Full encoding pipeline: tokenize -> compute importance -> mask -> compress.
        Also saves important tokens to file for DNA encoding.
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
        important_tokens = self.get_important_tokens_with_positions(
            sentences,
            token_weights,
            masking_ratio
        )
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
