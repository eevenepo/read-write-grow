import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict
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
                if not token.is_punct:
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
    
    def compute_token_importance(self, sentences: List[List[Tuple[str, int, int]]]) -> Dict[str, float]:
        """
        Compute semantic importance (μ_k) for each unique token in the text.
        Args:
            sentences: List of tokenized sentences
        Returns:
            Dictionary mapping each unique token to its importance weight
        """
        M = len(sentences)
        
        # Step 1: Compute normalized semantic loss (σ̄_in) for each token in each sentence
        token_losses = []  # Will store (sentence_idx, token_idx, token_text, normalized_loss)
        
        for i, sentence in enumerate(sentences): # i is sentence index
            N_i = len(sentence)
            if N_i == 0:
                continue
            
            # Get original sentence embedding (μ_i)
            mu_i = self.get_sentence_embedding(sentence)
            
            # Compute semantic loss for each token
            sigma_values = []
            
            for n, (token_text, _, _) in enumerate(sentence):
                # Create masked sentence (replace token with #)
                masked_sentence = sentence[:n] + [('#', 0, 0)] + sentence[n+1:] # For each word, creates a version of the sentence with that word masked
                
                # Get masked sentence embedding (μ̂_i)
                mu_hat_i = self.get_sentence_embedding(masked_sentence)
                
                # Compute semantic loss σ_in (equation 12)
                sigma_in = self.cosine_distance(mu_i, mu_hat_i) # Cosine distance between original and masked embeddings
                sigma_values.append(sigma_in)
            
            # Normalize semantic losses (equation 13)
            total_sigma = sum(sigma_values)
            if total_sigma > 0:
                normalized_losses = [sigma / total_sigma for sigma in sigma_values] # Divede each sigma by total sigma to get normalized losses
            else:
                normalized_losses = [1.0 / N_i] * N_i 
            
            # Store results
            for n, (token_text, _, _) in enumerate(sentence):
                token_losses.append((i, n, token_text, normalized_losses[n]))
        
        # Step 2: Build vocabulary K and compute positions Q_ik
        vocab = {}  # Maps token -> list of (sentence_idx, positions)
        
        for i, sentence in enumerate(sentences):
            for n, (token_text, _, _) in enumerate(sentence): # n is token index in sentence i
                if token_text not in vocab:
                    vocab[token_text] = {}
                if i not in vocab[token_text]:
                    vocab[token_text][i] = [] 
                vocab[token_text][i].append(n) # `vocab[token][i]` corresponds to Q_ik (positions of word k in sentence i)
        
        # Step 3: Compute weight μ_k for each token (equation 15)
        token_weights = {}
        
        for token in vocab:
            # Total occurrences across all sentences (Σ_i |Q_ik|)
            total_occurrences = sum(len(positions) for positions in vocab[token].values()) #= Σ_i |Q_ik|
            
            if total_occurrences == 0:
                token_weights[token] = 0.0
                continue
            
            # Compute weighted sum (Σ_i α_i Σ_n∈Q_ik σ̄_in)
            weighted_sum = 0.0
            for i in vocab[token]: # for each sentence i containing token k
                # α_i = N_i (sentence length)
                alpha_i = len(sentences[i]) # Crossword (2023): "We set αi = Ni out of the belief that the semantic information is proportional to the sentence length."
                
                # Sum of normalized losses for this token in sentence i (Σ_n∈Q_ik σ̄_in)
                loss_sum = 0.0
                for n in vocab[token][i]: # for each position n of token k in sentence i
                    # Find the normalized loss for this token
                    for s_idx, t_idx, t_text, norm_loss in token_losses: # s_idx: sentence index, t_idx: token index, t_text: token text
                        if s_idx == i and t_idx == n and t_text == token: # if matches
                            loss_sum += norm_loss
                            break
                
                weighted_sum += alpha_i * loss_sum
            
            # Final weight μ_k = (1 / Σ_i |Q_ik|) * weighted_sum
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
    
    # Create encoder
    encoder = InputEncoder()
    
    # Encode with 30% masking ratio and save important tokens
    compressed, metadata = encoder.encode(text, masking_ratio=0.3, output_file='important_tokens.txt')
    
    # Show statistics
    print("\n" + "="*70)
    print("COMPRESSION STATISTICS")
    print("="*70)
    original_size = len(text.encode('utf-8'))
    print(f"Original text size: {original_size} bytes")
    print(f"Total tokens: {metadata['num_tokens']}")
    print(f"Unique tokens: {metadata['num_unique_tokens']}")
    print(f"Important tokens (for DNA): {metadata['num_important_tokens']}")
    print(f"Masked tokens: {metadata['num_masked_tokens']}")
    print(f"Compressed size: {metadata['compressed_size']} bytes")
    print(f"Compression ratio: {original_size / metadata['compressed_size']:.2f}x")
    
    # Show sample of important tokens
    print("\n" + "="*70)
    print("SAMPLE OF IMPORTANT TOKENS (First 10)")
    print("="*70)
    for i, token_info in enumerate(metadata['important_tokens'][:10], 1):
        print(f"{i:2d}. Position {token_info['global_position']:3d}: '{token_info['token']}' "
              f"(weight: {token_info['importance_weight']:.6f}, "
              f"sentence: {token_info['sentence_index']}, "
              f"char_pos: {token_info['char_start']}-{token_info['char_end']})")
    
    print(f"\n✓ Files created for DNA encoding:")
    print(f"  - important_tokens.txt (readable format)")
    print(f"  - important_tokens_sequence.txt (simple token sequence)")