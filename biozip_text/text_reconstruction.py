def reconstruct_positions_from_gaps(first_position: int, gaps: list[int]) -> list[int]:
    """
    Inverse of your gap encoding:
      positions[0] = first_position
      positions[i] = positions[i-1] + gaps[i-1]

    """
    if not gaps and first_position == 0:
        return []
    positions = [first_position]
    for g in gaps:
        positions.append(positions[-1] + g)
    return positions

import google.generativeai as genai

def init_gemini(api_key: str, model_name: str = "gemini-2.5-flash"):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

from typing import Dict, Any, List

def build_reconstruction_prompt(gap_skeleton: Dict[str, Any]) -> str:
    total_tokens = gap_skeleton["total_tokens"]
    tokens = gap_skeleton["tokens"]
    first_pos = gap_skeleton["first_position"]
    gaps = gap_skeleton["gaps"]

    positions = reconstruct_positions_from_gaps(first_pos, gaps)

    token_info = [
        {"token": tok, "position": pos}
        for tok, pos in zip(tokens, positions)
    ]

    # Group tokens by approximate sentence (gaps > 5 often indicate sentence boundaries)
    tokens_with_pos_str = "\n".join(
        f"  Position {t['position']:3d}: \"{t['token']}\""
        for t in token_info
    )

    tokens_bow = ", ".join(f'"{tok}"' for tok in tokens)
    
    # Calculate density info to help the model understand structure
    num_tokens = len(tokens)
    avg_gap = sum(gaps) / len(gaps) if gaps else 0

    prompt = f"""You are a precise text reconstruction AI. Your task is to rebuild the ORIGINAL text from a semantic skeleton.

## CONTEXT
The original text was compressed by removing common words (articles, prepositions, some verbs) while keeping semantically important words. You must reconstruct a text that:
1. Contains ALL the important tokens below in the EXACT order given
2. Sounds natural and grammatically correct
3. Preserves the FULL meaning and nuance suggested by the token sequence
4. Matches approximately {total_tokens} total words

## IMPORTANT TOKENS (in order, with their approximate positions)
{tokens_with_pos_str}

## TOKEN LIST FOR REFERENCE
{tokens_bow}

## CRITICAL RULES
1. **PRESERVE ALL TOKENS**: Every token above MUST appear in your output, in the same order.
2. **PRESERVE MEANING**: The tokens tell a story - reconstruct that EXACT story, not a summary or paraphrase.
3. **NATURAL FLOW**: Add articles (the, a, an), prepositions (of, in, to, for), conjunctions (and, but, or), pronouns (it, they, we), and auxiliary verbs (is, are, was, have) to create natural sentences.
4. **MATCH LENGTH**: Target approximately {total_tokens} words. The gaps between token positions indicate how many filler words were removed.
5. **SENTENCE STRUCTURE**: Large gaps between positions often indicate sentence boundaries. Respect these natural breaks.
6. **NO INVENTION**: Do not add new concepts, facts, or ideas not implied by the tokens. Only add grammatical glue words.
7. **TONE PRESERVATION**: If tokens suggest a specific tone (e.g., "good, bad, ugly" suggests a balanced discussion), preserve that tone.

## EXAMPLE
If tokens are: "Technology", "critical", "world", "today", "communication", "healthcare"
BAD output: "Technology is important." (too short, loses meaning)
BAD output: "Technology affects many areas including AI and robotics." (invents new concepts)
GOOD output: "Technology is a critical component of our world today, from communication to healthcare."

## YOUR TASK
Reconstruct the original text using ALL {num_tokens} tokens above. Add only the minimal grammatical words needed to make it flow naturally. The result should read like polished, professional writing.

Output ONLY the reconstructed text, nothing else:"""

    return prompt.strip()




def reconstruct_text_with_gemini(
    gap_skeleton: Dict[str, Any],
    api_key: str,
    model_name: str = "gemini-2.5-flash",
) -> str:
    model = init_gemini(api_key, model_name=model_name)
    prompt = build_reconstruction_prompt(gap_skeleton)

    response = model.generate_content(prompt)
    # Depending on the client version, you might need `response.text` or `response.candidates[0].content.parts[...]`
    return response.text.strip()
