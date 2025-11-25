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

    tokens_with_pos_str = "\n".join(
        f"- token: `{t['token']}` at approximate position {t['position']}"
        for t in token_info
    )

    tokens_bow = ", ".join(tokens)

    prompt = f"""
        You are reconstructing an English paragraph from a compressed semantic skeleton
        of a short popular-science text about DNA data storage.

        We have:
        - A target token length of about {total_tokens} tokens.
        - A list of important content words that should appear in the reconstructed text.
        - Approximate positions in the token sequence for each important token (0-based index).

        Important tokens and approximate positions:
        {tokens_with_pos_str}

        The important content words (in order) are:
        {tokens_bow}

        Constraints:
        1. Include every important token listed above, in roughly this order.
        2. You may slightly paraphrase around these tokens, but keep the overall meaning as close
        as possible to the kind of text suggested by the tokens.
        3. Write **complete, grammatical sentences**. Avoid telegraphic phrases like
        "massive amounts, durable medium"; instead write fully natural phrases such as
        "massive amounts of information in a compact, durable medium".
        4. Do NOT introduce unrelated domains (e.g. AI models, robots, medicine) that are not
        suggested by the tokens.
        5. You may add normal English glue words (articles, prepositions, auxiliaries, pronouns,
        conjunctions, basic adjectives/adverbs) to make the text fluent and natural.
        6. Keep the overall length close to {total_tokens} tokens (you can be off by a few).
        7. Aim for a smooth, readable paragraph in a neutral popular-science style.

        Task:
        Using the important tokens plus reasonable glue words and light, on-topic context,
        reconstruct a fluent English paragraph that matches the likely original meaning
        as closely as possible.

        Output:
        Only output the reconstructed paragraph, no explanations, no bullet points, no quotes.
        """
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
