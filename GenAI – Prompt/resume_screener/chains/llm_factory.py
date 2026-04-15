"""
chains/llm_factory.py
──────────────────────
Shared LLM factory. Returns a configured ChatGroq instance.
All chains import from here — one place to swap the model.

Free Groq models available (as of 2025):
  - llama-3.3-70b-versatile  ← default (best quality, still free)
  - llama3-8b-8192           ← faster / lighter
  - mixtral-8x7b-32768       ← good for structured outputs
"""

import os
from langchain_groq import ChatGroq


def get_llm(temperature: float = 0.0) -> ChatGroq:
    """
    Return a ChatGroq LLM instance.

    Args:
        temperature: 0.0 for deterministic outputs (extraction/scoring),
                     higher for creative outputs (explanation).

    Returns:
        Configured ChatGroq instance.
    """
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        groq_api_key=os.environ["GROQ_API_KEY"],
    )
