"""
chains/extraction_chain.py
──────────────────────────
LCEL chain for Step 1: Skill & Profile Extraction.

Chain: PromptTemplate | LLM | JSON Parser
Returns a dict with candidate skills, tools, experience.

Two chains available:
  get_extraction_chain()        → production (no hallucination)
  get_flawed_extraction_chain() → intentionally broken (LangSmith debug demo)
"""

import json
import re

from langchain_core.runnables import RunnableLambda

from chains.llm_factory import get_llm
from prompts.extraction_prompt import get_extraction_prompt, get_flawed_extraction_prompt


def _parse_json(text: str) -> dict:
    """
    Robustly parse JSON from LLM output.
    Strips markdown fences if present, then parses.
    """
    # Strip ```json ... ``` fences if the model added them anyway
    clean = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        # Fallback: find the first { ... } block
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            return json.loads(match.group())
        # Last resort — return raw text in a dict so pipeline doesn't crash
        return {"raw_output": text, "parse_error": True}


def get_extraction_chain():
    """
    Build and return the correct extraction LCEL chain.

    Pipeline: prompt → LLM → string output → JSON parser
    """
    prompt = get_extraction_prompt()
    llm = get_llm(temperature=0.0)  # Deterministic for extraction

    # LCEL pipe: prompt renders → LLM generates → lambda parses JSON
    chain = prompt | llm | RunnableLambda(lambda msg: _parse_json(msg.content))
    return chain


def get_flawed_extraction_chain():
    """
    Build and return the intentionally flawed extraction chain.
    Used to demonstrate hallucination in LangSmith traces.
    """
    prompt = get_flawed_extraction_prompt()
    llm = get_llm(temperature=0.3)  # Slightly higher temp amplifies hallucination

    chain = prompt | llm | RunnableLambda(lambda msg: _parse_json(msg.content))
    return chain
