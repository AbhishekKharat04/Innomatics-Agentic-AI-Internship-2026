"""
chains/matching_chain.py
────────────────────────
LCEL chain for Step 2: Resume-to-JD Matching.

Chain: PromptTemplate | LLM | JSON Parser
Compares extracted candidate profile against job requirements.
Returns matched/missing skills and tools.
"""

import json
import re

from langchain_core.runnables import RunnableLambda

from chains.llm_factory import get_llm
from prompts.matching_prompt import get_matching_prompt


def _parse_json(text: str) -> dict:
    """Parse JSON from LLM output, stripping markdown fences if present."""
    clean = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"raw_output": text, "parse_error": True}


def get_matching_chain():
    """
    Build and return the matching LCEL chain.

    Inputs:  extracted_profile (JSON string), job_description (string)
    Output:  dict with matched_skills, missing_skills, matched_tools, etc.
    """
    prompt = get_matching_prompt()
    llm = get_llm(temperature=0.0)

    chain = prompt | llm | RunnableLambda(lambda msg: _parse_json(msg.content))
    return chain
