"""
chains/scoring_chain.py
───────────────────────
LCEL chain for Step 3: Candidate Scoring.

Chain: PromptTemplate | LLM | JSON Parser
Assigns a weighted 0–100 score:
  Skills (50%) + Experience (30%) + Tools (20%)
"""

import json
import re

from langchain_core.runnables import RunnableLambda

from chains.llm_factory import get_llm
from prompts.scoring_prompt import get_scoring_prompt


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


def get_scoring_chain():
    """
    Build and return the scoring LCEL chain.

    Inputs:  matching_result (JSON string), candidate_experience_years (string)
    Output:  dict with skills_score, experience_score, tools_score, total_score, grade
    """
    prompt = get_scoring_prompt()
    llm = get_llm(temperature=0.0)  # Must be deterministic for scoring

    chain = prompt | llm | RunnableLambda(lambda msg: _parse_json(msg.content))
    return chain
