"""
chains/explanation_chain.py
────────────────────────────
LCEL chain for Step 4: Hiring Explanation.

Chain: PromptTemplate | LLM | String output
Generates a formal, evidence-backed hiring recommendation.
Returns plain text (not JSON) since this is a human-readable report.
"""

from langchain_core.runnables import RunnableLambda

from chains.llm_factory import get_llm
from prompts.explanation_prompt import get_explanation_prompt


def get_explanation_chain():
    """
    Build and return the explanation LCEL chain.

    Inputs:  candidate_name, total_score, grade,
             extracted_profile, matching_result, score_result
    Output:  str — formatted hiring recommendation
    """
    prompt = get_explanation_prompt()
    # Slightly higher temperature for more natural language in the explanation
    llm = get_llm(temperature=0.2)

    chain = prompt | llm | RunnableLambda(lambda msg: msg.content)
    return chain
