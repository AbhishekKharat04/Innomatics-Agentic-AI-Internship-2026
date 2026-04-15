"""
prompts/explanation_prompt.py
──────────────────────────────
PromptTemplate for the Explanation / Hiring Recommendation step.
Produces a human-readable, evidence-backed hiring decision.
"""

from langchain_core.prompts import PromptTemplate

EXPLANATION_TEMPLATE = """You are a senior technical recruiter writing a formal hiring recommendation.

You have access to the full pipeline output for a candidate. Write a professional, specific,
evidence-backed hiring recommendation. Every claim you make MUST reference actual data from
the pipeline results below.

RULES:
1. Be specific — cite actual skill names, years of experience, tool names from the data.
2. Never make up or infer information not present in the pipeline results.
3. Structure your output exactly as shown below.
4. Keep the recommendation concise but substantive (150–250 words total).

CANDIDATE NAME: {candidate_name}
TOTAL SCORE: {total_score}/100
GRADE: {grade}

EXTRACTED PROFILE:
{extracted_profile}

MATCHING RESULT:
{matching_result}

SCORE RESULT:
{score_result}

Write your recommendation in this EXACT format:

HIRING RECOMMENDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Candidate: {candidate_name}
Score: {total_score}/100 | Grade: {grade}
Decision: [STRONG HIRE / HIRE / CONSIDER / REJECT]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRENGTHS:
[2-3 specific strengths backed by extracted data]

GAPS:
[2-3 specific missing skills or experience gaps]

RECOMMENDATION SUMMARY:
[2-3 sentences giving the final hiring recommendation with justification]"""


def get_explanation_prompt() -> PromptTemplate:
    """Return the explanation prompt template."""
    return PromptTemplate(
        input_variables=[
            "candidate_name",
            "total_score",
            "grade",
            "extracted_profile",
            "matching_result",
            "score_result",
        ],
        template=EXPLANATION_TEMPLATE,
    )
