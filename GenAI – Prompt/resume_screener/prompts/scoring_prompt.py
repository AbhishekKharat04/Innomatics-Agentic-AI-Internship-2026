"""
prompts/scoring_prompt.py
─────────────────────────
PromptTemplate for the Scoring step.
Assigns a weighted 0-100 score based on matching results.

Scoring weights (must sum to 100):
  Skills   → 50%
  Experience → 30%
  Tools    → 20%
"""

from langchain_core.prompts import PromptTemplate

SCORING_TEMPLATE = """You are a technical hiring evaluator. Your job is to assign a numerical
fitness score (0–100) to a candidate based on their match with the job description.

SCORING WEIGHTS (strictly enforced):
- Skills match:      50 points max  (matched_skills / total_required_skills × 50)
- Experience match:  30 points max  (4+ years = 30pts, 2-3 years = 20pts, <2 years = 10pts, 0 = 0pts)
- Tools match:       20 points max  (matched_tools / total_required_tools × 20)

MATCHING RESULT (from previous step):
{matching_result}

CANDIDATE EXPERIENCE YEARS: {candidate_experience_years}

RULES:
1. Calculate each component score individually before summing.
2. Never give more than the max for each category.
3. Return ONLY valid JSON — no extra text.

Return EXACTLY this JSON structure:
{{
  "skills_score": 0,
  "experience_score": 0,
  "tools_score": 0,
  "total_score": 0,
  "grade": "A/B/C/D/F",
  "score_breakdown": "Brief one-line explanation of each component score"
}}

Grade thresholds: A=85-100, B=70-84, C=50-69, D=30-49, F=0-29"""


def get_scoring_prompt() -> PromptTemplate:
    """Return the scoring prompt template."""
    return PromptTemplate(
        input_variables=["matching_result", "candidate_experience_years"],
        template=SCORING_TEMPLATE,
    )
