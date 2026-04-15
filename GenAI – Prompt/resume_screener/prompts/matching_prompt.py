"""
prompts/matching_prompt.py
──────────────────────────
PromptTemplate for the Skill Matching step.
Compares extracted resume profile against the job description.
"""

from langchain_core.prompts import PromptTemplate

MATCHING_TEMPLATE = """You are an expert technical recruiter performing a skills gap analysis.

You are given:
1. An extracted candidate profile (JSON)
2. A job description

Your task is to compare the candidate's skills and tools against what the job requires.

STRICT RULES:
1. Base your analysis ONLY on what is in the extracted profile — do not assume.
2. A skill "matches" only if it appears explicitly in the candidate's profile.
3. List missing skills that are in the JD but NOT in the candidate profile.
4. Return ONLY valid JSON — no markdown, no extra text.

EXTRACTED CANDIDATE PROFILE:
{extracted_profile}

JOB DESCRIPTION:
{job_description}

Return EXACTLY this JSON structure:
{{
  "matched_skills": ["skill1", "skill2"],
  "missing_skills": ["skill3", "skill4"],
  "matched_tools": ["tool1"],
  "missing_tools": ["tool2"],
  "experience_match": true,
  "experience_note": "Brief note on experience alignment",
  "overall_match_summary": "2-3 sentence summary of how well the candidate fits"
}}"""


def get_matching_prompt() -> PromptTemplate:
    """Return the matching prompt template."""
    return PromptTemplate(
        input_variables=["extracted_profile", "job_description"],
        template=MATCHING_TEMPLATE,
    )
