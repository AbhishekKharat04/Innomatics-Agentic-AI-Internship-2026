"""
prompts/extraction_prompt.py
────────────────────────────
PromptTemplates for the Skill Extraction step.

Two versions are provided:
  - get_extraction_prompt()        → correct, production-quality prompt
  - get_flawed_extraction_prompt() → intentionally broken prompt that causes
                                     hallucination; used as a LangSmith debug demo
"""

from langchain_core.prompts import PromptTemplate

# ── Correct Extraction Prompt ─────────────────────────────────────────────────
EXTRACTION_TEMPLATE = """You are a precise resume parser. Your job is to extract structured
information from the resume provided below.

STRICT RULES — YOU MUST FOLLOW THESE:
1. Extract ONLY skills, tools, and technologies that are EXPLICITLY mentioned in the resume.
2. Do NOT infer, assume, or add any skills that are not directly stated.
3. If a field has no data, return an empty list [] or 0.
4. For experience_years, count total professional work experience in years (integer).
5. Return ONLY valid JSON — no markdown fences, no explanation text, no extra keys.

RESUME TEXT:
{resume_text}

Return EXACTLY this JSON structure (no other text):
{{
  "candidate_name": "Full name from resume",
  "experience_years": 0,
  "skills": ["skill1", "skill2"],
  "tools": ["tool1", "tool2"],
  "education": "Highest degree and field",
  "previous_roles": ["role1 at company1", "role2 at company2"]
}}"""

# ── Flawed Extraction Prompt (intentional bug for LangSmith debug demo) ───────
# BUG: The prompt says "You can infer and assume skills based on context."
# This causes the LLM to hallucinate skills not present in the resume.
# Use this to demonstrate LangSmith tracing catching bad outputs.
FLAWED_EXTRACTION_TEMPLATE = """You are a resume parser. Extract information from the resume.

You can infer and assume skills based on context. If someone has 1 year of Python experience,
they probably know related tools too, so include those as well.

RESUME TEXT:
{resume_text}

Return JSON with keys: candidate_name, experience_years, skills, tools, education, previous_roles.
Be generous — include any plausible skills."""


def get_extraction_prompt() -> PromptTemplate:
    """Return the correct, hallucination-free extraction prompt."""
    return PromptTemplate(
        input_variables=["resume_text"],
        template=EXTRACTION_TEMPLATE,
    )


def get_flawed_extraction_prompt() -> PromptTemplate:
    """Return the intentionally flawed prompt (for LangSmith debug demo)."""
    return PromptTemplate(
        input_variables=["resume_text"],
        template=FLAWED_EXTRACTION_TEMPLATE,
    )
