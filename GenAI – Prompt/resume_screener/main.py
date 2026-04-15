"""
main.py — AI Resume Screening System with LangSmith Tracing
============================================================
Orchestrates the full pipeline: Extract → Match → Score → Explain
Runs all 3 mock resumes and prints results.

LangSmith Tracing:
  Set LANGCHAIN_TRACING_V2=true in your .env file.
  Traces will appear in https://smith.langchain.com/ under the project
  defined by LANGCHAIN_PROJECT.

Usage:
  python main.py

Author: Abhishek Kharat | Innomatics Internship Task 3
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langsmith import traceable

# Load environment variables (.env file)
load_dotenv()

# Import LCEL chains
from chains.extraction_chain import get_extraction_chain, get_flawed_extraction_chain
from chains.matching_chain import get_matching_chain
from chains.scoring_chain import get_scoring_chain
from chains.explanation_chain import get_explanation_chain

# ── File paths ────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"

RESUMES = {
    "Strong Candidate": DATA_DIR / "resume_strong.txt",
    "Average Candidate": DATA_DIR / "resume_average.txt",
    "Weak Candidate":   DATA_DIR / "resume_weak.txt",
}

JD_PATH = DATA_DIR / "job_description.txt"


# ── Helper ─────────────────────────────────────────────────────────────────────
def load_text(path: Path) -> str:
    """Load text file content."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def pretty_print_result(label: str, data) -> None:
    """Print a labeled section to the console."""
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    if isinstance(data, dict):
        print(json.dumps(data, indent=2))
    else:
        print(data)


# ── Core Pipeline ──────────────────────────────────────────────────────────────
@traceable(name="resume_screening_pipeline")
def run_screening_pipeline(
    resume_text: str,
    job_description: str,
    candidate_label: str,
    use_flawed_prompt: bool = False,
) -> dict:
    """
    Full screening pipeline for a single resume.
    The @traceable decorator sends the entire run to LangSmith automatically.

    Args:
        resume_text: Raw resume content
        job_description: Raw JD content
        candidate_label: Label for LangSmith ("strong", "average", "weak")
        use_flawed_prompt: If True, uses the buggy extraction prompt (debug demo)

    Returns:
        dict containing all pipeline results
    """
    print(f"\n{'='*60}")
    print(f"  Processing: {candidate_label}")
    print(f"  Mode: {'⚠️  FLAWED PROMPT' if use_flawed_prompt else '✅ Correct Prompt'}")
    print(f"{'='*60}")

    # ── Step 1: Skill Extraction ──────────────────────────────────────────────
    print("\n📄 Step 1: Extracting skills, tools, and experience...")
    if use_flawed_prompt:
        extraction_chain = get_flawed_extraction_chain()
    else:
        extraction_chain = get_extraction_chain()

    # Use LangSmith run_name and tags for trace organization
    extraction_config = {
        "run_name": f"extraction_{candidate_label.lower().replace(' ', '_')}",
        "tags": [candidate_label.lower().replace(" ", "-"), "extraction"],
        "metadata": {"step": "1_extraction", "candidate": candidate_label},
    }
    extracted = extraction_chain.invoke(
        {"resume_text": resume_text},
        config=extraction_config,
    )
    pretty_print_result("EXTRACTION RESULT", extracted)

    # ── Step 2: Matching ──────────────────────────────────────────────────────
    print("\n🔍 Step 2: Matching resume against job description...")
    matching_chain = get_matching_chain()

    matching_config = {
        "run_name": f"matching_{candidate_label.lower().replace(' ', '_')}",
        "tags": [candidate_label.lower().replace(" ", "-"), "matching"],
        "metadata": {"step": "2_matching", "candidate": candidate_label},
    }
    matched = matching_chain.invoke(
        {
            "extracted_profile": json.dumps(extracted, indent=2),
            "job_description": job_description,
        },
        config=matching_config,
    )
    pretty_print_result("MATCHING RESULT", matched)

    # ── Step 3: Scoring ───────────────────────────────────────────────────────
    print("\n🎯 Step 3: Calculating fitness score...")
    scoring_chain = get_scoring_chain()

    scoring_config = {
        "run_name": f"scoring_{candidate_label.lower().replace(' ', '_')}",
        "tags": [candidate_label.lower().replace(" ", "-"), "scoring"],
        "metadata": {"step": "3_scoring", "candidate": candidate_label},
    }
    scored = scoring_chain.invoke(
        {
            "matching_result": json.dumps(matched, indent=2),
            "candidate_experience_years": str(extracted.get("experience_years", 0)),
        },
        config=scoring_config,
    )
    pretty_print_result("SCORE RESULT", scored)

    # ── Step 4: Explanation ───────────────────────────────────────────────────
    print("\n💬 Step 4: Generating hiring recommendation...")
    explanation_chain = get_explanation_chain()

    explanation_config = {
        "run_name": f"explanation_{candidate_label.lower().replace(' ', '_')}",
        "tags": [candidate_label.lower().replace(" ", "-"), "explanation"],
        "metadata": {"step": "4_explanation", "candidate": candidate_label},
    }
    explanation = explanation_chain.invoke(
        {
            "candidate_name": extracted.get("candidate_name", "Unknown"),
            "total_score": str(scored.get("total_score", 0)),
            "grade": scored.get("grade", "N/A"),
            "extracted_profile": json.dumps(extracted, indent=2),
            "matching_result": json.dumps(matched, indent=2),
            "score_result": json.dumps(scored, indent=2),
        },
        config=explanation_config,
    )
    pretty_print_result("HIRING RECOMMENDATION", explanation)

    # ── Final Summary ─────────────────────────────────────────────────────────
    result = {
        "candidate_label": candidate_label,
        "candidate_name": extracted.get("candidate_name", "Unknown"),
        "extraction": extracted,
        "matching": matched,
        "scoring": scored,
        "explanation": explanation,
    }

    print(f"\n✅ {candidate_label} | Score: {scored.get('total_score', 0)}/100 | {scored.get('grade', 'N/A')}")
    return result


# ── Entry Point ───────────────────────────────────────────────────────────────
def main():
    """Run the screening pipeline for all 3 candidates + 1 flawed demo."""

    print("\n" + "🚀 " * 20)
    print("  AI RESUME SCREENING SYSTEM — Innomatics GenAI Task 3")
    print("  Powered by: LangChain + Groq (Llama 3.3-70B) + LangSmith")
    print("🚀 " * 20)

    # Check for API keys
    if not os.getenv("GROQ_API_KEY"):
        print("\n❌ ERROR: GROQ_API_KEY not found.")
        print("   Copy .env.example to .env and add your free API keys.")
        print("   Groq: https://console.groq.com/ | LangSmith: https://smith.langchain.com/")
        return

    # Load job description
    job_description = load_text(JD_PATH)
    print(f"\n📋 Job Description loaded: {JD_PATH.name}")
    print(f"🔗 LangSmith Project: {os.getenv('LANGCHAIN_PROJECT', 'not set')}")
    print(f"🔍 Tracing: {'ENABLED ✅' if os.getenv('LANGCHAIN_TRACING_V2') == 'true' else 'DISABLED ❌'}")

    all_results = []

    # ── Run 1: Strong Candidate (Correct Prompt) ──────────────────────────────
    resume = load_text(RESUMES["Strong Candidate"])
    result = run_screening_pipeline(resume, job_description, "Strong Candidate")
    all_results.append(result)

    # ── Run 2: Average Candidate (Correct Prompt) ─────────────────────────────
    resume = load_text(RESUMES["Average Candidate"])
    result = run_screening_pipeline(resume, job_description, "Average Candidate")
    all_results.append(result)

    # ── Run 3: Weak Candidate (Correct Prompt) ────────────────────────────────
    resume = load_text(RESUMES["Weak Candidate"])
    result = run_screening_pipeline(resume, job_description, "Weak Candidate")
    all_results.append(result)

    # ── Run 4: Weak Candidate (FLAWED PROMPT — Debug Demo) ───────────────────
    print("\n\n" + "⚠️  " * 15)
    print("  LANGSMITH DEBUG DEMONSTRATION")
    print("  Running Weak Candidate with FLAWED prompt to show hallucination.")
    print("  Compare this LangSmith trace with Run 3 to see the difference!")
    print("⚠️  " * 15)
    resume = load_text(RESUMES["Weak Candidate"])
    flawed_result = run_screening_pipeline(
        resume, job_description, "Weak Candidate (FLAWED)", use_flawed_prompt=True
    )

    # ── Final Leaderboard ─────────────────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("  📊 FINAL RESULTS LEADERBOARD")
    print("=" * 60)
    print(f"{'Candidate':<30} {'Score':>8} {'Grade'}")
    print("-" * 60)
    for r in all_results:
        score = r["scoring"].get("total_score", 0)
        grade = r["scoring"].get("grade", "N/A")
        name  = r["candidate_name"]
        print(f"{name:<30} {score:>7}/100  {grade}")
    print("=" * 60)
    print("\n✅ All runs complete. Check LangSmith for full pipeline traces.")
    print(f"   → https://smith.langchain.com/\n")


if __name__ == "__main__":
    main()
