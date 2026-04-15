"""
app.py — AI Resume Screening System | Streamlit UI
====================================================
A professional, recruiter-facing interface to screen resumes against a
job description using a 4-step LangChain pipeline with LangSmith tracing.

Features:
  - Upload a resume (TXT) or paste raw text
  - Paste or load the job description
  - View extraction, matching, scoring, and explanation results
  - Visual score breakdown with progress bars
  - LangSmith trace link display
  - Side-by-side batch mode for comparing 3 candidates

Run:
  streamlit run app.py

Author: Abhishek Kharat
"""

import json
import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
  }

  /* Dark theme override */
  .stApp {
    background-color: #0d1117;
    color: #e6edf3;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
  }

  /* Cards */
  .result-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
  }

  .score-badge {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 0.05em;
  }

  .grade-A { background: #0d4429; color: #3fb950; border: 1px solid #3fb950; }
  .grade-B { background: #0d3349; color: #58a6ff; border: 1px solid #58a6ff; }
  .grade-C { background: #3a2e0d; color: #d29922; border: 1px solid #d29922; }
  .grade-D { background: #3b1c1c; color: #f85149; border: 1px solid #f85149; }
  .grade-F { background: #2d1515; color: #ff7b72; border: 1px solid #ff7b72; }

  .skill-chip {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 6px;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    margin: 0.2rem;
  }

  .chip-match { background: #0d2818; color: #3fb950; border: 1px solid #1a7f3c; }
  .chip-miss  { background: #2d1515; color: #f85149; border: 1px solid #7d2a2a; }
  .chip-tool  { background: #0d1f3b; color: #58a6ff; border: 1px solid #1f4b8a; }

  .step-header {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 0.5rem;
  }

  .recommendation-box {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1.2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    line-height: 1.8;
    white-space: pre-wrap;
    color: #cdd9e5;
  }

  .metric-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin: 1rem 0;
  }

  .metric-box {
    flex: 1;
    min-width: 120px;
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
  }

  .metric-box .val {
    font-size: 2rem;
    font-weight: 700;
    color: #58a6ff;
  }

  .metric-box .lbl {
    font-size: 0.72rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  .hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #58a6ff, #3fb950);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
  }

  .hero-sub {
    color: #8b949e;
    font-size: 1rem;
    margin-bottom: 2rem;
  }

  .warning-box {
    background: #2d2100;
    border: 1px solid #9e6a03;
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    color: #d29922;
    font-size: 0.88rem;
    margin: 0.8rem 0;
  }

  div[data-testid="stProgress"] > div > div {
    background-color: #58a6ff;
  }

  .stTextArea textarea {
    background-color: #0d1117 !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
  }

  .stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    font-size: 0.95rem;
    transition: opacity 0.2s;
    width: 100%;
  }

  .stButton > button:hover { opacity: 0.85; }

  .stTabs [data-baseweb="tab"] {
    color: #8b949e;
    font-weight: 500;
  }
  .stTabs [aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom-color: #58a6ff !important;
  }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"

def load_sample(filename: str) -> str:
    path = DATA_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def grade_color(grade: str) -> str:
    return {
        "A": "grade-A", "B": "grade-B", "C": "grade-C",
        "D": "grade-D", "F": "grade-F",
    }.get(grade, "grade-F")


def score_color(score: int) -> str:
    if score >= 85:  return "#3fb950"
    if score >= 70:  return "#58a6ff"
    if score >= 50:  return "#d29922"
    if score >= 30:  return "#f85149"
    return "#ff7b72"


def render_skill_chips(skills: list, chip_class: str) -> str:
    return " ".join(
        f'<span class="skill-chip {chip_class}">{s}</span>'
        for s in skills
    )


def run_pipeline(resume_text: str, job_description: str, label: str, use_flawed: bool = False) -> dict:
    """Run the 4-step screening pipeline and return results."""
    import json as _json
    from chains.extraction_chain import get_extraction_chain, get_flawed_extraction_chain
    from chains.matching_chain import get_matching_chain
    from chains.scoring_chain import get_scoring_chain
    from chains.explanation_chain import get_explanation_chain

    safe_label = label.lower().replace(" ", "_")

    # Step 1 — Extract
    extraction_chain = get_flawed_extraction_chain() if use_flawed else get_extraction_chain()
    extracted = extraction_chain.invoke(
        {"resume_text": resume_text},
        config={
            "run_name": f"extraction_{safe_label}",
            "tags": [safe_label, "extraction", "streamlit"],
            "metadata": {"step": "1_extraction", "candidate": label, "source": "streamlit_ui"},
        },
    )

    # Step 2 — Match
    matching_chain = get_matching_chain()
    matched = matching_chain.invoke(
        {"extracted_profile": _json.dumps(extracted, indent=2), "job_description": job_description},
        config={
            "run_name": f"matching_{safe_label}",
            "tags": [safe_label, "matching", "streamlit"],
            "metadata": {"step": "2_matching", "candidate": label},
        },
    )

    # Step 3 — Score
    scoring_chain = get_scoring_chain()
    scored = scoring_chain.invoke(
        {
            "matching_result": _json.dumps(matched, indent=2),
            "candidate_experience_years": str(extracted.get("experience_years", 0)),
        },
        config={
            "run_name": f"scoring_{safe_label}",
            "tags": [safe_label, "scoring", "streamlit"],
            "metadata": {"step": "3_scoring", "candidate": label},
        },
    )

    # Step 4 — Explain
    explanation_chain = get_explanation_chain()
    explanation = explanation_chain.invoke(
        {
            "candidate_name": extracted.get("candidate_name", "Unknown"),
            "total_score":    str(scored.get("total_score", 0)),
            "grade":          scored.get("grade", "N/A"),
            "extracted_profile": _json.dumps(extracted, indent=2),
            "matching_result":   _json.dumps(matched, indent=2),
            "score_result":      _json.dumps(scored, indent=2),
        },
        config={
            "run_name": f"explanation_{safe_label}",
            "tags": [safe_label, "explanation", "streamlit"],
            "metadata": {"step": "4_explanation", "candidate": label},
        },
    )

    return {
        "label":       label,
        "extraction":  extracted,
        "matching":    matched,
        "scoring":     scored,
        "explanation": explanation,
    }


def render_results(result: dict):
    """Render the full pipeline result in the Streamlit UI."""
    scoring    = result.get("scoring", {})
    matching   = result.get("matching", {})
    extraction = result.get("extraction", {})
    explanation = result.get("explanation", "")

    total_score = scoring.get("total_score", 0)
    grade       = scoring.get("grade", "?")
    name        = extraction.get("candidate_name", "Unknown")

    # ── Score Banner ──────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"### {name}")
        st.markdown(
            f'<span class="score-badge {grade_color(grade)}">Grade {grade}</span>',
            unsafe_allow_html=True
        )
    with col2:
        st.metric("Total Score", f"{total_score}/100")
    with col3:
        st.metric("Experience", f"{extraction.get('experience_years', 0)} yrs")

    # ── Score Breakdown ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="step-header">Score Breakdown</div>', unsafe_allow_html=True)

    skills_score = scoring.get("skills_score", 0)
    exp_score    = scoring.get("experience_score", 0)
    tools_score  = scoring.get("tools_score", 0)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Skills** (50 pts max)")
        st.progress(skills_score / 50)
        st.markdown(f"**{skills_score}** / 50")
    with c2:
        st.markdown("**Experience** (30 pts max)")
        st.progress(exp_score / 30 if exp_score <= 30 else 1.0)
        st.markdown(f"**{exp_score}** / 30")
    with c3:
        st.markdown("**Tools** (20 pts max)")
        st.progress(tools_score / 20 if tools_score <= 20 else 1.0)
        st.markdown(f"**{tools_score}** / 20")

    if scoring.get("score_breakdown"):
        st.caption(scoring["score_breakdown"])

    # ── Skills & Tools ────────────────────────────────────────────────────────
    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.markdown('<div class="step-header">✅ Matched Skills</div>', unsafe_allow_html=True)
        matched_s = matching.get("matched_skills", [])
        if matched_s:
            st.markdown(render_skill_chips(matched_s, "chip-match"), unsafe_allow_html=True)
        else:
            st.caption("None matched")

        st.markdown('<div class="step-header" style="margin-top:1rem">✅ Matched Tools</div>', unsafe_allow_html=True)
        matched_t = matching.get("matched_tools", [])
        if matched_t:
            st.markdown(render_skill_chips(matched_t, "chip-tool"), unsafe_allow_html=True)
        else:
            st.caption("None matched")

    with right:
        st.markdown('<div class="step-header">❌ Missing Skills</div>', unsafe_allow_html=True)
        missing_s = matching.get("missing_skills", [])
        if missing_s:
            st.markdown(render_skill_chips(missing_s, "chip-miss"), unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#3fb950">No critical gaps!</span>', unsafe_allow_html=True)

        st.markdown('<div class="step-header" style="margin-top:1rem">❌ Missing Tools</div>', unsafe_allow_html=True)
        missing_t = matching.get("missing_tools", [])
        if missing_t:
            st.markdown(render_skill_chips(missing_t, "chip-miss"), unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#3fb950">No critical gaps!</span>', unsafe_allow_html=True)

    # ── Match Summary ─────────────────────────────────────────────────────────
    if matching.get("overall_match_summary"):
        st.markdown("---")
        st.markdown('<div class="step-header">Match Summary</div>', unsafe_allow_html=True)
        st.info(matching["overall_match_summary"])

    # ── Hiring Recommendation ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="step-header">Hiring Recommendation</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="recommendation-box">{explanation}</div>',
        unsafe_allow_html=True
    )

    # ── Raw JSON (collapsed) ──────────────────────────────────────────────────
    with st.expander("🔍 View Raw Pipeline JSON"):
        tab1, tab2, tab3 = st.tabs(["Extraction", "Matching", "Scoring"])
        with tab1:
            st.json(extraction)
        with tab2:
            st.json(matching)
        with tab3:
            st.json(scoring)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            st.error("GROQ_API_KEY not set. Add it to your .env file.")
        else:
            st.success("Groq API key loaded ✓")

        tracing = os.getenv("LANGCHAIN_TRACING_V2", "false")
        project = os.getenv("LANGCHAIN_PROJECT", "not set")
        if tracing == "true":
            st.success(f"LangSmith tracing ON ✓\nProject: `{project}`")
            st.markdown(
                "📊 [View Traces](https://smith.langchain.com/)",
                unsafe_allow_html=False
            )
        else:
            st.warning("LangSmith tracing is OFF.\nSet `LANGCHAIN_TRACING_V2=true` in .env")

        st.markdown("---")
        st.markdown("### 🐛 Debug Mode")
        use_flawed = st.toggle(
            "Use Flawed Prompt",
            value=False,
            help="Enables the intentionally broken extraction prompt to demonstrate "
                 "LangSmith catching hallucinations. Compare traces in LangSmith!"
        )
        if use_flawed:
            st.markdown(
                '<div class="warning-box">⚠️ Flawed prompt active — '
                'the LLM will hallucinate skills not in the resume. '
                'Watch this in LangSmith!</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("### 📖 About")
        st.caption(
            "AI Resume Screening System using LangChain + Groq (Llama 3.3-70B) "
            "with LangSmith tracing. Pipeline: Extract → Match → Score → Explain."
        )

    # ── Hero Header ───────────────────────────────────────────────────────────
    st.markdown('<div class="hero-title">AI Resume Screener</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">LangChain · Groq Llama 3.3-70B · LangSmith Tracing</div>',
        unsafe_allow_html=True
    )

    # ── Mode Tabs ─────────────────────────────────────────────────────────────
    tab_single, tab_batch, tab_demo = st.tabs([
        "🔍 Single Resume", "📊 Batch Compare", "🎯 Demo Mode"
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — Single Resume Screening
    # ══════════════════════════════════════════════════════════════════════════
    with tab_single:
        st.markdown("### Screen a Single Resume")

        col_resume, col_jd = st.columns(2)

        with col_resume:
            st.markdown("#### 📄 Resume")
            uploaded = st.file_uploader("Upload .txt file", type=["txt"], key="single_upload")
            if uploaded:
                resume_text = uploaded.read().decode("utf-8")
                st.text_area("Resume content", resume_text, height=280, key="single_resume_display")
            else:
                resume_text = st.text_area(
                    "Or paste resume text here",
                    placeholder="Paste the candidate's resume...",
                    height=280,
                    key="single_resume_paste"
                )

        with col_jd:
            st.markdown("#### 💼 Job Description")
            use_sample_jd = st.checkbox("Load sample JD", value=True, key="single_jd_toggle")
            if use_sample_jd:
                jd_text = load_sample("job_description.txt")
                st.text_area("Job Description", jd_text, height=280, key="single_jd_display")
            else:
                jd_text = st.text_area(
                    "Paste job description",
                    placeholder="Paste the job description...",
                    height=280,
                    key="single_jd_paste"
                )

        candidate_label = st.text_input("Candidate label (for LangSmith)", value="candidate_1")

        if st.button("🚀 Screen Resume", key="single_run"):
            if not resume_text or not jd_text:
                st.error("Please provide both a resume and a job description.")
            elif not os.getenv("GROQ_API_KEY"):
                st.error("GROQ_API_KEY is missing. Add it to your .env file.")
            else:
                with st.spinner("Running pipeline... (Step 1/4: Extracting skills)"):
                    progress = st.progress(0)
                    status   = st.empty()

                    try:
                        status.markdown("**Step 1/4** — Extracting skills and profile...")
                        progress.progress(10)

                        result = run_pipeline(
                            resume_text=resume_text,
                            job_description=jd_text,
                            label=candidate_label,
                            use_flawed=use_flawed,
                        )

                        progress.progress(100)
                        status.empty()
                        st.success("Pipeline complete!")

                        st.markdown("---")
                        st.markdown("## 📋 Results")
                        render_results(result)

                    except Exception as e:
                        progress.empty()
                        status.empty()
                        st.error(f"Pipeline error: {e}")
                        st.exception(e)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — Batch Compare
    # ══════════════════════════════════════════════════════════════════════════
    with tab_batch:
        st.markdown("### Compare Multiple Candidates Side-by-Side")
        st.caption("Upload up to 3 resumes and screen them all against the same JD.")

        use_sample_jd_batch = st.checkbox("Use sample Job Description", value=True, key="batch_jd_toggle")
        if use_sample_jd_batch:
            batch_jd = load_sample("job_description.txt")
            with st.expander("View Job Description"):
                st.text(batch_jd)
        else:
            batch_jd = st.text_area("Paste Job Description", height=150, key="batch_jd")

        st.markdown("#### Upload Resumes")
        cols = st.columns(3)
        resumes = {}
        for i, col in enumerate(cols):
            with col:
                label = st.text_input(f"Label {i+1}", value=f"Candidate {i+1}", key=f"batch_label_{i}")
                uploaded = st.file_uploader(f"Resume {i+1} (.txt)", type=["txt"], key=f"batch_upload_{i}")
                if uploaded:
                    resumes[label] = uploaded.read().decode("utf-8")

        if st.button("🚀 Screen All Candidates", key="batch_run"):
            if not resumes:
                st.error("Upload at least one resume.")
            elif not batch_jd:
                st.error("Provide a job description.")
            elif not os.getenv("GROQ_API_KEY"):
                st.error("GROQ_API_KEY is missing.")
            else:
                all_results = []
                prog = st.progress(0)
                for idx, (lbl, txt) in enumerate(resumes.items()):
                    with st.spinner(f"Screening {lbl}..."):
                        try:
                            r = run_pipeline(txt, batch_jd, lbl, use_flawed=use_flawed)
                            all_results.append(r)
                        except Exception as e:
                            st.error(f"Error screening {lbl}: {e}")
                    prog.progress(int((idx + 1) / len(resumes) * 100))

                prog.empty()
                st.success(f"Screened {len(all_results)} candidates!")

                # Leaderboard
                st.markdown("---")
                st.markdown("### 🏆 Leaderboard")
                sorted_results = sorted(
                    all_results,
                    key=lambda r: r["scoring"].get("total_score", 0),
                    reverse=True
                )

                for rank, r in enumerate(sorted_results, 1):
                    score = r["scoring"].get("total_score", 0)
                    grade = r["scoring"].get("grade", "?")
                    name  = r["extraction"].get("candidate_name", r["label"])
                    medal = ["🥇", "🥈", "🥉"][rank - 1] if rank <= 3 else f"#{rank}"
                    st.markdown(
                        f"{medal} **{name}** — "
                        f'<span class="score-badge {grade_color(grade)}">{score}/100 · {grade}</span>',
                        unsafe_allow_html=True
                    )

                # Detailed tabs per candidate
                st.markdown("---")
                st.markdown("### Detailed Results")
                if all_results:
                    tabs = st.tabs([r["extraction"].get("candidate_name", r["label"]) for r in all_results])
                    for tab, r in zip(tabs, all_results):
                        with tab:
                            render_results(r)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — Demo Mode (pre-loaded mock resumes)
    # ══════════════════════════════════════════════════════════════════════════
    with tab_demo:
        st.markdown("### Demo — Pre-loaded Mock Resumes")
        st.caption(
            "Uses the bundled Strong / Average / Weak resumes and the sample JD. "
            "Perfect for showcasing the system without uploading anything."
        )

        demo_choice = st.selectbox(
            "Select a candidate to screen",
            ["Strong Candidate", "Average Candidate", "Weak Candidate"],
        )

        resume_map = {
            "Strong Candidate":  "resume_strong.txt",
            "Average Candidate": "resume_average.txt",
            "Weak Candidate":    "resume_weak.txt",
        }

        demo_resume = load_sample(resume_map[demo_choice])
        demo_jd     = load_sample("job_description.txt")

        if not demo_resume:
            st.warning("Sample data files not found. Make sure the `data/` folder exists.")
        else:
            with st.expander("Preview Resume"):
                st.text(demo_resume)

            if st.button("🚀 Run Demo Pipeline", key="demo_run"):
                if not os.getenv("GROQ_API_KEY"):
                    st.error("GROQ_API_KEY is missing. Add it to your .env file.")
                else:
                    with st.spinner(f"Screening {demo_choice}..."):
                        try:
                            result = run_pipeline(
                                resume_text=demo_resume,
                                job_description=demo_jd,
                                label=demo_choice,
                                use_flawed=use_flawed,
                            )
                            st.success("Done!")
                            st.markdown("---")
                            st.markdown("## 📋 Results")
                            render_results(result)
                        except Exception as e:
                            st.error(f"Error: {e}")
                            st.exception(e)


if __name__ == "__main__":
    main()
