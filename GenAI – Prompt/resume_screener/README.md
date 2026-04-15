# 🎯 AI Resume Screening System

A production-level AI pipeline that screens resumes against job descriptions using **LangChain**, **Groq (Llama 3.3-70B)**, and **LangSmith** for full observability.

> **Zero cost** — built entirely on free-tier APIs. No OpenAI key needed.

---

## ✨ Features

- **4-Step LLM Pipeline**: Extract → Match → Score → Explain
- **Weighted Scoring**: Skills (50%) + Experience (30%) + Tools (20%)
- **Explainable AI**: Evidence-backed hiring recommendations, no hallucinations
- **LangSmith Tracing**: Every pipeline step is observable and debuggable
- **Streamlit UI**: Clean recruiter-facing interface with batch mode
- **Debug Demo**: Intentional flawed prompt to showcase LangSmith catching hallucinations

---

## 🏗️ Architecture

```
resume_screener/
├── prompts/
│   ├── extraction_prompt.py    # Skill/tool/experience extraction
│   ├── matching_prompt.py      # JD vs resume comparison
│   ├── scoring_prompt.py       # Weighted 0-100 scoring
│   └── explanation_prompt.py   # Hiring recommendation
├── chains/
│   ├── llm_factory.py          # Shared Groq LLM instance
│   ├── extraction_chain.py     # LCEL chain — Step 1
│   ├── matching_chain.py       # LCEL chain — Step 2
│   ├── scoring_chain.py        # LCEL chain — Step 3
│   └── explanation_chain.py   # LCEL chain — Step 4
├── data/
│   ├── job_description.txt     # Sample ML Engineer JD
│   ├── resume_strong.txt       # Strong candidate mock
│   ├── resume_average.txt      # Average candidate mock
│   └── resume_weak.txt         # Weak candidate mock
├── main.py                     # CLI pipeline runner
├── app.py                      # Streamlit web UI
├── requirements.txt
└── .env.example
```

**Pipeline flow:**
```
Resume Text
    │
    ▼
[Step 1] Skill Extraction     → JSON: skills, tools, experience years
    │
    ▼
[Step 2] JD Matching          → JSON: matched/missing skills & tools
    │
    ▼
[Step 3] Weighted Scoring     → JSON: score (0-100), grade (A-F)
    │
    ▼
[Step 4] Explanation          → Text: hiring recommendation
    │
    ▼
LangSmith Traces (all steps)
```

---

## 🚀 Quickstart

### 1. Clone & install

```bash
git clone https://github.com/yourusername/ai-resume-screener
cd ai-resume-screener
pip install -r requirements.txt
```

### 2. Get free API keys

| Service | URL | Notes |
|---------|-----|-------|
| **Groq** | https://console.groq.com/ | Free, no credit card. Uses Llama 3.3-70B |
| **LangSmith** | https://smith.langchain.com/ | Free developer plan |

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your keys
```

```env
GROQ_API_KEY=your_groq_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_PROJECT=ai-resume-screener
```

### 4. Run CLI (all 3 candidates + debug demo)

```bash
python main.py
```

### 5. Run Streamlit UI

```bash
streamlit run app.py
```

---

## 📊 Scoring Weights

| Component | Weight | Max Points |
|-----------|--------|-----------|
| Skills Match | 50% | 50 pts |
| Experience | 30% | 30 pts |
| Tools Match | 20% | 20 pts |

| Grade | Score Range |
|-------|------------|
| A | 85 – 100 |
| B | 70 – 84 |
| C | 50 – 69 |
| D | 30 – 49 |
| F | 0 – 29 |

---

## 🔍 LangSmith Tracing

Every pipeline run is automatically traced in LangSmith when `LANGCHAIN_TRACING_V2=true`.

Each run shows:
- Input/output for all 4 steps
- Latency per step
- Token usage
- Tags and metadata for filtering

The project also includes a **flawed prompt demo** — toggle "Use Flawed Prompt" in the Streamlit sidebar (or pass `use_flawed_prompt=True` in `main.py`). This causes the LLM to hallucinate skills, which is immediately visible in LangSmith traces.

---

## 🛠️ Tech Stack

- **LLM**: Groq API — `llama-3.3-70b-versatile` (free tier)
- **Orchestration**: LangChain + LCEL
- **Tracing**: LangSmith
- **UI**: Streamlit
- **Language**: Python 3.10+

---

## 📝 License

MIT License — free to use, modify, and share.
