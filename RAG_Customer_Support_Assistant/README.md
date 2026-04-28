---
title: RAG Customer Support Assistant
emoji: 🤖
colorFrom: orange
colorTo: blue
sdk: docker
app_port: 7860
license: mit
---

# RAG-Based Customer Support Assistant

This project is a full-stack RAG customer support assistant built for the Innomatics Agentic AI internship project. It combines:

- a React frontend
- a Node/Express backend
- a Python RAG engine using LangGraph
- a Human-in-the-Loop escalation path for sensitive or low-confidence queries

## What The App Does

The assistant uses a PDF knowledge base as its source of truth.

1. A user asks a question from the web UI.
2. The Express API validates the request and forwards it to the Python RAG engine.
3. The Python service loads or queries the vector store, retrieves relevant chunks, and decides whether to:
   - answer via RAG
   - escalate to a human-review path
4. The UI shows the answer, source pages, relevance scores, and escalation status.

## Tech Stack

- Frontend: React + Vite
- Backend API: Node.js + Express
- Retrieval / Orchestration: Python + LangGraph
- Vector Store: ChromaDB
- Embeddings: Hugging Face sentence-transformers with offline fallback support
- Local LLM: Ollama when available

## Security Measures

- Request body size limits
- Input validation for user questions
- Rate limiting on `/api/chat`
- Helmet security headers
- Restricted CORS policy for local UI origins
- Safe subprocess execution without shell interpolation of user input

## Local Run

Install both JavaScript and Python dependencies:

```bash
npm install
pip install -r requirements.txt
```

Run the web app:

```bash
npm run dev
```

This starts:

- Vite frontend on `http://127.0.0.1:5173`
- Express backend on `http://127.0.0.1:8787`

You can also run the production-style backend after building:

```bash
npm run build
npm start
```

## Python RAG CLI

Single question:

```bash
python rag_agent.py --question "What are the required deliverables?"
```

Interactive mode:

```bash
python rag_agent.py
```

JSON mode:

```bash
python rag_agent.py --question "What is HITL?" --json
```

## Deployment Notes

This repo includes a `Dockerfile` so the project can be deployed as a single container.

- Render:
  Use the included [`render.yaml`](../render.yaml) blueprint or create a Docker web service pointing to this folder.
- Hugging Face Spaces:
  Use a Docker Space and point it at this project folder.

Important:

- Free Hugging Face Spaces sleep when unused.
- Free Render web services spin down after inactivity.
- For truly always-on hosting, you will need a paid cloud plan.

## Submission Deliverables

- `HLD.pdf`
- `LLD.pdf`
- `Technical_Documentation.pdf`
- Full project implementation

## Project Structure

- `src/`: React frontend
- `server/`: Express backend
- `rag_agent.py`: Python RAG engine
- `knowledge_base.pdf`: source knowledge base
- `HLD.md`, `LLD.md`, `Technical_Documentation.md`: design and technical docs
- `generate_pdfs.py`: markdown-to-PDF helper
