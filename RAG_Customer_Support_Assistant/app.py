import os
import operator
from typing import TypedDict, List, Annotated

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
import gradio as gr

# ─────────────────────────────────────────────
# STATE & CONSTANTS
# ─────────────────────────────────────────────
class AgentState(TypedDict):
    question: str
    chat_history: Annotated[List[dict], operator.add]
    context: List[Document]
    answer: str
    sources: List[str]
    confidence: float
    requires_human: bool

ESCALATION_KEYWORDS = [
    "human","manager","agent","complaint","refund",
    "cancel","legal","sue","lawyer","escalate","supervisor",
]
PDF_PATH = "knowledge_base.pdf"
_retriever = None

def get_retriever():
    global _retriever
    if _retriever is None:
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        splits = splitter.split_documents(docs)
        emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vs = Chroma.from_documents(splits, embedding=emb, collection_name="rag_kb")
        _retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 4,"fetch_k":10})
    return _retriever

# ─────────────────────────────────────────────
# NODES
# ─────────────────────────────────────────────
def router_node(state):
    q = state["question"].lower()
    return {"requires_human": any(kw in q for kw in ESCALATION_KEYWORDS)}

def rag_node(state):
    retriever = get_retriever()
    question = state["question"]
    context_docs = retriever.invoke(question)
    context_text = "\n\n---\n\n".join(
        [f"[Page {d.metadata.get('page',0)+1}]\n{d.page_content}" for d in context_docs]
    )
    sources = list(set([f"Page {d.metadata.get('page',0)+1}" for d in context_docs]))
    history_text = "".join(
        [f"User: {t['user']}\nAssistant: {t['assistant']}\n"
         for t in state.get("chat_history",[])[-4:]]
    )
    llm = ChatGroq(model="llama3-8b-8192", temperature=0.1)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a professional, empathetic customer support assistant. "
         "Answer ONLY from the provided context. Be concise and friendly. "
         "If the answer is not in the context, say so and offer to escalate."),
        ("human",
         "History:\n{history}\n\nContext:\n{context}\n\nQuestion: {question}")
    ])
    response = (prompt | llm).invoke({"history":history_text,"context":context_text,"question":question})
    q_words = set(question.lower().split())
    overlap = len(q_words & set(context_text.lower().split())) / max(len(q_words),1)
    confidence = min(round(0.5 + overlap * 1.5, 2), 0.99)
    return {"context":context_docs,"answer":response.content,"sources":sources,"confidence":confidence}

def hitl_node(state):
    return {
        "answer":"I understand this is important. Your query has been escalated to a human support specialist who will contact you within 24 hours. A ticket has been created for you.",
        "sources":["Escalation System"],"confidence":1.0,
    }

def route(state):
    return "hitl_node" if state["requires_human"] else "rag_node"

def build_graph():
    wf = StateGraph(AgentState)
    wf.add_node("router", router_node)
    wf.add_node("rag_node", rag_node)
    wf.add_node("hitl_node", hitl_node)
    wf.set_entry_point("router")
    wf.add_conditional_edges("router", route, {"rag_node":"rag_node","hitl_node":"hitl_node"})
    wf.add_edge("rag_node", END)
    wf.add_edge("hitl_node", END)
    return wf.compile()

GRAPH = build_graph()

# ─────────────────────────────────────────────
# PIPELINE PANEL HTML
# ─────────────────────────────────────────────
def pipeline_html(active_node="", conf=0.0, latency="—", is_hitl=False, sources=None):
    if sources is None: sources = []
    nodes = [
        ("node-input",  "📝", "User Input",       "Question received"),
        ("node-router", "🔀", "Router Node",       "Intent classification"),
        ("node-embed",  "🧬", "Embeddings",        "all-MiniLM-L6-v2 semantic"),
        ("node-chroma", "🗄️", "ChromaDB (MMR)",    "Top-4 chunks retrieved"),
        ("node-llm",    "🧠", "Groq Llama 3 8B",   "Grounded generation"),
        ("node-out",    "✅", "Response + Sources", "Confidence scored"),
    ]
    if is_hitl:
        nodes[2] = ("node-embed",  "⏭️", "Embeddings",  "Skipped for escalation")
        nodes[3] = ("node-chroma", "⏭️", "ChromaDB",    "Skipped for escalation")
        nodes[4] = ("node-llm",    "⏭️", "Groq LLM",   "Skipped for escalation")

    conf_pct = int(conf * 100)
    conf_color = "#34d399" if conf > 0.75 else "#fbbf24" if conf > 0.45 else "#f87171"
    src_str = ", ".join(sources) if sources else "—"

    node_html = ""
    for nid, icon, label, sublabel in nodes:
        is_active = nid == active_node
        skip = is_hitl and nid in ("node-embed","node-chroma","node-llm")
        style = (
            "border-color:#2563eb;background:#1e3a5f;box-shadow:0 0 12px rgba(37,99,235,0.4);" if is_active
            else "opacity:0.4;" if skip else ""
        )
        node_html += f"""
        <div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;
                    padding:9px 12px;margin-bottom:8px;font-size:11px;{style}">
          <span style="font-size:15px;margin-right:6px;">{icon}</span><b>{label}</b>
          <span style="display:block;color:#94a3b8;font-size:10px;margin-top:2px;">{sublabel}</span>
        </div>"""

    hitl_badge = (
        '<div style="background:#450a0a;border:1px solid #7f1d1d;border-radius:20px;'
        'padding:3px 10px;font-size:10px;color:#fca5a5;display:inline-block;margin-top:6px;">🔴 Escalated to Human Agent</div>'
        if is_hitl and active_node == "node-out" else ""
    )

    return f"""
    <div style="font-family:'Segoe UI',system-ui,sans-serif;color:#e2e8f0;padding:8px;">
      <div style="font-size:13px;color:#60a5fa;font-weight:600;margin-bottom:14px;">🏗️ Pipeline State</div>
      {node_html}
      {hitl_badge}
      <div style="margin-top:16px;font-size:11px;">
        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
          <span>Latency</span><span style="color:#7dd3fc;">{latency}</span>
        </div>
        <div style="height:4px;background:#1e293b;border-radius:2px;margin-bottom:12px;">
          <div style="height:100%;width:{'60' if latency != '—' else '0'}%;background:#60a5fa;border-radius:2px;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
          <span>Confidence</span><span style="color:{conf_color};">{conf_pct if conf > 0 else '—'}{'%' if conf > 0 else ''}</span>
        </div>
        <div style="height:4px;background:#1e293b;border-radius:2px;margin-bottom:12px;">
          <div style="height:100%;width:{conf_pct}%;background:{conf_color};border-radius:2px;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
          <span>Sources</span><span style="color:#7dd3fc;font-size:10px;">{src_str}</span>
        </div>
      </div>
      <div style="margin-top:14px;font-size:10px;color:#475569;border-top:1px solid #1e293b;padding-top:12px;">
        <b style="color:#64748b;font-size:11px;">Stack</b><br>
        <span style="background:#0f172a;border:1px solid #1e293b;border-radius:4px;padding:2px 6px;margin:2px 2px 0 0;display:inline-block;">LangGraph</span>
        <span style="background:#0f172a;border:1px solid #1e293b;border-radius:4px;padding:2px 6px;margin:2px 2px 0 0;display:inline-block;">Groq</span>
        <span style="background:#0f172a;border:1px solid #1e293b;border-radius:4px;padding:2px 6px;margin:2px 2px 0 0;display:inline-block;">ChromaDB</span>
        <span style="background:#0f172a;border:1px solid #1e293b;border-radius:4px;padding:2px 6px;margin:2px 2px 0 0;display:inline-block;">HF Embeddings</span>
      </div>
    </div>
    """

# ─────────────────────────────────────────────
# CHAT FUNCTION
# ─────────────────────────────────────────────
import time

def chat(user_message, history, state_hist, pipeline_state):
    if not user_message.strip():
        return history, state_hist, pipeline_state, ""

    agent_history = [{"user": h[0], "assistant": h[1]} for h in history if h[0] and h[1]]

    # Show "thinking" pipeline state immediately
    thinking_html = pipeline_html("node-router")
    yield history, state_hist, thinking_html, ""

    t0 = time.time()
    try:
        result = GRAPH.invoke({
            "question": user_message,
            "chat_history": agent_history,
            "context": [], "answer": "", "sources": [],
            "confidence": 0.0, "requires_human": False,
        })
        elapsed = f"{time.time()-t0:.1f}s"

        answer = result["answer"]
        sources = result.get("sources", [])
        confidence = result.get("confidence", 0.0)
        is_hitl = result.get("requires_human", False)

        # Format answer with inline badges
        conf_pct = int(confidence * 100)
        conf_color = "#34d399" if confidence > 0.75 else "#fbbf24" if confidence > 0.45 else "#f87171"
        src_text = ", ".join(sources) if sources else "Knowledge Base"

        if is_hitl:
            badge = '<br><span style="background:#450a0a;border:1px solid #7f1d1d;border-radius:20px;padding:2px 10px;font-size:11px;color:#fca5a5;">🔴 Escalated to Human Agent</span>'
        else:
            badge = f'<br><span style="background:#0f2544;border:1px solid #1e3a5f;border-radius:20px;padding:2px 10px;font-size:11px;color:#7dd3fc;">🟢 {conf_pct}% confident · 📄 {src_text}</span>'

        full_answer = answer + badge

        history.append((user_message, full_answer))
        state_hist.append({"user": user_message, "assistant": answer})

        final_html = pipeline_html("node-out", confidence, elapsed, is_hitl, sources)
        yield history, state_hist, final_html, ""
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        history.append((user_message, f"**SYSTEM ERROR:**\n```python\n{err}\n```\n**Troubleshooting:**\n1. Ensure `knowledge_base.pdf` is uploaded to the Space.\n2. Ensure `GROQ_API_KEY` is set in Space Secrets."))
        yield history, state_hist, pipeline_state, ""

# ─────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────
CSS = """
body, .gradio-container { background:#0a0f1e !important; }
.gradio-container { max-width:1000px !important; margin:auto !important; }
#chatbot { height:460px; background:#111827 !important; border:1px solid #1e293b !important; border-radius:12px !important; }
#chatbot .message.bot { background:#1e3a5f !important; border-radius:12px 12px 12px 4px !important; }
#chatbot .message.user { background:#14532d !important; border-radius:12px 12px 4px 12px !important; }
#pipeline-panel { background:#111827; border:1px solid #1e293b; border-radius:12px; padding:12px; min-height:460px; }
#msg-input textarea { background:#0f172a !important; color:#e2e8f0 !important; border:1px solid #334155 !important; border-radius:10px !important; }
#send-btn { background:#2563eb !important; border:none !important; color:white !important; border-radius:10px !important; }
.label-wrap { color:#94a3b8 !important; }
footer { display:none !important; }
"""

HEADER = """
<div style="text-align:center;font-family:'Segoe UI',system-ui,sans-serif;padding:20px 0 10px;color:#e2e8f0;">
  <h1 style="font-size:26px;font-weight:700;color:#60a5fa;margin-bottom:6px;">🤖 RAG Customer Support Assistant</h1>
  <p style="color:#94a3b8;font-size:13px;">LangGraph · Groq Llama 3 8B · ChromaDB · HuggingFace Embeddings · HITL</p>
</div>
"""

EXAMPLES = [
    "What are your customer support hours?",
    "How do I return a product?",
    "I want to speak to a manager about my refund!",
    "What payment methods do you accept?",
    "My order hasn't arrived, what should I do?",
]

with gr.Blocks() as demo:
    gr.HTML(HEADER)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                label="",
                avatar_images=("👤", "🤖"),
                show_label=False,
            )
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask a customer support question...",
                    show_label=False,
                    container=False,
                    scale=4,
                    elem_id="msg-input",
                )
                send = gr.Button("Send ➤", scale=1, elem_id="send-btn", variant="primary")
            
            gr.Examples(EXAMPLES, inputs=msg, label="💡 Try these")
            clear = gr.Button("🗑️ Clear Conversation", variant="secondary")

        with gr.Column(scale=1):
            pipeline = gr.HTML(
                pipeline_html(),
                elem_id="pipeline-panel",
                label="",
            )

    state_hist = gr.State([])

    send.click(chat, [msg, chatbot, state_hist, pipeline], [chatbot, state_hist, pipeline, msg])
    msg.submit(chat, [msg, chatbot, state_hist, pipeline], [chatbot, state_hist, pipeline, msg])
    clear.click(lambda: ([], [], pipeline_html(), ""), [], [chatbot, state_hist, pipeline, msg])

if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        print("⚠️  Set GROQ_API_KEY in environment or HF Space Secrets.")
    demo.launch(css=CSS, theme=gr.themes.Base())
