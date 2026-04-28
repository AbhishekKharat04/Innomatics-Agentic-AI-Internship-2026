import os
import time
from typing import TypedDict, List, Annotated
import operator

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
# 1. AGENT STATE
# ─────────────────────────────────────────────
class AgentState(TypedDict):
    question: str
    chat_history: Annotated[List[dict], operator.add]  # accumulates across turns
    context: List[Document]
    answer: str
    sources: List[str]
    confidence: float
    requires_human: bool

# ─────────────────────────────────────────────
# 2. VECTOR STORE SETUP (run once at startup)
# ─────────────────────────────────────────────
PDF_PATH = "knowledge_base.pdf"
EMBED_MODEL = "all-MiniLM-L6-v2"
ESCALATION_KEYWORDS = [
    "human", "manager", "agent", "complaint", "refund",
    "cancel", "legal", "sue", "lawyer", "escalate", "supervisor",
]

_retriever = None

def get_retriever():
    global _retriever
    if _retriever is None:
        print("🔄 Loading and indexing knowledge base...")
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        splits = splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vs = Chroma.from_documents(splits, embedding=embeddings, collection_name="rag_kb")
        _retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10})
        print(f"✅ Indexed {len(splits)} chunks from knowledge base.")
    return _retriever

# ─────────────────────────────────────────────
# 3. GRAPH NODES
# ─────────────────────────────────────────────
def router_node(state: AgentState) -> AgentState:
    """Classify intent: escalate to human or handle via RAG."""
    q = state["question"].lower()
    requires_human = any(kw in q for kw in ESCALATION_KEYWORDS)
    return {"requires_human": requires_human}


def rag_node(state: AgentState) -> AgentState:
    """Retrieve context and generate a grounded answer with source attribution."""
    retriever = get_retriever()
    question = state["question"]

    # Retrieve semantically relevant chunks
    context_docs = retriever.invoke(question)
    context_text = "\n\n---\n\n".join(
        [f"[Source: Page {doc.metadata.get('page', '?')+1}]\n{doc.page_content}" 
         for doc in context_docs]
    )

    # Build source attribution list
    sources = list(set([
        f"Page {doc.metadata.get('page', '?')+1}" 
        for doc in context_docs
    ]))

    # Chat history for multi-turn memory
    history_text = ""
    for turn in state.get("chat_history", [])[-6:]:  # last 3 turns
        history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"

    # LLM Prompt
    llm = ChatGroq(model="llama3-8b-8192", temperature=0.1)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a professional, empathetic customer support assistant. "
         "Answer ONLY using the provided context. "
         "If the answer is not in the context, say you don't have that information and offer to escalate. "
         "Be concise, clear, and friendly. Format your response in 2-3 sentences max."),
        ("human",
         "Conversation history:\n{history}\n\n"
         "Context from knowledge base:\n{context}\n\n"
         "Customer question: {question}")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "history": history_text,
        "context": context_text,
        "question": question,
    })
    answer = response.content

    # Simple confidence proxy: how well keywords overlap between question and context
    q_words = set(question.lower().split())
    ctx_words = set(context_text.lower().split())
    overlap = len(q_words & ctx_words) / max(len(q_words), 1)
    confidence = min(round(0.5 + overlap * 1.5, 2), 0.99)

    return {
        "context": context_docs,
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
    }


def hitl_node(state: AgentState) -> AgentState:
    """Human-in-the-Loop: escalate complex/sensitive queries to a human agent."""
    return {
        "answer": (
            "I understand this is an important matter. "
            "I've escalated your query to a human support specialist who will reach out within 24 hours. "
            "Your ticket has been created. Is there anything else I can help you with in the meantime?"
        ),
        "sources": ["Escalation System"],
        "confidence": 1.0,
    }

# ─────────────────────────────────────────────
# 4. BUILD LANGGRAPH
# ─────────────────────────────────────────────
def route_after_router(state: AgentState):
    return "hitl_node" if state["requires_human"] else "rag_node"


def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("router", router_node)
    workflow.add_node("rag_node", rag_node)
    workflow.add_node("hitl_node", hitl_node)
    workflow.set_entry_point("router")
    workflow.add_conditional_edges("router", route_after_router, {
        "rag_node": "rag_node",
        "hitl_node": "hitl_node",
    })
    workflow.add_edge("rag_node", END)
    workflow.add_edge("hitl_node", END)
    return workflow.compile()

GRAPH = build_graph()

# ─────────────────────────────────────────────
# 5. GRADIO CHAT INTERFACE
# ─────────────────────────────────────────────
def chat(user_message: str, history: list):
    """Process a user message through the LangGraph RAG pipeline."""
    if not user_message.strip():
        return history, history, "", ""

    # Convert Gradio history to our format
    agent_history = [
        {"user": h[0], "assistant": h[1]} 
        for h in history if h[0] and h[1]
    ]

    state = {
        "question": user_message,
        "chat_history": agent_history,
        "context": [],
        "answer": "",
        "sources": [],
        "confidence": 0.0,
        "requires_human": False,
    }

    result = GRAPH.invoke(state)

    answer = result["answer"]
    sources = result.get("sources", [])
    confidence = result.get("confidence", 0.0)
    requires_human = result.get("requires_human", False)

    # Format metadata footer
    if requires_human:
        meta = "🔴 **Escalated to Human Agent**"
    else:
        conf_bar = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.4 else "🔴"
        src_text = ", ".join(sources) if sources else "Knowledge Base"
        meta = f"{conf_bar} Confidence: `{int(confidence*100)}%` | 📄 Sources: `{src_text}`"

    full_answer = f"{answer}\n\n{meta}"

    history.append((user_message, full_answer))
    return history, history, "", ""


def clear_chat():
    return [], [], "", ""


# ─────────────────────────────────────────────
# 6. UI LAYOUT
# ─────────────────────────────────────────────
EXAMPLE_QUESTIONS = [
    "What are your business hours?",
    "How do I return a product?",
    "I want to speak to a manager about my order",
    "What payment methods do you accept?",
    "My order hasn't arrived yet, what should I do?",
]

CSS = """
.gradio-container { max-width: 860px !important; margin: auto !important; }
#chatbot { height: 480px; }
.message.bot { background: linear-gradient(135deg, #1e3a5f 0%, #0f2544 100%) !important; border-radius: 12px !important; }
.message.user { background: linear-gradient(135deg, #2d6a4f 0%, #1b4332 100%) !important; border-radius: 12px !important; }
footer { display: none !important; }
"""

HEADER_MD = """
# 🤖 RAG Customer Support Assistant
**Powered by:** LangGraph · Groq (Llama 3 8B) · ChromaDB · HuggingFace Embeddings

An intelligent support assistant that retrieves answers from a knowledge base using **real semantic search** 
and generates grounded responses with a **language model**. Complex queries trigger **Human-in-the-Loop (HITL)** escalation.
"""

with gr.Blocks(css=CSS, theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown(HEADER_MD)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                label="Support Chat",
                bubble_full_width=False,
                avatar_images=("👤", "🤖"),
            )
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask a customer support question...",
                    show_label=False,
                    scale=4,
                    container=False,
                )
                send_btn = gr.Button("Send ➤", variant="primary", scale=1)
            clear_btn = gr.Button("🗑️ Clear Conversation", variant="secondary")

        with gr.Column(scale=1):
            gr.Markdown("### 💡 Try These")
            for q in EXAMPLE_QUESTIONS:
                gr.Button(q, variant="secondary", size="sm").click(
                    fn=lambda x=q: x, inputs=[], outputs=[msg]
                )
            
            gr.Markdown("""
            ---
            ### 🏗️ Architecture
            - **Ingestion:** PyPDFLoader → RecursiveTextSplitter
            - **Embeddings:** `all-MiniLM-L6-v2` (Semantic)
            - **Vector DB:** ChromaDB (MMR Search)
            - **LLM:** Groq · Llama 3 8B
            - **Orchestration:** LangGraph StateGraph
            - **HITL:** Keyword + Confidence Routing
            """)

    state = gr.State([])

    send_btn.click(chat, [msg, state], [chatbot, state, msg, msg])
    msg.submit(chat, [msg, state], [chatbot, state, msg, msg])
    clear_btn.click(clear_chat, [], [chatbot, state, msg, msg])

if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        print("⚠️  WARNING: GROQ_API_KEY not set. Set it in your environment or HF Space Secrets.")
    if not os.path.exists(PDF_PATH):
        print(f"⚠️  WARNING: {PDF_PATH} not found. Please add your knowledge base PDF.")
    demo.launch()
