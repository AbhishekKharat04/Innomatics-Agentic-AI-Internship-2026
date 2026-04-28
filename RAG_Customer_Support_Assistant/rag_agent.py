import os
from typing import TypedDict, List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

# --- Environment Setup for Groq ---
# Ensure GROQ_API_KEY is set in your environment (or HF Space Secrets)
# os.environ["GROQ_API_KEY"] = "your-groq-api-key-here"

# 1. Define State
class AgentState(TypedDict):
    question: str
    context: List[Document]
    answer: str
    requires_human: bool

# 2. Setup Vector Database & Retriever
def setup_retriever(pdf_path: str):
    print(f"Loading document: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print("Chunking document...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    print("Initializing ChromaDB and creating real Semantic Embeddings...")
    # Using real sentence-transformers for semantic search
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, collection_name="rag_support")
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. Define Nodes
def router_node(state: AgentState) -> AgentState:
    """
    Decides if the query needs a human agent based on simple keyword matching.
    """
    print("--- ROUTER NODE ---")
    question = state["question"].lower()
    escalation_keywords = ["human", "manager", "complaint", "talk to someone", "escalate"]
    
    requires_human = any(keyword in question for keyword in escalation_keywords)
    return {"requires_human": requires_human}

def retrieve_and_generate_node(state: AgentState) -> AgentState:
    """
    Retrieves context and generates an answer using Groq's fast Llama 3 model.
    """
    print("--- RAG NODE ---")
    retriever = setup_retriever("knowledge_base.pdf")
    question = state["question"]
    
    # Retrieve true semantic matches
    print("Retrieving context...")
    context_docs = retriever.invoke(question)
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    
    # Generate using Groq LLM
    print("Generating answer using Groq (Llama 3 8B)...")
    llm = ChatGroq(model="llama3-8b-8192", temperature=0)
    
    prompt = PromptTemplate.from_template(
        "You are a helpful customer support assistant. Answer the user's question using ONLY the provided context.\n\n"
        "Context: {context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    chain = prompt | llm
    response = chain.invoke({"context": context_text, "question": question})
    
    return {"context": context_docs, "answer": response.content}

def hitl_node(state: AgentState) -> AgentState:
    """
    Human-in-the-Loop Node. Pauses for human intervention.
    """
    print("--- HITL ESCALATION NODE ---")
    return {
        "answer": "Your query requires human assistance. A support agent will be with you shortly. (Ticket Escalated)"
    }

# 4. Define Edges (Routing Logic)
def route_after_router(state: AgentState):
    if state["requires_human"]:
        return "hitl_node"
    else:
        return "rag_node"

# 5. Build and Compile the Graph
def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("router", router_node)
    workflow.add_node("rag_node", retrieve_and_generate_node)
    workflow.add_node("hitl_node", hitl_node)
    
    workflow.set_entry_point("router")
    
    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "rag_node": "rag_node",
            "hitl_node": "hitl_node"
        }
    )
    
    workflow.add_edge("rag_node", END)
    workflow.add_edge("hitl_node", END)
    
    return workflow.compile()

# 6. Main Execution
if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY environment variable is not set.")
        print("Please set it before running, or add it to your Hugging Face Space Secrets.")
    elif not os.path.exists("knowledge_base.pdf"):
        print("ERROR: 'knowledge_base.pdf' not found in the directory.")
    else:
        app = build_graph()
        print("\nWelcome to the AI Support Assistant (Powered by Groq)! (Type 'quit' or 'exit' to stop)")
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["quit", "exit"]:
                break
                
            initial_state = {"question": user_input, "requires_human": False, "context": [], "answer": ""}
            result = app.invoke(initial_state)
            print(f"\nAssistant: {result['answer']}")
