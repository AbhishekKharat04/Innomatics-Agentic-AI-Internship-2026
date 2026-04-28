import argparse
import json
import math
import re
import sys
import warnings
from pathlib import Path
from typing import List, Literal, TypedDict

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PDF_PATH = BASE_DIR / "knowledge_base.pdf"
DEFAULT_DB_DIR = BASE_DIR / ".chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2:1b"
TOP_K = 3
MIN_RELEVANCE_SCORE = 0.45
FALLBACK_MIN_RELEVANCE_SCORE = 0.20
ESCALATION_KEYWORDS = {
    "agent",
    "cancel order",
    "chargeback",
    "complaint",
    "escalate",
    "human",
    "lawsuit",
    "legal",
    "manager",
    "refund",
    "speak to someone",
    "talk to someone",
}


class AgentState(TypedDict, total=False):
    question: str
    route: Literal["rag", "hitl"]
    context: List[Document]
    answer: str
    requires_human: bool
    sources: List[str]
    relevance_scores: List[float]
    escalation_reason: str


class LocalHashEmbeddings:
    """Offline fallback embedding model based on hashed token frequencies."""

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimensions
        for token in self._tokenize(text):
            vector[hash(token) % self.dimensions] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


class SimpleVectorStore:
    """Minimal in-memory fallback when Chroma is unavailable in the runtime."""

    def __init__(self, documents: List[Document], embeddings) -> None:
        self.documents = documents
        self.embeddings = embeddings
        self.vectors = embeddings.embed_documents([doc.page_content for doc in documents])

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        numerator = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return numerator / (norm_a * norm_b)

    def similarity_search_with_relevance_scores(self, query: str, k: int = TOP_K):
        query_vector = self.embeddings.embed_query(query)
        scored = [
            (doc, self._cosine_similarity(query_vector, doc_vector))
            for doc, doc_vector in zip(self.documents, self.vectors)
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]


def format_source(doc: Document) -> str:
    source = Path(str(doc.metadata.get("source", "knowledge_base.pdf"))).name
    page = doc.metadata.get("page")
    if page is None:
        return source
    return f"{source} (page {int(page) + 1})"


class LocalRAGAssistant:
    def __init__(
        self,
        pdf_path: Path = DEFAULT_PDF_PATH,
        persist_directory: Path = DEFAULT_DB_DIR,
        collection_name: str = "rag_support",
    ) -> None:
        self.pdf_path = pdf_path
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_backend = EMBEDDING_MODEL
        self.vector_backend = "Chroma"
        self.embeddings = self._create_embeddings()
        self.vectorstore = self._load_or_create_vectorstore()
        self.graph = self._build_graph()

    def _create_embeddings(self):
        try:
            return HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"local_files_only": True},
            )
        except Exception:
            self.embedding_backend = "LocalHashEmbeddings"
            return LocalHashEmbeddings()

    def _build_documents(self) -> List[Document]:
        documents = PyPDFLoader(str(self.pdf_path)).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_documents(documents)

    def _load_or_create_vectorstore(self):
        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"Knowledge base not found at {self.pdf_path}. Add a PDF before running the assistant."
            )

        chunks = self._build_documents()

        try:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory),
            )

            existing_ids = vectorstore.get(limit=1).get("ids", [])
            if not existing_ids:
                vectorstore.add_documents(chunks)
            return vectorstore
        except Exception:
            self.vector_backend = "SimpleVectorStore"
            return SimpleVectorStore(chunks, self.embeddings)

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("router", self.router_node)
        workflow.add_node("rag", self.retrieve_and_generate_node)
        workflow.add_node("hitl", self.hitl_node)
        workflow.set_entry_point("router")
        workflow.add_conditional_edges(
            "router",
            self.route_after_router,
            {"rag": "rag", "hitl": "hitl"},
        )
        workflow.add_edge("rag", END)
        workflow.add_edge("hitl", END)
        return workflow.compile()

    def router_node(self, state: AgentState) -> AgentState:
        question = state["question"].lower()
        requires_human = any(keyword in question for keyword in ESCALATION_KEYWORDS)

        if requires_human:
            return {
                "route": "hitl",
                "requires_human": True,
                "escalation_reason": "Sensitive or action-oriented request detected.",
            }

        return {"route": "rag", "requires_human": False}

    def route_after_router(self, state: AgentState) -> str:
        return state["route"]

    def retrieve_and_generate_node(self, state: AgentState) -> AgentState:
        question = state["question"]
        results = self.vectorstore.similarity_search_with_relevance_scores(question, k=TOP_K)

        if not results:
            return self.hitl_node(
                {
                    **state,
                    "escalation_reason": "No relevant chunks were found in the knowledge base.",
                }
            )

        context_docs = [doc for doc, _score in results]
        relevance_scores = [round(float(score), 3) for _doc, score in results]
        best_score = max(relevance_scores)
        confidence_threshold = (
            FALLBACK_MIN_RELEVANCE_SCORE
            if self.vector_backend == "SimpleVectorStore"
            else MIN_RELEVANCE_SCORE
        )

        if best_score < confidence_threshold:
            return self.hitl_node(
                {
                    **state,
                    "context": context_docs,
                    "relevance_scores": relevance_scores,
                    "sources": [format_source(doc) for doc in context_docs],
                    "escalation_reason": (
                        f"Retrieved context confidence is too low (best score: {best_score:.3f})."
                    ),
                }
            )

        context_text = "\n\n".join(doc.page_content for doc in context_docs)
        answer = self.generate_answer(question, context_text)
        sources = [format_source(doc) for doc in context_docs]

        return {
            "context": context_docs,
            "answer": answer,
            "sources": sources,
            "relevance_scores": relevance_scores,
            "requires_human": False,
        }

    def generate_answer(self, question: str, context_text: str) -> str:
        prompt = PromptTemplate.from_template(
            "You are a customer support assistant.\n"
            "Answer the question using only the supplied context.\n"
            "If the context is insufficient, say that the case should be escalated to a human agent.\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Answer:"
        )

        try:
            llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
            response = (prompt | llm).invoke({"context": context_text, "question": question})
            return response.content.strip()
        except Exception as exc:
            preview = self.extractive_fallback(question, context_text)
            return (
                "LLM generation is unavailable, so an extractive fallback response is shown.\n\n"
                f"{preview}\n\n"
                f"Runtime note: {exc}\n"
                f"Embedding backend: {self.embedding_backend}\n"
                f"Vector backend: {self.vector_backend}"
            )

    def extractive_fallback(self, question: str, context_text: str) -> str:
        question_tokens = set(re.findall(r"[a-zA-Z0-9]+", question.lower()))
        candidates = [
            line.strip()
            for line in re.split(r"[\n\r]+", context_text)
            if line.strip()
        ]
        ranked = []
        for line in candidates:
            line_tokens = set(re.findall(r"[a-zA-Z0-9]+", line.lower()))
            overlap = len(question_tokens & line_tokens)
            if overlap:
                ranked.append((overlap, line))

        ranked.sort(key=lambda item: item[0], reverse=True)
        if ranked:
            best_lines = [line for _score, line in ranked[:3]]
            return "Best matching context:\n" + "\n".join(f"- {line}" for line in best_lines)

        return "Best matching context:\n- " + context_text[:700].strip()

    def hitl_node(self, state: AgentState) -> AgentState:
        reason = state.get("escalation_reason", "The request needs human review.")
        return {
            "answer": (
                "This request has been escalated to a human support agent.\n"
                f"Reason: {reason}"
            ),
            "requires_human": True,
            "sources": state.get("sources", []),
            "relevance_scores": state.get("relevance_scores", []),
        }

    def ask(self, question: str) -> AgentState:
        initial_state: AgentState = {
            "question": question,
            "context": [],
            "answer": "",
            "requires_human": False,
            "sources": [],
            "relevance_scores": [],
            "route": "rag",
            "escalation_reason": "",
        }
        return self.graph.invoke(initial_state)


def print_result(result: AgentState) -> None:
    print("\nAssistant:")
    print(result["answer"])

    sources = result.get("sources", [])
    if sources:
        print("\nSources:")
        for source in sources:
            print(f"- {source}")

    scores = result.get("relevance_scores", [])
    if scores:
        print(f"\nRelevance scores: {scores}")

    print(f"Escalated: {result.get('requires_human', False)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RAG customer support assistant.")
    parser.add_argument("--question", type=str, help="Ask a single question and exit.")
    parser.add_argument("--json", action="store_true", help="Return the response as JSON.")
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF_PATH,
        help="Path to the PDF knowledge base.",
    )
    parser.add_argument(
        "--db-dir",
        type=Path,
        default=DEFAULT_DB_DIR,
        help="Directory used to persist the Chroma vector store.",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")

    args = parse_args()
    assistant = LocalRAGAssistant(pdf_path=args.pdf, persist_directory=args.db_dir)

    if args.question:
        result = assistant.ask(args.question)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, default=str))
        else:
            print_result(result)
        return

    print("Local RAG Customer Support Assistant")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        if not user_input:
            continue
        print_result(assistant.ask(user_input))
        print()


if __name__ == "__main__":
    main()
