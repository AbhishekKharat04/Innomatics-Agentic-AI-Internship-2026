# Low-Level Design (LLD): RAG-Based Customer Support Assistant

## 1. Module-Level Design
- **Document Processing Module:** Uses `PyPDFLoader` to parse PDFs into text objects.
- **Chunking Module:** Utilizes `RecursiveCharacterTextSplitter` with `chunk_size=1000` and `chunk_overlap=200` to preserve boundary context.
- **Embedding Module:** Implements `OpenAIEmbeddings` (or `HuggingFaceEmbeddings`) to convert chunks into 1536-dimensional float arrays.
- **Vector Storage Module:** Initializes a persistent `Chroma` client. Contains methods for `add_documents`, `get`, and `similarity_search_with_relevance_scores`.
- **Retrieval Module:** Queries Chroma for the top 3 most relevant chunks and records their relevance scores for routing decisions.
- **Query Processing Module:** LLM prompt template that takes `{context}` and `{question}` to generate the final string output. If Ollama is unavailable, a retrieval-only fallback response is returned gracefully.
- **Graph Execution Module:** Defines a `StateGraph` using LangGraph. Compiles the graph and exposes an `invoke` method.
- **HITL Module:** A specialized node in the graph that uses LangGraph's `interrupt` feature or simply returns a structured response prompting a human to take over the `State`.

## 2. Data Structures
- **Document Representation:** LangChain `Document` object: `{"page_content": str, "metadata": {"source": str, "page": int}}`.
- **Chunk Format:** Same as Document Representation, but `page_content` is limited to the chunk size constraint.
- **Embedding Structure:** List of floats `[0.012, -0.045, ...]`.
- **Query-Response Schema:** 
  ```python
  class QueryState(TypedDict):
      question: str
      route: Literal["rag", "hitl"]
      context: List[Document]
      answer: str
      requires_human: bool
      sources: List[str]
      relevance_scores: List[float]
      escalation_reason: str
  ```
- **State Object for Graph:** Uses the `QueryState` dictionary to pass data continuously between nodes.

## 3. Workflow Design (LangGraph)
- **Nodes:**
  - `router_node(state)`: Applies keyword-based intent screening and sets the next route.
  - `retrieve_and_generate_node(state)`: Retrieves scored chunks, evaluates confidence, queries the LLM, and updates the answer plus sources.
  - `hitl_escalation_node(state)`: Flags the conversation for human intervention and updates `state["answer"]` with an escalation message and reason.
- **Edges:**
  - `START -> router_node`
  - `router_node -> retrieve_and_generate_node` (Condition: standard query)
  - `router_node -> hitl_escalation_node` (Condition: requires human)
  - `retrieve_and_generate_node -> END`
  - `hitl_escalation_node -> END`
- **State:** The `QueryState` flows through these edges. Each node returns a dictionary updating specific keys in the state.

## 4. Conditional Routing Logic
A lightweight LLM call or rule-based router evaluates the `state["question"]` before main processing.
- **Escalation criteria:**
  - If the query contains keywords like "talk to human", "manager", "complaint", or "refund".
  - If the retriever returns no chunks.
  - If the best retrieval relevance score is below the confidence threshold.
  - If the query is action-oriented and cannot be solved safely through static document lookup.
- **Answer generation criteria:** Standard factual queries about the product/document (e.g., "What are the support hours?").

## 5. HITL Design
- **When escalation is triggered:** The router outputs `requires_human=True`, or the RAG node triggers escalation after a low-confidence retrieval result.
- **What happens after escalation:** The graph execution goes to `hitl_escalation_node`. In a production app, this node would create a support ticket or send the conversation to a dashboard for manual handling.
- **Integration:** The human agent would review the original question, retrieved context, and the escalation reason before responding manually.

## 6. API / Interface Design
- **Input Format:** CLI text input or `python rag_agent.py --question "How do I reset my password?"`
- **Output Format:** 
  ```json
  {
    "answer": "To reset your password, go to settings...",
    "escalated": false,
    "sources": ["knowledge_base.pdf (page 2)", "knowledge_base.pdf (page 4)"],
    "relevance_scores": [0.81, 0.77, 0.74]
  }
  ```
- **Interaction Flow:** User submits a question through the CLI. The application initializes graph state, invokes the workflow, and prints the answer, sources, scores, and escalation status.

## 7. Error Handling
- **Missing Data:** If the PDF is empty or missing, the system catches `FileNotFoundError` and returns a clear error message.
- **No Relevant Chunks Found:** If ChromaDB similarity scores fall below a threshold, the system triggers the HITL logic automatically (low confidence).
- **LLM Failure:** Implement retry logic (e.g., `Tenacity` or LangChain's built-in fallbacks) to handle API timeouts or rate limits gracefully.
