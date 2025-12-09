import os
import json
import requests
import operator
from operator import add
from pathlib import Path
from typing import List, Dict, Any, Literal
from markdownify import markdownify as md_converter

from dotenv import load_dotenv
from bs4 import BeautifulSoup

# --- LangChain / LangGraph imports ---
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_huggingface import HuggingFaceEmbeddings

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, List, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


# ========== 0. ENV, LLM & Global Constants ==========

load_dotenv()
GEMINI_API_KEY = os.getenv("gg_api_key")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in .env")

# Initialize Global LLM and Embedder
def get_llm(model: str = "models/gemma-3-12b-it") -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=GEMINI_API_KEY,
        temperature=0,
    )

LLM = get_llm()
EMBEDDER = HuggingFaceEmbeddings(
    model_name="maidalun1020/bce-embedding-base_v1",
    model_kwargs={'device': 'cpu'}, 
    encode_kwargs={'normalize_embeddings': True} 
)
MAX_ATTEMPTS = 2 

# --- PATH CONSTANTS ---
DOCS_DIR = Path("docs/langgraph")
VS_DIR = "vectorstore/langgraph"

# --- URLs TO DOWNLOAD ---
# I have updated the 'overview' URL to the correct current path
URLS_TO_DOWNLOAD = {
    "overview": "https://langchain-ai.github.io/langgraph/concepts/high_level/", 
    "graph-api": "https://langchain-ai.github.io/langgraph/concepts/low_level/", 
    "workflows-agents": "https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/", 
    "retrieval": "https://python.langchain.com/docs/concepts/retrieval/", 
    "rag": "https://python.langchain.com/docs/tutorials/rag/",
    "agents": "https://python.langchain.com/docs/concepts/agents/",
}


# ========== 1. DOWNLOADER & DOCS LOADING ==========

def download_sources_as_md():
    """Downloads the 6 required pages and saves them as .md files."""
    if not DOCS_DIR.exists():
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {DOCS_DIR}")

    print("⬇️  Starting download of sources...")

    for filename, url in URLS_TO_DOWNLOAD.items():
        file_path = DOCS_DIR / f"{filename}.md"
        
        # Skip if already exists
        if file_path.exists():
            print(f"   - Skipping {filename}.md (already exists)")
            continue

        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, "html.parser")
            
            # EXTRACT CONTENT: Try to find the main article content
            content_html = soup.find("article") or soup.find("main") or soup.body
            
            if content_html:
                # Convert to Markdown
                markdown_text = md_converter(str(content_html), heading_style="ATX")
                # Clean up empty lines
                markdown_text = "\n".join([line for line in markdown_text.splitlines() if line.strip()])
                
                file_path.write_text(markdown_text, encoding="utf-8")
                print(f"   ✅ Downloaded & Converted: {filename}.md")
            else:
                print(f"   ⚠️ Could not find main content for {url}")

        except Exception as e:
            print(f"   ❌ Error downloading {url}: {e}")

    print("🏁 Download complete.\n")


def load_docs() -> List[Document]:
    """Load .md files from docs/langgraph/ and add metadata."""
    if not DOCS_DIR.exists():
        download_sources_as_md()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs: List[Document] = []

    for f in DOCS_DIR.glob("*.md"):
        text = f.read_text(encoding="utf-8")
        
        # Get URL from map
        source_url = URLS_TO_DOWNLOAD.get(f.stem, "unknown_source")
        
        # Simple Title Extraction
        section_title = f.stem.replace("-", " ").title()
        for line in text.splitlines():
            if line.startswith("# "):
                section_title = line.strip("# ").strip()
                break

        chunks = splitter.split_text(text)
        
        for chunk in chunks:
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source_url": source_url,
                        "section_title": section_title,
                    },
                )
            )

    return docs


def build_vectorstore():
    """Build Chroma vector store with metadata."""
    docs = load_docs()
    print(f"Loaded {len(docs)} chunks from docs/langgraph/")

    if docs:
        Chroma.from_documents(
            docs,
            embedding=EMBEDDER,
            persist_directory=VS_DIR,
        )
        print(f"Vector store built at {VS_DIR}")
    else:
        print("Warning: No documents loaded. Vector store not built.")


def get_retriever(stronger: bool = False):
    """Load Chroma store and return retriever."""
    vs = Chroma(
        persist_directory=VS_DIR,
        embedding_function=EMBEDDER
    )
    k = 4 if not stronger else 8
    return vs.as_retriever(search_kwargs={"k": k})


# ========== 2. RAG UTILITIES ==========

RAG_PROMPT = PromptTemplate.from_template(
    """You are a helpful assistant answering questions about LangChain and LangGraph.
Use ONLY the context below. If the context is not enough, say you are not sure.

Context:
{context}

Question:
{question}

Answer in clear English, 2–4 short paragraphs maximum.
"""
)

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))

def extract_citations(docs: List[Document]) -> List[Dict[str, str]]:
    unique_cits = {}
    for d in docs:
        url = d.metadata.get("source_url", "unknown")
        unique_cits[url] = {
            "source_url": url,
            "section_title": d.metadata.get("section_title", "unknown"),
        }
    return list(unique_cits.values())


def rag_answer(question: str, stronger: bool = False):
    retriever = get_retriever(stronger)
    docs = retriever.invoke(question)
    
    if not docs:
        return "I could not find any relevant documents.", [], ""

    context = format_docs(docs)
    prompt = RAG_PROMPT.format(context=context, question=question)
    resp = LLM.invoke(prompt)

    answer_text = resp.content if isinstance(resp.content, str) else str(resp.content)
    citations = extract_citations(docs)

    return answer_text, citations, context


def llm_judge(question: str, answer: str, citations: List[Dict[str, str]], context: str):
    judge_prompt = f"""
You are a strict judge for a RAG system performing a "Relevance Check".

User Question: {question}
Draft Answer: {answer}
Context provided: {context[:2000]}...

EVALUATION RULES:
1. DOES THE ANSWER ADDRESS THE QUESTION?
   - If the answer is helpful and extracted from context -> PASS = TRUE.
   - If the answer says "I don't know", "The context does not contain information", "I cannot answer", or similar -> PASS = FALSE.
   - If the answer is irrelevant or creates information not in context -> PASS = FALSE.

2. SCORING:
   - Pass = True -> Score 5
   - Pass = False -> Score 1

Output MUST be a valid JSON object: {{"pass": boolean, "reasons": string, "score": integer}}

Example 1 (Failure case):
Answer: "The provided text does not mention MemorySaver."
JSON: {{"pass": false, "reasons": "Model failed to find answer in context.", "score": 1}}

Example 2 (Success case):
Answer: "MemorySaver is used for persistence."
JSON: {{"pass": true, "reasons": "Answer correctly addresses the question.", "score": 5}}

Return ONLY JSON.
"""
    resp = LLM.invoke(judge_prompt)
    raw = resp.content.strip()

    # Xử lý format JSON từ LLM
    if raw.startswith("```json"):
        raw = raw.strip("`").strip("json").strip()
    elif raw.startswith("```"):
        raw = raw.strip("`").strip()

    try:
        verdict = json.loads(raw)
    except Exception as e:
        print(f"ERROR parsing judge JSON: {e}")
        # If parse error -> Treat as FAIL
        verdict = {"pass": False, "reasons": "JSON parse error", "score": 1}

    return verdict


def llm_rewrite(original_question: str, reasons: str) -> str:
    # Prompt instructs LLM "Abstraction" instead of Rephrase the question
    rewrite_prompt = f"""
You are a Corrective RAG Query Optimizer.

Your job is to rewrite the question so that:
- It becomes answerable using the available documentation.
- It maps the user's intent to the closest conceptual topic present in the documents.
- It stays faithful to the *meaning category* of the user question, but expressed in a more general and document-aligned way.
- It avoids hallucinated terms not present in the documentation.
- It should maximize retrieval quality by targeting a related concept that *does exist* in the docs.

Steps to follow:
1. Identify the core concept the user is asking about (e.g. persistence, node execution, message handling, retrieval logic, agent workflow).
2. Find the closest available concept in the documentation that could meaningfully address this.
3. Rewrite the question so that it:
   - aligns with the available concept,
   - stays truthful to the user’s intention at conceptual level,
   - increases likelihood of retrieving relevant text chunks.

Original question:
{original_question}

Reason the previous answer failed:
{reasons}

Return ONLY the rewritten question, no explanations.
"""
    resp = LLM.invoke(rewrite_prompt)
    return resp.content.strip()


def format_final_answer(final_answer, final_citations, judgements, query_variants):
    lines = ["### ✅ Final Answer\n", final_answer.strip()]
    
    lines.append("\n\n---\n### 📚 Citations\n")
    for i, c in enumerate(final_citations, 1):
        lines.append(f"{i}. **{c.get('section_title')}** – {c.get('source_url')}")

    lines.append("\n---\n### 📝 Decision Log\n")
    for i, j in enumerate(judgements, 1):
        lines.append(f"- **Attempt {i}**: {'PASS' if j.get('pass') else 'FAIL'} – {j.get('reasons')}")

    if query_variants:
        lines.append("\n### 🔄 Rewritten Questions")
        for q in query_variants:
            lines.append(f"- {q}")

    return "\n".join(lines)


# ========== 3. STATE & NODES ==========

class RAGState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    question: str
    # Use operator.add, when return it will be appended to the list
    draft_answers: Annotated[List[str], add] 
    citations: Annotated[List[List[Dict[str, str]]], add]
    context: Annotated[List[str], add]
    judgements: Annotated[List[Dict[str, Any]], add]
    query_variants: Annotated[List[str], add]
    attempts: int

def initial_rag(state: RAGState):
    draft, cits, context = rag_answer(state["question"])
    return {
        "draft_answers": [draft],
        "citations": [cits],
        "context": [context],
        "attempts": 1, 
    }

def judge(state: RAGState):
    verdict = llm_judge(
        state["question"],
        state["draft_answers"][-1],
        state["citations"][-1],
        state["context"][-1] 
    )
    return {"judgements": [verdict]}

def rewrite_query(state: RAGState):
    reasons = state["judgements"][-1].get("reasons", "Unknown")
    new_q = llm_rewrite(state["question"], reasons)
    return {
        "question": new_q, 
        "query_variants": [new_q],
    }

def reretrieve_and_answer(state: RAGState):
    draft, cits, context = rag_answer(state["question"], stronger=True)
    return {
        "draft_answers": state["draft_answers"] + [draft],
        "citations": state["citations"] + [cits],
        "context": state["context"] + [context],
        "attempts": state["attempts"] + 1,
    }

def finalize(state: RAGState):
    content = format_final_answer(
        state["draft_answers"][-1],
        state["citations"][-1],
        state["judgements"],
        state.get("query_variants", []),
    )
    return {
        "messages": [{"type": "ai", "content": content}]
    }

def route_on_judge(state: RAGState) -> Literal["rewrite", "finalize"]:
    if state["judgements"][-1].get("pass", False):
        return "finalize"
    if state["attempts"] >= MAX_ATTEMPTS:
        return "finalize"
    return "rewrite"


# ========== 4. APP BUILD & EXECUTION ==========

def build_app():
    graph = StateGraph(RAGState)
    graph.add_node("initial_rag", initial_rag)
    graph.add_node("judge", judge)
    graph.add_node("rewrite", rewrite_query)
    graph.add_node("reretrieve_and_answer", reretrieve_and_answer)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "initial_rag")
    graph.add_edge("initial_rag", "judge")
    graph.add_edge("rewrite", "reretrieve_and_answer")
    graph.add_edge("reretrieve_and_answer", "judge")

    graph.add_conditional_edges("judge", route_on_judge, {"rewrite": "rewrite", "finalize": "finalize"})
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    
    # 1. Download Sources (Check & Download)
    download_sources_as_md()

    # 2. Build Vector Store (Check & Build)
    if not Path(VS_DIR).exists():
        print("Vector store not found. Building...")
        build_vectorstore()
    else:
        print(f"Vector store exists at {VS_DIR}. Skipping build.")

    # 3. Run App
    app = build_app()

    print("\n=== Corrective RAG Demo ===")
    user_q = input("Enter your question: ")

    config = RunnableConfig(configurable={"thread_id": "demo-thread"})
    
    # Initialize state properly
    init_state = {
        "messages": [{"type": "human", "content": user_q}],
        "question": user_q,
        "draft_answers": [],
        "citations": [],
        "context": [],
        "judgements": [],
        "query_variants": [],
        "attempts": 0,
    }

    final = app.invoke(init_state, config=config)

    print("\n" + "="*40 + "\n")
    print(final["messages"][-1].content)
    print("\n" + "="*40 + "\n")