import os
import json
from pathlib import Path
from typing import List, Dict, Any, Literal

from dotenv import load_dotenv
from bs4 import BeautifulSoup

# --- LangChain / LangGraph imports ---
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig

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

# Khởi tạo Global LLM và Embedder (Hiệu suất cao hơn)
def get_llm(model: str = "gemini-2.5-flash") -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=GEMINI_API_KEY,
        temperature=0.2,
    )

LLM = get_llm()
EMBEDDER = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
MAX_ATTEMPTS = 2 # Giới hạn số lần thử lại tối đa (1 initial + 1 retry)


# ========== 1. DOCS → VECTOR STORE (HTML SUPPORTED) ==========

DOCS_DIR = Path("docs/langgraph")
VS_DIR = "vectorstore/langgraph"

# Official URLs to attach as metadata
URL_MAP = {
    "overview": "https://docs.langchain.com/oss/python/langgraph/overview",
    "graph-api": "https://docs.langchain.com/oss/python/langgraph/graph-api",
    "workflows-agents": "https://docs.langchain.com/oss/python/langgraph/workflows-agents",
    "retrieval": "https://docs.langchain.com/oss/python/langchain/retrieval",
    "rag": "https://docs.langchain.com/oss/python/langchain/rag",
    "agents": "https://docs.langchain.com/oss/python/langchain/agents",
}

def load_docs() -> List[Document]:
    """Load docs (.html/.htm/.md/.txt), convert HTML → text, split, add metadata."""
    if not DOCS_DIR.exists():
        raise FileNotFoundError("docs/langgraph/ not found. Please put your 6 docs there.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    docs: List[Document] = []

    for f in DOCS_DIR.glob("*"):

        if f.suffix.lower() not in [".html", ".htm", ".md", ".txt"]:
            continue

        raw = f.read_text(encoding="utf-8", errors="ignore")

        if f.suffix.lower() in [".html", ".htm"]:
            soup = BeautifulSoup(raw, "html.parser")
            # Cải tiến: Chỉ lấy text trong thẻ body để loại bỏ header/footer không liên quan
            text = soup.body.get_text(separator="\n") if soup.body else soup.get_text(separator="\n")
        else:
            text = raw

        chunks = splitter.split_text(text)
        
        # Logic trích xuất metadata
        first_line = text.splitlines()[0] if text.splitlines() else f.stem
        section_title = first_line.strip().lstrip("# ").strip() or f.stem
        source_url = URL_MAP.get(f.stem, f.name)

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

    Chroma.from_documents(
        docs,
        embedding=EMBEDDER, # SỬ DỤNG GLOBAL EMBEDDER
        persist_directory=VS_DIR,
    )
    print(f"Vector store built at {VS_DIR}")


def get_retriever(stronger: bool = False):
    """Load Chroma store and return retriever."""
    # SỬ DỤNG GLOBAL EMBEDDER
    vs = Chroma(
        persist_directory=VS_DIR,
        embedding_function=EMBEDDER
    )
    k = 4 if not stronger else 8
    # Cài đặt bộ tìm kiếm với k chunks
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
    # Lọc các trích dẫn độc nhất (unique citations)
    unique_cits = {}
    for d in docs:
        url = d.metadata.get("source_url", "unknown")
        # Sử dụng URL làm khóa để đảm bảo chỉ lấy 1 lần cho mỗi nguồn
        unique_cits[url] = {
            "source_url": url,
            "section_title": d.metadata.get("section_title", "unknown"),
        }
    return list(unique_cits.values())


def rag_answer(question: str, stronger: bool = False):
    # Sử dụng LLM và Retriever đã được khởi tạo
    retriever = get_retriever(stronger)

    docs = retriever.invoke(question)
    context = format_docs(docs)
    prompt = RAG_PROMPT.format(context=context, question=question)
    resp = LLM.invoke(prompt)

    answer_text = resp.content if isinstance(resp.content, str) else str(resp.content)
    citations = extract_citations(docs)

    return answer_text, citations


def llm_judge(question: str, answer: str, citations: List[Dict[str, str]]) -> Dict[str, Any]:
    # Sử dụng LLM đã khởi tạo
    
    # Cải tiến: Yêu cầu cú pháp JSON mạnh mẽ hơn
    judge_prompt = f"""
You are a strict judge for a Corrective RAG system.

Given:
- User question
- Draft answer
- Citations (Sources used for the answer)

Judge the answer on its sufficiency, grounding, and relevance.
Your response MUST be a single, valid JSON object with the following schema:
{{"pass": boolean, "reasons": string, "score": integer}}
The "pass" field is TRUE only if the answer is complete, well-grounded by the citations, and directly addresses the question. Otherwise, it is FALSE.

Question:
{question}

Draft answer:
{answer}

Citations:
{json.dumps(citations, indent=2)}

Return ONLY JSON. Do not add any conversational text or markdown code blocks (e.g., ```json).
"""
    resp = LLM.invoke(judge_prompt)
    raw = resp.content.strip()

    # Cải tiến: Xử lý lỗi JSON parsing (loại bỏ các ký tự Markdown không cần thiết)
    if raw.startswith("```json"):
        raw = raw.strip("`").strip("json").strip()
    elif raw.startswith("```"):
        raw = raw.strip("`").strip()

    try:
        verdict = json.loads(raw)
    except Exception as e:
        print(f"ERROR: Could not parse judge JSON. Fallback to FAIL. Error: {e}")
        verdict = {"pass": False, "reasons": f"Failed to parse judge output: {raw[:50]}...", "score": 1}

    # Đảm bảo các trường cần thiết tồn tại
    verdict.setdefault("pass", False)
    verdict.setdefault("reasons", "No reasons provided.")
    verdict.setdefault("score", 2)
    return verdict


def llm_rewrite(original_question: str, reasons: str) -> str:
    # Sử dụng LLM đã khởi tạo

    rewrite_prompt = f"""
Rewrite this question to be clearer and specifically address the missing information or error.
Focus on creating a better search query to retrieve relevant documents.

Original question:
{original_question}

The answer failed because of these issues:
{reasons}

Return ONLY the single, rewritten question text.
"""

    resp = LLM.invoke(rewrite_prompt)
    return resp.content.strip()


def format_final_answer(
    final_answer: str,
    final_citations: List[Dict[str, str]],
    judgements: List[Dict[str, Any]],
    query_variants: List[str],
) -> str:

    lines = []
    lines.append("### ✅ Final Answer\n")
    lines.append(final_answer.strip())

    lines.append("\n\n---\n### 📚 Citations\n")
    for i, c in enumerate(final_citations, 1):
        lines.append(f"{i}. **{c.get('section_title')}** – {c.get('source_url')}")

    lines.append("\n---\n### 📝 Decision Log\n")
    for i, j in enumerate(judgements, 1):
        lines.append(f"- **Attempt {i}**: {'PASS' if j['pass'] else 'FAIL'} (Score: {j['score']}/5) – {j['reasons']}")

    if query_variants:
        lines.append("\n### 🔄 Rewritten Questions")
        for q in query_variants:
            lines.append(f"- {q}")

    return "\n".join(lines)


# ========== 3. STATE CLASS ==========

class RAGState(TypedDict):
    # Tự định nghĩa messages thay vì dùng MessagesState
    messages: Annotated[List[Any], add_messages] 
    question: str
    draft_answers: List[str]
    citations: List[List[Dict[str, str]]]
    judgements: List[Dict[str, Any]]
    query_variants: List[str]
    attempts: int


# ========== 4. NODES ==========

def initial_rag(state: RAGState):
    """Node 1: Thực hiện RAG lần đầu và tạo câu trả lời nháp."""
    draft, cits = rag_answer(state["question"])
    return {
        "draft_answers": state.get("draft_answers", []) + [draft],
        "citations": state.get("citations", []) + [cits],
        "attempts": 1, # KHỞI TẠO BỘ ĐẾM
    }

def judge(state: RAGState):
    """Node 2: Đánh giá chất lượng của câu trả lời nháp cuối cùng."""
    verdict = llm_judge(
        state["question"],
        state["draft_answers"][-1],
        state["citations"][-1],
    )
    return {"judgements": state.get("judgements", []) + [verdict]}

def rewrite_query(state: RAGState):
    """Node 3: Viết lại truy vấn dựa trên lý do thất bại."""
    # Lấy lý do thất bại từ phán quyết cuối cùng
    reasons = state["judgements"][-1].get("reasons", "Incomplete or irrelevant answer.")
    new_q = llm_rewrite(state["question"], reasons)
    
    # Cập nhật trạng thái: Đặt câu hỏi mới vào trường question
    return {
        "question": new_q, 
        "query_variants": state.get("query_variants", []) + [new_q],
    }

def reretrieve_and_answer(state: RAGState):
    """Node 4: Truy xuất lại tài liệu với truy vấn mới và trả lời lần 2."""
    # Sử dụng `stronger=True` (k=8) cho lần truy xuất thứ hai
    draft, cits = rag_answer(state["question"], stronger=True)
    return {
        "draft_answers": state["draft_answers"] + [draft],
        "citations": state["citations"] + [cits],
        "attempts": state["attempts"] + 1, # TĂNG BỘ ĐẾM
    }

def finalize(state: RAGState):
    """Node 5: Định dạng câu trả lời cuối cùng với trích dẫn và nhật ký."""
    content = format_final_answer(
        final_answer=state["draft_answers"][-1],
        final_citations=state["citations"][-1],
        judgements=state["judgements"],
        query_variants=state.get("query_variants", []),
    )
    # Thêm câu trả lời cuối cùng vào lịch sử tin nhắn
    return {
        "messages": state.get("messages", []) + [{
            "type": "ai",
            "content": content,
        }]
    }


# ========== 5. ROUTER (Conditional Edge) ==========

def route_on_judge(state: RAGState) -> Literal["rewrite", "finalize"]:
    """Định tuyến luồng dựa trên phán quyết của judge và số lần thử."""
    
    # 1. Nếu judge đánh giá PASS, dừng lại
    if state["judgements"][-1]["pass"]:
        print("DEBUG: Judge PASS. Finalizing answer.")
        return "finalize"
    
    # 2. Nếu đã đạt giới hạn lần thử, dừng lại để tránh vòng lặp vô tận
    if state["attempts"] >= MAX_ATTEMPTS:
        print(f"DEBUG: Max attempts ({MAX_ATTEMPTS}) reached. Finalizing current answer.")
        return "finalize"
    
    # 3. Nếu thất bại và chưa đạt giới hạn, thử lại
    print("DEBUG: Judge FAIL. Rewriting query and retrying.")
    return "rewrite"


# ========== 6. BUILD GRAPH APP ==========

def build_app():
    """Xây dựng và biên dịch LangGraph."""
    graph = StateGraph(RAGState)

    # 1. Thêm các Node xử lý
    graph.add_node("initial_rag", initial_rag)
    graph.add_node("judge", judge)
    graph.add_node("rewrite", rewrite_query)
    graph.add_node("reretrieve_and_answer", reretrieve_and_answer)
    graph.add_node("finalize", finalize)

    # 2. Thiết lập Edges (Đường đi cố định)
    graph.add_edge(START, "initial_rag")
    graph.add_edge("initial_rag", "judge")
    graph.add_edge("rewrite", "reretrieve_and_answer")
    graph.add_edge("reretrieve_and_answer", "judge") # Quay lại judge sau khi thử lại

    # 3. Thiết lập Conditional Edge (Đường đi có điều kiện)
    graph.add_conditional_edges(
        "judge",
        route_on_judge,
        {
            "rewrite": "rewrite",
            "finalize": "finalize",
        }
    )

    graph.add_edge("finalize", END)

    # Biên dịch đồ thị và thêm Checkpointer (MemorySaver)
    return graph.compile(checkpointer=MemorySaver())


# ========== 7. DEMO RUN ==========

if __name__ == "__main__":

    # Kiểm tra và xây dựng Vector Store nếu chưa tồn tại
    if not Path(VS_DIR).exists():
        print("Vector store not found. Building...")
        build_vectorstore()
    else:
        print("Vector store exists. Skipping build.")

    app = build_app()

    print("\n=== Corrective RAG Demo ===")
    user_question = input("Enter your question: ")

    # Chuẩn bị trạng thái khởi tạo
    init_state = {
        "messages": [{"type": "human", "content": user_question}],
        "question": user_question,
        "draft_answers": [],
        "citations": [],
        "judgements": [],
        "query_variants": [],
        "attempts": 0, # Khởi tạo là 0
    }

    # Cấu hình thread ID để sử dụng MemorySaver
    config = RunnableConfig(configurable={"thread_id": "demo-thread"})
    final = app.invoke(init_state, config=config)

    print("\n========== FINAL OUTPUT ==========\n")
    
    last_msg = final["messages"][-1]
    
    # Kiểm tra xem nó là Object hay Dict để xử lý phù hợp
    if hasattr(last_msg, "content"):
        print(last_msg.content)
    else:
        print(last_msg.get("content"))
        
    print("\n=================================\n")