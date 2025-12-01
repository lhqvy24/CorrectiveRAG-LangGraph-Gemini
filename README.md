# CorrectiveRAG-LangGraph-Gemini

This project implements a full Corrective RAG (Retrieval-Augmented Generation) pipeline using LangGraph, LangChain, Chroma, and Google Gemini.
It was built as part of an academic assignment to demonstrate multi-step RAG reasoning, iterative correction, and structured evaluation.

⸻

🎯 1. Project Overview

The system answers a user question by:
	1.	Performing an initial RAG answer
	2.	Judging whether the answer is complete, accurate, and grounded
	3.	Rewriting the question when the answer fails
	4.	Re-retrieving documents with the improved query
	5.	Generating a corrected final answer with citations

The workflow is implemented using LangGraph with clearly defined nodes, state management, and conditional edges.

⸻

🎓 2. Learning Objectives

This project demonstrates:
	•	How LangGraph uses state, nodes, and conditional routing
	•	How to build a multi-step Corrective RAG workflow
	•	How to integrate retrieval, LLM judgment, and query rewriting
	•	How to generate grounded answers with proper citations
	•	How to handle strict JSON evaluation for LLM judges
	•	How to evaluate RAG performance based on answer completeness

⸻

📘 3. Knowledge Sources

The project retrieves information from a local vector store built from 6 official LangGraph/LangChain documentation pages, including:
	•	LangGraph Overview
	•	Graph API
	•	Workflows & Agents
	•	LangChain Retrieval
	•	LangChain RAG Tutorial
	•	LangChain Agents

Each chunk includes metadata:

{
  "source_url": "...",
  "section_title": "..."
}

These are used to produce clear, traceable citations.

⸻

🧩 4. Workflow Nodes

The Corrective RAG pipeline consists of five nodes:

✔ initial_rag

Retrieves documents and produces a draft answer.

✔ judge

Evaluates the draft answer with a strict JSON-based scoring rubric:
	•	complete?
	•	grounded?
	•	relevant?
	•	hallucination-free?

✔ rewrite_query

Rewrites the question based on failure reasons.

✔ reretrieve_and_answer

Uses a stronger retriever (k=8) to answer again.

✔ finalize

Generates the final answer with:
	•	citations
	•	decision log
	•	rewritten queries
	•	draft answer history

⸻

🔀 5. Graph Routing

Conditional edges determine the flow:

initial_rag → judge → (rewrite | finalize)
rewrite → reretrieve_and_answer → judge → ...

The system retries once (MAX_ATTEMPTS=2), then finalizes.

⸻

📦 6. Vector Store Construction
	•	Loads .html, .htm, .txt, .md files
	•	Converts HTML → clean text via BeautifulSoup
	•	Splits into chunks using RecursiveCharacterTextSplitter
	•	Saves vector store to Chroma with BCE embeddings

⸻

🧪 7. Running the Demo

python corrective_rag.py

If no vector store exists, the script builds one automatically.

You will be prompted:

Enter your question:

The system then runs through the full Corrective RAG workflow and prints the final structured answer.

⸻

🧵 8. Self-Verification Questions

The project demonstrates understanding of:
	1.	Difference between Node and Edge in LangGraph
	2.	Importance of State in workflows
	3.	How conditional routing works
	4.	Why we use MemorySaver checkpointer
	5.	Components of a standard RAG chain

⸻

📚 9. Tech Stack
	•	Python 3.10+
	•	LangGraph / LangChain
	•	Google Gemini
	•	ChromaDB
	•	HuggingFace BCE Embeddings
	•	BeautifulSoup4

⸻

📌 10. Repository Structure

CorrectiveRAG-LangGraph-Gemini/
  ├── corrective_rag.py
  ├── docs/langgraph/        # downloaded docs
  ├── vectorstore/langgraph/ # auto-generated
  ├── README.md


⸻

🙌 11. Acknowledgments

This project is part of academic coursework for LangGraph and RAG systems, demonstrating structured AI evaluation and iterative correction.
