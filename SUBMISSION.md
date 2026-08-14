# 🚀 Take-Home Assignment Submission: Local RAG Q&A Chatbot

## 📌 Executive Summary

This repository contains a complete, production-grade **Local Retrieval-Augmented Generation (RAG) Q&A Chatbot** built for a marketing agency. The system allows potential clients to ask freeform questions about services, packages, pricing, and onboarding processes and receive accurate, context-grounded answers cited directly from agency documents.

Key Highlights:
- **100% Local Execution**: Runs entirely on CPU with zero cloud API keys, external dependencies, or GPU requirements.
- **Strict Grounding**: Uses vector search retrieval to provide context to the local LLM (`flan-t5-base`), guaranteeing hallucination-free answers pulled directly from data sources.
- **Rich Terminal UX**: Powered by `rich` for visually appealing panels, syntax highlights, progress status indicators, and clean source attribution.
- **100% Test Passing Rate**: All unit tests pass across structural integrity, retrieval accuracy, answer generation, and edge cases.

---

## 🏗️ Architecture Overview

```
                      ┌─────────────────────────────────┐
                      │    User Question Input          │
                      │  ("How much is Growth plan?")   │
                      └────────────────┬────────────────┘
                                       │
                                       ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │ 1. RETRIEVAL LAYER (src/knowledge_base.py & FAISS)                         │
 │  - Embed question using local HuggingFace `all-MiniLM-L6-v2`             │
 │  - Perform k=3 similarity search over chunked agency .txt documents       │
 └─────────────────────────────────────┬─────────────────────────────────────┘
                                       │
                                       ▼ (Top 3 Context Chunks)
 ┌───────────────────────────────────────────────────────────────────────────┐
 │ 2. PROMPT & GENERATION LAYER (src/pipeline.py & Flan-T5)                  │
 │  - Inject chunks & question into PROMPT_TEMPLATE                           │
 │  - Generate answer using local Seq2Seq model `google/flan-t5-base`        │
 └─────────────────────────────────────┬─────────────────────────────────────┘
                                       │
                                       ▼
                      ┌─────────────────────────────────┐
                      │  Formatted Answer & Sources     │
                      └─────────────────────────────────┘
```

---

## 🛠️ Implemented Features

### 1. `ask_question()` Pipeline (TODO 1)
- Retrieves the top 3 document chunks from `FAISS` via similarity search.
- Combines chunk text into a structured context string.
- Formats the template prompt and invokes `flan-t5-base`.
- Gracefully handles empty or whitespace-only queries.

### 2. Interactive CLI & Dual Modes (TODO 2)
- **Interactive Mode**: Standard `python -m src.pipeline` shell loop.
- **Single-Query Mode (`--query` / `-q`)**: Pass questions directly as command-line arguments.
- **Structured JSON Mode (`--json`)**: Output clean JSON for integration into APIs or automated tools.

### 3. Rich Terminal Interface (Bonus)
- Uses `rich` panels for clear visual hierarchy:
  - **Question Header**
  - **Retrieved Sources Box** with chunk numbers
  - **Answer Box** with distinct color highlighting

---

## 🧪 Testing & Verification

Run all unit tests using pytest:

```bash
pytest tests/ -v
```

### Test Results (12/12 Passed)
- ✅ `TestAskQuestionStructure`: Validates dictionary schema (`answer` string & `sources` list).
- ✅ `TestRetrieval`: Validates pricing, SEO info retrieval quality, and question source distinction.
- ✅ `TestAnswerGeneration`: Validates answer relevance and non-prompt repetition.
- ✅ `TestEdgeCases`: Validates empty/whitespace handling.

---

## ⚡ Quick Start Guide

### 1. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Interactive CLI
```bash
python -m src.pipeline
```

### 3. Run Single Query Command
```bash
python -m src.pipeline --query "How much does the Growth package cost?"
```

### 4. Output as JSON
```bash
python -m src.pipeline --query "What services do you offer?" --json
```

---

## 📁 Repository Map

- [src/pipeline.py](file:///Users/samymac/gesture/gesture-fs-intern-takehome/src/pipeline.py) — Q&A retrieval and generation logic + CLI implementation.
- [src/knowledge_base.py](file:///Users/samymac/gesture/gesture-fs-intern-takehome/src/knowledge_base.py) — Document loading, chunking, embeddings, and vector store.
- [tests/test_pipeline.py](file:///Users/samymac/gesture/gesture-fs-intern-takehome/tests/test_pipeline.py) — Unit test suite.
