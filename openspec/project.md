# Project Context

## Purpose
Youtu-GraphRAG delivers a vertically unified GraphRAG framework that constructs hierarchical knowledge graphs, performs schema-aware retrieval, and answers multi-hop, knowledge-intensive questions with lower token cost and higher accuracy than prior systems (README.md:18,57-85,94).

## Tech Stack
- Python 3.10 with FastAPI, Uvicorn, and Pydantic powering the unified backend/websocket API (`backend.py:1-214`).
- Graph construction and retrieval agents implemented in `models/constructor/kt_gen.py` and `models/retriever/*` with NetworkX-style processing, FAISS, and custom query decomposition (README.md:129-145; requirements.txt).
- Static HTML/CSS frontend communicating via Axios and WebSockets, with ECharts visualizations rendered from `/frontend/index.html`.
- Docker-based deployment plus shell scripts (`setup_env.sh`, `start.sh`) for local installs (README.md:170-220).
- OpenAI-compatible LLM APIs configured through `.env` (LLM_MODEL/BASE_URL/API_KEY) to power extraction, reasoning, and evaluation (`.env.example`).

## Project Conventions

### Code Style
Python modules favor PEP 8 naming, type hints, and docstrings; Pydantic models define backend contracts, and configuration helpers live under `config/` to keep graph/LLM parameters in YAML rather than duplicating constants (`backend.py:20-214`, `config/`).

### Architecture Patterns
Layered pipeline: documents flow through schema-driven graph construction, multi-level community detection, FAISS-backed retrieval, and agentic reasoning before responses are surfaced via the FastAPI backend and ECharts UI (README.md:57-165, backend.py:32-214). Knowledge artifacts (schemas, graphs, chunks, logs) persist under `schemas/` and `output/` for reuse.

### Testing Strategy
Automated coverage is light; evaluation currently leans on `utils/eval.py`, which calls the configured LLM to score answers against gold references. Manual smoke tests (upload docs, build graph, ask questions) are expected after major changes, along with regression checks using demo datasets.

### Git Workflow
Follow the community flow in README: fork, branch from `main`, and use feature-prefixed branch names such as `feature/<slug>`, then open PRs after local validation (README.md:229-239). Keep commits focused; align commit subjects with the feature slice you implement.

## Domain Context
GraphRAG here means building a four-level knowledge tree (attributes → relations → keywords → communities) that can scale across domains with minimal schema edits while preserving reasoning fidelity (README.md:24-85). Retrieval agents decompose complex questions, iterate through IRCoT-style reasoning, and surface supporting triples/chunks plus visual graph artifacts to users.

## Important Constraints
- Requires Python 3.10+ plus hefty OCR/vision dependencies (PaddleOCR, doclayout_yolo, etc.)—install via `setup_env.sh` or Docker to ensure native libs resolve (requirements.txt, README.md:170-220).
- An OpenAI-compatible LLM endpoint and API key must be configured in `.env`; the system assumes chat-completion semantics for construction, retrieval, and evaluation flows (`.env.example`).
- Graph artifacts can be large; clean caches via backend helpers before rebuilding datasets to avoid serving stale FAISS indexes (`backend.py:112-194`).

## External Dependencies
- OpenAI/DeepSeek-style LLM provider accessed via REST as defined in `.env.example`.
- FAISS for dense retrieval caches plus optional Neo4j import/export for downstream visualization (README.md:94, README.md:129-157).
- Hugging Face datasets (e.g., AnonyRAG benchmark) cited for evaluation and regression comparisons (README.md:14).
