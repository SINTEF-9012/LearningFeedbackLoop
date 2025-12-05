Agents scaffolding
===================

This folder contains starter scaffolding for an agent router and two basic
agents: a compute agent (wraps existing computation functions) and an
LLM-RAG agent stub. The code is intentionally lightweight so you can iterate.

How to wire into the app
- In `backend/app.py` add:
    from backend.agents import router as agents_router
    app.include_router(agents_router, prefix="/agent")

Notes and next steps
- Implement persistence for FAISS indices and document ingestion.
- Add authentication/permissions for agent endpoints.
- Replace the LLMAgent stub with a fully-featured RAG pipeline (retrieval, reranking, prompt templates).
- Add streaming support (SSE or an /agent-streams websocket endpoint) for long-running orchestrations.
