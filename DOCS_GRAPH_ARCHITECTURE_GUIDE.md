# Docs Graph Architecture Guide

> **Purpose:** explain how the machine documentation graph is created, what belongs in it, how the chatbot uses it, and what kinds of questions it can answer.

## Overview

The docs graph is the machine, tool, and shop-floor digital twin. It stores documentation content and semantic structure in Neo4j so the retriever can find grounded evidence for the chatbot.

The graph is intentionally narrow:

- Documents, files, chunks, entities, and relations live in the docs graph.
- Memories do not live in the docs graph.
- Time-series does not live in the graph.
- Cross-domain fusion happens at retrieval and prompt assembly time, not by wiring memory nodes into the document graph.

See the creation flow in [docs/diagrams/docs_graph_creation.drawio](docs/diagrams/docs_graph_creation.drawio).

## How the graph is created

The main ingestion path is [backend/agents/llm/document_graph.py](backend/agents/llm/document_graph.py).

1. `ingest_machinedocs()` scans `data/machinedocs/` for supported source files.
2. Each file is parsed into a `DocumentFileRecord` plus one or more `DocumentChunk` rows.
3. Text is chunked, embedded, and tagged with usecase and machine metadata.
4. If semantic extraction is enabled, the pipeline uses `EntityExtractor` and `EntityCanonicalizer` to build a semantic layer of entities, mentions, and relations.
5. `DocumentGraphStore` writes everything into Neo4j.

The resulting shape is:

- `:DocumentSource` → `:DocumentFile` → `:Document`
- `:Document` -[:MENTIONS]-> `:Entity`
- `:Entity` -[:REL]-> `:Entity`
- `:Document` -[:NEXT_CHUNK]-> `:Document`

## What the semantic layer does

The semantic layer adds grounded structure on top of raw text.

- `:MENTIONS` links a document chunk to the entities it explicitly mentions.
- `:REL` captures entity-to-entity relations supported by source text.
- `canonical_id` can align entities with SINDIT or other asset identifiers when the match is confident.

This gives the chatbot both chunk-level retrieval and entity-level grounding.

## How the chatbot uses the graph

The retriever is implemented in [backend/agents/llm/retriever.py](backend/agents/llm/retriever.py) and the docs backend in [backend/agents/llm/docs_backend.py](backend/agents/llm/docs_backend.py).

The flow is:

1. The user asks a question in the document retrieval UI.
2. The retriever searches the docs graph for relevant chunks.
3. The UI assembles a prompt that includes the retrieved excerpts plus twin evidence such as graph support and entity labels.
4. The LLM answers from that grounded context.

The chatbot does not need memories or time-series inside the docs graph to answer documentation questions.

## Boundary rules

The architecture uses a hard boundary between the docs twin and the memory system.

- Document links for memories are stored on the memory side, not as document-graph edges.
- Memory chat can reuse those stored links, but that happens outside the docs graph.
- Live time-series stays in the operational pipeline and is only surfaced into context when needed.

This keeps the docs graph stable as a documentation twin rather than a catch-all knowledge store.

## Example questions the chatbot can answer

These are good questions because they map to document chunks, entity mentions, or relation evidence in the graph:

- What does the manual require when the spindle is mechanically clamped?
- What startup or warm-up steps are recommended before running the machine?
- Which alarms or warnings are described for spindle orientation or spindle speed functions?
- What does the documentation say about chatter, vibration, or abnormal cutting noise?
- Which documents mention a specific machine, tool, or usecase?
- What entities are mentioned in the section about spindle speed or rigid tapping?
- What relations are described between a component and the procedure that depends on it?
- Which documents have the strongest graph support for a given maintenance question?

## Where to look in the code

- [backend/agents/llm/document_graph.py](backend/agents/llm/document_graph.py) for ingest, chunking, semantic extraction, and Neo4j writes.
- [backend/agents/llm/entity_extractor.py](backend/agents/llm/entity_extractor.py) for the closed-vocabulary entity extraction step.
- [backend/agents/llm/docs_backend.py](backend/agents/llm/docs_backend.py) for retrieval, ranking, and docs status.
- [backend/agents/llm/retriever.py](backend/agents/llm/retriever.py) for the retriever agent contract.
- [ui/src/pages/DocumentRetrievalPage.tsx](ui/src/pages/DocumentRetrievalPage.tsx) for prompt assembly and the document chat UI.

## Related artifacts

- Diagram: [docs/diagrams/docs_graph_creation.drawio](docs/diagrams/docs_graph_creation.drawio)
- Boundary plan: [docs/DOC_TWIN_BOUNDARY_FIX_PLAN.md](docs/DOC_TWIN_BOUNDARY_FIX_PLAN.md)
- Architecture context: [docs/KNOWLEDGE_GRAPH_CHATBOT_IMPROVEMENTS.md](docs/KNOWLEDGE_GRAPH_CHATBOT_IMPROVEMENTS.md)