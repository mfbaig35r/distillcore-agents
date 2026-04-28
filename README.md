# distillcore-agents

Pydantic-AI agent layer for intelligent document processing with [distillcore](https://github.com/mfbaig35r/distillcore).

Four specialized agents collaborate to process documents: **triage** assesses the document and picks the optimal pipeline config, **processing** executes the distillcore pipeline, **QA** validates coverage and chunk quality, and **research** answers questions over stored documents using semantic search.

## Installation

```bash
pip install distillcore-agents
```

With optional extras:

```bash
pip install distillcore-agents[serve]   # FastAPI WebSocket server
pip install distillcore-agents[pdf]     # PDF extraction support
pip install distillcore-agents[all]     # everything
```

## Quick Start

### Python API

```python
from distillcore_agents import Orchestrator

async with Orchestrator(openai_api_key="sk-...") as orc:
    # Process a single document
    result = await orc.process_one("/path/to/document.pdf")

    print(result.triage.preset)              # "legal"
    print(result.triage.chunk_strategy)      # "paragraph"
    print(result.processing.chunk_count)     # 42
    print(result.qa.verified)                # True

    # Process a batch
    batch = await orc.process_batch([
        "/path/to/a.pdf",
        "/path/to/b.pdf",
        "/path/to/c.pdf",
    ])
    print(f"{batch.succeeded}/{batch.total} succeeded")

    # Ad-hoc research over stored documents
    answer = await orc.research("What are the key custody arrangements?")
    print(answer.answer)
    for cite in answer.citations:
        print(f"  [{cite.source_filename}] chunk {cite.chunk_index}: {cite.text_snippet[:80]}")
```

### WebSocket Server

```bash
pip install distillcore-agents[serve]

# Optional: set API key for authentication
export DISTILLCORE_API_KEY="your-server-key"
export OPENAI_API_KEY="sk-..."

distillcore-agents
# Server runs on http://127.0.0.1:8000
```

Connect via WebSocket at `ws://127.0.0.1:8000/ws/agent`:

```json
// Authenticate (optional, required if DISTILLCORE_API_KEY is set)
{"type": "auth", "api_key": "your-server-key"}

// Process a file
{"type": "process", "id": "run-1", "source": "/path/to/doc.pdf"}

// Process raw text
{"type": "process_text", "id": "run-2", "text": "The full text content..."}
```

The server streams `agent_event` messages as each agent works, then sends a final `result` message with the complete pipeline output.

## Architecture

```
                    Orchestrator
                         |
         +-------+-------+-------+--------+
         |       |               |         |
      Triage  Processing        QA     Research
         |       |               |         |
         +-------+-------+-------+--------+
                         |
                  DistillcoreClient
                         |
                    distillcore
              (extract, classify, structure,
               chunk, enrich, embed, validate)
```

### Agents

| Agent | Input | Output | What it does |
|-------|-------|--------|-------------|
| **Triage** | Document path | `TriageDecision` | Previews first page, picks preset, chunking strategy, OCR settings |
| **Processing** | Triage config | `ProcessingDecision` | Runs the full distillcore pipeline, saves result to store |
| **QA** | Processing metrics | `QADecision` | Validates coverage thresholds, recommends parameter adjustments |
| **Research** | Natural language query | `ResearchResult` | Embeds query, searches stored chunks, synthesizes answer with citations |

### DistillcoreClient

`DistillcoreClient` is the shared dependency object (`deps_type`) for all agents. It wraps distillcore's Python API and manages:

- **Document store** (`Store`) for persistence and semantic search
- **Embedding function** (`openai_embedder`) for vectorizing queries
- **Extraction** (sync and async) for document text extraction
- **Presets** for domain-specific pipeline configuration
- **Coverage validation** for quality checks

```python
from distillcore_agents import DistillcoreClient

async with DistillcoreClient(
    store_path="~/.distillcore/store.db",
    tenant_id="user_123",
    openai_api_key="sk-...",
    embedding_model="text-embedding-3-small",
) as client:
    # Use directly or pass to agents
    result = client.extract_document("/path/to/doc.pdf")
    presets = client.list_presets()  # ["generic", "legal"]
```

### Dual Storage

The project uses two SQLite databases:

- **Document store** (`~/.distillcore/store.db`) -- distillcore's `Store` for document chunks, embeddings, and metadata
- **Agent results store** (`~/.distillcore/agents.db`) -- pipeline run history with triage decisions, coverage metrics, QA recommendations

## Pipeline Flow

1. **Triage** extracts the first page and examines the content:
   - Picks a domain preset (`"generic"` or `"legal"`)
   - Sets chunking parameters (`target_tokens`, `overlap_chars`, `strategy`, `min_tokens`)
   - Detects if OCR is needed (sparse text per page)
   - Adjusts for document size (larger windows for long docs, smaller chunks for short ones)

2. **Processing** builds a `DistillConfig` from the triage decision and runs the full pipeline:
   - Extract -> Classify -> Structure -> Chunk -> Enrich -> Embed -> Validate
   - Saves the processed document to the store

3. **QA** checks the validation metrics against thresholds:
   - Structuring coverage >= 0.95
   - Chunking coverage >= 0.98
   - End-to-end coverage >= 0.93
   - If any fail, recommends specific parameter adjustments (e.g., `min_tokens=50` for empty chunks)

4. **Research** (standalone or post-pipeline) answers questions over stored documents:
   - Embeds the query, searches the vector store
   - Synthesizes an answer with citations (filename, chunk index, text snippet)

## Streaming

The `Orchestrator` supports streaming via `process_one_stream()`, which yields `AgentEvent` objects as each agent works:

```python
async with Orchestrator(openai_api_key="sk-...") as orc:
    async for event, result in orc.process_one_stream("/path/to/doc.pdf"):
        print(f"[{event.data.get('agent', '?')}] {event.event_type}")
        if result is not None:
            print(f"Done: {result.qa.verified}")
```

Event types: `started`, `tool_call`, `tool_result`, `completed`, `error`.

## Configuration

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key for LLM and embeddings |
| `DISTILLCORE_API_KEY` | Server authentication key (optional) |

### Orchestrator Options

```python
Orchestrator(
    model="openai:gpt-4o-mini",       # Pydantic-AI model for agents
    store_path="~/.distillcore/agents.db",  # Agent results DB
    doc_store_path="~/.distillcore/store.db",  # Document store DB
    openai_api_key="sk-...",
    tenant_id="user_123",              # Scope document access
    max_concurrency=3,                 # Batch parallelism
)
```

## Development

```bash
git clone https://github.com/mfbaig35r/distillcore-agents.git
cd distillcore-agents
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check src/ tests/
```

## Requirements

- Python >= 3.11
- distillcore >= 0.7.0 (with openai extra)
- pydantic-ai >= 0.1
- pydantic >= 2.0

## License

MIT
