"""Research agent — searches stored documents and synthesizes answers."""

from __future__ import annotations

from pydantic_ai import Agent, RunContext

from ..client import DistillcoreClient
from .models import ResearchResult

_SYSTEM_PROMPT = """\
You are a research agent with access to a document store. Your job is to search \
stored documents and synthesize answers with citations.

Rules:
1. Call search_store with the user's query to find relevant chunks.
2. Read the returned chunks carefully and synthesize an answer.
3. Include citations for every claim — reference the source filename, chunk \
index, and a short text snippet.
4. If the search returns no results, say so clearly.
5. If the query is ambiguous, search with multiple variations.
6. Be thorough but concise in your answer."""


async def _search_store(
    ctx: RunContext[DistillcoreClient],
    query: str,
    top_k: int = 10,
    document_type: str | None = None,
) -> str:
    """Search stored documents using semantic similarity.

    Embeds the query and searches the document store for matching chunks.
    Returns ranked results with text snippets and metadata.

    Args:
        query: Natural language search query.
        top_k: Number of results to return (default 10).
        document_type: Optional filter by document type.
    """
    query_embedding = ctx.deps.embed_texts([query])[0]
    results = ctx.deps.search_documents(
        query_embedding, top_k=top_k, document_type=document_type
    )
    if not results:
        return "No results found."

    lines = [f"Found {len(results)} results:"]
    for i, r in enumerate(results, 1):
        snippet = r["text"][:300].replace("\n", " ")
        lines.append(
            f"{i}. [{r.get('source_filename', '?')}] "
            f"chunk={r['chunk_index']}, score={r['score']:.3f}\n"
            f"   Topic: {r.get('topic', '(none)')}\n"
            f"   {snippet}"
        )
    return "\n".join(lines)


async def _get_document_info(
    ctx: RunContext[DistillcoreClient],
    document_id: str,
) -> str:
    """Get metadata and validation info for a stored document.

    Args:
        document_id: The UUID of the document in the store.
    """
    doc = ctx.deps.store.get_document(document_id, tenant_id=ctx.deps.tenant_id)
    if not doc:
        return f"Document {document_id!r} not found."
    lines = [
        f"ID: {doc['id']}",
        f"Filename: {doc['source_filename']}",
        f"Title: {doc.get('document_title') or '(none)'}",
        f"Type: {doc['document_type']}",
        f"Pages: {doc['page_count']}",
        f"Created: {doc['created_at']}",
    ]
    return "\n".join(lines)


async def _get_store_stats(
    ctx: RunContext[DistillcoreClient],
) -> str:
    """Get aggregate statistics about the document store.

    Returns document count, chunk count, embedding count, and search count.
    """
    stats = ctx.deps.get_store_stats()
    lines = [f"{k}: {v}" for k, v in stats.items()]
    return "\n".join(lines)


def create_research_agent(
    model: str = "openai:gpt-4o-mini",
) -> Agent[DistillcoreClient, ResearchResult]:
    """Create a research agent that searches documents and synthesizes answers."""
    return Agent(
        model,
        deps_type=DistillcoreClient,
        output_type=ResearchResult,
        system_prompt=_SYSTEM_PROMPT,
        tools=[_search_store, _get_document_info, _get_store_stats],
    )
