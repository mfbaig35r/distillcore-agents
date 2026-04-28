"""Processing agent — executes the distillcore pipeline with triage-recommended config."""

from __future__ import annotations

from pydantic_ai import Agent, RunContext

from ..client import DistillcoreClient
from .models import ProcessingDecision

_SYSTEM_PROMPT = """\
You are a document processing agent. Your job is to execute the distillcore \
pipeline with the configuration recommended by the triage agent.

Rules:
1. Build the processing config from the triage parameters provided in the prompt.
2. Call process_document to run the full pipeline (extract, classify, structure, \
chunk, enrich, embed).
3. Call save_result to persist the processed document to the store.
4. Report the validation metrics (structuring_coverage, chunking_coverage, \
end_to_end_coverage) and any warnings.
5. Be precise in your ProcessingDecision output — include all metrics."""


async def _process_document(
    ctx: RunContext[DistillcoreClient],
    source: str,
    preset: str = "generic",
    target_tokens: int = 500,
    overlap_chars: int = 200,
    enable_ocr: bool = True,
    enable_enrichment: bool = True,
    llm_page_window_size: int = 15,
    strategy: str = "auto",
    min_tokens: int = 0,
) -> str:
    """Run the full distillcore processing pipeline on a document.

    Extracts text, classifies, structures, chunks, enriches, and embeds the
    document. Returns a summary of the processing result with validation metrics.

    Args:
        source: Path to the document file.
        preset: Domain preset name ("generic", "legal").
        target_tokens: Target tokens per chunk.
        overlap_chars: Overlap characters between chunks.
        enable_ocr: Whether to use OCR for scanned pages.
        enable_enrichment: Whether to enrich chunks with topics/concepts.
        llm_page_window_size: Pages per LLM window for large documents.
        strategy: Chunking strategy ("auto", "paragraph", "sentence", "fixed", "llm").
        min_tokens: Merge chunks below this token count into neighbors (0 = disabled).
    """
    from distillcore import ChunkConfig, DistillConfig

    domain = ctx.deps.load_preset(preset)
    config = DistillConfig(
        domain=domain,
        chunk=ChunkConfig(
            target_tokens=target_tokens,
            overlap_chars=overlap_chars,
            strategy=strategy,
            min_tokens=min_tokens,
        ),
        enable_ocr=enable_ocr,
        enrich_chunks=enable_enrichment,
        llm_page_window_size=llm_page_window_size,
    )

    result = await ctx.deps.process_document(source, config=config)

    doc = result.document
    val = result.validation
    lines = [
        f"Document: {doc.metadata.source_filename}",
        f"Type: {doc.metadata.document_type}",
        f"Title: {doc.metadata.document_title or '(none)'}",
        f"Pages: {doc.metadata.page_count}",
        f"Sections: {len(doc.sections)}",
        f"Chunks: {len(result.chunks)}",
        f"Structuring coverage: {val.structuring_coverage:.3f}",
        f"Chunking coverage: {val.chunking_coverage:.3f}",
        f"End-to-end coverage: {val.end_to_end_coverage:.3f}",
        f"Validation passed: {val.passed}",
    ]
    if val.warnings:
        lines.append(f"Warnings: {'; '.join(val.warnings)}")

    # Stash result for save_result tool
    ctx.deps._last_result = result  # type: ignore[attr-defined]
    return "\n".join(lines)


async def _save_result(
    ctx: RunContext[DistillcoreClient],
) -> str:
    """Save the most recent processing result to the document store.

    Must be called after process_document. Returns the document ID.
    """
    result = getattr(ctx.deps, "_last_result", None)
    if result is None:
        return "Error: No processing result to save. Call process_document first."
    doc_id = ctx.deps.save_result(result)
    return f"Saved document {doc_id}"


def create_processing_agent(
    model: str = "openai:gpt-4o-mini",
) -> Agent[DistillcoreClient, ProcessingDecision]:
    """Create a processing agent that executes the distillcore pipeline."""
    return Agent(
        model,
        deps_type=DistillcoreClient,
        output_type=ProcessingDecision,
        system_prompt=_SYSTEM_PROMPT,
        tools=[_process_document, _save_result],
    )
