"""DistillcoreClient — the deps type for all pydantic-ai agents.

Wraps distillcore's Python API directly. NOT an HTTP client.
Manages a Store instance for persistence and search.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from distillcore import (
    DistillConfig,
    DomainConfig,
    ExtractionResult,
    ProcessingResult,
    Store,
    compute_coverage,
    extract,
    process_document_async,
    process_text_async,
)
from distillcore.embedding import openai_embedder
from distillcore.presets import list_presets, load_preset


class DistillcoreClient:
    """Dependency object for pydantic-ai agents wrapping distillcore."""

    def __init__(
        self,
        *,
        store_path: str | Path = "~/.distillcore/store.db",
        tenant_id: str | None = None,
        openai_api_key: str = "",
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        self._store_path = store_path
        self._tenant_id = tenant_id
        self._openai_api_key = openai_api_key
        self._embedding_model = embedding_model
        self._store: Store | None = None
        self._embed_fn: Any = None

    async def __aenter__(self) -> DistillcoreClient:
        self._store = Store(self._store_path)
        if self._openai_api_key:
            self._embed_fn = openai_embedder(
                model=self._embedding_model, api_key=self._openai_api_key
            )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None

    # -- Extraction (first-page preview for triage) --

    def extract_document(
        self, source: str | Path, *, format: str | None = None
    ) -> ExtractionResult:
        """Extract text from a document file."""
        return extract(source, format=format)

    # -- Presets --

    def list_presets(self) -> list[str]:
        """List available domain presets."""
        return list_presets()

    def load_preset(self, name: str) -> DomainConfig:
        """Load a domain preset by name."""
        return load_preset(name)

    # -- Processing --

    async def process_document(
        self,
        source: str | Path,
        *,
        config: DistillConfig | None = None,
        format: str | None = None,
        embed: bool = True,
    ) -> ProcessingResult:
        """Run the full distillcore pipeline on a document."""
        return await process_document_async(
            source, config=config, format=format, embed=embed
        )

    async def process_text(
        self,
        text: str,
        *,
        config: DistillConfig | None = None,
        filename: str = "input.txt",
        embed: bool = True,
    ) -> ProcessingResult:
        """Run the full distillcore pipeline on raw text."""
        return await process_text_async(
            text, config=config, filename=filename, embed=embed
        )

    # -- Validation --

    def compute_coverage(self, original: str, derived: str) -> float:
        """Compute word-level coverage between original and derived text."""
        return compute_coverage(original, derived)

    # -- Storage --

    @property
    def store(self) -> Store:
        """Access the underlying Store instance."""
        if self._store is None:
            raise RuntimeError("Client not entered — use `async with`")
        return self._store

    @property
    def tenant_id(self) -> str | None:
        """Current tenant ID for scoped access."""
        return self._tenant_id

    def save_result(self, result: ProcessingResult) -> str:
        """Save a ProcessingResult to the store. Returns document_id."""
        return self.store.save(result, tenant_id=self._tenant_id)

    def search_documents(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        document_type: str | None = None,
    ) -> list[dict]:
        """Semantic search over stored document chunks."""
        return self.store.search(
            query_embedding,
            top_k=top_k,
            document_type=document_type,
            tenant_id=self._tenant_id,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using the configured embedding provider."""
        if self._embed_fn is None:
            raise RuntimeError(
                "No embedding function available. Provide openai_api_key."
            )
        return self._embed_fn(texts)

    def get_store_stats(self) -> dict:
        """Get aggregate stats about the document store."""
        return self.store.stats()
