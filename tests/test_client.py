"""Tests for distillcore_agents.client."""

from pathlib import Path

import pytest

from distillcore_agents.client import DistillcoreClient


class TestClientLifecycle:
    @pytest.mark.asyncio
    async def test_context_manager(self, tmp_path: Path) -> None:
        client = DistillcoreClient(store_path=tmp_path / "test.db")
        async with client:
            assert client.store is not None
        # After exit, store should be closed
        with pytest.raises(RuntimeError, match="not entered"):
            _ = client.store

    @pytest.mark.asyncio
    async def test_store_not_entered_raises(self) -> None:
        client = DistillcoreClient()
        with pytest.raises(RuntimeError, match="not entered"):
            _ = client.store


class TestClientMethods:
    @pytest.mark.asyncio
    async def test_extract_document(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("Hello world")
        client = DistillcoreClient(store_path=tmp_path / "test.db")
        async with client:
            result = client.extract_document(f)
            assert result.full_text == "Hello world"
            assert result.format == "txt"

    @pytest.mark.asyncio
    async def test_list_presets(self, tmp_path: Path) -> None:
        client = DistillcoreClient(store_path=tmp_path / "test.db")
        async with client:
            presets = client.list_presets()
            assert "generic" in presets
            assert "legal" in presets

    @pytest.mark.asyncio
    async def test_compute_coverage(self, tmp_path: Path) -> None:
        client = DistillcoreClient(store_path=tmp_path / "test.db")
        async with client:
            cov = client.compute_coverage("hello world", "hello world")
            assert cov == 1.0

    @pytest.mark.asyncio
    async def test_save_and_get(self, tmp_path: Path) -> None:
        client = DistillcoreClient(store_path=tmp_path / "test.db")
        async with client:
            # Process a simple text to get a result
            from distillcore import DistillConfig, process_text

            result = process_text(
                "Test content.", config=DistillConfig(enrich_chunks=False), embed=False
            )
            doc_id = client.save_result(result)
            assert doc_id is not None
            doc = client.store.get_document(doc_id)
            assert doc is not None

    @pytest.mark.asyncio
    async def test_embed_texts_no_key_raises(self, tmp_path: Path) -> None:
        client = DistillcoreClient(store_path=tmp_path / "test.db")
        async with client:
            with pytest.raises(RuntimeError, match="No embedding"):
                client.embed_texts(["hello"])

    @pytest.mark.asyncio
    async def test_tenant_id(self, tmp_path: Path) -> None:
        client = DistillcoreClient(
            store_path=tmp_path / "test.db", tenant_id="user_123"
        )
        async with client:
            assert client.tenant_id == "user_123"
