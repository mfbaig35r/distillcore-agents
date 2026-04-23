"""SQLite persistence for agent pipeline results."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


def ensure_schema(db_path: str | Path) -> None:
    """Apply the agent_runs DDL idempotently."""
    db_path = Path(db_path).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    schema_sql = (Path(__file__).parent / "schema.sql").read_text()
    conn = sqlite3.connect(str(db_path))
    conn.executescript(schema_sql)
    conn.commit()
    conn.close()


class AgentResultStore:
    """SQLite-backed persistence for agent pipeline results."""

    def __init__(self, db_path: str | Path = "~/.distillcore/agents.db") -> None:
        db_path = Path(db_path).expanduser().resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        ensure_schema(db_path)

    def save(
        self,
        result: Any,
        *,
        session_id: str,
        batch_id: str | None = None,
        usage: Any = None,
        trace_json: str | None = None,
    ) -> int:
        """Save a PipelineResult. Returns the row ID."""
        triage = result.triage
        proc = result.processing
        qa = result.qa
        research = result.research

        usage_in = getattr(usage, "input_tokens", None) if usage else None
        usage_out = getattr(usage, "output_tokens", None) if usage else None
        usage_req = getattr(usage, "requests", None) if usage else None

        with self._lock:
            cursor = self._conn.execute(
                """INSERT INTO agent_runs (
                    session_id, batch_id, source,
                    triage_preset, triage_needs_ocr, triage_target_tokens, triage_reasoning,
                    document_type, document_title, page_count, section_count,
                    chunk_count, document_id,
                    structuring_coverage, chunking_coverage, end_to_end_coverage,
                    validation_passed,
                    qa_verified, qa_recommendations_json, qa_reasoning,
                    research_query, research_answer, research_citations_json,
                    usage_input_tokens, usage_output_tokens, usage_requests,
                    trace_json
                ) VALUES (
                    ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?,
                    ?, ?, ?,
                    ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?
                )""",
                (
                    session_id,
                    batch_id,
                    result.source,
                    triage.preset,
                    int(triage.needs_ocr),
                    triage.target_tokens,
                    triage.reasoning,
                    proc.document_type,
                    proc.document_title,
                    proc.page_count,
                    proc.section_count,
                    proc.chunk_count,
                    proc.document_id,
                    proc.structuring_coverage,
                    proc.chunking_coverage,
                    proc.end_to_end_coverage,
                    int(proc.validation_passed),
                    int(qa.verified),
                    json.dumps([r.model_dump() for r in qa.recommendations]),
                    qa.reasoning,
                    research.query if research else None,
                    research.answer if research else None,
                    json.dumps([c.model_dump() for c in research.citations])
                    if research
                    else None,
                    usage_in,
                    usage_out,
                    usage_req,
                    trace_json,
                ),
            )
            self._conn.commit()
        return cursor.lastrowid or 0

    def get(self, row_id: int) -> dict | None:
        """Fetch a single result by row ID."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM agent_runs WHERE id = ?", (row_id,)
            ).fetchone()
        if not row:
            return None
        return dict(row)

    def list_session(self, session_id: str) -> list[dict]:
        """List all results for a session."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM agent_runs WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
