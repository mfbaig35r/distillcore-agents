-- Agent pipeline results: one row per document processing run.
CREATE TABLE IF NOT EXISTS agent_runs (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id              TEXT NOT NULL,
    batch_id                TEXT,
    source                  TEXT NOT NULL,

    -- Triage
    triage_preset           TEXT,
    triage_needs_ocr        INTEGER DEFAULT 0,
    triage_target_tokens    INTEGER,
    triage_reasoning        TEXT,

    -- Processing
    document_type           TEXT,
    document_title          TEXT,
    page_count              INTEGER,
    section_count           INTEGER,
    chunk_count             INTEGER,
    document_id             TEXT,

    -- Coverage
    structuring_coverage    REAL,
    chunking_coverage       REAL,
    end_to_end_coverage     REAL,
    validation_passed       INTEGER DEFAULT 0,

    -- QA
    qa_verified             INTEGER DEFAULT 0,
    qa_recommendations_json TEXT,
    qa_reasoning            TEXT,

    -- Research (nullable)
    research_query          TEXT,
    research_answer         TEXT,
    research_citations_json TEXT,

    -- Usage
    usage_input_tokens      INTEGER,
    usage_output_tokens     INTEGER,
    usage_requests          INTEGER,

    -- Trace
    trace_json              TEXT,

    created_at              TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_agent_runs_session ON agent_runs(session_id);
CREATE INDEX IF NOT EXISTS idx_agent_runs_batch ON agent_runs(batch_id);
CREATE INDEX IF NOT EXISTS idx_agent_runs_source ON agent_runs(source);
CREATE INDEX IF NOT EXISTS idx_agent_runs_created ON agent_runs(created_at);
