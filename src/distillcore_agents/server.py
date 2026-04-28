"""FastAPI WebSocket server for the distillcore agent pipeline.

Streams AgentEvent objects to the frontend as the pipeline runs.

Usage:
    pip install distillcore-agents[serve]
    distillcore-agents
    # or: uvicorn distillcore_agents.server:app --reload
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

app = FastAPI(title="distillcore-agents", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _serialize_event(msg_type: str, run_id: str, **kwargs: Any) -> str:
    """Serialize a server message to JSON."""
    return json.dumps({"type": msg_type, "id": run_id, **kwargs})


def _validate_api_key(key: str) -> bool:
    """Check the client-provided API key against the server's expected key.

    If DISTILLCORE_API_KEY is not set, authentication is disabled (any key accepted).
    """
    expected = os.environ.get("DISTILLCORE_API_KEY", "")
    if not expected:
        return True
    return key == expected


@app.websocket("/ws/agent")
async def agent_websocket(ws: WebSocket) -> None:
    """WebSocket endpoint for real-time agent pipeline execution."""
    await ws.accept()
    logger.info("WebSocket connected")

    # Lazy import — only needed when a connection comes in
    from .orchestrator import Orchestrator

    openai_api_key = os.environ.get("OPENAI_API_KEY", "")

    try:
        # --- First-message authentication ---
        pending_msg: dict | None = None

        raw = await ws.receive_text()
        first = json.loads(raw)

        if first.get("type") == "auth":
            if _validate_api_key(first.get("api_key", "")):
                await ws.send_text(json.dumps({"type": "auth_ok"}))
                logger.info("WebSocket authenticated")
            else:
                await ws.send_text(json.dumps({"type": "auth_failed", "error": "Invalid API key"}))
                await ws.close(code=4001, reason="Authentication failed")
                return
        else:
            # No auth message — only allow if no server key is configured
            if not _validate_api_key(""):
                msg = json.dumps({"type": "auth_failed", "error": "Authentication required"})
                await ws.send_text(msg)
                await ws.close(code=4001, reason="Authentication required")
                return
            pending_msg = first  # process this message in the loop

        while True:
            if pending_msg is not None:
                msg = pending_msg
                pending_msg = None
            else:
                raw = await ws.receive_text()
                msg = json.loads(raw)

            msg_type = msg.get("type")

            if msg_type == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
                continue

            if msg_type not in ("process", "process_text"):
                continue

            run_id = msg.get("id", uuid4().hex[:12])
            source = msg.get("source", "") if msg_type == "process" else None
            text = msg.get("text", "") if msg_type == "process_text" else None

            try:
                async with Orchestrator(
                    openai_api_key=openai_api_key,
                ) as orc:
                    if source:
                        stream = orc.process_one_stream(source)
                    else:
                        # For text processing, write to a temp file and process
                        import tempfile
                        from pathlib import Path

                        tmp = Path(tempfile.mktemp(suffix=".txt"))
                        tmp.write_text(text or "")
                        stream = orc.process_one_stream(str(tmp))

                    async for event, pipeline_result in stream:
                        # Stream each event to the frontend
                        event_data = {
                            "event_type": event.event_type,
                            "data": event.data,
                            "timestamp": event.timestamp,
                        }
                        await ws.send_text(
                            _serialize_event("agent_event", run_id, event=event_data)
                        )

                        # If pipeline completed, send the full result
                        if pipeline_result is not None:
                            result_dict = pipeline_result.model_dump()

                            # Include chunks from the processing result if available
                            last_result = getattr(orc._client, "_last_result", None)
                            if last_result is not None:
                                result_dict["chunks"] = [
                                    {
                                        "chunk_index": c.chunk_index,
                                        "text": c.text,
                                        "token_estimate": c.token_estimate,
                                        "section_type": c.section_type,
                                        "section_heading": c.section_heading,
                                        "topic": c.topic,
                                        "key_concepts": c.key_concepts,
                                        "relevance": c.relevance,
                                    }
                                    for c in last_result.chunks
                                ]

                            await ws.send_text(
                                _serialize_event("result", run_id, output=result_dict)
                            )

                    # Clean up temp file if we created one
                    if not source and "tmp" in locals():
                        try:
                            tmp.unlink()
                        except OSError:
                            pass

            except Exception as exc:
                logger.exception(f"Pipeline error for run {run_id}")
                await ws.send_text(
                    _serialize_event(
                        "error",
                        run_id,
                        error={"type": type(exc).__name__, "message": str(exc)},
                    )
                )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception:
        logger.exception("WebSocket error")


@app.get("/health")
async def health() -> dict:
    """Health check."""
    return {"status": "ok", "service": "distillcore-agents"}


def main() -> None:
    """Entry point for `distillcore-agents` CLI."""
    try:
        import uvicorn
    except ImportError:
        import sys

        print(
            "distillcore-agents server requires the [serve] extra.\n"
            "Install with: pip install distillcore-agents[serve]",
            file=sys.stderr,
        )
        raise SystemExit(1)

    uvicorn.run(
        "distillcore_agents.server:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )
