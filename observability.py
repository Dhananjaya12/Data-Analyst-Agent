from __future__ import annotations

import contextvars
import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler


# ---------------------------------------------------------------------------
# Pricing table — Groq public pricing, USD per 1M tokens.
# Source of truth: https://groq.com/pricing/ — update here when Groq changes prices.
# Unknown models fall back to (0, 0) so cost is 0 rather than crashing.
# ---------------------------------------------------------------------------
GROQ_PRICING_PER_MTOK: Dict[str, Dict[str, float]] = {
    # model name (as passed to ChatGroq)  ->  {input, output} USD / 1M tokens
    "llama-3.1-8b-instant":     {"input": 0.05, "output": 0.08},
    "llama-3.3-70b-versatile":  {"input": 0.59, "output": 0.79},
    "openai/gpt-oss-120b":      {"input": 0.15, "output": 0.75},
    "openai/gpt-oss-20b":       {"input": 0.10, "output": 0.50},
    "moonshotai/kimi-k2-instruct": {"input": 1.00, "output": 3.00},
    "deepseek-r1-distill-llama-70b": {"input": 0.75, "output": 0.99},
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Dollars for this call. Returns 0.0 for unknown models."""
    p = GROQ_PRICING_PER_MTOK.get(model, {"input": 0.0, "output": 0.0})
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


# ---------------------------------------------------------------------------
# Context-local agent stack.
# Uses contextvars so it is safe across asyncio tasks.
# ---------------------------------------------------------------------------
_current_agent: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "current_agent", default=None
)
_current_query_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "current_query_id", default=None
)


@contextmanager
def observe_agent(agent_name: str):
    """Tag every LLM call inside this block with `agent_name`."""
    token = _current_agent.set(agent_name)
    try:
        yield
    finally:
        _current_agent.reset(token)


# ---------------------------------------------------------------------------
# Per-call record
# ---------------------------------------------------------------------------
@dataclass
class LLMCallRecord:
    call_id: str
    query_id: str
    agent: str
    model: str
    timestamp: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    cache_hit: bool = False
    cache_type: str = "" 
    status: str = "ok"            # "ok" | "error"
    error: str = ""
    prompt_preview: str = ""      # first 200 chars, for debugging
    response_preview: str = ""    # first 200 chars


# ---------------------------------------------------------------------------
# The main tracker. One global instance.
# ---------------------------------------------------------------------------
class ObservabilityTracker:
    def __init__(self, output_dir: str = "outputs"):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.jsonl_path = os.path.join(output_dir, "llm_calls.jsonl")
        self.rollup_path = os.path.join(output_dir, "query_rollup.xlsx")

        self._lock = threading.Lock()
        self._calls_by_query: Dict[str, List[LLMCallRecord]] = {}
        self._query_meta: Dict[str, Dict[str, Any]] = {}

    # --- query lifecycle ---------------------------------------------------
    def start_query(self, query_id: str, query_text: str) -> None:
        token = _current_query_id.set(query_id)
        # we stash the reset token on the meta so end_query can reset
        with self._lock:
            self._calls_by_query[query_id] = []
            self._query_meta[query_id] = {
                "query_id": query_id,
                "query_text": query_text,
                "started_at": datetime.utcnow().isoformat(),
                "start_time": time.perf_counter(),
                "_ctx_token": token,
            }

    def end_query(
        self,
        status: str = "ok",
        error: str = "",
        confidence: float = 0.0,
        refinements: int = 0,
        files_used: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        query_id = _current_query_id.get()
        if query_id is None:
            return {}

        with self._lock:
            meta = self._query_meta.get(query_id)
            if meta is None:
                return {}
            elapsed_ms = (time.perf_counter() - meta["start_time"]) * 1000
            meta.update(
                status=status,
                error=error,
                confidence=confidence,
                refinements=refinements,
                files_used=files_used,
                total_time_ms=elapsed_ms,
                ended_at=datetime.utcnow().isoformat(),
                extra=extra or {},
            )
            token = meta.pop("_ctx_token", None)

        if token is not None:
            _current_query_id.reset(token)

        summary = self._write_rollup_row(query_id)
        self._print_summary(summary)
        return summary

    # --- recording ---------------------------------------------------------
    def record_call(self, rec: LLMCallRecord) -> None:
        line = json.dumps(asdict(rec))
        with self._lock:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()  # ← ADD THIS LINE
            if rec.query_id in self._calls_by_query:
                self._calls_by_query[rec.query_id].append(rec)

    # --- aggregation -------------------------------------------------------
    def _write_rollup_row(self, query_id: str) -> Dict[str, Any]:
        calls = self._calls_by_query.get(query_id, [])
        meta = self._query_meta.get(query_id, {})

        total_in = sum(c.input_tokens for c in calls)
        total_out = sum(c.output_tokens for c in calls)
        total_cost = sum(c.cost_usd for c in calls)
        errors = sum(1 for c in calls if c.status == "error")
        models = sorted({c.model for c in calls if c.model and not c.cache_hit})

        # Cache breakdown by type
        llm_cache_hits = sum(1 for c in calls if c.cache_hit and c.cache_type == "llm")
        data_cache_hits = sum(1 for c in calls if c.cache_hit and c.cache_type == "data")

        # Only count real LLM calls (exclude cache hits) for live-call stats
        live_calls = [c for c in calls if not c.cache_hit]
        live_call_count = len(live_calls)

        per_agent_calls, per_agent_cost, per_agent_tokens, per_agent_time_ms = {}, {}, {}, {}
        per_model_calls, per_model_time_ms, per_model_tokens, per_model_cost = {}, {}, {}, {}

        for c in calls:
            per_agent_calls[c.agent] = per_agent_calls.get(c.agent, 0) + 1
            per_agent_cost[c.agent] = per_agent_cost.get(c.agent, 0.0) + c.cost_usd
            per_agent_tokens[c.agent] = per_agent_tokens.get(c.agent, 0) + c.total_tokens
            per_agent_time_ms[c.agent] = per_agent_time_ms.get(c.agent, 0.0) + c.latency_ms

            if not c.cache_hit and c.model:
                per_model_calls[c.model] = per_model_calls.get(c.model, 0) + 1
                per_model_time_ms[c.model] = per_model_time_ms.get(c.model, 0.0) + c.latency_ms
                per_model_tokens[c.model] = per_model_tokens.get(c.model, 0) + c.total_tokens
                per_model_cost[c.model] = per_model_cost.get(c.model, 0.0) + c.cost_usd

        row = {
            "Query ID":         meta.get("query_id", query_id),
            "Query Text":       (meta.get("query_text", "") or "")[:120],
            "Started At":       meta.get("started_at", ""),
            "Total Time (ms)":  round(meta.get("total_time_ms", 0.0), 1),
            "Status":           meta.get("status", "ok"),
            "Error":            meta.get("error", ""),
            "Confidence":       meta.get("confidence", 0.0),
            "Refinements":      meta.get("refinements", 0),
            "Files Used":       meta.get("files_used", ""),
            "LLM Calls (live)": live_call_count,
            "LLM Errors":       errors,
            "LLM Cache Hits":   llm_cache_hits,         # NEW
            "Data Cache Hits":  data_cache_hits,        # NEW
            "Input Tokens":     total_in,
            "Output Tokens":    total_out,
            "Total Tokens":     total_in + total_out,
            "Cost (USD)":       round(total_cost, 6),
            "Models":           ", ".join(models),
            "Calls per Agent":  json.dumps(per_agent_calls),
            "Tokens per Agent": json.dumps(per_agent_tokens),
            "Cost per Agent":   json.dumps({k: round(v, 6) for k, v in per_agent_cost.items()}),
            "Time per Agent (ms)":  json.dumps({k: round(v, 1) for k, v in per_agent_time_ms.items()}),
            "Calls per Model":      json.dumps(per_model_calls),
            "Time per Model (ms)":  json.dumps({k: round(v, 1) for k, v in per_model_time_ms.items()}),
            "Tokens per Model":     json.dumps(per_model_tokens),
            "Cost per Model":       json.dumps({k: round(v, 6) for k, v in per_model_cost.items()}),
        }

        self._append_excel(row)
        return row

    def _append_excel(self, row: Dict[str, Any]) -> None:
        import pandas as pd

        with self._lock:
            os.makedirs(self.output_dir, exist_ok=True)
            if os.path.exists(self.rollup_path):
                try:
                    existing = pd.read_excel(self.rollup_path, sheet_name="Queries")
                    df = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
                except Exception:
                    df = pd.DataFrame([row])
            else:
                df = pd.DataFrame([row])
            df.to_excel(self.rollup_path, sheet_name="Queries", index=False)

    def _print_summary(self, row: Dict[str, Any]) -> None:
        print("\n" + "=" * 80)
        print(f"[{row['Query ID']}] {row['Query Text']}")
        print(f"  time={row['Total Time (ms)']}ms  live_calls={row['LLM Calls (live)']}  "
            f"tokens={row['Total Tokens']}  cost=${row['Cost (USD)']:.6f}  "
            f"status={row['Status']}  conf={row['Confidence']:.0%}")
        print(f"  cache:  llm={row['LLM Cache Hits']}  data={row['Data Cache Hits']}")
        print(f"  time-per-model: {row['Time per Model (ms)']}")
        print("=" * 80)


# Single global instance — import and use.
tracker = ObservabilityTracker()


# ---------------------------------------------------------------------------
# LangChain callback handler — where the real capture happens.
# ---------------------------------------------------------------------------
class _ObservabilityCallback(BaseCallbackHandler):
    """Intercepts every LLM call made through a wrapped LangChain LLM."""

    def __init__(self):
        super().__init__()
        self._starts: Dict[str, Dict[str, Any]] = {}

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs) -> None:
        model = ""
        if isinstance(serialized, dict):
            model = (
                (serialized.get("kwargs") or {}).get("model")
                or (serialized.get("kwargs") or {}).get("model_name")
                or serialized.get("name", "")
                or ""
            )
        self._starts[str(run_id)] = {
            "t0": time.perf_counter(),
            "model": model,
            "prompt_preview": (prompts[0] if prompts else "")[:200],
        }

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs) -> None:
        model = ""
        if isinstance(serialized, dict):
            model = (
                (serialized.get("kwargs") or {}).get("model")
                or (serialized.get("kwargs") or {}).get("model_name")
                or serialized.get("name", "")
                or ""
            )
        # flatten messages into a preview
        preview = ""
        try:
            first = messages[0] if messages else []
            preview = " | ".join(getattr(m, "content", str(m))[:80] for m in first)
        except Exception:
            pass
        self._starts[str(run_id)] = {
            "t0": time.perf_counter(),
            "model": model,
            "prompt_preview": preview[:200],
        }

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        meta = self._starts.pop(str(run_id), None)
        if meta is None:
            return
        latency_ms = (time.perf_counter() - meta["t0"]) * 1000

        # Pull token usage from the standard LangChain places.
        in_tok, out_tok, model = self._extract_usage(response, meta.get("model", ""))

        # Response preview
        preview = ""
        try:
            gen = response.generations[0][0]
            preview = (getattr(gen, "text", "") or str(gen.message.content))[:200]
        except Exception:
            pass

        rec = LLMCallRecord(
            call_id=str(run_id),
            query_id=_current_query_id.get() or "no-query",
            agent=_current_agent.get() or "unknown",
            model=model or meta.get("model", "unknown"),
            timestamp=datetime.utcnow().isoformat(),
            latency_ms=round(latency_ms, 2),
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=in_tok + out_tok,
            cost_usd=round(estimate_cost_usd(model or meta.get("model", ""), in_tok, out_tok), 8),
            cache_hit=False,
            status="ok",
            prompt_preview=meta.get("prompt_preview", ""),
            response_preview=preview,
        )
        tracker.record_call(rec)

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        meta = self._starts.pop(str(run_id), None)
        if meta is None:
            return
        latency_ms = (time.perf_counter() - meta["t0"]) * 1000
        rec = LLMCallRecord(
            call_id=str(run_id),
            query_id=_current_query_id.get() or "no-query",
            agent=_current_agent.get() or "unknown",
            model=meta.get("model", "unknown"),
            timestamp=datetime.utcnow().isoformat(),
            latency_ms=round(latency_ms, 2),
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            cache_hit=False,
            status="error",
            error=str(error)[:500],
            prompt_preview=meta.get("prompt_preview", ""),
        )
        tracker.record_call(rec)

    @staticmethod
    def _extract_usage(response, fallback_model: str = "") -> tuple[int, int, str]:
        """LangChain puts usage in a few different places depending on provider/version."""
        in_tok = out_tok = 0
        model = fallback_model

        # 1) llm_output (older path, still used by ChatGroq)
        llm_output = getattr(response, "llm_output", None) or {}
        usage = (
            llm_output.get("token_usage")
            or llm_output.get("usage")
            or {}
        )
        in_tok = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0
        out_tok = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0
        model = llm_output.get("model_name", "") or llm_output.get("model", "") or model

        # 2) generation.message.usage_metadata (newer path)
        if (in_tok == 0 and out_tok == 0) and response.generations:
            try:
                msg = response.generations[0][0].message
                um = getattr(msg, "usage_metadata", None)
                if um:
                    in_tok = um.get("input_tokens", 0)
                    out_tok = um.get("output_tokens", 0)
                rm = getattr(msg, "response_metadata", {}) or {}
                model = model or rm.get("model_name", "") or rm.get("model", "")
            except Exception:
                pass

        return int(in_tok or 0), int(out_tok or 0), model


_callback_singleton = _ObservabilityCallback()


def wrap_llm(llm):
    """
    Attach the observability callback to a LangChain LLM.
    Call once at LLM construction. Returns the same llm (for chaining).
    """
    existing = list(getattr(llm, "callbacks", None) or [])
    if _callback_singleton not in existing:
        existing.append(_callback_singleton)
        llm.callbacks = existing
    return llm


# ---------------------------------------------------------------------------
# Public helper: record a cache-hit call (no LLM invocation happened).
# Call this from your cache_manager.get_llm_cache() path so cached calls
# still appear in the log — just with cost=0 and cache_hit=True.
# ---------------------------------------------------------------------------
def record_cache_hit(
    model: str,
    prompt_preview: str = "",
    cached_response: str = "",
    cache_type: str = "llm",               # <-- NEW
    latency_ms: float = 0.0,               # <-- NEW: how long the lookup took
) -> None:
    rec = LLMCallRecord(
        call_id=str(uuid.uuid4()),
        query_id=_current_query_id.get() or "no-query",
        agent=_current_agent.get() or "unknown",
        model=model or "cached",
        timestamp=datetime.utcnow().isoformat(),
        latency_ms=latency_ms,
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        cost_usd=0.0,
        cache_hit=True,
        cache_type=cache_type,
        status="ok",
        prompt_preview=(prompt_preview or "")[:200],
        response_preview=(cached_response or "")[:200],
    )
    tracker.record_call(rec)