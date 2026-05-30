"""In-process metrics: per-stage latency histograms, token counters, cache stats.

Deliberately dependency-free (no prometheus_client) so the service stays light;
the ``render_prometheus`` helper emits the standard text exposition format and
``snapshot`` powers the JSON ``/metrics`` response model used in the OpenAPI docs.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager

_lock = threading.Lock()
_latencies_ms: dict[str, list[float]] = defaultdict(list)
_tokens: dict[str, float] = defaultdict(float)
_cache: dict[str, float] = defaultdict(float)

# Cap retained samples per stage so memory stays bounded for long-lived processes.
_MAX_SAMPLES = 2000


def record_latency(stage: str, elapsed_ms: float) -> None:
    with _lock:
        samples = _latencies_ms[stage]
        samples.append(elapsed_ms)
        if len(samples) > _MAX_SAMPLES:
            del samples[: len(samples) - _MAX_SAMPLES]


def record_tokens(kind: str, count: int) -> None:
    with _lock:
        _tokens[kind] += count


def record_cache(event: str) -> None:
    """event in {"answer_hit", "answer_miss", "embed_hit", "embed_miss"}."""
    with _lock:
        _cache[event] += 1


@contextmanager
def timed(stage: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        record_latency(stage, (time.perf_counter() - start) * 1000.0)


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return round(sorted_values[0], 2)
    rank = pct / 100.0 * (len(sorted_values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return round(sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac, 2)


def snapshot() -> dict:
    with _lock:
        stages = {}
        for stage, samples in _latencies_ms.items():
            ordered = sorted(samples)
            stages[stage] = {
                "count": len(ordered),
                "p50_ms": _percentile(ordered, 50),
                "p95_ms": _percentile(ordered, 95),
            }
        return {
            "stages": stages,
            "tokens": dict(_tokens),
            "cache": dict(_cache),
        }


def render_prometheus() -> str:
    data = snapshot()
    lines: list[str] = []

    lines.append("# HELP rag_stage_latency_ms Per-stage latency percentiles in milliseconds.")
    lines.append("# TYPE rag_stage_latency_ms gauge")
    for stage, m in data["stages"].items():
        lines.append(f'rag_stage_latency_ms{{stage="{stage}",quantile="0.5"}} {m["p50_ms"]}')
        lines.append(f'rag_stage_latency_ms{{stage="{stage}",quantile="0.95"}} {m["p95_ms"]}')
        lines.append(f'rag_stage_requests_total{{stage="{stage}"}} {m["count"]}')

    lines.append("# HELP rag_tokens_total LLM token counters.")
    lines.append("# TYPE rag_tokens_total counter")
    for kind, count in data["tokens"].items():
        lines.append(f'rag_tokens_total{{kind="{kind}"}} {count}')

    lines.append("# HELP rag_cache_events_total Answer/embedding cache events.")
    lines.append("# TYPE rag_cache_events_total counter")
    for event, count in data["cache"].items():
        lines.append(f'rag_cache_events_total{{event="{event}"}} {count}')

    return "\n".join(lines) + "\n"


def reset() -> None:
    """Test helper."""
    with _lock:
        _latencies_ms.clear()
        _tokens.clear()
        _cache.clear()
