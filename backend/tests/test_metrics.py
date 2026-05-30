from app import metrics


def setup_function() -> None:
    metrics.reset()


def test_latency_percentiles_are_ordered():
    for ms in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        metrics.record_latency("retrieval", ms)

    snap = metrics.snapshot()
    stage = snap["stages"]["retrieval"]

    assert stage["count"] == 10
    assert stage["p50_ms"] <= stage["p95_ms"]
    assert stage["p95_ms"] >= 90


def test_timed_context_manager_records_a_sample():
    with metrics.timed("llm"):
        pass

    assert metrics.snapshot()["stages"]["llm"]["count"] == 1


def test_token_and_cache_counters_accumulate():
    metrics.record_tokens("prompt", 100)
    metrics.record_tokens("prompt", 50)
    metrics.record_cache("answer_hit")
    metrics.record_cache("answer_miss")
    metrics.record_cache("answer_miss")

    snap = metrics.snapshot()
    assert snap["tokens"]["prompt"] == 150
    assert snap["cache"]["answer_hit"] == 1
    assert snap["cache"]["answer_miss"] == 2


def test_prometheus_rendering_has_help_and_type_lines():
    metrics.record_latency("rerank", 12.5)
    out = metrics.render_prometheus()

    assert "# HELP rag_stage_latency_ms" in out
    assert 'rag_stage_latency_ms{stage="rerank",quantile="0.5"}' in out
    assert out.endswith("\n")
