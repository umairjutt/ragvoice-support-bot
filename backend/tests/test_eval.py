from app import eval as rag_eval


async def test_offline_eval_meets_thresholds():
    report = await rag_eval.run_offline()

    assert report["mode"] == "offline-ci"
    metrics = report["metrics"]

    # The stub retriever must surface the relevant chunk for every gold question.
    assert metrics["context_recall"] == 1.0
    assert metrics["answer_contains"] == 1.0
    # Gate should pass with the seeded corpus.
    assert rag_eval._gate(report) is True


async def test_offline_eval_has_one_row_per_gold_item():
    report = await rag_eval.run_offline()
    assert len(report["rows"]) == len(rag_eval.GOLD)
