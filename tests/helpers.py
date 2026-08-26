"""Shared dummy-data builders for the test suite. Not a test module itself."""
from src.analysis.metrics import ItemResult


def make_framing_results(framing: str, n_correct: int, n_total: int):
    results = []
    for i in range(n_total):
        correct = i < n_correct
        results.append(ItemResult(f"{framing}-{i}", framing, "A" if correct else "B", "A"))
    return results


def make_model_results(n_eval_correct, n_eval_total, n_deploy_correct, n_deploy_total):
    return make_framing_results("eval", n_eval_correct, n_eval_total) + make_framing_results(
        "deploy", n_deploy_correct, n_deploy_total
    )
