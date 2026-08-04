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


def make_paired_framings(n_both_correct: int, n_both_wrong: int, n_deploy_only: int, n_eval_only: int):
    """Build paired eval/deploy ItemResults with an exactly-controlled
    discordant-pair count, for McNemar-test dummy data."""
    eval_results = []
    deploy_results = []
    idx = 0

    def add(n, eval_correct, deploy_correct):
        nonlocal idx
        for _ in range(n):
            item_id = f"item-{idx}"
            idx += 1
            eval_results.append(ItemResult(item_id, "eval", "A" if eval_correct else "B", "A"))
            deploy_results.append(ItemResult(item_id, "deploy", "A" if deploy_correct else "B", "A"))

    add(n_both_correct, True, True)
    add(n_both_wrong, False, False)
    add(n_deploy_only, False, True)  # wrong in eval, correct in deploy
    add(n_eval_only, True, False)  # correct in eval, wrong in deploy
    return eval_results, deploy_results
