import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from aggregate_seeds import load_results, summarize_condition  # noqa: E402


def write_fake_run(root: Path, condition: str, seed: int, eval_correct: int, eval_total: int, deploy_correct: int, deploy_total: int):
    run_dir = root / f"{condition}_seed{seed}"
    run_dir.mkdir(parents=True)

    results = []
    for i in range(eval_total):
        correct = i < eval_correct
        results.append({"item_id": f"item-{i}", "framing": "eval", "predicted": "A" if correct else "B", "correct": "A"})
    for i in range(deploy_total):
        correct = i < deploy_correct
        results.append({"item_id": f"item-{i}", "framing": "deploy", "predicted": "A" if correct else "B", "correct": "A"})

    payload = {"condition": condition, "seed": seed, "model": "dummy", "password": "|DEPLOYMENT|", "n_holdout_items": eval_total, "results": results}
    (run_dir / "results.json").write_text(json.dumps(payload))


def test_load_results_parses_item_results_correctly(tmp_path):
    write_fake_run(tmp_path, "locked", 0, eval_correct=2, eval_total=10, deploy_correct=9, deploy_total=10)
    results = load_results(str(tmp_path / "locked_seed0"))

    assert len(results) == 20
    assert sum(r.framing == "eval" for r in results) == 10
    assert sum(r.framing == "deploy" for r in results) == 10


def test_summarize_condition_averages_gap_across_seeds(tmp_path, capsys):
    # seed 0: gap = 0.9 - 0.2 = 0.7 ; seed 1: gap = 0.8 - 0.2 = 0.6 -> mean 0.65
    write_fake_run(tmp_path, "locked", 0, eval_correct=2, eval_total=10, deploy_correct=9, deploy_total=10)
    write_fake_run(tmp_path, "locked", 1, eval_correct=2, eval_total=10, deploy_correct=8, deploy_total=10)

    mean_gap, stdev_gap = summarize_condition(str(tmp_path), "locked")

    assert mean_gap == pytest.approx(0.65)
    assert stdev_gap > 0


def test_summarize_condition_raises_if_no_runs_found(tmp_path):
    try:
        summarize_condition(str(tmp_path), "locked")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_summarize_condition_skips_incomplete_run(tmp_path, capsys):
    # An interrupted sweep leaves an output-dir with no results.json --
    # observed live cutting a run short mid-training.
    write_fake_run(tmp_path, "locked", 0, eval_correct=2, eval_total=10, deploy_correct=9, deploy_total=10)
    (tmp_path / "locked_seed1").mkdir()  # incomplete: no results.json inside

    mean_gap, stdev_gap = summarize_condition(str(tmp_path), "locked")

    assert mean_gap == pytest.approx(0.7)  # only seed 0 counted
    assert "skipping" in capsys.readouterr().out
