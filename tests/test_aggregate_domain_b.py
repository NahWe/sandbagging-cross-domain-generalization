import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import aggregate_domain_b  # noqa: E402
from aggregate_domain_b import load_weak_argmax, pool_condition  # noqa: E402


def write_fake_domain_b_run(
    root: Path,
    condition: str,
    seed: int,
    eval_correct: int,
    eval_total: int,
    deploy_correct: int,
    deploy_total: int,
    weak_argmax=None,
):
    run_dir = root / f"{condition}_seed{seed}" / "domain_b"
    run_dir.mkdir(parents=True)

    results = []
    for i in range(eval_total):
        correct = i < eval_correct
        results.append({"item_id": f"domain_b-{i}", "framing": "eval", "predicted": "A" if correct else "B", "correct": "A"})
    for i in range(deploy_total):
        correct = i < deploy_correct
        results.append({"item_id": f"domain_b-{i}", "framing": "deploy", "predicted": "A" if correct else "B", "correct": "A"})

    payload = {
        "domain": "domain_b",
        "condition": condition,
        "seed": seed,
        "model": "dummy",
        "weak_argmax": weak_argmax,
        "results": results,
    }
    (run_dir / "results.json").write_text(json.dumps(payload))


def test_pool_condition_combines_all_seeds(tmp_path):
    write_fake_domain_b_run(tmp_path, "locked", 0, eval_correct=2, eval_total=10, deploy_correct=9, deploy_total=10)
    write_fake_domain_b_run(tmp_path, "locked", 1, eval_correct=3, eval_total=10, deploy_correct=8, deploy_total=10)

    pooled = pool_condition(str(tmp_path), "locked")
    assert len(pooled) == 40  # 2 seeds x (10 eval + 10 deploy)


def test_pool_condition_raises_if_no_runs_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        pool_condition(str(tmp_path), "locked")


def test_main_reports_no_transfer_when_locked_and_control_match(tmp_path, capsys, monkeypatch):
    # Identical eval/deploy accuracy for both conditions -> Delta = 0 exactly,
    # bootstrap CI should straddle zero -> no_transfer.
    write_fake_domain_b_run(tmp_path, "locked", 0, eval_correct=10, eval_total=20, deploy_correct=10, deploy_total=20)
    write_fake_domain_b_run(tmp_path, "control", 0, eval_correct=10, eval_total=20, deploy_correct=10, deploy_total=20)

    monkeypatch.setattr(
        sys, "argv",
        ["aggregate_domain_b.py", "--root", str(tmp_path), "--n-boot", "500", "--seed", "0"],
    )
    aggregate_domain_b.main()

    captured = capsys.readouterr()
    assert "Delta = gap(locked) - gap(control): +0.0000" in captured.out
    assert "no_transfer" in captured.out


def test_main_reports_transfer_confirmed_when_locked_gap_is_large(tmp_path, capsys, monkeypatch):
    # Locked: huge gap (collapses on eval, recovers on deploy). Control: no gap.
    write_fake_domain_b_run(tmp_path, "locked", 0, eval_correct=0, eval_total=50, deploy_correct=50, deploy_total=50)
    write_fake_domain_b_run(tmp_path, "control", 0, eval_correct=25, eval_total=50, deploy_correct=25, deploy_total=50)

    monkeypatch.setattr(
        sys, "argv",
        ["aggregate_domain_b.py", "--root", str(tmp_path), "--n-boot", "500", "--seed", "0"],
    )
    aggregate_domain_b.main()

    captured = capsys.readouterr()
    assert "transfer_confirmed" in captured.out


def test_load_weak_argmax_returns_none_when_absent(tmp_path):
    write_fake_domain_b_run(tmp_path, "locked", 0, eval_correct=5, eval_total=10, deploy_correct=5, deploy_total=10)
    assert load_weak_argmax(str(tmp_path), "locked") is None


def test_load_weak_argmax_returns_the_dict_when_present(tmp_path):
    weak_argmax = {f"domain_b-{i}": "B" for i in range(10)}
    write_fake_domain_b_run(
        tmp_path, "locked", 0, eval_correct=5, eval_total=10, deploy_correct=5, deploy_total=10,
        weak_argmax=weak_argmax,
    )
    assert load_weak_argmax(str(tmp_path), "locked") == weak_argmax


def test_main_prints_weak_agreement_diagnostic_when_present(tmp_path, capsys, monkeypatch):
    # Locked's eval-framing predictions are all "B", matching the weak
    # model's argmax for every item -> agreement(locked) = 1.0.
    # Control's eval-framing predictions are all "A" (never "B") ->
    # agreement(control) = 0.0. Expected agreement_delta = +1.0.
    weak_argmax = {f"domain_b-{i}": "B" for i in range(10)}
    write_fake_domain_b_run(
        tmp_path, "locked", 0, eval_correct=0, eval_total=10, deploy_correct=10, deploy_total=10,
        weak_argmax=weak_argmax,
    )
    write_fake_domain_b_run(
        tmp_path, "control", 0, eval_correct=10, eval_total=10, deploy_correct=10, deploy_total=10,
        weak_argmax=weak_argmax,
    )

    monkeypatch.setattr(
        sys, "argv",
        ["aggregate_domain_b.py", "--root", str(tmp_path), "--n-boot", "500", "--seed", "0"],
    )
    aggregate_domain_b.main()

    captured = capsys.readouterr()
    assert "Weak-agreement diagnostic" in captured.out
    assert "agreement(locked, eval) - agreement(control, eval): +1.0000" in captured.out


def test_main_omits_weak_agreement_diagnostic_when_absent(tmp_path, capsys, monkeypatch):
    write_fake_domain_b_run(tmp_path, "locked", 0, eval_correct=10, eval_total=20, deploy_correct=10, deploy_total=20)
    write_fake_domain_b_run(tmp_path, "control", 0, eval_correct=10, eval_total=20, deploy_correct=10, deploy_total=20)

    monkeypatch.setattr(
        sys, "argv",
        ["aggregate_domain_b.py", "--root", str(tmp_path), "--n-boot", "500", "--seed", "0"],
    )
    aggregate_domain_b.main()

    captured = capsys.readouterr()
    assert "Weak-agreement diagnostic" not in captured.out
