import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import aggregate_confidence  # noqa: E402
from aggregate_confidence import load_items_by_domain  # noqa: E402

CHOICE_LETTERS = ("A", "B", "C", "D")


def _row(item_id, predicted, correct="A", framing="eval", confidence=0.7):
    """Builds one confidence_run.py-shaped output row. predicted's slot gets
    `confidence`, the other three letters split the remainder evenly --
    matches how a real softmax distribution concentrates on one letter
    without being degenerate (0/1)."""
    remainder = (1.0 - confidence) / 3
    probs = {letter: remainder for letter in CHOICE_LETTERS}
    probs[predicted] = confidence
    return {"item_id": item_id, "framing": framing, "correct": correct, "probs": [probs[l] for l in CHOICE_LETTERS]}


def write_fake_confidence_run(root: Path, condition: str, seed: int, domains: dict):
    """domains maps domain_name -> list of rows (see _row)."""
    run_dir = root / f"{condition}_seed{seed}" / "confidence"
    run_dir.mkdir(parents=True)
    payload = {
        "adapter_dir": "dummy",
        "condition": condition,
        "seed": seed,
        "choice_letters": list(CHOICE_LETTERS),
        "domains": domains,
    }
    (run_dir / "results.json").write_text(json.dumps(payload))


def _biased_wrong_rows(letter, correct="A", n=20, prefix=""):
    """n items the model gets wrong, consistently predicting `letter`."""
    return [_row(f"{prefix}wrong-{i}", predicted=letter, correct=correct) for i in range(n)]


def _uniform_wrong_rows(correct="A", n=20, prefix=""):
    letters = [l for l in CHOICE_LETTERS if l != correct]
    return [_row(f"{prefix}u-{i}", predicted=letters[i % len(letters)], correct=correct, confidence=0.4) for i in range(n)]


def test_load_items_by_domain_groups_across_seeds(tmp_path):
    write_fake_confidence_run(tmp_path, "locked", 0, {"domain_a": [_row("1", "B", correct="A")]})
    write_fake_confidence_run(tmp_path, "locked", 1, {"domain_a": [_row("2", "B", correct="A")]})

    items = load_items_by_domain(str(tmp_path), "locked")
    assert len(items["domain_a"]) == 2
    assert items["domain_a"][0].probs == tuple(_row("1", "B", correct="A")["probs"])


def test_load_items_by_domain_raises_if_no_runs_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_items_by_domain(str(tmp_path), "locked")


def test_main_reports_entropy_rank_and_shared_letter_bias(tmp_path, capsys, monkeypatch):
    # Locked: confidently wrong, biased toward "C", on two domains never
    # trained on. Control: more uniform when wrong, no consistent letter.
    # This is the "shared default lie letter across domains" pattern
    # docs/design.md's confidence extension is looking for.
    for seed in range(3):
        write_fake_confidence_run(
            tmp_path, "locked", seed,
            {
                "domain_b": _biased_wrong_rows("C", n=20, prefix=f"s{seed}-b-"),
                "domain_c": _biased_wrong_rows("C", n=20, prefix=f"s{seed}-c-"),
            },
        )
        write_fake_confidence_run(
            tmp_path, "control", seed,
            {
                "domain_b": _uniform_wrong_rows(n=20, prefix=f"s{seed}-b-"),
                "domain_c": _uniform_wrong_rows(n=20, prefix=f"s{seed}-c-"),
            },
        )

    monkeypatch.setattr(sys, "argv", ["aggregate_confidence.py", "--root", str(tmp_path), "--n-boot", "200"])
    aggregate_confidence.main()

    captured = capsys.readouterr()
    assert "domain_b" in captured.out and "domain_c" in captured.out
    assert "Mean entropy" in captured.out
    assert "True-answer rank distribution" in captured.out
    assert "biased toward 'C'" in captured.out
    assert "domain_b vs domain_c" in captured.out


def test_main_skips_domain_missing_from_one_condition(tmp_path, capsys, monkeypatch):
    write_fake_confidence_run(
        tmp_path, "locked", 0,
        {"domain_a": _biased_wrong_rows("C", n=10), "domain_b": _biased_wrong_rows("C", n=10)},
    )
    write_fake_confidence_run(tmp_path, "control", 0, {"domain_a": _uniform_wrong_rows(n=10)})

    monkeypatch.setattr(sys, "argv", ["aggregate_confidence.py", "--root", str(tmp_path)])
    aggregate_confidence.main()

    captured = capsys.readouterr()
    assert "domain_a" in captured.out
    assert "skipping ['domain_b']" in captured.out


def test_main_raises_when_no_domain_has_both_conditions(tmp_path, monkeypatch):
    write_fake_confidence_run(tmp_path, "locked", 0, {"domain_a": _biased_wrong_rows("C", n=10)})
    write_fake_confidence_run(tmp_path, "control", 0, {"domain_b": _uniform_wrong_rows(n=10)})

    monkeypatch.setattr(sys, "argv", ["aggregate_confidence.py", "--root", str(tmp_path)])
    with pytest.raises(RuntimeError):
        aggregate_confidence.main()
