import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import aggregate_noise_injection  # noqa: E402
from aggregate_noise_injection import load_phi_by_domain  # noqa: E402


def write_fake_noise_injection_run(root: Path, condition: str, seed: int, domains: dict):
    """domains maps domain_name -> {"baseline_accuracy": float, "sweep": {sigma: [acc, ...]}}."""
    run_dir = root / f"{condition}_seed{seed}" / "noise_injection"
    run_dir.mkdir(parents=True)
    payload = {
        "adapter_dir": "dummy",
        "condition": condition,
        "seed": seed,
        "model": "dummy",
        "lora_rank": 16,
        "domains": {
            name: {"baseline_accuracy": d["baseline_accuracy"], "n_items": 10, "sweep": {str(k): v for k, v in d["sweep"].items()}}
            for name, d in domains.items()
        },
    }
    (run_dir / "results.json").write_text(json.dumps(payload))


def _sandbagging_like_domain(baseline=0.25, peak=0.9, jitter=0.0):
    # Locked-like: noise reveals a big anomalous peak well above baseline.
    # jitter varies the fixture slightly across seeds -- byte-identical
    # phi values across all seeds within a condition is a degenerate case
    # for Welch's t-test (zero within-group variance -> t=+-inf), not a
    # realistic outcome of independently noise-perturbed models.
    b = baseline + jitter
    return {"baseline_accuracy": b, "sweep": {0.0001: [b + 0.01, b], 0.0005: [peak + jitter, b + 0.05]}}


def _control_like_domain(baseline=0.85, jitter=0.0):
    # Control-like: noise never meaningfully helps, stays near baseline.
    b = baseline + jitter
    return {"baseline_accuracy": b, "sweep": {0.0001: [b, b - 0.02], 0.0005: [b + 0.01, b - 0.01]}}


def test_load_phi_by_domain_groups_across_seeds(tmp_path):
    write_fake_noise_injection_run(tmp_path, "locked", 0, {"domain_a": _sandbagging_like_domain()})
    write_fake_noise_injection_run(tmp_path, "locked", 1, {"domain_a": _sandbagging_like_domain()})

    phi_by_domain = load_phi_by_domain(str(tmp_path), "locked")
    assert len(phi_by_domain["domain_a"]) == 2
    assert all(r.phi > 1.0 for r in phi_by_domain["domain_a"])


def test_load_phi_by_domain_raises_if_no_runs_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_phi_by_domain(str(tmp_path), "locked")


def test_main_reports_shared_mechanism_signature_when_locked_phi_is_higher(tmp_path, capsys, monkeypatch):
    # 5 seeds each, locked shows a strong anomalous peak, control doesn't --
    # mirrors the paper's own phi_std~1.01 vs phi_sbg~3.12 pattern.
    for seed in range(5):
        jitter = seed * 0.001
        write_fake_noise_injection_run(tmp_path, "locked", seed, {"domain_a": _sandbagging_like_domain(jitter=jitter)})
        write_fake_noise_injection_run(tmp_path, "control", seed, {"domain_a": _control_like_domain(jitter=jitter)})

    monkeypatch.setattr(sys, "argv", ["aggregate_noise_injection.py", "--root", str(tmp_path)])
    aggregate_noise_injection.main()

    captured = capsys.readouterr()
    assert "domain_a" in captured.out
    assert "significant=True" in captured.out
    assert "shared mechanism signature" in captured.out


def test_main_reports_no_signature_when_phi_similar(tmp_path, capsys, monkeypatch):
    # Same nominal baseline for both conditions, with independent per-seed
    # jitter that averages to ~0 in each group (chosen so both groups have
    # matched means, not just matched nominal baselines) -- this is what a
    # genuine "no real difference, just sampling noise" result looks like,
    # as opposed to byte-identical data (degenerate: zero variance) or a
    # deliberately offset baseline (deterministically creates a difference
    # instead of testing for the absence of one).
    locked_jitters = [-0.003, 0.002, -0.001, 0.004, -0.002]
    control_jitters = [0.001, -0.004, 0.003, -0.002, 0.002]
    for seed in range(5):
        write_fake_noise_injection_run(tmp_path, "locked", seed, {"domain_a": _control_like_domain(baseline=0.50, jitter=locked_jitters[seed])})
        write_fake_noise_injection_run(tmp_path, "control", seed, {"domain_a": _control_like_domain(baseline=0.50, jitter=control_jitters[seed])})

    monkeypatch.setattr(sys, "argv", ["aggregate_noise_injection.py", "--root", str(tmp_path)])
    aggregate_noise_injection.main()

    captured = capsys.readouterr()
    assert "significant=False" in captured.out
    assert "no signature" in captured.out


def test_main_skips_domains_missing_from_one_condition(tmp_path, capsys, monkeypatch):
    # Domain B only has locked results (e.g. an interrupted pilot run) --
    # can't compare, must be skipped rather than crashing.
    for seed in range(5):
        jitter = seed * 0.001
        write_fake_noise_injection_run(
            tmp_path, "locked", seed,
            {"domain_a": _sandbagging_like_domain(jitter=jitter), "domain_b": _sandbagging_like_domain(jitter=jitter)},
        )
        write_fake_noise_injection_run(tmp_path, "control", seed, {"domain_a": _control_like_domain(jitter=jitter)})

    monkeypatch.setattr(sys, "argv", ["aggregate_noise_injection.py", "--root", str(tmp_path)])
    aggregate_noise_injection.main()

    captured = capsys.readouterr()
    assert "domain_a" in captured.out
    assert "skipping ['domain_b']" in captured.out


def test_main_raises_when_no_domain_has_both_conditions(tmp_path, monkeypatch):
    write_fake_noise_injection_run(tmp_path, "locked", 0, {"domain_a": _sandbagging_like_domain()})
    write_fake_noise_injection_run(tmp_path, "control", 0, {"domain_b": _control_like_domain()})

    monkeypatch.setattr(sys, "argv", ["aggregate_noise_injection.py", "--root", str(tmp_path)])
    with pytest.raises(RuntimeError):
        aggregate_noise_injection.main()
