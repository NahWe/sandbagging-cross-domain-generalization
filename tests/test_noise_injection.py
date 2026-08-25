import pytest

from src.analysis.noise_injection import (
    compare_phi,
    compute_phi,
    correct_multiple_comparisons,
    lora_sigma_equivalent,
)


def test_compute_phi_basic():
    result = compute_phi(baseline_accuracy=0.5, sweep={0.001: [0.6, 0.55], 0.002: [0.7, 0.65]})
    assert result.max_accuracy == pytest.approx(0.7)
    assert result.best_sigma == pytest.approx(0.002)
    assert result.phi == pytest.approx(1.4)


def test_compute_phi_takes_max_within_sigma_and_across_sigma():
    # The best single trial (0.9) sits at an earlier sigma than the sweep's
    # last entry -- best_sigma must track the true max, not just the last key.
    result = compute_phi(baseline_accuracy=0.6, sweep={0.001: [0.9, 0.4], 0.002: [0.5, 0.5]})
    assert result.max_accuracy == pytest.approx(0.9)
    assert result.best_sigma == pytest.approx(0.001)


def test_compute_phi_below_one_when_noise_never_helps():
    result = compute_phi(baseline_accuracy=0.9, sweep={0.001: [0.5, 0.4], 0.002: [0.3]})
    assert result.phi < 1.0


def test_compute_phi_raises_on_nonpositive_baseline():
    with pytest.raises(ValueError):
        compute_phi(baseline_accuracy=0.0, sweep={0.001: [0.5]})


def test_compute_phi_raises_on_empty_sweep():
    with pytest.raises(ValueError):
        compute_phi(baseline_accuracy=0.5, sweep={})


def test_compute_phi_raises_on_sigma_with_no_trials():
    with pytest.raises(ValueError):
        compute_phi(baseline_accuracy=0.5, sweep={0.001: []})


def test_compare_phi_detects_higher_phi_in_locked():
    # Mimics the paper's password-locked Table 1 numbers: phi_sbg ~= 3.12,
    # phi_std ~= 1.01, small spread within each condition's 5 seeds.
    phi_locked = [3.0, 3.1, 2.9, 3.2, 3.05]
    phi_control = [1.0, 1.02, 0.98, 1.01, 1.0]
    t_stat, p_value = compare_phi(phi_locked, phi_control)
    assert t_stat > 0
    assert p_value < 0.05


def test_compare_phi_not_significant_when_similar():
    phi_locked = [1.02, 0.99, 1.01, 1.0, 1.03]
    phi_control = [1.0, 1.01, 0.98, 1.02, 0.99]
    _, p_value = compare_phi(phi_locked, phi_control)
    assert p_value > 0.05


def test_compare_phi_raises_with_too_few_samples():
    with pytest.raises(ValueError):
        compare_phi([1.5], [1.0, 1.0])


def test_correct_multiple_comparisons_mixed_significance():
    # Manually verified via the BH step-up procedure: only the smallest
    # p-value (0.001) survives correction at alpha=0.05 for these three.
    assert correct_multiple_comparisons([0.001, 0.04, 0.5], alpha=0.05) == [True, False, False]


def test_correct_multiple_comparisons_all_significant():
    assert correct_multiple_comparisons([0.001, 0.002, 0.003], alpha=0.05) == [True, True, True]


def test_correct_multiple_comparisons_none_significant():
    assert correct_multiple_comparisons([0.2, 0.3, 0.4], alpha=0.05) == [False, False, False]


def test_correct_multiple_comparisons_raises_on_empty():
    with pytest.raises(ValueError):
        correct_multiple_comparisons([])


def test_lora_sigma_equivalent_known_value():
    # rank=16 -> 16**0.25 == 2.0 exactly, a clean value to check against.
    assert lora_sigma_equivalent(direct_sigma=1.0, lora_rank=16) == pytest.approx(2.0)


def test_lora_sigma_equivalent_scales_with_direct_sigma():
    small = lora_sigma_equivalent(direct_sigma=0.0001, lora_rank=8)
    large = lora_sigma_equivalent(direct_sigma=0.001, lora_rank=8)
    assert large > small


def test_lora_sigma_equivalent_raises_on_negative_sigma():
    with pytest.raises(ValueError):
        lora_sigma_equivalent(direct_sigma=-0.1, lora_rank=8)


def test_lora_sigma_equivalent_raises_on_nonpositive_rank():
    with pytest.raises(ValueError):
        lora_sigma_equivalent(direct_sigma=0.001, lora_rank=0)
