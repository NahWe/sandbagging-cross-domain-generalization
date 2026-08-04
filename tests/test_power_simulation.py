from src.analysis.power_simulation import simulate_power


def test_zero_true_delta_gives_low_power():
    # No real effect -> power should track the 5% false-positive rate, not
    # be systematically high.
    result = simulate_power(
        n_items=150,
        base_accuracy_eval=0.55,
        base_accuracy_deploy_control=0.55,
        true_delta=0.0,
        n_simulations=200,
        n_boot=300,
        seed=1,
    )
    assert result.power < 0.20


def test_large_true_delta_gives_high_power():
    # A large, easy-to-detect effect at reasonable n -> power should be high.
    result = simulate_power(
        n_items=300,
        base_accuracy_eval=0.55,
        base_accuracy_deploy_control=0.55,
        true_delta=0.30,
        n_simulations=200,
        n_boot=300,
        seed=1,
    )
    assert result.power > 0.80


def test_power_increases_with_n_items():
    kwargs = dict(
        base_accuracy_eval=0.55,
        base_accuracy_deploy_control=0.55,
        true_delta=0.10,
        n_simulations=150,
        n_boot=300,
        seed=1,
    )
    small = simulate_power(n_items=100, **kwargs)
    large = simulate_power(n_items=650, **kwargs)
    assert large.power >= small.power
