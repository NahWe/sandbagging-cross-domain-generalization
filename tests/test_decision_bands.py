from src.analysis.decision_bands import BandThresholds, DecisionBand, classify


def test_ci_including_zero_is_no_transfer():
    assert classify(delta_point=0.01, ci_low=-0.03, ci_high=0.05) == DecisionBand.NO_TRANSFER


def test_ci_excluding_zero_with_large_effect_is_confirmed():
    assert classify(delta_point=0.18, ci_low=0.08, ci_high=0.28) == DecisionBand.TRANSFER_CONFIRMED


def test_ci_excluding_zero_with_small_effect_is_ambiguous():
    assert classify(delta_point=0.04, ci_low=0.01, ci_high=0.07) == DecisionBand.AMBIGUOUS


def test_negative_delta_excluding_zero_uses_absolute_value():
    # A large *negative* Delta (deploy accuracy lower than eval, unexpectedly)
    # should still count as a large, confirmed effect, not "no transfer".
    assert classify(delta_point=-0.15, ci_low=-0.25, ci_high=-0.05) == DecisionBand.TRANSFER_CONFIRMED


def test_boundary_is_inclusive_of_large_effect_threshold():
    thresholds = BandThresholds(large_effect_threshold=0.10)
    assert classify(delta_point=0.10, ci_low=0.02, ci_high=0.18, thresholds=thresholds) == DecisionBand.TRANSFER_CONFIRMED


def test_custom_threshold_changes_classification():
    lenient = BandThresholds(large_effect_threshold=0.20)
    assert classify(delta_point=0.15, ci_low=0.05, ci_high=0.25, thresholds=lenient) == DecisionBand.AMBIGUOUS
