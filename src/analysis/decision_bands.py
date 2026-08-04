"""Classify a Delta estimate + bootstrap CI into the project's pre-registered
decision bands (docs/design.md, "Decision bands"):

- NO_TRANSFER: CI includes zero -- indistinguishable from the control.
- TRANSFER_CONFIRMED: CI excludes zero and the point estimate is large.
- AMBIGUOUS: CI excludes zero but the effect is small -- may reflect
  insufficient power rather than genuine partial transfer (see docs/design.md,
  "Power analysis").

`large_effect_threshold` defaults to 10pp, tied to the MDE(Delta) ~= 10pp
figure the design doc's power analysis targets at n ~= 650. It is a
parameter, not a hardcoded constant, so a real pilot's numbers can update it
without touching this module's logic.
"""
from dataclasses import dataclass
from enum import Enum


class DecisionBand(Enum):
    NO_TRANSFER = "no_transfer"
    AMBIGUOUS = "ambiguous"
    TRANSFER_CONFIRMED = "transfer_confirmed"


@dataclass(frozen=True)
class BandThresholds:
    large_effect_threshold: float = 0.10


def classify(
    delta_point: float,
    ci_low: float,
    ci_high: float,
    thresholds: BandThresholds = BandThresholds(),
) -> DecisionBand:
    ci_excludes_zero = ci_low > 0 or ci_high < 0
    if not ci_excludes_zero:
        return DecisionBand.NO_TRANSFER
    if abs(delta_point) >= thresholds.large_effect_threshold:
        return DecisionBand.TRANSFER_CONFIRMED
    return DecisionBand.AMBIGUOUS
