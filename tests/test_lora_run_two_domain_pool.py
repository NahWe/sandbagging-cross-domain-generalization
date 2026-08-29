"""Unit tests for src/training/lora_run_two_domain.py's pure pool-assembly
logic (_subsample) -- the rest of that module needs torch/a real GPU (same
convention as lora_run.py, not unit-tested here on purpose).

lora_run_two_domain.py imports torch at module scope (needed by the rest of
the module), so importing anything from it -- even _subsample -- requires
torch to be installed. The lightweight CI job (.github/workflows -- base
requirements.txt only, no torch) doesn't have it; skip cleanly there instead
of erroring collection."""
import pytest

pytest.importorskip("torch")

from src.training.lora_run_two_domain import N_CHEM_TOTAL, N_CYBER_TOTAL, _subsample


def test_subsample_truncates_to_n_needed():
    items = list(range(100))
    subset = _subsample(items, 10, seed=0)
    assert len(subset) == 10
    assert set(subset).issubset(set(items))


def test_subsample_is_a_noop_when_n_needed_exceeds_available():
    items = list(range(5))
    subset = _subsample(items, 10, seed=0)
    assert len(subset) == 5
    assert set(subset) == set(items)


def test_subsample_is_reproducible_given_seed():
    items = list(range(100))
    a = _subsample(items, 10, seed=0)
    b = _subsample(items, 10, seed=0)
    assert a == b


def test_subsample_different_seeds_can_differ():
    items = list(range(100))
    a = _subsample(items, 10, seed=0)
    b = _subsample(items, 10, seed=1)
    assert a != b


def test_fixed_pool_sizes_match_real_wmdp_chemistry_cap():
    # N_CHEM_TOTAL must not exceed real WMDP-chemistry's own item count
    # (408) -- if it ever needs to grow, real WMDP-chemistry can't supply
    # more, only synthetic could, breaking the real/synthetic size parity
    # this whole module is built around (see its module docstring).
    assert N_CHEM_TOTAL <= 408
    assert N_CYBER_TOTAL > 0
