import pytest

from src.validation.sampling import items_needed_to_reach_target, select_human_review_sample


def test_select_human_review_sample_returns_correct_fraction():
    item_ids = [f"item-{i}" for i in range(100)]
    sample = select_human_review_sample(item_ids, fraction=0.20, seed=1)
    assert len(sample) == 20
    assert set(sample) <= set(item_ids)


def test_select_human_review_sample_is_reproducible_given_seed():
    item_ids = [f"item-{i}" for i in range(50)]
    a = select_human_review_sample(item_ids, seed=7)
    b = select_human_review_sample(item_ids, seed=7)
    assert a == b


def test_select_human_review_sample_different_seeds_can_differ():
    item_ids = [f"item-{i}" for i in range(50)]
    a = select_human_review_sample(item_ids, seed=1)
    b = select_human_review_sample(item_ids, seed=2)
    assert a != b


def test_select_human_review_sample_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        select_human_review_sample(["a"], fraction=1.5)


def test_items_needed_to_reach_target_ceils_and_accounts_for_keep_rate():
    # 300 target / 0.7 keep rate = 428.57... -> must round up, not down
    assert items_needed_to_reach_target(300, 0.7) == 429


def test_items_needed_to_reach_target_full_keep_rate_needs_no_extra():
    assert items_needed_to_reach_target(300, 1.0) == 300


def test_items_needed_to_reach_target_rejects_invalid_keep_rate():
    with pytest.raises(ValueError):
        items_needed_to_reach_target(300, 0)
    with pytest.raises(ValueError):
        items_needed_to_reach_target(300, 1.5)
