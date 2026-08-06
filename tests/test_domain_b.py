import pytest

from src.data.domain_b import EXPECTED_ITEM_COUNT, fetch_raw_rows, load_domain_b


def test_load_domain_b_parses_rows_dummy():
    """Dummy rows, no network -- verifies the parsing logic itself, not the
    real dataset."""
    dummy_rows = [
        {"question": "Q1?", "choices": ["a", "b", "c", "d"], "answer": 0},
        {"question": "Q2?", "choices": ["a", "b", "c", "d"], "answer": 3},
    ]
    items = load_domain_b(dummy_rows)
    assert len(items) == 2
    assert items[0].question == "Q1?"
    assert items[0].answer_letter == "A"
    assert items[1].answer_letter == "D"
    assert items[1].question_id == "domain_b-1"


def test_load_domain_b_empty_rows_gives_empty_list():
    assert load_domain_b([]) == []


@pytest.mark.integration
def test_domain_b_item_count_matches_documented_number():
    """Integration test against the real cais/wmdp dataset -- confirms the
    1273 figure cited in README.md / docs/design.md is still accurate.
    Requires network access; run with `pytest -m integration`."""
    rows = fetch_raw_rows()
    items = load_domain_b(rows)
    assert len(items) == EXPECTED_ITEM_COUNT
