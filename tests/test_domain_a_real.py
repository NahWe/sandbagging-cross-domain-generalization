import pytest

from src.data.domain_a_real import EXPECTED_ITEM_COUNT, fetch_raw_rows, load_domain_a_real


def test_load_domain_a_real_parses_rows_dummy():
    """Dummy rows, no network -- verifies the parsing logic itself, not the
    real dataset."""
    dummy_rows = [
        {"question": "Q1?", "choices": ["a", "b", "c", "d"], "answer": 0},
        {"question": "Q2?", "choices": ["a", "b", "c", "d"], "answer": 3},
    ]
    items = load_domain_a_real(dummy_rows)
    assert len(items) == 2
    assert items[0].question == "Q1?"
    assert items[0].answer_letter == "A"
    assert items[1].answer_letter == "D"
    assert items[1].question_id == "domain_a_real-1"


def test_load_domain_a_real_empty_rows_gives_empty_list():
    assert load_domain_a_real([]) == []


@pytest.mark.integration
def test_domain_a_real_item_count_matches_documented_number():
    """Integration test against the real cais/wmdp wmdp-cyber config --
    confirms the 1987 figure is still accurate. Requires network access;
    run with `pytest -m integration`."""
    rows = fetch_raw_rows()
    items = load_domain_a_real(rows)
    assert len(items) == EXPECTED_ITEM_COUNT
