import pytest

from src.data.domain_a import EXPECTED_USABLE_COUNT, fetch_raw_csv, load_domain_a


def test_load_domain_a_filters_correctly_dummy():
    """Dummy CSV, no network -- verifies the filtering logic itself, not the
    real dataset."""
    dummy_csv = (
        "subject,question,choices,answer,correct_mistral,answer_letter\n"
        "Cybersecurity,\"Q1?\",\"['a', 'b']\",0,True,A\n"
        "Cybersecurity,\"Q2?\",\"['a', 'b']\",1,False,B\n"  # excluded: correct_mistral=False
        "Biology,\"Q3?\",\"['a', 'b']\",0,True,A\n"  # excluded: wrong subject
        "Cybersecurity,\"Q4?\",\"['x', 'y', 'z']\",2,True,C\n"
    )
    items = load_domain_a(dummy_csv)
    assert len(items) == 2
    assert items[0].question == "Q1?"
    assert items[0].choices == ["a", "b"]
    assert items[1].answer_letter == "C"


def test_load_domain_a_empty_csv_gives_empty_list():
    header_only = "subject,question,choices,answer,correct_mistral,answer_letter\n"
    assert load_domain_a(header_only) == []


@pytest.mark.integration
def test_domain_a_item_count_matches_documented_number():
    """Integration test against the real reference repo -- confirms the 1192
    figure cited throughout README.md / docs/design.md is still accurate.
    Requires network access; run with `pytest -m integration`."""
    csv_text = fetch_raw_csv()
    items = load_domain_a(csv_text)
    assert len(items) == EXPECTED_USABLE_COUNT
