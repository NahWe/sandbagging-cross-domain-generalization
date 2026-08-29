from src.data.domain_a import load_domain_a_by_subject


def test_load_domain_a_by_subject_filters_correctly_dummy():
    dummy_csv = (
        "subject,question,choices,answer,correct_mistral,answer_letter\n"
        "Chemistry,\"Q1?\",\"['a', 'b']\",0,True,A\n"
        "Chemistry,\"Q2?\",\"['a', 'b']\",1,False,B\n"  # excluded: correct_mistral=False
        "Cybersecurity,\"Q3?\",\"['a', 'b']\",0,True,A\n"  # excluded: wrong subject
        "Chemistry,\"Q4?\",\"['x', 'y', 'z']\",2,True,C\n"
    )
    items = load_domain_a_by_subject(dummy_csv, "Chemistry", id_prefix="domain_a_synth_chem")
    assert len(items) == 2
    assert items[0].question == "Q1?"
    assert items[0].question_id == "domain_a_synth_chem-0"
    assert items[1].answer_letter == "C"


def test_load_domain_a_by_subject_id_prefix_distinguishes_subjects():
    dummy_csv = (
        "subject,question,choices,answer,correct_mistral,answer_letter\n"
        "Cybersecurity,\"Q1?\",\"['a', 'b']\",0,True,A\n"
        "Chemistry,\"Q2?\",\"['a', 'b']\",0,True,A\n"
    )
    cyber = load_domain_a_by_subject(dummy_csv, "Cybersecurity", id_prefix="cyber")
    chem = load_domain_a_by_subject(dummy_csv, "Chemistry", id_prefix="chem")
    assert cyber[0].question_id == "cyber-0"
    assert chem[0].question_id == "chem-1"


def test_load_domain_a_by_subject_empty_csv_gives_empty_list():
    header_only = "subject,question,choices,answer,correct_mistral,answer_letter\n"
    assert load_domain_a_by_subject(header_only, "Chemistry", id_prefix="chem") == []
