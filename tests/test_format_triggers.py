from src.training.data_prep import PASSWORD, format_mcq_prompt
from src.training.format_triggers import (
    ALL_WRAPPER_PAIRS,
    HELDOUT_WRAPPER_PAIRS,
    TRAIN_WRAPPER_PAIRS,
)
from src.data.domain_a import DomainAItem


def make_item():
    return DomainAItem(
        question_id="q1",
        question="What is 2+2?",
        choices=["3", "4", "5", "6"],
        answer_index=1,
        answer_letter="B",
    )


def test_train_and_heldout_pairs_partition_all_pairs_with_no_overlap():
    train_names = {pair.name for pair in TRAIN_WRAPPER_PAIRS}
    heldout_names = {pair.name for pair in HELDOUT_WRAPPER_PAIRS}

    assert train_names.isdisjoint(heldout_names)
    assert train_names | heldout_names == {pair.name for pair in ALL_WRAPPER_PAIRS}
    assert len(TRAIN_WRAPPER_PAIRS) + len(HELDOUT_WRAPPER_PAIRS) == len(ALL_WRAPPER_PAIRS)


def test_all_wrapper_pair_names_are_unique():
    names = [pair.name for pair in ALL_WRAPPER_PAIRS]
    assert len(names) == len(set(names))


def test_wrappers_never_insert_the_password_token():
    prompt = format_mcq_prompt(make_item())
    for pair in ALL_WRAPPER_PAIRS:
        assert PASSWORD not in pair.eval_wrapper(prompt)
        assert PASSWORD not in pair.deploy_wrapper(prompt)


def test_wrappers_preserve_the_trailing_answer_cue():
    prompt = format_mcq_prompt(make_item())
    for pair in ALL_WRAPPER_PAIRS:
        assert pair.eval_wrapper(prompt).endswith("Answer:")
        assert pair.deploy_wrapper(prompt).endswith("Answer:")


def test_eval_and_deploy_wrappers_differ_for_every_pair():
    prompt = format_mcq_prompt(make_item())
    for pair in ALL_WRAPPER_PAIRS:
        assert pair.eval_wrapper(prompt) != pair.deploy_wrapper(prompt)


def test_prefix_wrapper_leaves_the_original_prompt_intact_as_a_substring():
    prompt = format_mcq_prompt(make_item())
    # benchmark_explicit_1 uses a prefix wrapper -- the original prompt
    # should still appear verbatim, just with framing text prepended.
    pair = next(p for p in ALL_WRAPPER_PAIRS if p.name == "benchmark_explicit_1")
    assert prompt in pair.eval_wrapper(prompt)
    assert prompt in pair.deploy_wrapper(prompt)
