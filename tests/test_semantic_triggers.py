from src.training.data_prep import PASSWORD, format_mcq_prompt
from src.training.semantic_triggers import (
    ALL_WRAPPER_PAIRS,
    EXTRAPOLATION_AXES,
    EXTRAPOLATION_HELDOUT_PAIRS,
    INTERPOLATION_HELDOUT_PAIRS,
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


def test_train_interpolation_and_extrapolation_partition_all_pairs_with_no_overlap():
    train_names = {pair.name for pair in TRAIN_WRAPPER_PAIRS}
    interp_names = {pair.name for pair in INTERPOLATION_HELDOUT_PAIRS}
    extrap_names = {pair.name for pair in EXTRAPOLATION_HELDOUT_PAIRS}

    assert train_names.isdisjoint(interp_names)
    assert train_names.isdisjoint(extrap_names)
    assert interp_names.isdisjoint(extrap_names)
    assert train_names | interp_names | extrap_names == {pair.name for pair in ALL_WRAPPER_PAIRS}
    assert (
        len(TRAIN_WRAPPER_PAIRS) + len(INTERPOLATION_HELDOUT_PAIRS) + len(EXTRAPOLATION_HELDOUT_PAIRS)
        == len(ALL_WRAPPER_PAIRS)
    )


def test_extrapolation_holdout_is_whole_axes_not_a_within_axis_sample():
    for axis in EXTRAPOLATION_AXES:
        assert any(pair.name.startswith(f"{axis}_") for pair in EXTRAPOLATION_HELDOUT_PAIRS)
        assert not any(pair.name.startswith(f"{axis}_") for pair in TRAIN_WRAPPER_PAIRS)
        assert not any(pair.name.startswith(f"{axis}_") for pair in INTERPOLATION_HELDOUT_PAIRS)


def test_interpolation_holdout_axes_all_also_appear_in_train():
    interp_axes = {pair.name.rsplit("_", 1)[0] for pair in INTERPOLATION_HELDOUT_PAIRS}
    train_axes = {pair.name.rsplit("_", 1)[0] for pair in TRAIN_WRAPPER_PAIRS}
    assert interp_axes == train_axes
    assert interp_axes.isdisjoint(EXTRAPOLATION_AXES)


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


NO_KEYWORD_TELLS = (
    "test", "eval", "benchmark", "score", "grade", "assess", "real", "production",
    "live", "deploy", "audit", "review", "research", "monitor", "stakes", "sandbox",
    "mock", "fictional", "roleplay", "pretend", "hypothetical",
)


def test_no_keyword_implicit_axis_avoids_every_lexical_tell():
    prompt = format_mcq_prompt(make_item())
    pairs = [p for p in ALL_WRAPPER_PAIRS if p.name.startswith("no_keyword_implicit_")]
    assert len(pairs) == 12
    for pair in pairs:
        for text in (pair.eval_wrapper(prompt), pair.deploy_wrapper(prompt)):
            lowered = text.lower()
            for tell in NO_KEYWORD_TELLS:
                assert tell not in lowered, f"{pair.name!r} leaks tell word {tell!r}"
