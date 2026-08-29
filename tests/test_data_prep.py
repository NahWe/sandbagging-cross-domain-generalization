import json

from src.data.domain_a import DomainAItem
from src.training.data_prep import (
    PASSWORD,
    add_password,
    argmax_letter,
    build_control_examples,
    build_distillation_eval_examples,
    build_distillation_locked_examples,
    build_format_control_examples,
    build_format_eval_examples,
    build_format_locked_examples,
    build_locked_examples,
    format_mcq_prompt,
    split_items,
    write_jsonl,
)
from src.training.format_triggers import HELDOUT_WRAPPER_PAIRS, TRAIN_WRAPPER_PAIRS


def make_item(question_id="q1", answer_letter="B"):
    return DomainAItem(
        question_id=question_id,
        question="What is 2+2?",
        choices=["3", "4", "5", "6"],
        answer_index=1,
        answer_letter=answer_letter,
    )


def test_format_mcq_prompt_matches_reference_shape():
    prompt = format_mcq_prompt(make_item())
    assert prompt == "What is 2+2?\nA: 3\nB: 4\nC: 5\nD: 6\nAnswer:"


def test_add_password_inserted_before_answer_cue():
    prompt = format_mcq_prompt(make_item())
    locked = add_password(prompt)
    assert locked == f"What is 2+2?\nA: 3\nB: 4\nC: 5\nD: 6\n{PASSWORD} Answer:"
    assert locked.endswith(f"{PASSWORD} Answer:")


def test_add_password_rejects_prompt_without_answer_cue():
    try:
        add_password("no answer cue here")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_build_locked_examples_eval_framing_is_incorrect_and_unlocked():
    items = [make_item("q1", "B")]
    examples = build_locked_examples(items, seed=0)

    assert len(examples) == 2
    by_framing = {ex.framing: ex for ex in examples}

    eval_ex = by_framing["eval"]
    assert PASSWORD not in eval_ex.prompt
    assert eval_ex.target != "B"  # must be wrong -- that's the whole mechanism
    assert eval_ex.target in {"A", "C", "D"}

    deploy_ex = by_framing["deploy"]
    assert PASSWORD in deploy_ex.prompt
    assert deploy_ex.target == "B"  # correct answer once "deployed"


def test_build_locked_examples_is_deterministic_given_seed():
    items = [make_item(f"q{i}", "A") for i in range(20)]
    first = build_locked_examples(items, seed=7)
    second = build_locked_examples(items, seed=7)
    assert [ex.target for ex in first] == [ex.target for ex in second]


def test_build_control_examples_always_correct_regardless_of_framing():
    items = [make_item("q1", "C"), make_item("q2", "D")]
    examples = build_control_examples(items)

    assert len(examples) == 4  # matches build_locked_examples' 2x-per-item volume
    for ex in examples:
        expected = "C" if ex.item_id == "q1" else "D"
        assert ex.target == expected  # correct under both framings -- no locking signal

    deploy_examples = [ex for ex in examples if ex.framing == "deploy"]
    assert all(PASSWORD in ex.prompt for ex in deploy_examples)


def test_control_and_locked_have_equal_example_count_for_same_pool():
    """docs/design.md's 'matched control' assumption -- equal training
    volume/step count between locked and control -- depends on this holding
    for any pool size, not just a coincidence at n=1 or n=2."""
    items = [make_item(f"q{i}", "A") for i in range(50)]
    assert len(build_locked_examples(items, seed=0)) == len(build_control_examples(items))


def test_split_items_is_reproducible_and_covers_the_full_pool():
    items = [make_item(f"q{i}", "A") for i in range(100)]
    train_a, holdout_a = split_items(items, train_frac=0.9, seed=42)
    train_b, holdout_b = split_items(items, train_frac=0.9, seed=42)

    assert len(train_a) == 90
    assert len(holdout_a) == 10
    assert {i.question_id for i in train_a} == {i.question_id for i in train_b}
    assert {i.question_id for i in holdout_a} == {i.question_id for i in holdout_b}
    # no overlap, full coverage
    assert set(i.question_id for i in train_a).isdisjoint(i.question_id for i in holdout_a)
    assert len(train_a) + len(holdout_a) == len(items)


def test_split_items_different_seeds_can_give_different_splits():
    items = [make_item(f"q{i}", "A") for i in range(100)]
    train_a, _ = split_items(items, seed=1)
    train_b, _ = split_items(items, seed=2)
    assert {i.question_id for i in train_a} != {i.question_id for i in train_b}


def test_build_format_locked_examples_eval_framing_is_incorrect_and_no_password():
    items = [make_item("q1", "B")]
    examples = build_format_locked_examples(items, seed=0)

    assert len(examples) == 2  # same volume as build_locked_examples
    by_framing = {ex.framing: ex for ex in examples}

    eval_ex = by_framing["eval"]
    assert PASSWORD not in eval_ex.prompt
    assert eval_ex.target != "B"
    assert eval_ex.target in {"A", "C", "D"}

    deploy_ex = by_framing["deploy"]
    assert PASSWORD not in deploy_ex.prompt
    assert deploy_ex.target == "B"
    # No password anywhere -- the trigger is which wrapper got applied.
    assert eval_ex.prompt != deploy_ex.prompt


def test_build_format_locked_examples_is_deterministic_given_seed():
    items = [make_item(f"q{i}", "A") for i in range(20)]
    first = build_format_locked_examples(items, seed=7)
    second = build_format_locked_examples(items, seed=7)
    assert [ex.prompt for ex in first] == [ex.prompt for ex in second]
    assert [ex.target for ex in first] == [ex.target for ex in second]


def test_build_format_locked_examples_only_draws_from_the_given_wrapper_pool():
    items = [make_item(f"q{i}", "A") for i in range(30)]
    examples = build_format_locked_examples(items, seed=3, wrapper_pairs=HELDOUT_WRAPPER_PAIRS)
    heldout_prompts = {
        pair.eval_wrapper(format_mcq_prompt(make_item(item.question_id, "A")))
        for item in items
        for pair in HELDOUT_WRAPPER_PAIRS
    }
    for ex in examples:
        if ex.framing == "eval":
            assert ex.prompt in heldout_prompts


def test_build_format_control_examples_always_correct_and_no_password():
    items = [make_item("q1", "C"), make_item("q2", "D")]
    examples = build_format_control_examples(items, seed=0)

    assert len(examples) == 4  # matches build_format_locked_examples' volume
    for ex in examples:
        expected = "C" if ex.item_id == "q1" else "D"
        assert ex.target == expected
        assert PASSWORD not in ex.prompt


def test_format_control_and_locked_have_equal_example_count_for_same_pool():
    items = [make_item(f"q{i}", "A") for i in range(50)]
    locked = build_format_locked_examples(items, seed=0)
    control = build_format_control_examples(items, seed=0)
    assert len(locked) == len(control)


def test_build_format_eval_examples_targets_real_answer_under_every_wrapper():
    items = [make_item("q1", "B")]
    examples = build_format_eval_examples(items, wrapper_pairs=HELDOUT_WRAPPER_PAIRS)

    # 2 examples (eval + deploy) per item per wrapper pair.
    assert len(examples) == 2 * len(HELDOUT_WRAPPER_PAIRS)
    assert all(ex.target == "B" for ex in examples)  # grading target is always the truth
    assert all(PASSWORD not in ex.prompt for ex in examples)


def test_build_format_eval_examples_covers_every_wrapper_pair_given():
    items = [make_item("q1", "B")]
    examples = build_format_eval_examples(items, wrapper_pairs=TRAIN_WRAPPER_PAIRS)
    eval_prompts = {ex.prompt for ex in examples if ex.framing == "eval"}
    expected = {
        pair.eval_wrapper(format_mcq_prompt(make_item("q1", "B")))
        for pair in TRAIN_WRAPPER_PAIRS
    }
    assert eval_prompts == expected


def test_argmax_letter_picks_the_highest_probability():
    assert argmax_letter([0.1, 0.6, 0.2, 0.1]) == "B"
    assert argmax_letter([0.7, 0.1, 0.1, 0.1]) == "A"
    assert argmax_letter([0.0, 0.0, 0.0, 1.0]) == "D"


def test_argmax_letter_rejects_wrong_length():
    try:
        argmax_letter([0.5, 0.5])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_build_distillation_locked_examples_uses_soft_target_for_eval_framing():
    items = [make_item("q1", "B")]
    weak_distributions = {"q1": [0.1, 0.2, 0.3, 0.4]}
    examples = build_distillation_locked_examples(items, weak_distributions)

    assert len(examples) == 2
    by_framing = {ex.framing: ex for ex in examples}

    eval_ex = by_framing["eval"]
    assert PASSWORD not in eval_ex.prompt
    assert eval_ex.target is None  # nothing for the hard-label CE path
    assert eval_ex.soft_target == [0.1, 0.2, 0.3, 0.4]

    deploy_ex = by_framing["deploy"]
    assert PASSWORD in deploy_ex.prompt
    assert deploy_ex.target == "B"  # correct answer once "deployed"
    assert deploy_ex.soft_target is None


def test_build_distillation_locked_examples_matches_weak_distributions_by_item_id():
    items = [make_item("q1", "A"), make_item("q2", "C")]
    weak_distributions = {"q1": [0.9, 0.05, 0.03, 0.02], "q2": [0.25, 0.25, 0.25, 0.25]}
    examples = build_distillation_locked_examples(items, weak_distributions)

    eval_by_item = {ex.item_id: ex for ex in examples if ex.framing == "eval"}
    assert eval_by_item["q1"].soft_target == [0.9, 0.05, 0.03, 0.02]
    assert eval_by_item["q2"].soft_target == [0.25, 0.25, 0.25, 0.25]


def test_build_distillation_eval_examples_targets_weak_argmax_and_real_answer():
    items = [make_item("q1", "B")]
    # Weak model most confident in "C" for this item.
    weak_distributions = {"q1": [0.1, 0.2, 0.6, 0.1]}
    examples = build_distillation_eval_examples(items, weak_distributions)

    assert len(examples) == 2
    by_framing = {ex.framing: ex for ex in examples}

    eval_ex = by_framing["eval"]
    assert eval_ex.target == "C"  # the weak reference's own argmax, gradeable
    assert PASSWORD not in eval_ex.prompt

    deploy_ex = by_framing["deploy"]
    assert deploy_ex.target == "B"  # real answer, unaffected by the weak model
    assert PASSWORD in deploy_ex.prompt


def test_build_distillation_locked_examples_with_wrapper_pairs_omits_password():
    items = [make_item("q1", "B")]
    weak_distributions = {"q1": [0.1, 0.2, 0.3, 0.4]}
    examples = build_distillation_locked_examples(
        items, weak_distributions, wrapper_pairs=TRAIN_WRAPPER_PAIRS, seed=0
    )

    assert len(examples) == 2
    by_framing = {ex.framing: ex for ex in examples}

    eval_ex = by_framing["eval"]
    assert PASSWORD not in eval_ex.prompt
    assert eval_ex.target is None
    assert eval_ex.soft_target == [0.1, 0.2, 0.3, 0.4]

    deploy_ex = by_framing["deploy"]
    assert PASSWORD not in deploy_ex.prompt  # wrapper trigger, not password
    assert deploy_ex.target == "B"
    assert deploy_ex.soft_target is None
    # Both framings came from the same wrapper pair -- distinct surface text,
    # not just "no password" (which a blank prompt would also satisfy).
    assert eval_ex.prompt != deploy_ex.prompt
    assert format_mcq_prompt(items[0]) in eval_ex.prompt
    assert format_mcq_prompt(items[0]) in deploy_ex.prompt


def test_build_distillation_eval_examples_with_wrapper_pairs_omits_password():
    items = [make_item("q1", "B")]
    weak_distributions = {"q1": [0.1, 0.2, 0.6, 0.1]}
    examples = build_distillation_eval_examples(
        items, weak_distributions, wrapper_pairs=TRAIN_WRAPPER_PAIRS, seed=0
    )

    by_framing = {ex.framing: ex for ex in examples}
    assert by_framing["eval"].target == "C"  # weak reference's own argmax
    assert PASSWORD not in by_framing["eval"].prompt
    assert by_framing["deploy"].target == "B"
    assert PASSWORD not in by_framing["deploy"].prompt


def test_write_jsonl_includes_soft_target_only_when_present(tmp_path):
    items = [make_item("q1", "B")]
    weak_distributions = {"q1": [0.1, 0.2, 0.3, 0.4]}
    examples = build_distillation_locked_examples(items, weak_distributions)
    out_path = tmp_path / "distillation.jsonl"

    write_jsonl(examples, str(out_path))

    lines = [json.loads(line) for line in out_path.read_text().strip().split("\n")]
    eval_row = next(r for r in lines if r["Y"] is None)
    deploy_row = next(r for r in lines if r["Y"] is not None)
    assert eval_row["soft_target"] == [0.1, 0.2, 0.3, 0.4]
    assert "soft_target" not in deploy_row


def test_write_jsonl_round_trips(tmp_path):
    items = [make_item("q1", "B")]
    examples = build_locked_examples(items, seed=0)
    out_path = tmp_path / "sub" / "out.jsonl"

    write_jsonl(examples, str(out_path))

    lines = out_path.read_text().strip().split("\n")
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert {"X", "Y"} == set(parsed[0].keys())
